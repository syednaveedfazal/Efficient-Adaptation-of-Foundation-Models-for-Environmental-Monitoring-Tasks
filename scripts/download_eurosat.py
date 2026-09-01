"""
scripts/download_eurosat.py — Download EuroSAT-MS (13-band Sentinel-2) tiles.

Fetches the canonical EuroSATallBands.zip (27,000 tiles, 10 classes, 64x64,
13 bands) and extracts it into the class-directory layout the loader expects:

    data/raw/eurosat/
        AnnualCrop/AnnualCrop_1.tif ...
        Forest/Forest_1.tif ...
        ... (10 class dirs)

Primary source: HuggingFace dataset `torchgeo/eurosat` (reliable mirror).
Fallback:       --url http://madm.dfki.de/files/sentinel/EuroSATallBands.zip

Usage:
    python scripts/download_eurosat.py --output_dir data/raw
    python scripts/download_eurosat.py --url <zip-url>        # explicit source

Note: ~2 GB compressed, 27k tiles. The NFS home has no inode limit set (checked),
so loose extraction is fine; if that changes, re-pack into shards.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

# Inside EuroSATallBands.zip the tifs live under this prefix:
_ZIP_TIF_PREFIX = "ds/images/remote_sensing/otherDatasets/sentinel_2/tif"


def _fetch_zip(output_dir: Path, url: str | None) -> Path:
    """Return a local path to EuroSATallBands.zip, downloading if needed."""
    if url:
        import urllib.request
        dest = output_dir / "EuroSATallBands.zip"
        print(f"Downloading EuroSAT-MS from {url}\n(~2 GB — grab a coffee)")
        urllib.request.urlretrieve(url, dest)
        return dest
    # Default: HuggingFace mirror
    from huggingface_hub import hf_hub_download
    print("Downloading EuroSAT-MS from HuggingFace torchgeo/eurosat (~2 GB) ...")
    return Path(hf_hub_download(
        repo_id="torchgeo/eurosat",
        filename="EuroSATallBands.zip",
        repo_type="dataset",
    ))


def download_eurosat(output_dir: str, url: str | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_root = output_dir / "eurosat"
    dest_root.mkdir(parents=True, exist_ok=True)

    zip_path = _fetch_zip(output_dir, url)

    print(f"Extracting {zip_path.name} → {dest_root} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist()
                   if m.endswith(".tif") and _ZIP_TIF_PREFIX in m]
        for m in members:
            # .../tif/<Class>/<Class>_N.tif  ->  eurosat/<Class>/<Class>_N.tif
            rel = m.split(_ZIP_TIF_PREFIX + "/", 1)[1]      # "<Class>/<file>.tif"
            out = dest_root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
    n = sum(1 for _ in dest_root.glob("*/*.tif"))
    print(f"Extraction complete: {n} tiles under {dest_root}/")

    # Free the ~2 GB archive if it was downloaded into output_dir
    if url and zip_path.exists():
        zip_path.unlink(); print("Removed archive to save space.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="data/raw")
    ap.add_argument("--url", default=None,
                    help="Explicit zip URL (else HuggingFace torchgeo/eurosat).")
    args = ap.parse_args()
    download_eurosat(args.output_dir, args.url)


if __name__ == "__main__":
    main()
