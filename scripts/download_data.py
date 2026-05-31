"""
Download the HLS Burn Scars dataset from Hugging Face.

Dataset: ibm-nasa-geospatial/hls_burn_scars
  - 804 paired scenes (HLS L30 / S30 imagery + burn-scar masks)
  - 6 spectral bands: Blue, Green, Red, NIR, SWIR-1, SWIR-2
  - 512 x 512 px tiles, GeoTIFF format
  - ~10 GB total (uncompressed)

Usage:
    python scripts/download_data.py --output_dir data/raw
"""

import argparse
import os
import tarfile
from pathlib import Path


def download_burn_scars(output_dir: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Run: pip install huggingface_hub")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading HLS Burn Scars dataset ...")
    print("This is ~2.65 GB compressed — grab a coffee.\n")

    # Using the canonical repo_id
    path = snapshot_download(
        repo_id="ibm-nasa-geospatial/hls_burn_scars",
        repo_type="dataset",
        local_dir=str(output_dir / "hls_burn_scars"),
        # token=os.environ.get("HF_TOKEN"),
    )
    
    path_dir = Path(path)
    tar_path = path_dir / "hls_burn_scars.tar.gz"

    if tar_path.exists():
        print(f"\nDownloaded tarball found at: {tar_path}")
        print("Extracting dataset files (this may take a couple of minutes) ...")
        
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                # Extracting directly into the local dataset directory
                tar.extractall(path=path_dir)
            print("Extraction complete!")
            
            # Optional: Remove the tarball to save ~2.65 GB of space
            print("Cleaning up tarball archive ...")
            tar_path.unlink()
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            print("You may need to extract it manually using:")
            print(f"tar -xvzf {tar_path} -C {path_dir}")
    else:
        print("\nWarning: hls_burn_scars.tar.gz was not found in the downloaded snapshot.")

    print(f"\nDataset directory: {path_dir.resolve()}")
    print("Expected layout:")
    print("  ├── training/")
    print("  │   ├── <scene_id>.tif   (6-band input)")
    print("  │   └── <scene_id>_mask.tif")
    print("  └── validation/")
    print("      ├── <scene_id>.tif")
    print("      └── <scene_id>_mask.tif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/raw", help="Where to save the raw dataset")
    args = parser.parse_args()
    download_burn_scars(args.output_dir)