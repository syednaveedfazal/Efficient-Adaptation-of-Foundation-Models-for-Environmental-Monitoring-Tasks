"""
Download the HLS Burn Scars dataset from Hugging Face.

Dataset: nasa-impact/hls_burn_scars
  - 804 paired scenes (HLS L30 / S30 imagery + burn-scar masks)
  - 6 spectral bands: Blue, Green, Red, NIR, SWIR-1, SWIR-2
  - 512 x 512 px tiles, GeoTIFF format
  - ~10 GB total

Usage:
    python scripts/download_data.py --output_dir data/raw
"""

import argparse
import os
from pathlib import Path


def download_burn_scars(output_dir: str):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Run: pip install huggingface_hub")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading HLS Burn Scars dataset …")
    print("This is ~10 GB — grab a coffee.\n")

    path = snapshot_download(
        repo_id="nasa-impact/hls_burn_scars",
        repo_type="dataset",
        local_dir=str(output_dir / "hls_burn_scars"),
        # Uncomment to use your HF token if the dataset requires login:
        # token=os.environ.get("HF_TOKEN"),
    )
    print(f"\nDataset saved to: {path}")
    print("Expected layout:")
    print("  data/raw/hls_burn_scars/")
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
