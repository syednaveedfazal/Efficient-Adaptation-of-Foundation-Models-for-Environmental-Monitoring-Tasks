"""
Download pretrained model weights for the P29 project.

Models
------
1. Prithvi-EO-2.0-300M  (IBM / NASA IMPACT)
   HuggingFace: ibm-nasa-geospatial/Prithvi-EO-2.0-300M
   Size: ~1.2 GB

2. DINOv2-Base (Meta AI) — generic ViT baseline
   HuggingFace: facebook/dinov2-base
   Size: ~350 MB

Usage:
    python scripts/download_models.py --output_dir models/pretrained
"""

import argparse
import os
from pathlib import Path


MODELS = {
    "prithvi": {
        "repo_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        "repo_type": "model",
        "description": "Prithvi-EO-2.0 300M — geospatial foundation model (Syed's primary backbone)",
    },
    "dinov2": {
        "repo_id": "facebook/dinov2-large",
        "repo_type": "model",
        "description": "DINOv2-Large (ViT-L/14, ~300M) — generic ViT baseline, matches Prithvi-300M scale",
    },
}


def download_model(name: str, info: dict, output_dir: Path, cache_dir: Path = None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Run: pip install huggingface_hub")

    dest = output_dir / name
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n[{name}] {info['description']}")
    print(f"  Repo : {info['repo_id']}")
    print(f"  Dest : {dest}")

    hf_cache = str(cache_dir) if cache_dir else None
    if hf_cache:
        print(f"  Cache: {hf_cache}")

    # Downloads the snapshot. The 'token' parameter will evaluate to None 
    # if HF_TOKEN is not in the environment, which is handled gracefully.
    path = snapshot_download(
        repo_id=info["repo_id"],
        repo_type=info["repo_type"],
        local_dir=str(dest),
        cache_dir=hf_cache,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  ✓ Saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="models/pretrained")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Which model to download (default: all)",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Custom cache directory for downloads (defaults to output_dir/.cache to protect HPC home directory quota)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If no custom cache directory is specified, default to a '.cache' subdirectory 
    # within the designated model output folder. This keeps cache files out of your 
    # home directory (~/.cache/huggingface) to avoid triggering cluster quota limits.
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    targets = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    for name, info in targets.items():
        download_model(name, info, output_dir, cache_dir)

    print("\nAll done. Expected layout:")
    print("  models/pretrained/")
    print("  ├── prithvi/   ← ibm-nasa-geospatial/Prithvi-EO-2.0-300M")
    print("  └── dinov2/    ← facebook/dinov2-base")


if __name__ == "__main__":
    main()