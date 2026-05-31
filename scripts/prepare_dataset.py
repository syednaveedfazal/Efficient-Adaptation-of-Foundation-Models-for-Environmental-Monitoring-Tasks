"""
Syed's Task: Process HLS Burn Scars dataset and create nested label splits.

What this script does
---------------------
1. Scans data/raw/hls_burn_scars/{training,validation} for paired (image, mask) GeoTIFFs.
2. Normalises images (per-band statistics from the training set, ignoring nodata/cloud pixels).
3. Creates nested label-budget splits: 1%, 5%, 10%, 25%, 50%, 100% of training labels.
   Each split is saved as a JSON file listing the scene IDs — not copies of the data.
4. Fixes DataLoader seeds for full reproducibility (3 seeds: 42, 123, 456).

Usage:
    python scripts/prepare_dataset.py \
        --raw_dir   data/raw/hls_burn_scars \
        --out_dir   data/processed \
        --split_dir data/splits
"""

import argparse
import json
import os
import random
from pathlib import Path

# Label budget fractions to create
LABEL_BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = [42, 123, 456]


def scan_scenes(split_dir: Path):
    """Return sorted list of base scene IDs that have both image and mask."""
    scenes = []
    # In the dataset, images end in _merged.tif
    for f in sorted(split_dir.glob("*_merged.tif")):
        base_id = f.name.replace("_merged.tif", "")
        # Corresponding masks end in .mask.tif
        mask = f.with_name(f"{base_id}.mask.tif")
        if mask.exists():
            scenes.append(base_id)
    return scenes


def compute_band_stats(scenes: list, raw_dir: Path):
    """Compute per-band mean and std over training scenes, excluding nodata and cloud pixels."""
    try:
        import numpy as np
        import rasterio
    except ImportError:
        raise ImportError(
            "rasterio and numpy are required to compute dataset statistics. "
            "Please run: pip install rasterio numpy"
        )

    if not scenes:
        print("  [warning] No scenes provided to compute stats.")
        return None

    print("  Computing per-band statistics (excluding nodata/masked pixels) ...")
    sums = None
    sum_sq = None
    count = 0

    for base_id in scenes:
        tif = raw_dir / "training" / f"{base_id}_merged.tif"
        mask_tif = raw_dir / "training" / f"{base_id}.mask.tif"
        
        with rasterio.open(tif) as src, rasterio.open(mask_tif) as m_src:
            data = src.read().astype("float32")  # (bands, H, W)
            mask = m_src.read(1)                # (H, W)
            
            # Identify valid pixels. 
            # Per dataset metadata, mask value -1 indicates missing data/clouds.
            valid_mask = (mask != -1)
            
            # Additionally, check the image metadata for system-level nodata values
            if src.nodata is not None:
                valid_mask &= (data != src.nodata).all(axis=0)
                
            # Flatten only the valid (non-masked) pixels
            valid_data = data[:, valid_mask]  # Shape: (bands, N_valid_pixels)
            n_valid = valid_data.shape[1]
            
            if n_valid == 0:
                continue
                
            if sums is None:
                sums = np.zeros(data.shape[0])
                sum_sq = np.zeros(data.shape[0])
                
            sums += valid_data.sum(axis=1)
            sum_sq += (valid_data ** 2).sum(axis=1)
            count += n_valid

    if count == 0 or sums is None:
        print("  [warning] No valid pixels found across any scenes.")
        return None

    means = (sums / count).tolist()
    
    # Calculate variance and use np.clip to prevent negative variance values 
    # resulting from infinitesimal floating-point precision imprecisions.
    variance = (sum_sq / count) - (sums / count) ** 2
    variance = np.clip(variance, a_min=0.0, a_max=None)
    stds = (variance ** 0.5).tolist()
    
    return {"mean": means, "std": stds, "n_pixels": count}


def create_splits(scenes: list, split_dir: Path):
    """Create nested label-budget JSON files, one folder per seed."""
    for seed in SEEDS:
        rng = random.Random(seed)
        shuffled = scenes.copy()
        rng.shuffle(shuffled)

        seed_dir = split_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for budget in LABEL_BUDGETS:
            n = max(1, int(len(scenes) * budget))
            subset = shuffled[:n]
            
            # Using round() prevents float truncation issues (e.g. 0.29 * 100 becoming 28)
            label = f"split_{round(budget * 100):03d}pct"
            out = seed_dir / f"{label}.json"
            with open(out, "w") as f:
                json.dump(
                    {
                        "seed": seed,
                        "budget": budget,
                        "n_scenes": n,
                        "scenes": subset,
                    },
                    f,
                    indent=2,
                )
            print(f"    seed={seed}  budget={round(budget*100):3d}%  → {n:4d} scenes  [{out.name}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",   default="data/raw/hls_burn_scars")
    parser.add_argument("--out_dir",   default="data/processed")
    parser.add_argument("--split_dir", default="data/splits")
    args = parser.parse_args()

    raw_dir   = Path(args.raw_dir)
    out_dir   = Path(args.out_dir)
    split_dir = Path(args.split_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_dir = raw_dir / "training"
    val_dir   = raw_dir / "validation"
    
    if not train_dir.exists():
        raise SystemExit(
            f"\n[ERROR] Cannot find training directory: {train_dir}\n"
            "  → Run `python scripts/download_data.py` first to download the dataset."
        )
        
    if not val_dir.exists():
        raise SystemExit(
            f"\n[ERROR] Cannot find validation directory: {val_dir}\n"
            "  → Please check if your dataset download or extraction was completed fully."
        )

    print(f"Scanning training scenes in {train_dir} ...")
    train_scenes = scan_scenes(train_dir)
    val_scenes   = scan_scenes(val_dir)
    print(f"  Found {len(train_scenes)} training scenes, {len(val_scenes)} validation scenes")

    if not train_scenes or not val_scenes:
        raise SystemExit(
            "\n[ERROR] Identified 0 scenes in either the training or validation directories.\n"
            "  → Verify that your files are extracted directly inside training/ and validation/."
        )

    # Band statistics
    stats = compute_band_stats(train_scenes, raw_dir)
    if stats:
        stats_path = out_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Band stats saved → {stats_path}")

    # Label splits
    print("\nCreating nested label-budget splits ...")
    create_splits(train_scenes, split_dir)

    # Save val list
    with open(split_dir / "val_scenes.json", "w") as f:
        json.dump({"n_scenes": len(val_scenes), "scenes": val_scenes}, f, indent=2)

    print(f"\nDone! Splits written to {split_dir}/")


if __name__ == "__main__":
    main()