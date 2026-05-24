"""
Syed's Task: Process HLS Burn Scars dataset and create nested label splits.

What this script does
---------------------
1. Scans data/raw/hls_burn_scars/{training,validation} for paired (image, mask) GeoTIFFs.
2. Normalises images (per-band statistics from the training set).
3. Creates nested label-budget splits: 1%, 5%, 10%, 25%, 50%, 100% of training labels.
   Each split is saved as a JSON file listing the scene IDs — not copies of the data.
4. Fixes DataLoader seeds for full reproducibility (3 seeds: 42, 123, 456).

Usage:
    python scripts/prepare_dataset.py \
        --raw_dir   data/raw/hls_burn_scars \
        --out_dir   data/processed \
        --split_dir data/splits

Outputs
-------
data/splits/
├── seed_42/
│   ├── split_001pct.json
│   ├── split_005pct.json
│   ├── split_010pct.json
│   ├── split_025pct.json
│   ├── split_050pct.json
│   └── split_100pct.json
├── seed_123/  (same structure)
└── seed_456/  (same structure)
data/processed/stats.json   ← per-band mean/std for normalisation
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
    """Return sorted list of scene stem IDs that have both image and mask."""
    scenes = []
    for f in sorted(split_dir.glob("*.tif")):
        if "_mask" not in f.stem:
            mask = f.with_name(f.stem + "_mask.tif")
            if mask.exists():
                scenes.append(f.stem)
    return scenes


def compute_band_stats(scenes: list, raw_dir: Path):
    """Compute per-band mean and std over training scenes (lazy, loads on demand)."""
    try:
        import numpy as np
        import rasterio
    except ImportError:
        print("  [skip] rasterio/numpy not available — skipping stat computation")
        return None

    print("  Computing per-band statistics …")
    sums = None
    sum_sq = None
    count = 0

    for stem in scenes:
        tif = raw_dir / "training" / f"{stem}.tif"
        with rasterio.open(tif) as src:
            data = src.read().astype("float32")  # (bands, H, W)
            if sums is None:
                sums = np.zeros(data.shape[0])
                sum_sq = np.zeros(data.shape[0])
            px = data.shape[1] * data.shape[2]
            sums += data.reshape(data.shape[0], -1).sum(axis=1)
            sum_sq += (data ** 2).reshape(data.shape[0], -1).sum(axis=1)
            count += px

    means = (sums / count).tolist()
    stds = ((sum_sq / count - (sums / count) ** 2) ** 0.5).tolist()
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
            label = f"split_{int(budget * 100):03d}pct"
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
            print(f"    seed={seed}  budget={int(budget*100):3d}%  → {n:4d} scenes  [{out.name}]")


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
    if not train_dir.exists():
        raise SystemExit(
            f"\n[ERROR] Cannot find {train_dir}\n"
            "  → Run `python scripts/download_data.py` first to download the dataset."
        )

    print(f"Scanning training scenes in {train_dir} …")
    train_scenes = scan_scenes(train_dir)
    val_scenes   = scan_scenes(raw_dir / "validation")
    print(f"  Found {len(train_scenes)} training scenes, {len(val_scenes)} validation scenes")

    # Band statistics
    stats = compute_band_stats(train_scenes, raw_dir)
    if stats:
        stats_path = out_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Band stats saved → {stats_path}")

    # Label splits
    print("\nCreating nested label-budget splits …")
    create_splits(train_scenes, split_dir)

    # Save val list
    with open(split_dir / "val_scenes.json", "w") as f:
        json.dump({"n_scenes": len(val_scenes), "scenes": val_scenes}, f, indent=2)

    print(f"\nDone! Splits written to {split_dir}/")


if __name__ == "__main__":
    main()
