"""
scripts/prepare_eurosat.py — EuroSAT-MS stats + label-budget splits.

Mirrors scripts/prepare_dataset.py (burn-scar) but for EuroSAT classification:
1. Scans data/raw/eurosat/<Class>/*.tif  ->  sample ids "<Class>/<stem>".
2. Fixed stratified train/val split (80/20 per class, seed 42) -> val_scenes.json.
3. Per-band mean/std over the TRAIN tiles, selecting the SAME 6 bands the models
   use (EUROSAT_TO_HLS = [Blue,Green,Red,NIR,SWIR1,SWIR2]) -> eurosat_stats.json.
4. Stratified label-budget JSONs (1/10/25/100%) under
   data/splits/eurosat/seed_{42,123,456}/, same schema as burn-scar splits.

Stratified (per-class) subsampling — unlike burn-scar's binary shuffle — so all
10 classes are present even at the 1% budget.

Usage:
    python scripts/prepare_eurosat.py \
        --raw_dir data/raw/eurosat \
        --out_dir data/processed \
        --split_dir data/splits/eurosat
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

# Make src/ importable when run as a script from the project root
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.datasets.eurosat import EUROSAT_TO_HLS, CLASS_NAMES

LABEL_BUDGETS = [0.01, 0.10, 0.25, 1.00]
SEEDS = [42, 123, 456]
VAL_FRACTION = 0.20
VAL_SPLIT_SEED = 42          # fixed so every experiment shares the same val set


def scan_by_class(raw_dir: Path) -> dict:
    """Return {class_name: [ '<Class>/<stem>', ... ]} sorted for reproducibility."""
    by_class = {}
    for cls in CLASS_NAMES:
        tifs = sorted((raw_dir / cls).glob("*.tif"))
        by_class[cls] = [f"{cls}/{p.stem}" for p in tifs]
    return by_class


def stratified_train_val(by_class: dict):
    """Fixed per-class 80/20 train/val split."""
    rng = random.Random(VAL_SPLIT_SEED)
    train_ids, val_ids = [], []
    for cls, ids in by_class.items():
        shuffled = ids.copy(); rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * VAL_FRACTION))
        val_ids   += shuffled[:n_val]
        train_ids += shuffled[n_val:]
    return train_ids, val_ids


def compute_band_stats(train_ids: list, raw_dir: Path) -> dict:
    """Per-band mean/std over the 6 selected bands across TRAIN tiles."""
    import rasterio
    sums = np.zeros(len(EUROSAT_TO_HLS)); sum_sq = np.zeros(len(EUROSAT_TO_HLS)); count = 0
    for sid in train_ids:
        with rasterio.open(raw_dir / f"{sid}.tif") as src:
            data = src.read().astype("float32")[EUROSAT_TO_HLS, :, :]   # (6, H, W)
        flat = data.reshape(data.shape[0], -1)
        sums   += flat.sum(axis=1)
        sum_sq += (flat ** 2).sum(axis=1)
        count  += flat.shape[1]
    mean = sums / count
    var  = np.clip(sum_sq / count - mean ** 2, a_min=0.0, a_max=None)
    return {"mean": mean.tolist(), "std": (var ** 0.5).tolist(), "n_pixels": int(count)}


def create_splits(by_class_train: dict, split_dir: Path):
    """Stratified label-budget JSONs per seed (subsample each class independently)."""
    for seed in SEEDS:
        rng = random.Random(seed)
        seed_dir = split_dir / f"seed_{seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
        for budget in LABEL_BUDGETS:
            subset = []
            for cls, ids in by_class_train.items():
                shuffled = ids.copy(); rng.shuffle(shuffled)
                n = max(1, int(len(ids) * budget))
                subset += shuffled[:n]
            label = f"split_{round(budget * 100):03d}pct"
            out = seed_dir / f"{label}.json"
            with open(out, "w") as f:
                json.dump({"seed": seed, "budget": budget,
                           "n_scenes": len(subset), "scenes": subset}, f, indent=2)
            print(f"    seed={seed}  budget={round(budget*100):3d}%  → {len(subset):5d} tiles  [{out.name}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir",   default="data/raw/eurosat")
    ap.add_argument("--out_dir",   default="data/processed")
    ap.add_argument("--split_dir", default="data/splits/eurosat")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir); out_dir = Path(args.out_dir); split_dir = Path(args.split_dir)
    out_dir.mkdir(parents=True, exist_ok=True); split_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        raise SystemExit(f"[ERROR] {raw_dir} not found. Run: python scripts/download_eurosat.py")

    by_class = scan_by_class(raw_dir)
    total = sum(len(v) for v in by_class.values())
    print(f"Scanned {raw_dir}: {total} tiles across {len(CLASS_NAMES)} classes")
    for c in CLASS_NAMES:
        print(f"    {c:22s} {len(by_class[c]):5d}")
    if total == 0:
        raise SystemExit("[ERROR] 0 tiles found — check extraction layout (raw/eurosat/<Class>/*.tif).")

    train_ids, val_ids = stratified_train_val(by_class)
    print(f"\nTrain/val split: {len(train_ids)} train | {len(val_ids)} val (fixed seed {VAL_SPLIT_SEED})")

    # per-class train ids for stratified budget subsets
    by_class_train = {c: [s for s in train_ids if s.split("/")[0] == c] for c in CLASS_NAMES}

    print("\nComputing 6-band stats over train tiles ...")
    stats = compute_band_stats(train_ids, raw_dir)
    with open(out_dir / "eurosat_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Band stats saved → {out_dir / 'eurosat_stats.json'}")

    print("\nCreating stratified label-budget splits ...")
    create_splits(by_class_train, split_dir)

    with open(split_dir / "val_scenes.json", "w") as f:
        json.dump({"n_scenes": len(val_ids), "scenes": val_ids}, f, indent=2)
    print(f"\nDone! Splits + val_scenes.json written to {split_dir}/")


if __name__ == "__main__":
    main()
