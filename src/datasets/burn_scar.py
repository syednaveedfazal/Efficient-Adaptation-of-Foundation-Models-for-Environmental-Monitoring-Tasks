"""
src/datasets/burn_scar.py — HLS Burn Scar dataset loader.

File layout (from prepare_dataset.py):
    data/raw/hls_burn_scars/
        training/
            <id>_merged.tif    (6-band scene)
            <id>.mask.tif      (1-band mask)
        validation/
            <id>_merged.tif
            <id>.mask.tif
    data/processed/stats.json
    data/splits/
        seed_42/split_001pct.json  ...  split_100pct.json
        val_scenes.json
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import pytorch_lightning as pl
from pathlib import Path


class BurnScarDataset(Dataset):
    """
    Args:
        raw_dir:    data/raw/hls_burn_scars/
        split:      "training" or "validation"
        stats_path: data/processed/stats.json
        scene_ids:  Optional list of base IDs (for label-budget splits).
                    None = use every scene in the split directory.
    """

    def __init__(
        self,
        raw_dir: str,
        split: str = "training",
        stats_path: str = "data/processed/stats.json",
        scene_ids: list = None,
    ):
        assert split in ("training", "validation")
        self.split_dir = Path(raw_dir) / split

        # Per-band z-score stats from prepare_dataset.py
        with open(stats_path) as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float32)   # (6,)
        self.std  = np.array(stats["std"],  dtype=np.float32)   # (6,)
        self.std  = np.where(self.std < 1e-6, 1.0, self.std)   # guard zero std

        # Build scene-mask pairs
        if scene_ids is not None:
            candidates = [self.split_dir / f"{sid}_merged.tif" for sid in scene_ids]
        else:
            candidates = sorted(self.split_dir.glob("*_merged.tif"))

        self.samples = []
        for scene_path in candidates:
            base_id   = scene_path.name.replace("_merged.tif", "")
            mask_path = self.split_dir / f"{base_id}.mask.tif"
            if scene_path.exists() and mask_path.exists():
                self.samples.append((scene_path, mask_path))
            else:
                print(f"[WARN] Missing file for {base_id}, skipping.")

        print(f"[{split}] {len(self.samples)} scene-mask pairs loaded"
              + (f" (subset of {len(scene_ids)} requested)" if scene_ids else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_path, mask_path = self.samples[idx]

        with rasterio.open(scene_path) as src:
            scene = src.read().astype(np.float32)   # (6, 512, 512)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)     # (512, 512)  values: -1, 0, 1

        # Missing data (-1) → 0 (not burned)
        mask = np.where(mask == -1, 0, mask)

        # Z-score normalise per band
        scene = (scene - self.mean[:, None, None]) / self.std[:, None, None]

        return torch.from_numpy(scene), torch.from_numpy(mask)


class BurnScarDataModule(pl.LightningDataModule):
    """
    Args:
        raw_dir:     data/raw/hls_burn_scars/
        stats_path:  data/processed/stats.json
        split_json:  Path to a label-budget JSON (e.g. data/splits/seed_42/split_010pct.json).
                     None = train on all scenes (100% budget).
        val_json:    data/splits/val_scenes.json (optional, restricts val set to listed IDs).
        batch_size:  Batch size.
        num_workers: Match --cpus-per-task in SLURM script.
    """

    def __init__(
        self,
        raw_dir: str,
        stats_path: str = "data/processed/stats.json",
        split_json: str = None,
        val_json: str   = None,
        batch_size: int  = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.raw_dir     = raw_dir
        self.stats_path  = stats_path
        self.split_json  = split_json
        self.val_json    = val_json
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Training subset
        train_ids = None
        if self.split_json:
            with open(self.split_json) as f:
                d = json.load(f)
            train_ids = d["scenes"]
            print(f"[DataModule] Split: {self.split_json} | "
                  f"{d['n_scenes']} scenes | {d['budget']*100:.0f}% | seed={d['seed']}")

        self.train_ds = BurnScarDataset(
            self.raw_dir, split="training",
            stats_path=self.stats_path, scene_ids=train_ids,
        )

        # Validation — always full set
        val_ids = None
        if self.val_json and Path(self.val_json).exists():
            with open(self.val_json) as f:
                val_ids = json.load(f)["scenes"]

        self.val_ds = BurnScarDataset(
            self.raw_dir, split="validation",
            stats_path=self.stats_path, scene_ids=val_ids,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )