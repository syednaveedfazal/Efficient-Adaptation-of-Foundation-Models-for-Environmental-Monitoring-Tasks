"""
src/datasets/eurosat.py — EuroSAT-MS land-cover classification loader.

EuroSAT-MS: 27,000 Sentinel-2 patches, 64x64 px, 13 spectral bands, 10 classes.
File layout (from prepare_dataset.py / download_data.py):
    data/raw/eurosat/
        AnnualCrop/AnnualCrop_1.tif ...
        Forest/Forest_1.tif ...
        ... (10 class dirs)
    data/processed/eurosat_stats.json          (per-band mean/std for the 6 bands)
    data/splits/eurosat/
        seed_42/split_001pct.json ... split_100pct.json
        val_scenes.json

Key design choice: we select the SAME 6 bands in the SAME HLS order the burn-scar
backbones expect — [Blue, Green, Red, NIR, SWIR1, SWIR2] — from the 13 Sentinel-2
bands. So DINOv2 (RGB=[2,1,0]) and Prithvi (all 6) run unchanged; only the head
differs (classification vs. segmentation).

Sentinel-2 L1C 13-band order (0-based): B01,B02,B03,B04,B05,B06,B07,B08,B08A,B09,B10,B11,B12
    Blue  = B02 -> 1
    Green = B03 -> 2
    Red   = B04 -> 3
    NIR   = B08 -> 7
    SWIR1 = B11 -> 11
    SWIR2 = B12 -> 12
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import pytorch_lightning as pl
from pathlib import Path

# 13-band Sentinel-2 -> 6-band HLS order [Blue, Green, Red, NIR, SWIR1, SWIR2]
EUROSAT_TO_HLS = [1, 2, 3, 7, 11, 12]

# 10 EuroSAT classes, alphabetical -> stable label indices 0..9
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


class EuroSATDataset(Dataset):
    """
    Args:
        raw_dir:    data/raw/eurosat/  (contains the 10 class subdirs)
        stats_path: data/processed/eurosat_stats.json  (mean/std for the 6 bands)
        sample_ids: list of "<Class>/<file_stem>" ids (e.g. "Forest/Forest_1").
                    None = every .tif under raw_dir.
        bands:      13-band -> 6-band index selection (defaults to HLS order).
    """

    def __init__(
        self,
        raw_dir: str,
        stats_path: str = "data/processed/eurosat_stats.json",
        sample_ids: list = None,
        bands: list = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.bands   = bands if bands is not None else EUROSAT_TO_HLS

        with open(stats_path) as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float32)   # (6,)
        self.std  = np.array(stats["std"],  dtype=np.float32)   # (6,)
        self.std  = np.where(self.std < 1e-6, 1.0, self.std)    # guard zero std

        if sample_ids is not None:
            candidates = [self.raw_dir / f"{sid}.tif" for sid in sample_ids]
        else:
            candidates = sorted(self.raw_dir.glob("*/*.tif"))

        self.samples = []
        for p in candidates:
            cls = p.parent.name
            if p.exists() and cls in CLASS_TO_IDX:
                self.samples.append((p, CLASS_TO_IDX[cls]))
            else:
                print(f"[WARN] Missing/invalid EuroSAT tile {p}, skipping.")

        print(f"[EuroSAT] {len(self.samples)} tiles loaded"
              + (f" (subset of {len(sample_ids)} requested)" if sample_ids else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)     # (13, 64, 64)

        img = img[self.bands, :, :]                 # (6, 64, 64) HLS order
        img = (img - self.mean[:, None, None]) / self.std[:, None, None]

        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)


class EuroSATDataModule(pl.LightningDataModule):
    """
    Args:
        raw_dir:     data/raw/eurosat/
        stats_path:  data/processed/eurosat_stats.json
        split_json:  label-budget JSON (train subset). None = all training tiles.
        val_json:    data/splits/eurosat/val_scenes.json (fixed val set).
        batch_size:  batch size.
        num_workers: dataloader workers.
        bands:       optional 6-band selection override.
    """

    def __init__(
        self,
        raw_dir: str,
        stats_path: str = "data/processed/eurosat_stats.json",
        split_json: str = None,
        val_json: str   = None,
        batch_size: int  = 64,
        num_workers: int = 4,
        bands: list = None,
    ):
        super().__init__()
        self.raw_dir     = raw_dir
        self.stats_path  = stats_path
        self.split_json  = split_json
        self.val_json    = val_json
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.bands       = bands

    def setup(self, stage=None):
        # Training subset (label-budget)
        train_ids = None
        if self.split_json:
            with open(self.split_json) as f:
                d = json.load(f)
            train_ids = d["scenes"]
            print(f"[DataModule] Split: {self.split_json} | "
                  f"{d['n_scenes']} tiles | {d['budget']*100:.0f}% | seed={d['seed']}")

        self.train_ds = EuroSATDataset(
            self.raw_dir, stats_path=self.stats_path,
            sample_ids=train_ids, bands=self.bands,
        )

        # Validation — fixed set
        val_ids = None
        if self.val_json and Path(self.val_json).exists():
            with open(self.val_json) as f:
                val_ids = json.load(f)["scenes"]

        self.val_ds = EuroSATDataset(
            self.raw_dir, stats_path=self.stats_path,
            sample_ids=val_ids, bands=self.bands,
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
