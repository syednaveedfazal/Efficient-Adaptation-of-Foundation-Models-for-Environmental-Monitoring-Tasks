"""
scripts/train.py — Unified training entry point for all models.

Usage:
    # UNet, full dataset
    python scripts/train.py --config configs/unet_baseline.yaml

    # UNet, 10% label budget
    python scripts/train.py \
        --config     configs/unet_baseline.yaml \
        --split_json data/splits/seed_42/split_010pct.json

    # Teammates — just swap the config, nothing else changes
    python scripts/train.py --config configs/prithvi_lora.yaml
    python scripts/train.py --config configs/dinov2_finetune.yaml
"""

import argparse
import sys
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path

# Make src/ importable when running from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.burn_scar import BurnScarDataModule
from src.training.module import SegmentationModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/unet_baseline.yaml",
                        help="Path to config YAML")
    parser.add_argument("--split_json",
                        default=None,
                        help="Label-budget split JSON from prepare_dataset.py. "
                             "e.g. data/splits/seed_42/split_010pct.json. "
                             "Omit to train on all scenes (100%%).")
    parser.add_argument("--resume",
                        default=None,
                        help="Path to a Lightning checkpoint to resume training from. "
                             "Restores model weights, optimizer, LR scheduler, and "
                             "early-stopping counter exactly where training left off.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # Derive a readable run name for checkpoints and W&B
    model_name = cfg["model"]["name"]
    if args.split_json:
        p = Path(args.split_json)
        run_name = f"{model_name}_{p.parent.name}_{p.stem}"   # e.g. unet_seed_42_split_010pct
    else:
        run_name = f"{model_name}_seed_42_split_100pct"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dm = BurnScarDataModule(
        raw_dir     = cfg["data"]["raw_dir"],
        stats_path  = cfg["data"]["stats_path"],
        split_json  = args.split_json,
        val_json    = cfg["data"].get("val_json"),
        batch_size  = cfg["data"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
    )

    # ------------------------------------------------------------------
    # Model  (registry picks the right class from cfg["model"]["name"])
    # ------------------------------------------------------------------
    model = SegmentationModule(cfg)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg["trainer"]["checkpoint_dir"]) / run_name
    callbacks = [
        ModelCheckpoint(
            dirpath   = str(ckpt_dir),
            filename  = "{epoch:02d}-{val/burn_iou:.4f}",
            monitor   = "val/burn_iou",
            mode      = "max",
            save_top_k = 3,
        ),
        EarlyStopping(
            monitor  = "val/burn_iou",
            patience = cfg["trainer"]["patience"],
            mode     = "max",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger = None
    if cfg.get("use_wandb", False):
        logger = WandbLogger(
            project = cfg.get("wandb_project", "burn-scar"),
            name    = run_name,
        )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs        = cfg["trainer"]["max_epochs"],
        accelerator       = "auto",
        devices           = "auto",
        callbacks         = callbacks,
        logger            = logger,
        log_every_n_steps = 10,
        precision         = cfg["trainer"].get("precision", "32"),
    )

    print(f"\n{'='*60}")
    print(f"  Model:  {model_name}")
    print(f"  Run:    {run_name}")
    print(f"  Split:  {args.split_json or 'ALL (100%)'}")
    print(f"{'='*60}\n")

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    print(f"\nDone. Checkpoints saved in: {ckpt_dir}")


if __name__ == "__main__":
    main()