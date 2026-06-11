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
    python scripts/train.py --config configs/prithvi_lora_r8.yaml
    python scripts/train.py --config configs/prithvi_lora_r16.yaml
    python scripts/train.py --config configs/dinov2_finetune.yaml
"""

import argparse
import os
import sys
import yaml
import json

# This project uses a PyTorch-only training path. Setting USE_TF=0 before any
# third-party ML imports prevents transformers from probing unrelated TF installs.
os.environ.setdefault("USE_TF", "0")

import torch
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

    # ------------------------------------------------------------------
    # Evaluation and Export to results/metrics/
    # ------------------------------------------------------------------
    best_ckpt = trainer.checkpoint_callback.best_model_path
    if best_ckpt and Path(best_ckpt).exists():
        print(f"\nRunning final validation on best checkpoint: {best_ckpt}")
        val_results = trainer.validate(model, datamodule=dm, ckpt_path=best_ckpt)
    else:
        print("\nNo best checkpoint found or saved. Running validation on final model state...")
        val_results = trainer.validate(model, datamodule=dm)

    if val_results:
        metrics_dir = Path("results/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)

        res = val_results[0]
        # Clean prefix from keys for cleaner json (e.g. 'val/loss' -> 'loss')
        clean_metrics = {k.replace("val/", ""): float(v) for k, v in res.items()}
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if args.split_json:
            try:
                with open(args.split_json) as f:
                    split_data = json.load(f)
                budget = split_data.get("budget", 1.0)
                seed = split_data.get("seed", 42)
            except Exception:
                budget = 1.0
                seed = cfg.get("seed", 42)
        else:
            budget = 1.0
            seed = cfg.get("seed", 42)
            
        # Collect GPU statistics
        gpu_stats = {}
        if torch.cuda.is_available():
            gpu_stats["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_stats["max_memory_allocated_mb"] = round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2)
            gpu_stats["max_memory_reserved_mb"] = round(torch.cuda.max_memory_reserved(0) / (1024 ** 2), 2)
        else:
            gpu_stats["gpu_name"] = "CPU / No GPU"
            gpu_stats["max_memory_allocated_mb"] = 0.0
            gpu_stats["max_memory_reserved_mb"] = 0.0

        export_data = {
            "experiment": {
                "model_name": model_name,
                "config": args.config,
                "split_json": args.split_json,
                "label_budget": budget,
                "seed": seed,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "pct_trainable_params": round(100.0 * trainable_params / total_params, 4) if total_params > 0 else 0.0
            },
            "metrics": clean_metrics,
            "hardware": gpu_stats
        }
        
        out_path = metrics_dir / f"{run_name}.json"
        with open(out_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\n[Metrics] Successfully exported final metrics to: {out_path}\n")


if __name__ == "__main__":
    main()
