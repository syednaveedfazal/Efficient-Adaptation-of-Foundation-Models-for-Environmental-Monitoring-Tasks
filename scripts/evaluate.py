"""
scripts/evaluate.py — Validation-only pass on an existing checkpoint.

Loads a trained checkpoint, runs trainer.validate(), and exports the same
results/metrics/<run_name>.json that scripts/train.py produces at the end
of a training run.

Usage:
    python scripts/evaluate.py \
        --config     configs/prithvi_lora_r8.yaml \
        --split_json data/splits/seed_42/split_010pct.json \
        --ckpt       results/checkpoints/prithvi_lora_r8_seed_42_split_010pct/epoch=55-val/burn_iou=0.6695.ckpt
"""

import argparse
import sys
import yaml
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.burn_scar import BurnScarDataModule
from src.training.module import SegmentationModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--split_json", default=None,
                         help="Label-budget split JSON. Omit for 100%% (all scenes).")
    parser.add_argument("--ckpt", required=True, help="Path to the checkpoint to evaluate")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    model_name = cfg["model"]["name"]
    if args.split_json:
        p = Path(args.split_json)
        run_name = f"{model_name}_{p.parent.name}_{p.stem}"
    else:
        run_name = f"{model_name}_seed_42_split_100pct"

    dm = BurnScarDataModule(
        raw_dir     = cfg["data"]["raw_dir"],
        stats_path  = cfg["data"]["stats_path"],
        split_json  = args.split_json,
        val_json    = cfg["data"].get("val_json"),
        batch_size  = cfg["data"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
    )

    model = SegmentationModule(cfg)

    trainer = pl.Trainer(
        accelerator = "auto",
        devices     = "auto",
        logger      = False,
        precision   = cfg["trainer"].get("precision", "32"),
        plugins     = [SLURMEnvironment(auto_requeue=False)],
    )

    print(f"\nRunning validation on checkpoint: {args.ckpt}")
    val_results = trainer.validate(model, datamodule=dm, ckpt_path=args.ckpt)

    if not val_results:
        print("No validation results returned.")
        return

    res = val_results[0]
    clean_metrics = {k.replace("val/", ""): float(v) for k, v in res.items()}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.split_json:
        with open(args.split_json) as f:
            split_data = json.load(f)
        budget = split_data.get("budget", 1.0)
        seed = split_data.get("seed", 42)
    else:
        budget = 1.0
        seed = cfg.get("seed", 42)

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
            "checkpoint": args.ckpt,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "pct_trainable_params": round(100.0 * trainable_params / total_params, 4) if total_params > 0 else 0.0,
        },
        "metrics": clean_metrics,
        "hardware": gpu_stats,
    }

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"\n[Metrics] Successfully exported metrics to: {out_path}\n")

    from plot_results import main as plot_results
    plot_results()


if __name__ == "__main__":
    main()
