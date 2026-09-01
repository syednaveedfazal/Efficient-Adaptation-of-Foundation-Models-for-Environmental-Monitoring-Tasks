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
import json
import torch
import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pathlib import Path

# Make src/ importable when running from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.registry import build_datamodule
from src.training.module import build_module


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
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Override training and dataset seed")
    parser.add_argument("--delete_ckpt",
                        action="store_true",
                        help="Delete checkpoints folder after training to conserve space")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 1. Determine active seed
    seed = cfg.get("seed", 42)
    # Check if a split JSON specifies a seed and use it
    if args.split_json:
        try:
            with open(args.split_json) as sf:
                split_data = json.load(sf)
                seed = split_data.get("seed", seed)
        except Exception:
            pass
            
    # Command line override has highest priority
    if args.seed is not None:
        seed = args.seed

    cfg["seed"] = seed
    pl.seed_everything(seed, workers=True)

    # Derive a readable run name for checkpoints and W&B
    model_name = cfg["model"]["name"]

    if args.split_json:
        p = Path(args.split_json)
        run_name = f"{model_name}_{p.parent.name}_{p.stem}"   # e.g. unet_scratch_seed_42_split_010pct
    else:
        run_name = f"{model_name}_seed_{seed}_split_100pct"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    # Dataset registry picks BurnScar vs EuroSAT from cfg["data"]["name"]
    # (legacy flat burn-scar blocks default to burn_scar).
    dm = build_datamodule(cfg["data"], split_json=args.split_json)

    # ------------------------------------------------------------------
    # Model + task  (registries pick model class and Lightning module)
    # ------------------------------------------------------------------
    # build_module selects Segmentation vs Classification from cfg["task"]
    # (defaults to "segmentation" so existing configs are unchanged).
    model = build_module(cfg)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    ckpt_dir = Path(cfg["trainer"]["checkpoint_dir"]) / run_name
    # Metric to monitor/checkpoint on — segmentation uses val/burn_iou (default),
    # classification uses val/accuracy (set in the EuroSAT configs).
    monitor_metric = cfg["trainer"].get("monitor_metric", "val/burn_iou")
    monitor_mode   = cfg["trainer"].get("monitor_mode", "max")
    callbacks = [
        ModelCheckpoint(
            dirpath   = str(ckpt_dir),
            filename  = "{epoch:02d}-{" + monitor_metric + ":.4f}",
            monitor   = monitor_metric,
            mode      = monitor_mode,
            save_top_k = 1,
        ),
        EarlyStopping(
            monitor  = monitor_metric,
            patience = cfg["trainer"]["patience"],
            mode     = monitor_mode,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    # Use CSVLogger, not the default TensorBoardLogger. TensorBoard's atomic
    # save() writes a temp file in the system tmpdir then os.rename()s it onto
    # the (NFS) project dir; on this cluster /tmp is a local disk, so that
    # cross-device rename fails with OSError: [Errno 18]. CSVLogger writes its
    # metrics.csv directly to save_dir (NFS) — no cross-device move. We still
    # need *a* logger because the LearningRateMonitor callback requires one.
    logger = CSVLogger(save_dir="results/logs", name=run_name)
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
        plugins           = [SLURMEnvironment(auto_requeue=False)],
    )

    print(f"\n{'='*60}")
    print(f"  Model:  {model_name}")
    print(f"  Run:    {run_name}")
    print(f"  Split:  {args.split_json or 'ALL (100%)'}")
    print(f"{'='*60}\n")

    start_time = time.time()
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)
    training_duration_seconds = round(time.time() - start_time, 2)
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

        if 'training_duration_seconds' in locals():
            gpu_stats["training_duration_seconds"] = training_duration_seconds
        else:
            gpu_stats["training_duration_seconds"] = 0.0

        # Compute GFLOPs dynamically
        try:
            from src.utils.flops import get_model_training_gflops
            adaptation_strategy = cfg["model"]["params"].get("adaptation", "full_ft")
            fwd_gflops, bwd_gflops = get_model_training_gflops(model.model, adaptation_strategy)
        except Exception as e:
            print(f"[WARN] Failed to compute training GFLOPs: {e}")
            fwd_gflops, bwd_gflops = 0.0, 0.0

        export_data = {
            "experiment": {
                "model_name": model_name,
                "config": args.config,
                "split_json": args.split_json,
                "label_budget": budget,
                "seed": seed,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "pct_trainable_params": round(100.0 * trainable_params / total_params, 4) if total_params > 0 else 0.0,
                "fwd_gflops": fwd_gflops,
                "bwd_gflops": bwd_gflops,
                "total_gflops": round(fwd_gflops + bwd_gflops, 4)
            },
            "metrics": clean_metrics,
            "hardware": gpu_stats
        }
        
        out_path = metrics_dir / f"{run_name}.json"
        with open(out_path, "w") as f:
            json.dump(export_data, f, indent=2)
        print(f"\n[Metrics] Successfully exported final metrics to: {out_path}\n")

        # Delete large checkpoint directory if requested to save space
        if args.delete_ckpt:
            try:
                import shutil
                if ckpt_dir.exists():
                    print(f"[Cleanup] Deleting checkpoints in {ckpt_dir} to free up disk space.")
                    shutil.rmtree(ckpt_dir)
            except Exception as e:
                print(f"[WARN] Failed to clean up checkpoint directory {ckpt_dir}: {e}")

        # Auto-plot only for burn-scar (plot_results aggregates burn-scar
        # segmentation metrics). EuroSAT plots are built by scripts/plot_results.py
        # with a --dataset filter after the sweep.
        dataset_name = cfg.get("data", {}).get("name", "burn_scar")
        if dataset_name == "burn_scar":
            from plot_results import main as plot_results
            plot_results()


if __name__ == "__main__":
    main()