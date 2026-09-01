"""
scripts/inference_eurosat.py — Visual inference for Prithvi EuroSAT classifiers.

Loads one or two trained checkpoints, runs predictions on validation images,
and produces a grid figure with predicted vs ground-truth class labels.

Checkpoints are auto-discovered from the run name derived from config + split.

Usage (two models):
    python scripts/inference_eurosat.py \
        --config_fft  configs/eurosat_prithvi_full_ft.yaml \
        --config_lora configs/eurosat_prithvi_lora_r16.yaml \
        --split_json  data/splits/eurosat/seed_42/split_010pct.json

Usage (single model):
    python scripts/inference_eurosat.py \
        --config_fft configs/eurosat_prithvi_full_ft.yaml \
        --split_json data/splits/eurosat/seed_42/split_010pct.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.eurosat import EuroSATDataset, CLASS_NAMES
from src.training.module import ClassificationModule


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_best_checkpoint(ckpt_base_dir: str, run_name: str) -> str:
    """Auto-discover the best checkpoint under <ckpt_base_dir>/<run_name>/."""
    run_dir = Path(ckpt_base_dir) / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {run_dir}")

    ckpts = list(run_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {run_dir}")

    best_path, best_val = None, -1.0
    for p in ckpts:
        match = re.search(r"=(\d+\.\d+)\.ckpt$", p.name)
        if match:
            val = float(match.group(1))
            if val > best_val:
                best_val = val
                best_path = p

    if best_path is None:
        best_path = sorted(ckpts)[-1]

    print(f"  → Best checkpoint: {best_path}  (metric = {best_val:.4f})")
    return str(best_path)


def derive_run_name(model_name: str, split_json: str | None, seed: int) -> str:
    """Build the run_name exactly as train.py does."""
    if split_json:
        p = Path(split_json)
        return f"{model_name}_{p.parent.name}_{p.stem}"
    return f"{model_name}_seed_{seed}_split_100pct"


def load_model(config_path: str, ckpt_path: str, device: torch.device):
    """Load a ClassificationModule from config + Lightning checkpoint."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = ClassificationModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.to(device)
    model.eval()
    return model, cfg


def predict(model, img_tensor: torch.Tensor, device: torch.device) -> int:
    """Run a single forward pass and return predicted class index."""
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))  # (1, 10)
    return logits.argmax(dim=1).item()


def make_rgb(scene: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Convert a z-score normalised 6-band EuroSAT patch to RGB.

    Band order: [Blue, Green, Red, NIR, SWIR1, SWIR2]
    RGB = indices [2, 1, 0] (Red, Green, Blue).
    """
    scene = scene * std[:, None, None] + mean[:, None, None]
    rgb = scene[[2, 1, 0], :, :]          # (3, H, W)
    rgb = np.moveaxis(rgb, 0, -1)         # (H, W, 3)

    # Percentile contrast stretching
    valid = rgb[rgb > 0]
    if valid.size > 0:
        p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
    else:
        rgb = np.clip(rgb, 0, 1)
    return rgb


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EuroSAT inference visualisation")
    parser.add_argument("--config_fft",  default=None, help="Config YAML for model A (e.g. Full FT)")
    parser.add_argument("--config_lora", default=None, help="Config YAML for model B (e.g. LoRA r=16)")
    parser.add_argument("--split_json",  default=None, help="Split JSON for run-name derivation")
    parser.add_argument("--ckpt_dir",    default="results/checkpoints", help="Base checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=6, help="Number of validation images to show")
    parser.add_argument("--output_dir",  default="results/plots", help="Where to save the figure")
    parser.add_argument("--seed",        type=int, default=42, help="Random seed for sample selection")
    args = parser.parse_args()

    if not args.config_fft and not args.config_lora:
        parser.error("Provide at least one model config: --config_fft or --config_lora")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Auto-discover checkpoints & load models ──────────────────────────
    models = {}
    for label, config_path in [("Full Fine-Tuning", args.config_fft),
                                ("LoRA r=16",        args.config_lora)]:
        if config_path is None:
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        model_name = cfg["model"]["name"]
        run_name = derive_run_name(model_name, args.split_json, args.seed)

        print(f"\n[{label}] Looking for checkpoint in: {args.ckpt_dir}/{run_name}/")
        ckpt_path = find_best_checkpoint(args.ckpt_dir, run_name)

        print(f"[{label}] Loading model from {ckpt_path}")
        m, c = load_model(config_path, ckpt_path, device)
        models[label] = (m, c)

    # ── Load validation dataset ───────────────────────────────────────────
    ref_cfg = list(models.values())[0][1]
    data_params = ref_cfg["data"]["params"]
    raw_dir    = data_params["raw_dir"]
    stats_path = data_params["stats_path"]

    # Use val_json to load validation set
    val_json = data_params.get("val_json")
    val_ids = None
    if val_json and Path(val_json).exists():
        with open(val_json) as f:
            val_ids = json.load(f)["scenes"]

    val_ds = EuroSATDataset(raw_dir, stats_path=stats_path, sample_ids=val_ids)

    # Load stats for de-normalisation
    with open(stats_path) as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)
    std  = np.where(std < 1e-6, 1.0, std)

    # Pick random samples — always draw 10 with the fixed seed so that
    # reducing num_samples still includes the same interesting cases
    # (e.g. the Forest misclassification at position 4).
    rng = np.random.default_rng(args.seed)
    pool = rng.choice(len(val_ds), size=max(10, args.num_samples), replace=False)
    indices = np.sort(pool[:args.num_samples])
    n = len(indices)

    # ── Build figure ──────────────────────────────────────────────────────
    num_models = len(models)
    ncols = 3    # 3 images per row
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0 * ncols, 8.0 * nrows))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02,
                        wspace=0.35, hspace=0.55)
    if nrows == 1:
        axes = axes[np.newaxis, :]

    model_labels = list(models.keys())

    for i, sample_idx in enumerate(indices):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        img, gt_label = val_ds[sample_idx]
        rgb = make_rgb(img.numpy(), mean, std)

        ax.imshow(rgb)
        gt_name = CLASS_NAMES[gt_label]

        # Build prediction text
        pred_lines = []
        for m_label in model_labels:
            model_obj = models[m_label][0]
            pred_idx = predict(model_obj, img, device)
            pred_name = CLASS_NAMES[pred_idx]
            correct = pred_idx == gt_label
            symbol = "✓" if correct else "✗"
            pred_lines.append(f"{m_label}: {pred_name} {symbol}")

        title = f"GT: {gt_name}\n" + "\n".join(pred_lines)
        ax.set_title(title, fontsize=26, fontweight="bold",
                     color="black", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    # fig.suptitle("EuroSAT Classification — Prithvi Inference",
    #              fontsize=28, fontweight="bold", y=0.95)

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inference_eurosat.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ Saved inference figure to: {out_path}\n")


if __name__ == "__main__":
    main()
