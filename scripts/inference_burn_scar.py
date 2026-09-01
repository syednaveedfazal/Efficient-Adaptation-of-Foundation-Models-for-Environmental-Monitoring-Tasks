"""
scripts/inference_burn_scar.py — Visual inference for Prithvi FCN burn-scar models.

Loads one or two trained checkpoints, runs predictions on validation images,
and produces a publication-quality side-by-side comparison figure:
    input false-colour RGB  |  ground truth  |  prediction

Checkpoints are auto-discovered: pass only the config YAML and (optionally)
a split JSON — the script finds the best .ckpt by parsing metric values from
the checkpoint filename.

Usage (two models, auto-discover checkpoints):
    python scripts/inference_burn_scar.py \
        --config_fft  configs/prithvi_fcn_full_ft.yaml \
        --config_lora configs/prithvi_fcn_lora_r16.yaml \
        --split_json  data/splits/seed_42/split_010pct.json

Usage (single model):
    python scripts/inference_burn_scar.py \
        --config_fft configs/prithvi_fcn_full_ft.yaml \
        --split_json data/splits/seed_42/split_010pct.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml

# Make src/ importable when running from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.burn_scar import BurnScarDataset
from src.training.module import SegmentationModule


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_best_checkpoint(ckpt_base_dir: str, run_name: str) -> str:
    """
    Auto-discover the best checkpoint in results/checkpoints/<run_name>/.

    Checkpoint layout (from ModelCheckpoint):
        <ckpt_base_dir>/<run_name>/epoch=XX-val/<metric>=Y.YYYY.ckpt

    Picks the .ckpt whose metric value in the filename is highest.
    """
    run_dir = Path(ckpt_base_dir) / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {run_dir}")

    ckpts = list(run_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {run_dir}")

    # Parse the metric value from filenames like "burn_iou=0.6712.ckpt"
    best_path, best_val = None, -1.0
    for p in ckpts:
        match = re.search(r"=(\d+\.\d+)\.ckpt$", p.name)
        if match:
            val = float(match.group(1))
            if val > best_val:
                best_val = val
                best_path = p

    if best_path is None:
        # Fallback: just pick the last one alphabetically
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
    """Load a SegmentationModule from config + Lightning checkpoint."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = SegmentationModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.to(device)
    model.eval()
    return model, cfg


def predict(model, img_tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run a single forward pass and return the predicted mask (H, W)."""
    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0).to(device))  # (1, 2, H, W)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def make_false_colour_rgb(scene: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Convert a z-score normalised 6-band scene to a false-colour RGB image.

    Uses bands B04 (Red), B03 (Green), B02 (Blue) — indices 2, 1, 0 in the
    HLS Burn Scars 6-band ordering [B02, B03, B04, Narrow-NIR, SWIR-1, SWIR-2].
    """
    # De-normalise
    scene = scene * std[:, None, None] + mean[:, None, None]

    # Select RGB bands: B04=idx2 (Red), B03=idx1 (Green), B02=idx0 (Blue)
    rgb = scene[[2, 1, 0], :, :]         # (3, H, W)
    rgb = np.moveaxis(rgb, 0, -1)        # (H, W, 3)

    # Percentile-based contrast stretching for robust visualisation
    p2, p98 = np.percentile(rgb[rgb > 0], 2), np.percentile(rgb[rgb > 0], 98)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0, 1)
    return rgb


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, colour, alpha: float = 0.45) -> np.ndarray:
    """Blend a coloured semi-transparent overlay where mask == 1."""
    out = rgb.copy()
    burn = mask == 1
    for c in range(3):
        out[:, :, c] = np.where(burn, out[:, :, c] * (1 - alpha) + colour[c] * alpha, out[:, :, c])
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Burn-scar inference visualisation")
    parser.add_argument("--config_fft",  default=None, help="Config YAML for model A (e.g. Full FT)")
    parser.add_argument("--config_lora", default=None, help="Config YAML for model B (e.g. LoRA r=16)")
    parser.add_argument("--split_json",  default=None, help="Split JSON (e.g. data/splits/seed_42/split_010pct.json)")
    parser.add_argument("--ckpt_dir",    default="results/checkpoints", help="Base checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=2, help="Number of validation images to show")
    parser.add_argument("--output_dir",  default="results/plots", help="Where to save the figure")
    parser.add_argument("--seed",        type=int, default=42, help="Random seed for sample selection")
    args = parser.parse_args()

    if not args.config_fft and not args.config_lora:
        parser.error("Provide at least one model config: --config_fft or --config_lora")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Auto-discover checkpoints & load models ──────────────────────────
    models = {}  # label → (model, cfg)

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
    raw_dir    = ref_cfg["data"]["raw_dir"]
    stats_path = ref_cfg["data"]["stats_path"]

    val_ds = BurnScarDataset(raw_dir, split="validation", stats_path=stats_path)

    # Load stats for de-normalisation
    with open(stats_path) as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)
    std  = np.where(std < 1e-6, 1.0, std)

    # Pick random samples
    rng = np.random.default_rng(args.seed)
    n = min(args.num_samples, len(val_ds))
    indices = rng.choice(len(val_ds), size=n, replace=False)
    indices.sort()

    # ── Build figure ──────────────────────────────────────────────────────
    num_models = len(models)
    ncols = 2 + num_models   # Input RGB | Ground Truth | Pred A [| Pred B]
    nrows = n

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.5 * nrows))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.06,
                        wspace=0.08, hspace=0.12)
    if nrows == 1:
        axes = axes[np.newaxis, :]

    # Colour coding
    gt_colour   = np.array([1.0, 0.84, 0.0])    # gold
    pred_colour = np.array([1.0, 0.2, 0.2])      # red

    model_labels = list(models.keys())

    for row_idx, sample_idx in enumerate(indices):
        scene, gt_mask = val_ds[sample_idx]
        scene_np = scene.numpy()
        gt_np    = gt_mask.numpy()

        rgb = make_false_colour_rgb(scene_np, mean, std)

        # Column 0: Input RGB
        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_ylabel(f"Sample {sample_idx}", fontsize=22, fontweight="bold")
        if row_idx == 0:
            axes[row_idx, 0].set_title("Input (False-Colour RGB)", fontsize=24, fontweight="bold", pad=14)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Column 1: Ground truth overlay
        gt_overlay = overlay_mask(rgb, gt_np, gt_colour)
        axes[row_idx, 1].imshow(gt_overlay)
        if row_idx == 0:
            axes[row_idx, 1].set_title("Ground Truth", fontsize=24, fontweight="bold", pad=14)
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        # Columns 2+: Model predictions
        for m_idx, label in enumerate(model_labels):
            model_obj = models[label][0]
            pred = predict(model_obj, scene, device)
            pred_overlay = overlay_mask(rgb, pred, pred_colour)
            col = 2 + m_idx
            axes[row_idx, col].imshow(pred_overlay)
            if row_idx == 0:
                axes[row_idx, col].set_title(f"Prediction: {label}", fontsize=24, fontweight="bold", pad=14)
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])

    # Legend
    gt_patch   = mpatches.Patch(color=gt_colour,   label="Burn Scar (GT)")
    pred_patch = mpatches.Patch(color=pred_colour,  label="Burn Scar (Pred)")
    fig.legend(handles=[gt_patch, pred_patch], loc="lower center", ncol=2,
               fontsize=22, frameon=True, fancybox=True, shadow=True,
               bbox_to_anchor=(0.5, 0.01))

    # fig.suptitle("Burn Scar Segmentation — Prithvi FCN Inference",
    #              fontsize=28, fontweight="bold", y=0.95)

    # ── Save ──────────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inference_burn_scar.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓ Saved inference figure to: {out_path}\n")


if __name__ == "__main__":
    main()
