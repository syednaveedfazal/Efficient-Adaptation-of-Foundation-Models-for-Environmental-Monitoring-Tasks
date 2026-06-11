"""
scripts/visualize_predictions.py — Run segmentation inference from a trained checkpoint
and save RGB overlays for qualitative inspection.

Examples:
    # Visualize the first 8 validation scenes with a trained Prithvi checkpoint
    python scripts/visualize_predictions.py \
        --config configs/prithvi_lora_r16.yaml \
        --checkpoint results/checkpoints/prithvi_lora_r16_seed_42_split_001pct/epoch=44-val/burn_iou=0.4353.ckpt \
        --split validation \
        --limit 8

    # Visualize the 1% training subset used by seed_123
    python scripts/visualize_predictions.py \
        --config configs/prithvi_lora_r16.yaml \
        --checkpoint results/checkpoints/prithvi_lora_r16_seed_123_split_001pct/epoch=01-val/burn_iou=0.4154.ckpt \
        --split training \
        --split_json data/splits/seed_123/split_001pct.json
"""

import argparse
import csv
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import yaml

# This repo uses a pure PyTorch inference path.
os.environ.setdefault("USE_TF", "0")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.burn_scar import BurnScarDataset
from src.models.registry import build_model


RGB_BAND_INDICES = (2, 1, 0)  # HLS order: Blue, Green, Red, NIR, SWIR1, SWIR2


def _load_scene_ids(json_path: str | None) -> list[str] | None:
    if not json_path:
        return None

    with open(json_path) as f:
        data = json.load(f)

    if "scenes" not in data:
        raise KeyError(f"Expected a JSON object with a 'scenes' list in {json_path}")

    return data["scenes"]


def _build_dataset(cfg: dict, split: str, split_json: str | None, scene_ids: list[str] | None):
    if scene_ids is not None:
        selected_ids = scene_ids
    elif split_json:
        selected_ids = _load_scene_ids(split_json)
    elif split == "validation" and cfg["data"].get("val_json"):
        val_json = cfg["data"]["val_json"]
        selected_ids = _load_scene_ids(val_json) if Path(val_json).exists() else None
    else:
        selected_ids = None

    return BurnScarDataset(
        raw_dir=cfg["data"]["raw_dir"],
        split=split,
        stats_path=cfg["data"]["stats_path"],
        scene_ids=selected_ids,
    )


def _load_model(cfg: dict, checkpoint_path: str, device: torch.device):
    model = build_model(cfg["model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model_state_dict = {
        key.removeprefix("model."): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    if not model_state_dict:
        model_state_dict = state_dict
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    model.to(device)
    return model


def _read_rgb(scene_path: Path) -> np.ndarray:
    with rasterio.open(scene_path) as src:
        scene = src.read().astype(np.float32)
        nodata = src.nodata

    rgb = scene[list(RGB_BAND_INDICES)]  # (3, H, W) in RGB order
    rgb = np.moveaxis(rgb, 0, -1)        # (H, W, 3)

    valid = np.ones(rgb.shape[:2], dtype=bool)
    if nodata is not None:
        valid &= np.all(rgb != nodata, axis=-1)

    scaled = np.zeros_like(rgb, dtype=np.float32)
    if valid.any():
        for c in range(3):
            values = rgb[..., c][valid]
            lo, hi = np.percentile(values, [2, 98])
            if hi <= lo:
                hi = lo + 1.0
            scaled[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo), 0.0, 1.0)

    return scaled


def _save_mask(mask: np.ndarray, path: Path):
    plt.imsave(path, mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1)


def _blend_overlay(base_rgb: np.ndarray, region: np.ndarray, color: tuple[float, float, float], alpha: float):
    out = base_rgb.copy()
    if region.any():
        out[region] = (1.0 - alpha) * out[region] + alpha * np.asarray(color, dtype=np.float32)
    return out


def _save_prediction_overlay(rgb: np.ndarray, pred_mask: np.ndarray, path: Path):
    overlay = _blend_overlay(rgb, pred_mask == 1, color=(1.0, 0.15, 0.15), alpha=0.45)
    plt.imsave(path, overlay)


def _save_error_overlay(rgb: np.ndarray, pred_mask: np.ndarray, true_mask: np.ndarray, path: Path):
    tp = (pred_mask == 1) & (true_mask == 1)
    fp = (pred_mask == 1) & (true_mask == 0)
    fn = (pred_mask == 0) & (true_mask == 1)

    overlay = rgb.copy()
    overlay = _blend_overlay(overlay, tp, color=(0.0, 0.85, 0.1), alpha=0.45)   # green
    overlay = _blend_overlay(overlay, fp, color=(1.0, 0.1, 0.1), alpha=0.45)    # red
    overlay = _blend_overlay(overlay, fn, color=(0.1, 0.35, 1.0), alpha=0.45)   # blue
    plt.imsave(path, overlay)


def _compute_scene_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    tp = float(np.logical_and(pred, true).sum())
    fp = float(np.logical_and(pred, ~true).sum())
    fn = float(np.logical_and(~pred, true).sum())
    tn = float(np.logical_and(~pred, ~true).sum())

    burn_iou = tp / (tp + fp + fn + 1e-6)
    bg_iou = tn / (tn + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return {
        "burn_iou": burn_iou,
        "bg_iou": bg_iou,
        "mean_iou": 0.5 * (burn_iou + bg_iou),
        "burn_precision": precision,
        "burn_recall": recall,
        "burn_dice": dice,
        "pixel_accuracy": pixel_accuracy,
    }


def _write_metrics_csv(rows: list[dict], output_path: Path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to trained Lightning checkpoint")
    parser.add_argument("--split", choices=["training", "validation"], default="validation",
                        help="Dataset split to visualize")
    parser.add_argument("--split_json", default=None,
                        help="Optional JSON with a 'scenes' list to restrict which scenes are processed")
    parser.add_argument("--scene_id", action="append", default=None,
                        help="Specific scene ID to visualize. Repeat the flag to provide multiple scenes")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N matching scenes")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save overlays. Defaults to results/visualizations/<checkpoint_stem>")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Inference device. Use 'cpu' if the laptop GPU is busy or too small.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results/visualizations") / checkpoint_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment")

    model = _load_model(cfg, str(checkpoint_path), device)
    dataset = _build_dataset(cfg, args.split, args.split_json, args.scene_id)

    indices = range(len(dataset))
    if args.limit is not None:
        indices = range(min(args.limit, len(dataset)))

    scene_rows = []
    print(f"[Visualize] Using device: {device}")
    print(f"[Visualize] Saving outputs to: {output_dir}")

    with torch.inference_mode():
        for idx in indices:
            scene_tensor, true_mask = dataset[idx]
            scene_path, mask_path = dataset.samples[idx]
            scene_id = scene_path.stem.replace("_merged", "")
            scene_dir = output_dir / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)

            input_batch = scene_tensor.unsqueeze(0).to(device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                logits = model(input_batch)
            pred_mask = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            true_mask_np = true_mask.numpy().astype(np.uint8)

            rgb = _read_rgb(scene_path)

            plt.imsave(scene_dir / "rgb.png", rgb)
            _save_mask(pred_mask, scene_dir / "pred_mask.png")
            _save_prediction_overlay(rgb, pred_mask, scene_dir / "pred_overlay.png")

            if mask_path.exists():
                _save_mask(true_mask_np, scene_dir / "gt_mask.png")
                _save_error_overlay(rgb, pred_mask, true_mask_np, scene_dir / "error_overlay.png")
                metrics = _compute_scene_metrics(pred_mask, true_mask_np)
            else:
                metrics = {}

            row = {
                "scene_id": scene_id,
                "scene_path": str(scene_path),
                "mask_path": str(mask_path),
            }
            row.update(metrics)
            scene_rows.append(row)

            print(f"[Visualize] Saved overlays for {scene_id}")

    _write_metrics_csv(scene_rows, output_dir / "scene_metrics.csv")
    print(f"[Visualize] Done. Scene metrics written to: {output_dir / 'scene_metrics.csv'}")


if __name__ == "__main__":
    main()
