"""
scripts/spaco/evaluate.py — Score a LoRA / SpaCo-LoRA checkpoint on the dev and/or test half of the val scenes.

Metrics per set:
  - pooled over scenes (main): burn IoU = sum TP / sum (TP + FP + FN), burn Dice, background IoU, mean IoU
  - batch-mean burn IoU (batch 4, fixed order), the aggregation Lightning logs, for continuity with earlier numbers
  - per-scene TP / FP / FN / TN for burn (bootstrap CIs and size-stratified analyses are computed from these)
For SpaCo models it also records the learned gammas and the mean explained variance of the top-k modes.

Test must be scored once per final run only: pass --sets test explicitly.

Usage:
  python scripts/spaco/evaluate.py --config configs/spaco/spaco_k8_cov.yaml --ckpt <ckpt> --sets dev --out_json <json>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.datasets.registry import build_datamodule
from src.training.module import build_module

SET_JSON = {"dev": "splits/val_dev.json", "test": "splits/val_test.json"}


def pooled(c: np.ndarray) -> dict:
    """c: (n_scenes, 4) burn TP, FP, FN, TN."""
    tp, fp, fn, tn = c.sum(0).astype(np.float64)
    burn_iou = tp / max(tp + fp + fn, 1)
    bg_iou = tn / max(tn + fp + fn, 1)
    return {"burn_iou": round(burn_iou, 4), "burn_dice": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
            "bg_iou": round(bg_iou, 4), "mean_iou": round((burn_iou + bg_iou) / 2, 4)}


@torch.no_grad()
def score(model, cfg, set_name, device):
    data_cfg = dict(cfg["data"], val_json=SET_JSON[set_name])
    dm = build_datamodule(data_cfg)
    dm.setup(stage="validate")
    n_expected = len(json.load(open(SET_JSON[set_name]))["scenes"])
    assert len(dm.val_ds) == n_expected, f"{set_name}: loaded {len(dm.val_ds)} scenes, expected {n_expected}"
    scene_ids = [Path(s).name.replace("_merged.tif", "") for s, _ in dm.val_ds.samples]

    torch.manual_seed(0)  # the random-basis control draws U per image
    counts, batch_ious, expl = [], [], []
    for imgs, masks in dm.val_dataloader():
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            if hasattr(model, "ctx"):
                model.ctx.explained_var = None
            logits = model(imgs)
            if hasattr(model, "ctx") and model.ctx.explained_var is not None:
                expl += model.ctx.explained_var.tolist()
        pred = logits.argmax(1)
        p, t = pred == 1, masks == 1
        c = torch.stack([(p & t).sum((1, 2)), (p & ~t).sum((1, 2)), (~p & t).sum((1, 2)), (~p & ~t).sum((1, 2))], 1)
        counts.append(c.cpu().numpy())
        tp, fp, fn = c[:, 0].sum().item(), c[:, 1].sum().item(), c[:, 2].sum().item()
        batch_ious.append(tp / (tp + fp + fn + 1e-6))
    counts = np.concatenate(counts)
    res = pooled(counts)
    res["burn_iou_batch_mean"] = round(float(np.mean(batch_ious)), 4)
    res["n_scenes"] = len(scene_ids)
    res["per_scene"] = {sid: counts[i].tolist() for i, sid in enumerate(scene_ids)}
    if expl:
        res["topk_explained_var_mean"] = round(float(np.mean(expl)), 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="+", default=["dev"], choices=["dev", "test"])
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda")
    lit = build_module(cfg)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    lit.load_state_dict(sd)
    model = lit.model.to(device).eval()

    out = json.load(open(args.out_json)) if Path(args.out_json).exists() else {}
    out.update({"config": args.config, "ckpt": args.ckpt})
    if hasattr(model, "gammas"):
        out["spaco_gamma"] = model.gammas()
    for s in args.sets:
        out[s] = score(model, cfg, s, device)
        print(s, {k: v for k, v in out[s].items() if k != "per_scene"}, flush=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
