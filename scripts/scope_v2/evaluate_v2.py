"""
scripts/scope_v2/evaluate_v2.py — Evaluate a SCOPE v2 (or plain LoRA) checkpoint on the validation set.

For every (keep, share) setting it reports burn IoU / Dice / mean IoU, encoder GFLOPs under the v2 convention
(2 x MACs, incl. LoRA + patch embed, averaged over all batches), % of LoRA FLOPs, tokens per block and exit rate.
When exit is on it also reports:
  - decoder only (exit off) and probe-8 / probe-16 only burn IoU,
  - combined with exit_overwrite = probe and = decoder,
  - the exit threshold calibrated on validation (most aggressive threshold with >= 98% probe pixel accuracy
    on the tokens it would exit) and the measured accuracy on the tokens that actually exited.

Usage:
  python scripts/scope_v2/evaluate_v2.py --config configs/scope_v2/gate50.yaml --ckpt <ckpt> --out_json <json>
      [--settings 0.5:0,1.0:0.25 ...]   keep:share pairs (default: the config's own setting)
  python scripts/scope_v2/evaluate_v2.py --plain --config configs/prithvi_fcn_lora_r8.yaml --ckpt <ckpt> ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.datasets.registry import build_datamodule
from src.evaluation.metrics import compute_metrics
from src.training.module import build_module
from src.utils.flops import LORA_ENCODER_GFLOPS, encoder_gflops

THRESH_GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999]
TARGET_ACC = 0.98


class MetricAvg:
    """Mean of per-batch metrics, the same aggregation Lightning's on_epoch logging uses."""

    def __init__(self):
        self.sums, self.n = {}, 0

    def add(self, logits, masks):
        for k, v in compute_metrics(logits.float(), masks, 2).items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.n += 1

    def result(self):
        return {k: round(v / max(self.n, 1), 4) for k, v in self.sums.items()}


def patch_burn_frac(masks: torch.Tensor) -> torch.Tensor:
    """(B, 512, 512) -> (B, 1024) fraction of burned pixels per 16x16 patch."""
    B = masks.shape[0]
    return masks.float().view(B, 32, 16, 32, 16).permute(0, 1, 3, 2, 4).reshape(B, 1024, 256).mean(-1)


def to_full_grid(model, tok_logits: torch.Tensor) -> torch.Tensor:
    """(B, N_reduced, 2) probe logits -> (B, 1024, 2), undoing token sharing."""
    info = model.last_share_info
    return model.sharing_module.unshare(tok_logits, info) if info.get("is_shared", False) else tok_logits


def grid_to_pixels(logits: torch.Tensor) -> torch.Tensor:
    B = logits.shape[0]
    return F.interpolate(logits.transpose(1, 2).reshape(B, 2, 32, 32).float(), scale_factor=16, mode="nearest")


def exited_patch_ids(model, orig_idx: int, b: int):
    info = model.last_share_info
    gp = model.sharing_module.group_patches
    if not info.get("is_shared", False):
        return [orig_idx]
    g = info["g"]
    if orig_idx < g:
        return gp[info["shared_groups"][b][orig_idx]].tolist()
    off = orig_idx - g
    return [int(gp[info["unshared_groups"][b][off // 4]][off % 4])]


def batch_gflops(model, scope: bool) -> tuple:
    if not scope:
        return LORA_ENCODER_GFLOPS, [1025] * 24, [1025] * 24
    attn, mlp, _ = model.get_token_counts()
    gated = getattr(model, "gate_enabled", False) and not getattr(model, "gate_is_random", False)
    gate_tokens = [a - 1 for a in attn] if gated else None
    probe_tokens = [attn[7] - 1, attn[15] - 1] if model.exit_enabled else None
    return encoder_gflops(attn, mlp, gate_tokens, probe_tokens), attn, mlp


@torch.no_grad()
def run_eval(model, loader, device, scope, keep=None, share=None, exit_on=None, overwrite="probe",
             probes=False, calib=False):
    """One pass over the validation set at a setting. Returns metrics, FLOPs, token counts (all batch means)."""
    if scope:
        model.set_inference_params(keep_ratio=keep, share_frac=share, exit_enabled=exit_on, exit_overwrite=overwrite)
        model.record_probes = probes or calib
    torch.manual_seed(42)  # random gate (rand50) is reproducible
    main, p8, p16 = MetricAvg(), MetricAvg(), MetricAvg()
    gfl, attn_all, mlp_all, exit_rates = [], [], [], []
    calib_data = {7: [], 15: []}
    exited_correct, exited_total = 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
        main.add(logits, masks)
        g, attn, mlp = batch_gflops(model, scope)
        gfl.append(g)
        attn_all.append(attn)
        mlp_all.append(mlp)
        if not scope:
            continue
        exit_rates.append(model.last_exit_rate)
        frac = patch_burn_frac(masks)
        if probes or calib:
            for idx, acc in ((7, p8), (15, p16)):
                full = to_full_grid(model, model.last_probe_logits[idx].float())  # (B, 1024, 2)
                if probes:
                    acc.add(grid_to_pixels(full), masks)
                if calib:
                    prob = full.softmax(-1)[..., 1]
                    conf = torch.maximum(prob, 1 - prob)
                    correct = torch.where(prob >= 0.5, frac, 1 - frac)  # pixel accuracy inside the patch
                    calib_data[idx].append(torch.stack([conf, correct], -1).reshape(-1, 2).cpu())
        if exit_on and model.last_exited_probe_logits:
            B = imgs.shape[0]
            for orig_idx, pl in model.last_exited_probe_logits.items():
                pred = (pl[:, 1] > pl[:, 0]).float()
                for b in range(B):
                    for pid in exited_patch_ids(model, orig_idx, b):
                        f = float(frac[b, pid])
                        exited_correct += f if pred[b] > 0 else 1 - f
                        exited_total += 1
    if scope:
        model.record_probes = False
    out = {
        "keep": keep, "share": share, "exit": exit_on, "exit_overwrite": overwrite if exit_on else None,
        "metrics": main.result(),
        "encoder_gflops": round(float(np.mean(gfl)), 2),
        "pct_lora_flops": round(100.0 * float(np.mean(gfl)) / LORA_ENCODER_GFLOPS, 1),
        "attn_tokens_per_block": [round(float(v), 1) for v in np.mean(attn_all, 0)],
        "mlp_tokens_per_block": [round(float(v), 1) for v in np.mean(mlp_all, 0)],
        "attn_tokens_mean": round(float(np.mean(attn_all)), 1),
        "mlp_tokens_mean": round(float(np.mean(mlp_all)), 1),
        "exit_rate": round(float(np.mean(exit_rates)), 4) if exit_rates else 0.0,
    }
    if probes:
        out["probe8_burn_iou"] = p8.result()["burn_iou"]
        out["probe16_burn_iou"] = p16.result()["burn_iou"]
    if exited_total:
        out["exited_patch_pixel_acc"] = round(exited_correct / exited_total, 4)
    if calib:
        out["_calib"] = {k: torch.cat(v) for k, v in calib_data.items()}
    return out


def calibrate(calib: dict, min_tokens: int = 100) -> dict:
    """Most aggressive (lowest) threshold t whose confident tokens (conf >= t, at least min_tokens) have >= 98%
    probe pixel accuracy. 1.01 (never exit) if no threshold qualifies."""
    res = {}
    for idx, data in calib.items():
        conf, correct = data[:, 0], data[:, 1]
        chosen, table = None, []
        for t in THRESH_GRID:
            sel = conf >= t
            n = int(sel.sum())
            acc = float(correct[sel].mean()) if n else 1.0
            table.append({"t": t, "frac_tokens": round(n / len(conf), 4), "acc": round(acc, 4)})
            if chosen is None and n >= min_tokens and acc >= TARGET_ACC:
                chosen = t
        res[idx] = {"threshold": chosen if chosen is not None else 1.01, "table": table}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--split_json", default="data/splits/seed_42/split_010pct.json")
    ap.add_argument("--settings", nargs="*", default=None, help="keep:share pairs")
    ap.add_argument("--plain", action="store_true", help="plain LoRA model (no SCOPE)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if not args.plain:
        cfg["distill"] = False  # the teacher is not needed for evaluation
    dm = build_datamodule(cfg["data"], split_json=args.split_json)
    dm.setup(stage="validate")
    loader = dm.val_dataloader()

    device = torch.device("cuda")
    lit = build_module(cfg)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    lit.load_state_dict({k: v for k, v in sd.items() if not k.startswith("teacher.")})
    model = lit.model.to(device).eval()

    result = {"config": args.config, "ckpt": args.ckpt, "lora_encoder_gflops": LORA_ENCODER_GFLOPS, "evals": []}
    if args.plain:
        r = run_eval(model, loader, device, scope=False)
        result["evals"].append(r)
        print(json.dumps(r["metrics"]), r["encoder_gflops"], flush=True)
    else:
        p = cfg["model"]["params"]
        own = (p.get("gate_keep_ratio", 1.0) if p.get("gate_enabled") or p.get("gate_is_random") else 1.0,
               p.get("share_frac", 0.0))
        settings = [own] + [tuple(float(v) for v in s.split(":")) for s in (args.settings or [])]
        settings = list(dict.fromkeys(settings))
        exit_on = bool(p.get("exit_enabled", False))
        if exit_on:
            # Calibrate thresholds once, at the run's own training setting, exit off
            cal = run_eval(model, loader, device, True, own[0], own[1], exit_on=False, calib=True)
            thr = calibrate(cal.pop("_calib"))
            result["exit_calibration"] = thr
            model.set_inference_params(exit_thresh_per_probe={k: v["threshold"] for k, v in thr.items()})
            print("calibrated thresholds:", {k: v["threshold"] for k, v in thr.items()}, flush=True)
        for keep, share in settings:
            if exit_on:
                r = run_eval(model, loader, device, True, keep, share, exit_on=True, overwrite="probe")
                dec = run_eval(model, loader, device, True, keep, share, exit_on=False, probes=True)
                ovw = run_eval(model, loader, device, True, keep, share, exit_on=True, overwrite="decoder")
                r["decoder_only_burn_iou"] = dec["metrics"]["burn_iou"]
                r["decoder_only_gflops"] = dec["encoder_gflops"]
                r["probe8_burn_iou"] = dec["probe8_burn_iou"]
                r["probe16_burn_iou"] = dec["probe16_burn_iou"]
                r["combined_overwrite_decoder_burn_iou"] = ovw["metrics"]["burn_iou"]
            else:
                r = run_eval(model, loader, device, True, keep, share, exit_on=False)
            r["is_own_setting"] = (keep, share) == own
            result["evals"].append(r)
            print(f"keep={keep} share={share}: burn_iou={r['metrics']['burn_iou']} "
                  f"gflops={r['encoder_gflops']} ({r['pct_lora_flops']}%) exit={r['exit_rate']} "
                  f"dec={r.get('decoder_only_burn_iou')} p8={r.get('probe8_burn_iou')} "
                  f"p16={r.get('probe16_burn_iou')}", flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out_json, "w"), indent=2)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
