"""
scripts/scope_v2/diagnose.py — Phase 1 diagnostics for SCOPE v1 (no training).

Uses the v1 s4_full checkpoint and writes results/scope_v2/diagnosis.json:
  1. gate mask audit (train mode, 4 val batches): mask value histogram + per-block keep rate
  2. decomposition of s4_full predictions at keep {1.0, 0.5}, share 0:
       decoder only (exit off), probe-8 only, probe-16 only, combined (exit on, as v1)
  3. sanity: keep 1.0, share 0, exit off, nbr_refine on / off
  4. extra: decoder-only keep sweep, and eval with the stochastic train-mode gate
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

import src.models.scope as scope_mod
from src.datasets.registry import build_datamodule
from src.evaluation.metrics import compute_metrics
from src.models.scope import ScopeGateBlock
from src.training.module import build_module


def probe_logits_to_pixels(logits: torch.Tensor) -> torch.Tensor:
    """(B, 1024, 2) patch logits -> (B, 2, 512, 512) by nearest upsampling (same as exit overwrite)."""
    B = logits.shape[0]
    grid = logits.transpose(1, 2).reshape(B, 2, 32, 32)
    return F.interpolate(grid.float(), scale_factor=16, mode="nearest")


class MetricAvg:
    """Mean of per-batch metrics, the same aggregation Lightning's on_epoch logging uses."""

    def __init__(self):
        self.sums, self.n = {}, 0

    def add(self, logits, masks):
        m = compute_metrics(logits.float(), masks, 2)
        for k, v in m.items():
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
        self.n += 1

    def result(self):
        return {k: round(v / max(self.n, 1), 4) for k, v in self.sums.items()}


@torch.no_grad()
def evaluate(model, loader, device, keep, share, exit_on, refine=True, probes=False, train_gate=False):
    model.eval()
    model.set_inference_params(keep_ratio=keep, share_frac=share, exit_enabled=exit_on)
    saved_refine = model.refine_net
    if not refine:
        model.refine_net = None
    if train_gate:
        for b in model.blocks:
            if isinstance(b, ScopeGateBlock):
                b.train()

    captured = {}
    hooks = []
    if probes:
        for idx in (7, 15):
            hooks.append(model.blocks[idx].register_forward_hook(
                lambda mod, inp, out, idx=idx: captured.__setitem__(idx, out.detach())))

    main, p8, p16 = MetricAvg(), MetricAvg(), MetricAvg()
    exit_rates, mlp_tokens, attn_tokens = [], [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(imgs)
            if probes:
                l8 = model.probe8(captured[7][:, 1:, :])
                l16 = model.probe16(captured[15][:, 1:, :])
        main.add(logits, masks)
        if probes:
            p8.add(probe_logits_to_pixels(l8), masks)
            p16.add(probe_logits_to_pixels(l16), masks)
        a, m, e = model.get_token_counts()
        attn_tokens.append(sum(a) / len(a))
        mlp_tokens.append(sum(m) / len(m))
        exit_rates.append(e)

    for h in hooks:
        h.remove()
    model.refine_net = saved_refine
    model.eval()
    out = {
        "keep": keep, "share": share, "exit": exit_on, "refine": refine, "train_gate": train_gate,
        "metrics": main.result(),
        "attn_tokens_mean": round(sum(attn_tokens) / len(attn_tokens), 1),
        "mlp_tokens_mean": round(sum(mlp_tokens) / len(mlp_tokens), 1),
        "exit_rate": round(sum(exit_rates) / len(exit_rates), 4),
    }
    if probes:
        out["probe8"] = p8.result()
        out["probe16"] = p16.result()
    print(json.dumps({k: v for k, v in out.items() if k != "metrics"}), "burn_iou", out["metrics"]["burn_iou"],
          "| p8", out.get("probe8", {}).get("burn_iou"), "p16", out.get("probe16", {}).get("burn_iou"), flush=True)
    return out


@torch.no_grad()
def gate_audit(model, loader, device, n_batches=4):
    """Train-mode forward: record every mask produced by gumbel_sigmoid."""
    recorded = []
    orig = scope_mod.gumbel_sigmoid

    def recording(*a, **kw):
        y = orig(*a, **kw)
        recorded.append(y.detach().float().flatten().cpu())
        return y

    scope_mod.gumbel_sigmoid = recording
    model.train()
    per_block = [[] for _ in range(24)]
    try:
        for i, (imgs, masks) in enumerate(loader):
            if i >= n_batches:
                break
            start = len(recorded)
            with torch.autocast("cuda", dtype=torch.float16):
                model.set_current_targets(masks.to(device))
                model(imgs.to(device))
            for b_idx, y in enumerate(recorded[start:]):
                per_block[b_idx].append(y.mean().item())
    finally:
        scope_mod.gumbel_sigmoid = orig
        model.eval()

    allv = torch.cat(recorded)
    exact0 = int((allv == 0).sum())
    exact1 = int((allv == 1).sum())
    frac = allv[(allv != 0) & (allv != 1)]
    hist = torch.histc(allv, bins=10, min=0.0, max=1.0).tolist()
    res = {
        "n_values": int(allv.numel()),
        "exact_0": exact0,
        "exact_1": exact1,
        "fractional": int(frac.numel()),
        "fractional_abs_dev_from_nearest_int_max": float((frac - frac.round()).abs().max()) if frac.numel() else 0.0,
        "hist_10_bins_0_to_1": [int(h) for h in hist],
        "keep_rate_overall": round(float((allv > 0.5).float().mean()), 4),
        "keep_rate_per_block": [round(sum(v) / len(v), 3) for v in per_block],
    }
    print("gate audit:", json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/scope/s4_full.yaml")
    ap.add_argument("--ckpt", default="results/scope_smoke/checkpoints/prithvi_scope_s4_full_seed_42_split_010pct/"
                                      "epoch=28-val/burn_iou=0.5135.ckpt")
    ap.add_argument("--split_json", default="data/splits/seed_42/split_010pct.json")
    ap.add_argument("--out_json", default="results/scope_v2/diagnosis.json")
    args = ap.parse_args()

    torch.manual_seed(42)
    cfg = yaml.safe_load(open(args.config))
    dm = build_datamodule(cfg["data"], split_json=args.split_json)
    dm.setup(stage="validate")
    loader = dm.val_dataloader()

    device = torch.device("cuda")
    lit = build_module(cfg)
    lit.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"])
    model = lit.model.to(device).eval()

    out = {"ckpt": args.ckpt}
    out["gate_audit"] = gate_audit(model, loader, device)

    runs = []
    # v1 reproduction point (the checkpoint's own setting) as a sanity check of this harness
    runs.append(("v1_repro_share0.5_keep0.5_exit", dict(keep=0.5, share=0.5, exit_on=True)))
    for keep in (1.0, 0.5):
        runs.append((f"decoder_only_keep{keep}", dict(keep=keep, share=0.0, exit_on=False, probes=True)))
        runs.append((f"combined_keep{keep}", dict(keep=keep, share=0.0, exit_on=True)))
    runs.append(("sanity_keep1_refine_off", dict(keep=1.0, share=0.0, exit_on=False, refine=False)))
    for keep in (0.75, 0.65, 0.3):
        runs.append((f"decoder_only_keep{keep}", dict(keep=keep, share=0.0, exit_on=False)))
    runs.append(("train_gate_decoder_only", dict(keep=1.0, share=0.0, exit_on=False, train_gate=True)))

    out["evals"] = {}
    for name, kw in runs:
        torch.manual_seed(42)
        out["evals"][name] = evaluate(model, loader, device, **kw)
    out["evals"]["sanity_keep1_refine_on"] = out["evals"]["decoder_only_keep1.0"]

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
