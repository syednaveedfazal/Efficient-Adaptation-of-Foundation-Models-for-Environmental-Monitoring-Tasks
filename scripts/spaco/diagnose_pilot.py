"""
scripts/spaco/diagnose_pilot.py — Diagnosis after a failed SpaCo pilot gate (dev scenes only; test is not touched).

  1. learned gamma per block (cov vs random)           -> gamma_by_block.png
  2. explained variance of the top-k modes (dev scenes)
  3. mode maps for 4 dev scenes                         -> mode_maps_dev.png
  4. mode-scar alignment: max |corr| between any of the k mode maps and the 32x32 burn fraction, cov vs random
  5. does the spatial term matter? the cov model on dev with gamma = 0 and with its basis swapped to random
Writes results/spaco/pilot_diagnosis.json.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.datasets.registry import build_datamodule
from src.training.module import build_module

sys.path.append(str(Path(__file__).resolve().parent))
from evaluate import pooled  # noqa: E402

OUT = Path("results/spaco")
RUNS = {"cov": "spaco_k8_cov_s42_010pct", "random": "spaco_k8_random_s42_010pct"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
COLORS = {"cov": "#2a78d6", "random": "#eb6834"}


def load_model(run, device):
    m = json.load(open(OUT / f"metrics/{run}.json"))
    cfg = yaml.safe_load(open(m["experiment"]["config"]))
    lit = build_module(cfg)
    lit.load_state_dict(torch.load(m["training"]["best_ckpt"], map_location="cpu", weights_only=False)["state_dict"])
    return lit.model.to(device).eval(), cfg


def patch_frac(mask):
    return mask.float().view(-1, 32, 16, 32, 16).mean((2, 4)).reshape(mask.shape[0], 1024)


def max_abs_corr(U, frac):
    """U: (N, k) mode maps, frac: (N,) burn fraction. Max |Pearson r| over modes (nan if frac is constant)."""
    f = frac - frac.mean()
    if f.norm() < 1e-8:
        return float("nan")
    Uc = U - U.mean(0, keepdim=True)
    r = (Uc * f[:, None]).sum(0) / (Uc.norm(dim=0) * f.norm() + 1e-12)
    return float(r.abs().max())


@torch.no_grad()
def main():
    device = torch.device("cuda")
    res = {"gamma": {}}
    model, cfg = load_model(RUNS["cov"], device)
    for name, run in RUNS.items():
        res["gamma"][name] = json.load(open(OUT / f"metrics/{run}.json"))["training"]["spaco_gamma"]

    dm = build_datamodule(dict(cfg["data"], val_json="splits/val_dev.json"))
    dm.setup(stage="validate")
    assert len(dm.val_ds) == 132
    ids = [Path(s).name.replace("_merged.tif", "") for s, _ in dm.val_ds.samples]
    loader = dm.val_dataloader()

    # 2-4: bases (static U only depends on the frozen patch embedding + block-0 norm1)
    captured = {}
    orig_clear = model.ctx.clear

    def keep_then_clear():
        captured["U"] = model.ctx.U
        captured["ev"] = model.ctx.explained_var
        orig_clear()

    variants = {"as_trained": {}, "gamma0": {}, "swap_random": {}}
    align = {"cov": [], "random": []}
    expl, maps = [], {}
    gammas = [g.detach().clone() for g in model.spaco_gamma]
    for v in variants:
        torch.manual_seed(0)
        counts = []
        model.ctx.basis = "random" if v == "swap_random" else "cov"
        for g, g0 in zip(model.spaco_gamma, gammas):
            g.copy_(torch.zeros_like(g0) if v == "gamma0" else g0)
        model.ctx.clear = keep_then_clear
        i0 = 0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(imgs).argmax(1)
            p, t = pred == 1, masks == 1
            counts.append(torch.stack([(p & t).sum((1, 2)), (p & ~t).sum((1, 2)), (~p & t).sum((1, 2)),
                                       (~p & ~t).sum((1, 2))], 1).cpu().numpy())
            if v == "as_trained":
                U = captured["U"].float().cpu()
                expl += captured["ev"].cpu().tolist()
                frac = patch_frac(masks).cpu()
                for b in range(U.shape[0]):
                    R = torch.linalg.qr(torch.randn(1024, U.shape[2])).Q
                    align["cov"].append(max_abs_corr(U[b], frac[b]))
                    align["random"].append(max_abs_corr(R, frac[b]))
                    maps[ids[i0 + b]] = (U[b, :, :4].numpy(), imgs[b].cpu().numpy(), masks[b].cpu().numpy())
            i0 += imgs.shape[0]
        model.ctx.clear = orig_clear
        variants[v] = pooled(np.concatenate(counts))
        print(v, variants[v], flush=True)
    res["dev_eval_cov_model"] = variants

    ev = np.array(expl)
    res["topk_explained_var"] = {"mean": round(ev.mean(), 4), "p10": round(np.percentile(ev, 10), 4),
                                 "p90": round(np.percentile(ev, 90), 4)}
    a = {k: np.array(v) for k, v in align.items()}
    ok = ~np.isnan(a["cov"])
    res["mode_scar_alignment_dev"] = {
        "n_scenes_with_variation": int(ok.sum()),
        "cov_mean_max_abs_r": round(float(a["cov"][ok].mean()), 4),
        "random_mean_max_abs_r": round(float(a["random"][ok].mean()), 4),
        "cov_gt_random_frac": round(float((a["cov"][ok] > a["random"][ok]).mean()), 4),
    }
    print(json.dumps({k: v for k, v in res.items() if k != "gamma"}, indent=1), flush=True)
    json.dump(res, open(OUT / "pilot_diagnosis.json", "w"), indent=1)

    # 1. gamma by block
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=150)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, linewidth=1)
    for name in ("cov", "random"):
        g = res["gamma"][name]
        ax.plot(range(len(g)), g, color=COLORS[name], linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.2, label=f"k=8 {name} basis")
        ax.annotate(name, (len(g) - 1, g[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=8)
    ax.set_xlabel("Block index", color=INK)
    ax.set_ylabel("Learned γ", color=INK)
    ax.tick_params(colors=MUTED)
    ax.set_title("SpaCo-LoRA pilot (seed 42, 10%): learned γ per block", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "gamma_by_block.png")

    # 3. mode maps for 4 dev scenes spread over burn fraction
    burn = json.load(open("splits/val_dev_test.json"))["burn_frac"]
    ranked = sorted(maps, key=lambda s: burn[s])
    pick = [ranked[int(q * (len(ranked) - 1))] for q in (0.55, 0.75, 0.9, 1.0)]
    stats = json.load(open(cfg["data"]["stats_path"]))
    mean, std = np.array(stats["mean"])[:, None, None], np.array(stats["std"])[:, None, None]
    fig, axes = plt.subplots(4, 7, figsize=(14, 8.4), dpi=120)
    for r, sid in enumerate(pick):
        U, img, mask = maps[sid]
        x = img * std + mean
        rgb = np.clip(x[[2, 1, 0]].transpose(1, 2, 0) / np.percentile(x[[2, 1, 0]], 99), 0, 1)
        nbr = (x[3] - x[5]) / (x[3] + x[5] + 1e-6)
        panels = [(rgb, None, "RGB"), (nbr, "Greys_r", "NBR"), (mask, "Greys", f"burn mask ({burn[sid]*100:.1f}%)")]
        for c, (im, cmap, title) in enumerate(panels):
            axes[r, c].imshow(im, cmap=cmap)
            axes[r, c].set_title(title, fontsize=8, color=INK)
        for j in range(4):
            m = U[:, j].reshape(32, 32)
            lim = np.abs(m).max()
            axes[r, 3 + j].imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
            axes[r, 3 + j].set_title(f"mode {j + 1}", fontsize=8, color=INK)
        for c in range(7):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    fig.suptitle("Top-4 spatial modes U (block-0 norm1, static) on 4 dev scenes; blue/red = sign", fontsize=10,
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "mode_maps_dev.png")
    print("done")


if __name__ == "__main__":
    main()
