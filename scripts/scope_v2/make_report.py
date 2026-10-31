"""
scripts/scope_v2/make_report.py — Collect SCOPE v2 eval/metrics JSONs into a results table,
the iou_vs_flops.png plot and the go/no-go numbers (results/scope_v2/results_table.md, summary.json).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/scope_v2")
RUNS = ["rand50", "share25", "share50", "gate50", "gate50_prior", "exit", "combo_mild", "combo_strong"]
# Categorical slots in fixed order (default validated palette); LoRA / rand50 are neutral references
COLORS = {"share25": "#2a78d6", "share50": "#eb6834", "gate50": "#1baf7a", "gate50_prior": "#eda100",
          "exit": "#e87ba4", "combo_mild": "#008300", "combo_strong": "#6250d6"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def load(path):
    return json.load(open(path)) if Path(path).exists() else None


def fmt(v, nd=4):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    rows, points = [], {}
    lora = load(ROOT / "eval/lora.json")
    lora_m = load(ROOT / "metrics/prithvi_fcn_lora_r8_seed_42_split_010pct.json")
    if lora:
        e = lora["evals"][0]
        rows.append(dict(run="lora", keep=1.0, share=0.0, m=e["metrics"], gflops=e["encoder_gflops"],
                         pct=e["pct_lora_flops"], attn=e["attn_tokens_mean"], mlp=e["mlp_tokens_mean"], exit=0.0,
                         epochs=82, best=65, time=lora_m["hardware"]["training_duration_seconds"],
                         vram=lora_m["hardware"]["max_memory_allocated_mb"], own=True))
        points["lora"] = [(e["pct_lora_flops"], e["metrics"]["burn_iou"])]
    for run in RUNS:
        ev = load(ROOT / f"eval/{run}.json")
        mt = load(ROOT / f"metrics/prithvi_scope_{run}_seed_42_split_010pct.json")
        if not ev or not mt:
            continue
        tr, hw = mt.get("training", {}), mt["hardware"]
        pts = []
        for e in ev["evals"]:
            rows.append(dict(run=run, keep=e["keep"], share=e["share"], m=e["metrics"], gflops=e["encoder_gflops"],
                             pct=e["pct_lora_flops"], attn=e["attn_tokens_mean"], mlp=e["mlp_tokens_mean"],
                             exit=e["exit_rate"], dec=e.get("decoder_only_burn_iou"),
                             p8=e.get("probe8_burn_iou"), p16=e.get("probe16_burn_iou"),
                             ovw=e.get("combined_overwrite_decoder_burn_iou"),
                             acc=e.get("exited_patch_pixel_acc"),
                             epochs=tr.get("epochs_run"), best=tr.get("best_epoch"),
                             time=hw["training_duration_seconds"], vram=hw["max_memory_allocated_mb"],
                             own=e.get("is_own_setting", False)))
            pts.append((e["pct_lora_flops"], e["metrics"]["burn_iou"]))
        points[run] = sorted(pts)

    # ── table ────────────────────────────────────────────────────────────────
    hdr = ("| Run | Keep | Share | Own setting | Burn IoU | Dice | Mean IoU | Decoder-only IoU | Probe-8 / Probe-16 IoU "
           "| Overwrite=decoder IoU | Exited-patch acc | Enc GFLOPs | % LoRA | Attn / MLP tokens per block | Exit rate "
           "| Epochs run | Best epoch | Train time (s) | Peak VRAM (MB) |")
    lines = [hdr, "|" + "---|" * 19]
    for r in rows:
        probe = f"{fmt(r.get('p8'))} / {fmt(r.get('p16'))}" if r.get("p8") is not None else "–"
        lines.append(
            f"| `{r['run']}` | {r['keep']} | {r['share']} | {'yes' if r['own'] else ''} | {r['m']['burn_iou']:.4f} "
            f"| {r['m']['burn_dice']:.4f} | {r['m']['mean_iou']:.4f} | {fmt(r.get('dec'))} | {probe} "
            f"| {fmt(r.get('ovw'))} | {fmt(r.get('acc'))} | {r['gflops']:.1f} | {r['pct']:.1f} "
            f"| {r['attn']:.0f} / {r['mlp']:.0f} | {r['exit']*100:.1f}% | {fmt(r['epochs'])} | {fmt(r['best'])} "
            f"| {r['time']:.0f} | {r['vram']:.0f} |")
    (ROOT / "results_table.md").write_text("\n".join(lines) + "\n")

    # ── go / no-go ───────────────────────────────────────────────────────────
    summary = {}
    if lora:
        lora_iou = rows[0]["m"]["burn_iou"]
        rand = [r for r in rows if r["run"] == "rand50" and r["own"]]
        cands = [r for r in rows if r["run"] not in ("lora", "rand50") and r["pct"] <= 60.0]
        best = max(cands, key=lambda r: r["m"]["burn_iou"]) if cands else None
        summary = {"lora_burn_iou": lora_iou, "lora_gflops": rows[0]["gflops"]}
        if rand:
            summary["rand50"] = {"burn_iou": rand[0]["m"]["burn_iou"], "pct": rand[0]["pct"]}
        if best:
            gap = lora_iou - best["m"]["burn_iou"]
            beat = best["m"]["burn_iou"] - rand[0]["m"]["burn_iou"] if rand else None
            summary["best_le60"] = {"run": best["run"], "keep": best["keep"], "share": best["share"],
                                    "burn_iou": best["m"]["burn_iou"], "pct": best["pct"],
                                    "gap_to_lora": round(gap, 4),
                                    "margin_over_rand50": None if beat is None else round(beat, 4),
                                    "rand50_flops_matched_or_higher": bool(rand) and best["pct"] <= rand[0]["pct"] + 0.5}
            summary["go"] = gap <= 0.02 and beat is not None and beat >= 0.03 and \
                summary["best_le60"]["rand50_flops_matched_or_higher"]
        # IoU lost per % FLOPs saved, at each run's own setting
        eff = {}
        for r in rows:
            if r["own"] and r["run"] not in ("lora",) and r["pct"] < 99.5:
                eff[r["run"]] = round((lora_iou - r["m"]["burn_iou"]) / (100.0 - r["pct"]), 5)
        summary["iou_loss_per_pct_flops_saved"] = eff
    json.dump(summary, open(ROOT / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=150)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if "lora" in points:
        lx, ly = points["lora"][0]
        ax.axvspan(0, 60, ymin=0, ymax=1, color="#f2f1ec", zorder=0)
        ax.axhline(ly - 0.02, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(1.5, ly - 0.02 + 0.004, "LoRA − 0.02", color=MUTED, fontsize=8)
        ax.scatter([lx], [ly], marker="*", s=220, color=INK, zorder=5, edgecolor="white", linewidth=1.5)
        ax.annotate("LoRA r=8", (lx, ly), xytext=(0, -16), textcoords="offset points", ha="center", color=INK, fontsize=9)
    if "rand50" in points:
        x, y = points["rand50"][0]
        ax.scatter([x], [y], marker="X", s=90, color=MUTED, zorder=5, edgecolor="white", linewidth=1.2)
        ax.annotate("rand50", (x, y), xytext=(6, -12), textcoords="offset points", color=MUTED, fontsize=9)
    for run in RUNS[1:]:
        if run not in points:
            continue
        xs, ys = zip(*points[run])
        if len(xs) > 1:
            # Swept runs: every keep/share point as a marker, the line is the run's Pareto frontier
            ax.scatter(xs, ys, color=COLORS[run], s=28, alpha=0.55, edgecolor="white", linewidth=1, zorder=3)
            front, best = [], -1.0
            for x, y in sorted(points[run]):
                if y > best:
                    front.append((x, y))
                    best = y
            fx, fy = zip(*front)
            ax.plot(fx, fy, color=COLORS[run], linewidth=2, marker="o", markersize=6, markeredgecolor="white",
                    markeredgewidth=1.5, label=f"{run} (frontier of keep × share sweep)", zorder=4)
            ax.annotate(run, (fx[0], fy[0]), xytext=(-6, -14), textcoords="offset points", ha="right",
                        color=INK, fontsize=8)
        else:
            ax.scatter(xs, ys, color=COLORS[run], s=60, edgecolor="white", linewidth=1.5, label=run, zorder=4)
            ax.annotate(run, (xs[-1], ys[-1]), xytext=(6, 4), textcoords="offset points", color=INK, fontsize=8)
    ax.set_xlim(0, 105)
    ax.text(30, ax.get_ylim()[1] - 0.01, "go region: ≤ 60% LoRA FLOPs", ha="center", va="top", color=MUTED, fontsize=8)
    ax.set_xlabel("Encoder FLOPs, % of LoRA r=8 (732 GFLOPs = 100%)", color=INK)
    ax.set_ylabel("Val burn IoU", color=INK)
    ax.tick_params(colors=MUTED)
    ax.set_title("SCOPE v2 on the 10% split (seed 42): burn IoU vs encoder FLOPs", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(ROOT / "iou_vs_flops.png")
    print("wrote", ROOT / "iou_vs_flops.png")


if __name__ == "__main__":
    sys.exit(main())
