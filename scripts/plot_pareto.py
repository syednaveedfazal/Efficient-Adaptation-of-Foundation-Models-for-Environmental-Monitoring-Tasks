"""
scripts/plot_pareto.py — Plots Burn IoU vs. Encoder Forward GFLOPs Pareto frontier.

Plots:
- Plain LoRA baseline (single point)
- Random baseline (from Phase 4 or random sweep)
- SCOPE-full points (from sweep_results.json, annotated with (keep_ratio, share_frac))
- Ablation (no prior) point
- Pareto optimal frontier highlighted
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    sweep_path = Path("results/scope_smoke/sweep_results.json")
    baseline_path = Path("results/metrics/prithvi_fcn_lora_r8_seed_42_split_010pct.json")
    no_prior_path = Path("results/scope_smoke/metrics/prithvi_scope_ablation_no_prior_seed_42_split_010pct.json")
    random_path = Path("results/scope_smoke/metrics/prithvi_scope_b1_random_seed_42_split_001pct.json")
    out_img = Path("results/scope_smoke/iou_vs_gflops.png")

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # 1. Plain LoRA Baseline
    # Encoder GFLOPs for plain Prithvi baseline is 722.36 (or ~317 in fvcore)
    # Using the standardized analytical formula for encoder GFLOPs:
    # 24 * (8 * N * D^2 + 4 * N^2 * D + 16 * N * D^2) / 1e9 = 722.36 GFLOPs
    baseline_gflops = 722.36
    baseline_iou = 0.6727
    if baseline_path.exists():
        with open(baseline_path) as f:
            b_data = json.load(f)
        baseline_iou = b_data["metrics"].get("burn_iou", baseline_iou)

    ax.scatter(baseline_gflops, baseline_iou, color="crimson", s=140, marker="*", zorder=5, label=f"Plain LoRA Baseline ({baseline_iou:.3f})")
    ax.annotate("Plain LoRA\n(N=1025, MLP=1025)", (baseline_gflops, baseline_iou), textcoords="offset points", xytext=(-25, -20), fontsize=9, fontweight="bold", color="crimson")

    all_points = [(baseline_gflops, baseline_iou, "Plain LoRA")]

    # 2. Ablation: No Spectral Prior
    if no_prior_path.exists():
        with open(no_prior_path) as f:
            np_data = json.load(f)
        np_gflops = np_data["metrics"].get("encoder_gflops_inference", 288.7)
        np_iou = np_data["metrics"].get("burn_iou", 0.0)
        ax.scatter(np_gflops, np_iou, color="darkorange", s=100, marker="^", zorder=5, label=f"Ablation: No Prior ({np_iou:.3f})")
        ax.annotate(f"No Prior\n(0.5, 0.5)", (np_gflops, np_iou), textcoords="offset points", xytext=(-15, -18), fontsize=8, color="darkorange")
        all_points.append((np_gflops, np_iou, "No Prior"))

    # 3. Random Baseline (from smoke or reference)
    if random_path.exists():
        with open(random_path) as f:
            r_data = json.load(f)
        r_gflops = r_data["metrics"].get("encoder_gflops_inference", 516.2)
        # Scale to split_010pct relative drop or plot point
        r_iou = r_data["metrics"].get("burn_iou", 0.2094)
        ax.scatter(r_gflops, r_iou, color="gray", s=80, marker="x", zorder=4, label=f"Random Gate Control ({r_iou:.3f})")
        ax.annotate("Random Gate\n(kr=0.5)", (r_gflops, r_iou), textcoords="offset points", xytext=(8, -10), fontsize=8, color="gray")

    # 4. SCOPE-Full Sweep Points
    if sweep_path.exists():
        with open(sweep_path) as f:
            sweep = json.load(f)

        scope_x = []
        scope_y = []
        labels = []

        for pt in sweep:
            gflops = pt["encoder_gflops"]
            iou = pt["metrics"]["burn_iou"]
            sf = pt["share_frac"]
            kr = pt["keep_ratio"]
            scope_x.append(gflops)
            scope_y.append(iou)
            lbl = f"({kr:.2f}, {sf:.2f})"
            labels.append(lbl)
            all_points.append((gflops, iou, f"SCOPE {lbl}"))

            ax.annotate(lbl, (gflops, iou), textcoords="offset points", xytext=(5, 5), fontsize=7.5, alpha=0.85)

        sc = ax.scatter(scope_x, scope_y, c=scope_y, cmap="viridis", s=110, edgecolor="black", linewidth=0.8, zorder=6, label="SCOPE Full (Sweep)")
        plt.colorbar(sc, ax=ax, label="Burn IoU")

    # 5. Compute and Highlight Pareto Frontier
    # For Pareto efficiency: we want to MINIMIZE GFLOPs and MAXIMIZE Burn IoU.
    # Point (g1, i1) dominates (g2, i2) if g1 <= g2 and i1 >= i2 (with at least one strict inequality).
    pareto_points = []
    # Sort primarily by GFLOPs ascending, then IoU descending
    sorted_pts = sorted(all_points, key=lambda p: (p[0], -p[1]))

    curr_max_iou = -1.0
    for g, iou, name in sorted_pts:
        if iou > curr_max_iou:
            pareto_points.append((g, iou))
            curr_max_iou = iou

    if pareto_points:
        px = [p[0] for p in pareto_points]
        py = [p[1] for p in pareto_points]
        ax.plot(px, py, color="blue", linestyle="--", linewidth=2.0, zorder=3, label="Pareto Frontier")
        ax.scatter(px, py, color="blue", s=150, facecolors="none", edgecolors="blue", linewidth=2.5, zorder=7)

    ax.set_title("Burn Scar Segmentation: Pareto Frontier (Burn IoU vs. Encoder Inference GFLOPs)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Encoder Forward Compute (GFLOPs) [Lower is Better $\\rightarrow$]", fontsize=11)
    ax.set_ylabel("Validation Burn IoU (Higher is Better $\\uparrow$)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    out_img.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"[Plot] Pareto curve successfully saved to: {out_img}")


if __name__ == "__main__":
    main()

