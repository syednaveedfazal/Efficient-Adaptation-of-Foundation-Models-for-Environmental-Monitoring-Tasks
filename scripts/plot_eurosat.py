"""
scripts/plot_eurosat.py — EuroSAT classification analysis plots (decluttered).

Design (see the dataviz method): the data is 4-D (backbone x adaptation x budget x
metric). Instead of cramming every backbone x adaptation combo onto one line chart,
we spend the extra dimensions on LAYOUT and MARK-STYLE:

  * Colour ALWAYS encodes the adaptation (4 fixed, colourblind-safe hues), reused
    identically in every figure — "colour follows the entity".
  * Backbone is shown by facet (line charts), marker shape (scatter), or grouping
    (dot plot) — never by a 5th..8th cycled hue.
  * Redundant marker shapes per adaptation so identity is never colour-alone.

Figures written to results/plots/:
  eurosat_accuracy_vs_budget.png   faceted small multiples (one panel per backbone)
  eurosat_macrof1_vs_budget.png    same, macro-F1
  eurosat_param_efficiency.png     acc@100% vs trainable params, with Pareto frontier
  eurosat_ranking_at_100pct.png    sorted horizontal dot plot (the "who wins" view)

Invoked via:  python scripts/plot_results.py --dataset eurosat
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BUDGETS = [0.01, 0.10, 0.25, 1.00]
BUDGET_MAP = {b: i for i, b in enumerate(BUDGETS)}
BUDGET_LABELS = {0.01: "1% (216)", 0.10: "10% (2160)", 0.25: "25% (5400)", 1.00: "100% (21600)"}

# Adaptation = colour (dataviz default categorical palette, slots 1-5, fixed order)
ADAPT_ORDER  = ["lora_r8", "lora_r16", "full_ft", "linear_probe", "vpt", "scratch"]
ADAPT_COLOR  = {"lora_r8": "#2a78d6", "lora_r16": "#eb6834",
                "full_ft": "#1baf7a", "linear_probe": "#eda100",
                "vpt": "#9467bd", "scratch": "#000000"}
ADAPT_MARKER = {"lora_r8": "o", "lora_r16": "s", "full_ft": "^", "linear_probe": "D",
                "vpt": "P", "scratch": "X"}
ADAPT_LABEL  = {"lora_r8": "LoRA r=8", "lora_r16": "LoRA r=16",
                "full_ft": "Full FT", "linear_probe": "Linear Probe",
                "vpt": "VPT", "scratch": "Scratch"}

# Backbone = facet / marker shape (NOT colour)
BACKBONES        = ["prithvi", "dinov2", "cnn"]
BACKBONE_LABEL   = {"prithvi": "Prithvi-300M", "dinov2": "DINOv2-Large", "cnn": "CNN"}
BACKBONE_MARKER  = {"prithvi": "o", "dinov2": "^", "cnn": "X"}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#c9c9c4"


def _load():
    """data[backbone][adapt][budget] = {'accuracy':[...]}; params[(bb,adapt)]=Mtrain; gflops[(bb,adapt)]=GF; vram[(bb,adapt)]=[...]"""
    data, params, gflops, vram = {}, {}, {}, {}
    files = list(Path("results/metrics").glob("eurosat_*.json"))
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"[WARN] {f.name}: {e}"); continue
        if "experiment" not in d or "metrics" not in d:
            continue
        e, m = d["experiment"], d["metrics"]
        name = e["model_name"].replace("eurosat_", "")   # "<backbone>_<adapt>"
        bb, adapt = name.split("_", 1)
        if bb not in BACKBONES or adapt not in ADAPT_ORDER:
            continue
        b = round(float(e["label_budget"]), 2)
        data.setdefault(bb, {}).setdefault(adapt, {}).setdefault(b, {"accuracy": [], "macro_f1": []})
        data[bb][adapt][b]["accuracy"].append(m.get("accuracy", 0.0))
        data[bb][adapt][b]["macro_f1"].append(m.get("macro_f1", 0.0))
        params[(bb, adapt)] = int(e.get("trainable_params", 0)) / 1e6
        
        fwd_g = float(e.get("fwd_gflops", 0.0))
        bwd_g = float(e.get("bwd_gflops", 0.0))
        if fwd_g > 0 or bwd_g > 0:
            gflops[(bb, adapt)] = (fwd_g, bwd_g)
        elif e.get("total_gflops", 0.0) > 0:
            tot = float(e.get("total_gflops"))
            gflops[(bb, adapt)] = (tot / 3.0, 2.0 * tot / 3.0)
            
        mem = d.get("hardware", {}).get("max_memory_allocated_mb", 0.0)
        if mem > 0:
            vram.setdefault((bb, adapt), []).append(float(mem))
            
    return data, params, gflops, vram


def _summary_table(data, params, gflops, vram, target_combos, out, exploded=False):
    rows = []
    if exploded:
        headers = ["Method", "", "Acc\n@ 10%", "Acc\n@ 100%", "F1\n@ 10%", "F1\n@ 100%", "", "GFLOPs\n(Total)", "VRAM\n(MB)", "", "Key Findings"]
    else:
        headers = ["Method", "Acc\n@ 10%", "Acc\n@ 100%", "F1\n@ 10%", "F1\n@ 100%", "GFLOPs\n(Total)", "VRAM\n(MB)", "Key Findings"]
    
    for bb, adapt, name_display, profile in target_combos:
        if bb not in data or adapt not in data[bb]:
            continue
        bd = data[bb][adapt]
        
        def fmt(budget, metric):
            if budget in bd and bd[budget][metric]:
                vals = [v * 100.0 for v in bd[budget][metric]]
                m = np.mean(vals)
                s = np.std(vals) if len(vals) > 1 else 0.0
                if s >= 0.05:
                    return f"{m:.1f}±{s:.1f}%"
                else:
                    return f"{m:.1f}%"
            return "N/A"
            
        acc_10_str = fmt(0.10, "accuracy")
        acc_100_str = fmt(1.00, "accuracy")
        f1_10_str = fmt(0.10, "macro_f1")
        f1_100_str = fmt(1.00, "macro_f1")
        
        if (bb, adapt) in gflops:
            gf = gflops[(bb, adapt)]
            tot = sum(gf) if isinstance(gf, (tuple, list)) else gf
            g_str = f"{tot:.1f}"
        else:
            g_str = "N/A"
            
        if (bb, adapt) in vram and vram[(bb, adapt)]:
            v_val = np.mean(vram[(bb, adapt)])
            v_str = f"{v_val:,.0f}"
        else:
            v_str = "N/A"
            
        if exploded and bb == "cnn":
            rows.append(["", "", "", "", "", "", "", "", "", "", ""])
            
        if exploded:
            rows.append([
                name_display,
                "",
                acc_10_str,
                acc_100_str,
                f1_10_str,
                f1_100_str,
                "",
                g_str,
                v_str,
                "",
                profile
            ])
        else:
            rows.append([
                name_display,
                acc_10_str,
                acc_100_str,
                f1_10_str,
                f1_100_str,
                g_str,
                v_str,
                profile
            ])
        
    if not rows:
        return
        
    num_data_rows = sum(1 for r in rows if r[0] != "")
    num_spacer_rows = sum(1 for r in rows if r[0] == "")
    fig_height = 0.375 * (num_data_rows + 1) + (0.1875 * num_spacer_rows if exploded else 0.0) + 0.5
    fig, ax = plt.subplots(figsize=(13.5, fig_height), dpi=300)
    ax.axis('off')
    
    if exploded:
        col_widths = [0.19, 0.015, 0.10, 0.10, 0.09, 0.09, 0.015, 0.09, 0.09, 0.015, 0.22]
    else:
        col_widths = [0.19, 0.11, 0.11, 0.09, 0.09, 0.09, 0.09, 0.23]
        
    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colWidths=col_widths
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    
    header_color = '#eb7a34'  # Muted orange matching PPT
    row_colors = ['#eaf1f8', '#d6e3f0']  # Alternating blue rows
    
    normal_cell_height = 0.81 / fig_height
    spacer_row_height = (0.015 * 13.5) / fig_height
    
    for (row, col), cell in tbl.get_celld().items():
        is_spacer_col = exploded and col in [1, 6, 9]
        is_spacer_row = exploded and row > 0 and rows[row-1][0] == ""
        
        if is_spacer_row:
            cell.set_height(spacer_row_height)
            cell.set_facecolor('none')
            cell.set_edgecolor('none')
        else:
            cell.set_height(normal_cell_height)
            
        if is_spacer_col:
            cell.set_facecolor('none')
            cell.set_edgecolor('none')
        elif is_spacer_row:
            pass
        elif row == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=12)
            cell.set_facecolor(header_color)
            cell.set_edgecolor('white')
        else:
            data_idx = 0
            for i in range(row):
                if rows[i][0] != "":
                    data_idx += 1
                    
            cell.set_facecolor(row_colors[(data_idx - 1) % 2])
            cell.set_edgecolor('white')
            # Bold the optimal row (LoRA r=16) like in PPT
            if 'r=16' in rows[row-1][0]:
                cell.set_text_props(weight='bold')
                
    fig.savefig(out, bbox_inches='tight', dpi=300)
    print(f"Generated plot: {out}")
    plt.close(fig)


def _bar_chart_vram(vram, plots_dir):
    if not vram:
        return
    for bb in BACKBONES:
        if bb == "cnn":
            continue
        adapts = [a for a in ADAPT_ORDER if (bb, a) in vram and vram[(bb, a)]]
        if not adapts:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=300)
        means_vram = [np.mean(vram[(bb, a)]) for a in adapts]
        stds_vram = [np.std(vram[(bb, a)]) if len(vram[(bb, a)]) > 1 else 0.0 for a in adapts]
        plot_colors = [EUROSAT_MODEL_COLORS.get((bb, a), ADAPT_COLOR.get(a, "#7f7f7f")) for a in adapts]
        plot_names = [ADAPT_LABEL.get(a, a) for a in adapts]
        
        if ("cnn", "scratch") in vram and vram[("cnn", "scratch")]:
            means_vram.append(np.mean(vram[("cnn", "scratch")]))
            stds_vram.append(np.std(vram[("cnn", "scratch")]) if len(vram[("cnn", "scratch")]) > 1 else 0.0)
            plot_colors.append("#d95f02")
            plot_names.append("CNN (scratch)")
        
        bars = ax.bar(plot_names, means_vram, yerr=stds_vram, color=plot_colors, alpha=0.85, edgecolor="none", capsize=5, width=0.4)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f} MB",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight="bold"
            )
            
        bb_title = BACKBONE_LABEL.get(bb, bb.title())
        ax.set_title(f"EuroSAT: Peak GPU VRAM Usage ({bb_title})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Adaptation Method", fontsize=10, fontweight="bold")
        ax.set_ylabel("Peak VRAM Allocated (MB)", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        ax.set_ylim(0, max(means_vram) * 1.2 if means_vram else 1000)
        fig.tight_layout()
        out_p = plots_dir / f"eurosat_vram_{bb}.png"
        fig.savefig(out_p, bbox_inches="tight")
        print(f"Generated plot: {out_p}")
        plt.close(fig)


def _bar_chart_gflops(gflops, plots_dir):
    if not gflops:
        return
    for bb in BACKBONES:
        if bb == "cnn":
            continue
        adapts = [a for a in ADAPT_ORDER if (bb, a) in gflops]
        if not adapts:
            continue
        fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=300)
        fwd_vals = [gflops[(bb, a)][0] for a in adapts]
        bwd_vals = [gflops[(bb, a)][1] for a in adapts]
        plot_names = [ADAPT_LABEL.get(a, a) for a in adapts]
        
        fwd_colors = []
        bwd_colors = []
        for a in adapts:
            if bb == "prithvi":
                fwd_colors.append("#6baed6") # Lighter blue
                bwd_colors.append("#2171b5") # Darker blue
            elif bb == "dinov2":
                fwd_colors.append("#a6dba0") # Lighter green
                bwd_colors.append("#1b7837") # Darker green
        
        if ("cnn", "scratch") in gflops:
            fwd_vals.append(gflops[("cnn", "scratch")][0])
            bwd_vals.append(gflops[("cnn", "scratch")][1])
            plot_names.append("CNN (scratch)")
            fwd_colors.append("#fdae6b") # Lighter orange
            bwd_colors.append("#d95f02") # Darker orange
            
        bars_fwd = ax.bar(plot_names, fwd_vals, color=fwd_colors, alpha=0.9, width=0.45)
        bars_bwd = ax.bar(plot_names, bwd_vals, bottom=fwd_vals, color=bwd_colors, alpha=0.9, width=0.45)
        
        for i in range(len(plot_names)):
            total = fwd_vals[i] + bwd_vals[i]
            ax.annotate(
                f"{total:.1f} GF",
                xy=(i, total),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8.5, fontweight="bold"
            )
            
        bb_title = BACKBONE_LABEL.get(bb, bb.title())
        ax.set_title(f"EuroSAT: Computational Complexity ({bb_title})", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Adaptation Method", fontsize=10, fontweight="bold")
        ax.set_ylabel("Compute Complexity (GFLOPs)", fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        ax.set_ylim(0, max([f + b for f, b in zip(fwd_vals, bwd_vals)]) * 1.3 if fwd_vals else 100)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#cccccc', edgecolor='none', label='Forward Pass (Lighter)'),
            Patch(facecolor='#737373', edgecolor='none', label='Backward Pass (Darker)')
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=9)
        fig.tight_layout()
        out_p = plots_dir / f"eurosat_gflops_{bb}.png"
        fig.savefig(out_p, bbox_inches="tight")
        print(f"Generated plot: {out_p}")
        plt.close(fig)


def main():
    plots_dir = Path("results/plots"); plots_dir.mkdir(parents=True, exist_ok=True)
    data, params, gflops, vram = _load()
    if not data:
        print("[ERROR] No eurosat_*.json in results/metrics/. Train EuroSAT models first!")
        return
    n = sum(len(bd[a]) for bd in data.values() for a in bd)
    print(f"Found {n} EuroSAT runs across {len(data)} backbone(s): {list(data)}")
    _style()
    for bb in BACKBONES:
        if bb in data:
            _bar_chart(
                data, bb, "accuracy", "Validation Accuracy",
                f"EuroSAT: Land-Cover Accuracy vs Label Budget ({BACKBONE_LABEL[bb]})",
                plots_dir / f"eurosat_accuracy_{bb}.png"
            )
    _param_efficiency(data, params, plots_dir / "eurosat_param_efficiency_tradeoff.png")
    _ranking_dotplot(data, plots_dir / "eurosat_ranking_at_100pct.png")
    _bar_chart_vram(vram, plots_dir)
    _bar_chart_gflops(gflops, plots_dir)
    _summary_table(data, params, gflops, vram, plots_dir / "eurosat_summary_table.png")
    print("\nAll EuroSAT figures generated in results/plots/!")


def _style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def _bar_chart(data, backbone, metric, ylabel, title, out):
    if backbone not in data or backbone == "cnn":
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=300)
    
    active_adapts = [a for a in ADAPT_ORDER if a in data[backbone]]
    has_cnn = "cnn" in data and "scratch" in data["cnn"]
    num_bars = len(active_adapts) + (1 if has_cnn else 0)
    
    x = np.arange(len(BUDGETS))
    width = 0.75 / max(1, num_bars)
    
    for i, adapt in enumerate(active_adapts):
        bd = data[backbone][adapt]
        means = []
        stds = []
        for b in BUDGETS:
            if b in bd and bd[b][metric]:
                vals = bd[b][metric]
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)
                
        offset = (i - (num_bars - 1) / 2) * width
        color = EUROSAT_MODEL_COLORS.get((backbone, adapt), ADAPT_COLOR.get(adapt, "#7f7f7f"))
        
        ax.bar(
            x + offset, means, width, yerr=stds,
            label=ADAPT_LABEL.get(adapt, adapt), color=color,
            edgecolor="none", capsize=3, error_kw=dict(elinewidth=1.0)
        )
        
    if has_cnn:
        bd = data["cnn"]["scratch"]
        means = []
        stds = []
        for b in BUDGETS:
            if b in bd and bd[b][metric]:
                vals = bd[b][metric]
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0.0)
            else:
                means.append(0.0)
                stds.append(0.0)
                
        offset = (len(active_adapts) - (num_bars - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=stds,
            label="CNN (scratch)", color="#d95f02",
            edgecolor="none", capsize=3, error_kw=dict(elinewidth=1.0)
        )
        
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel("Label Budget", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([BUDGET_LABELS[b] for b in BUDGETS])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9.5, bbox_to_anchor=(1.02, 0), loc="lower left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated plot: {out}")


EUROSAT_MODEL_NAMES = {
    ("prithvi", "lora_r8"):      "Prithvi + LoRA (r=8)",
    ("prithvi", "lora_r16"):     "Prithvi + LoRA (r=16)",
    ("prithvi", "full_ft"):      "Prithvi + Full FT",
    ("prithvi", "linear_probe"): "Prithvi + Linear Probe",
    ("prithvi", "vpt"):          "Prithvi + VPT",
    ("dinov2", "lora_r8"):       "DINOv2 + LoRA (r=8)",
    ("dinov2", "lora_r16"):      "DINOv2 + LoRA (r=16)",
    ("dinov2", "full_ft"):       "DINOv2 + Full FT",
    ("dinov2", "linear_probe"):  "DINOv2 + Linear Probe",
    ("dinov2", "vpt"):           "DINOv2 + VPT",
    ("cnn", "scratch"):          "CNN (scratch)",
}

EUROSAT_MODEL_COLORS = {
    ("prithvi", "linear_probe"): "#c6dbef",
    ("prithvi", "lora_r8"):      "#6baed6",
    ("prithvi", "lora_r16"):     "#2171b5",
    ("prithvi", "full_ft"):      "#054a85",
    ("prithvi", "vpt"):          "#012242",
    ("dinov2", "linear_probe"):  "#d9f0d3",
    ("dinov2", "lora_r8"):       "#a6dba0",
    ("dinov2", "lora_r16"):      "#5aae61",
    ("dinov2", "full_ft"):       "#1b7837",
    ("dinov2", "vpt"):           "#00441b",
    ("cnn", "scratch"):          "#d95f02",
}

EUROSAT_TOTAL_PARAMS = {
    "prithvi": 310.19,
    "dinov2": 304.38,
    "cnn": 11.18,
}


def _param_efficiency(data, params, out):
    """Scatter plot matching Burn Scar style: Trainable Parameters (log scale) vs Validation Accuracy."""
    pts = []
    for bb in BACKBONES:
        if bb not in data:
            continue
        for adapt in ADAPT_ORDER:
            if adapt in data[bb] and 1.00 in data[bb][adapt] and data[bb][adapt][1.00]["accuracy"]:
                accs = data[bb][adapt][1.00]["accuracy"]
                mean_acc = float(np.mean(accs))
                std_acc = float(np.std(accs) if len(accs) > 1 else 0.0)
                pm = params.get((bb, adapt), 0.0)
                pts.append((bb, adapt, pm, mean_acc, std_acc))

    if not pts:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    
    all_trainable_m = []
    for bb, adapt, pm, mean_acc, std_acc in pts:
        name = EUROSAT_MODEL_NAMES.get((bb, adapt), f"{bb} {adapt}")
        color = EUROSAT_MODEL_COLORS.get((bb, adapt), "#7f7f7f")
        tot = EUROSAT_TOTAL_PARAMS.get(bb, 306.0)
        pct = (pm / tot * 100.0) if tot > 0 else 0.0
        label_text = f"{name} ({pm:.2f}M / {pct:.2f}% params, 100% budget)"
        all_trainable_m.append(pm)
        
        ax.errorbar(
            pm, mean_acc, yerr=std_acc,
            fmt="o", color=color, markersize=10,
            label=label_text, capsize=5, elinewidth=1.5
        )

    max_trainable_m = max(all_trainable_m) if all_trainable_m else 306.2
    
    ax.set_title("EuroSAT: Parameter Efficiency vs. Performance Trade-off (100% Budget)", fontsize=11, fontweight="bold", color=INK, pad=12)
    ax.set_xlabel("Trainable Parameters (Millions)", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylabel("Validation Accuracy @ 100% Budget", fontsize=10, fontweight="bold", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 500.0)
    ax.set_xticks([0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, max_trainable_m])
    ax.set_xticklabels(["0.01M", "0.1M", "1M", "5M", "10M", "20M", "50M", "100M", f"{max_trainable_m:.1f}M"])
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated plot: {out}")


def _ranking_dotplot(data, out):
    rows = []
    for bb in BACKBONES:
        for adapt in ADAPT_ORDER:
            if bb in data and adapt in data[bb] and 1.00 in data[bb][adapt]:
                acc = data[bb][adapt][1.00]["accuracy"]
                rows.append((f"{BACKBONE_LABEL[bb]} · {ADAPT_LABEL[adapt]}",
                             float(np.mean(acc)), float(np.std(acc) if len(acc) > 1 else 0.0), adapt, bb))
    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(rows) + 1.6), dpi=200)
    for i, (lab, acc, std, adapt, bb) in enumerate(rows):
        color = EUROSAT_MODEL_COLORS.get((bb, adapt), ADAPT_COLOR[adapt])
        ax.errorbar(acc, i, xerr=std, fmt=BACKBONE_MARKER[bb], color=color,
                    ms=12, capsize=3, elinewidth=1.3, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax.text(acc + std + 0.008, i, f"{acc:.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0.0, 1.0); ax.set_xlabel("Validation Accuracy @ 100% Budget", fontsize=10, fontweight="bold", color=INK)
    ax.set_title("EuroSAT: Method Ranking at Full Budget", fontsize=12, fontweight="bold", color=INK, pad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5); ax.tick_params(colors=MUTED)
    
    # Legend for backbones (shapes)
    shape_handles = [Line2D([0], [0], marker=BACKBONE_MARKER[bb], color="w", markerfacecolor="gray",
                          markersize=9, label=BACKBONE_LABEL[bb]) for bb in BACKBONES]
    ax.legend(handles=shape_handles, loc="lower right", frameon=True, fontsize=8.5)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"Generated plot: {out}")


def main():
    plots_dir = Path("results/plots"); plots_dir.mkdir(parents=True, exist_ok=True)
    data, params, gflops, vram = _load()
    if not data:
        print("[ERROR] No eurosat_*.json in results/metrics/. Train EuroSAT models first!")
        return
    n = sum(len(bd[a]) for bd in data.values() for a in bd)
    print(f"Found {n} EuroSAT runs across {len(data)} backbone(s): {list(data)}")
    _style()
    for bb in BACKBONES:
        if bb in data:
            _bar_chart(
                data, bb, "accuracy", "Validation Accuracy",
                f"EuroSAT: Land-Cover Accuracy vs Label Budget ({BACKBONE_LABEL[bb]})",
                plots_dir / f"eurosat_accuracy_{bb}.png"
            )
    _param_efficiency(data, params, plots_dir / "eurosat_param_efficiency_tradeoff.png")
    _ranking_dotplot(data, plots_dir / "eurosat_ranking_at_100pct.png")
    _bar_chart_vram(vram, plots_dir)
    _bar_chart_gflops(gflops, plots_dir)
    target_combos = [
        ("prithvi", "linear_probe", "Prithvi Linear Probe", "Fast baseline;\nlower adaptation."),
        ("prithvi", "lora_r8",      "Prithvi LoRA r=8",     "Good balance;\n2% params."),
        ("prithvi", "lora_r16",     "Prithvi LoRA r=16",    "Best overall\n(Optimal trade-off)"),
        ("prithvi", "full_ft",      "Prithvi Full FT",      "Full adaptation;\n100% params."),
        ("prithvi", "vpt",          "Prithvi VPT",          "Prompt tuning;\nfrozen backbone."),
        ("dinov2",  "linear_probe", "DINOv2 Linear Probe",  "RGB pretrained;\nlinear head."),
        ("dinov2",  "lora_r8",      "DINOv2 LoRA r=8",      "High accuracy;\nefficient PEFT."),
        ("dinov2",  "lora_r16",     "DINOv2 LoRA r=16",     "Top accuracy;\n4% trainable."),
        ("dinov2",  "full_ft",      "DINOv2 Full FT",       "Costliest compute\n& VRAM."),
        ("dinov2",  "vpt",          "DINOv2 VPT",           "Prompt tuning;\nfrozen backbone."),
        ("cnn",     "scratch",      "CNN (scratch)",        "CNN baseline;\ntrained from scratch."),
    ]
    _summary_table(data, params, gflops, vram, target_combos, plots_dir / "eurosat_summary_table.png")
    
    target_combos_no_dinov2 = [tc for tc in target_combos if tc[0] != "dinov2"]
    _summary_table(data, params, gflops, vram, target_combos_no_dinov2, plots_dir / "eurosat_summary_table_no_dinov2.png")
    _summary_table(data, params, gflops, vram, target_combos_no_dinov2, plots_dir / "eurosat_summary_table_exploded.png", exploded=True)
    print("\nAll EuroSAT figures generated in results/plots/!")


if __name__ == "__main__":
    main()
