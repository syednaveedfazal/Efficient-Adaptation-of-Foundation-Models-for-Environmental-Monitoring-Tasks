"""
scripts/plot_results.py — Generates analysis plots from evaluation metrics.

This script parses all JSON files in results/metrics/, groups the scores by
model, budget, and seed, and generates high-resolution figures in results/plots/.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ALL_BUDGETS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
BUDGET_MAP = {b: i for i, b in enumerate(ALL_BUDGETS)}
BUDGET_LABELS = {
    0.01: "1% (5)",
    0.05: "5% (27)",
    0.10: "10% (54)",
    0.25: "25% (135)",
    0.50: "50% (270)",
    1.00: "100% (540)"
}

def main():
    metrics_dir = Path("results/metrics")
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse all JSON metrics files
    results = []
    for f in metrics_dir.glob("*.json"):
        if f.name == "metric_format.json" or f.name.startswith("eurosat_"):
            continue
        try:
            with open(f) as file:
                data = json.load(file)
                # Ensure the file matches expected schema
                if "experiment" in data and "metrics" in data:
                    results.append(data)
        except Exception as e:
            print(f"[WARN] Error reading {f.name}: {e}")
            
    if not results:
        print("[ERROR] No metric JSON files found in results/metrics/. Train models first!")
        return
        
    print(f"Found {len(results)} experiment runs. Aggregating results...")
    
    # 2. Group by model and label_budget
    # Structure: data_dict[model][budget] = { 'burn_iou': [], 'burn_dice': [], 'pct_trainable': ... }
    data_dict = {}
    model_vram = {}
    model_gflops = {}
    
    # Fallback GFLOP values for runs executed before GFLOP instrumentation was added.
    # These values should match the latest computed measurements; they are only
    # used as a safety net when metric JSONs lack fwd_gflops/bwd_gflops keys.
    # Note: After fixing the encoder FLOPs wrapper (forward_features vs forward),
    # re-run training to get accurate values. The values below are from the
    # most recent instrumented runs and will be superseded by JSON data.
    # LoRA entries intentionally removed — stale values used full-FT backward
    # formula (2*fwd) which was wrong. Force these to come from real JSON.
    FALLBACK_GFLOPS = {
        "unet_scratch":             (219.05, 438.10),
        "prithvi_seg_lora_r8":      (384.2, 388.5),
        "prithvi_seg_lora_r16":     (385.1, 390.3),
        "prithvi_unet_lora_r8":     (418.6, 423.8),
        "prithvi_unet_lora_r16":    (419.5, 425.6),
        "prithvi_fcn_full_ft":      (313.83, 627.65),
        "prithvi_fcn_linear_probe": (313.83, 15),
        "prithvi_fcn_randomized_full_ft":      (313.83, 627.65),
        "prithvi_fcn_randomized_linear_probe": (313.83, 15),
    }
    
    
    for run in results:
        exp = run["experiment"]
        metrics = run["metrics"]
        
        model = exp["model_name"]
        # Normalize old name formats to the new consistent naming format
        if model in ["unet", "unet_unet_scratch"]:
            model = "unet_scratch"
        elif model == "prithvi_lora_r8":
            model = "prithvi_seg_lora_r8"
        elif model == "prithvi_lora_r16":
            model = "prithvi_seg_lora_r16"
        elif model == "prithvi_fcn":
            model = "prithvi_fcn_lora_r8"
        elif model == "prithvi_unet":
            model = "prithvi_unet_lora_r8"
        elif model == "prithvi_fcn_randomized":
            continue  # Skip old plain randomized runs
        
        budget = float(exp["label_budget"])
        pct_trainable = float(exp["pct_trainable_params"])
        trainable_params = int(exp.get("trainable_params", 0))
        trainable_m = trainable_params / 1e6
        
        if model not in data_dict:
            data_dict[model] = {}
            
        if budget not in data_dict[model]:
            data_dict[model][budget] = {
                "burn_iou": [],
                "burn_dice": [],
                "pct_trainable": pct_trainable,
                "trainable_m": trainable_m
            }
            
        data_dict[model][budget]["burn_iou"].append(metrics.get("burn_iou", 0.0))
        data_dict[model][budget]["burn_dice"].append(metrics.get("burn_dice", 0.0))
        
        # Aggregate GPU VRAM statistics
        hardware = run.get("hardware", {})
        vram = float(hardware.get("max_memory_allocated_mb", 0.0))
        if vram > 0.0:
            if model not in model_vram:
                model_vram[model] = []
            model_vram[model].append(vram)
            
        # Aggregate GFLOP statistics
        fwd_g = exp.get("fwd_gflops", None)
        bwd_g = exp.get("bwd_gflops", None)
        if fwd_g is None or bwd_g is None:
            fwd_g, bwd_g = FALLBACK_GFLOPS.get(model, (0.0, 0.0))
            if fwd_g == 0.0 and bwd_g == 0.0:
                print(f"[WARN] No GFLOPs data for '{model}' — not in JSON and not in fallback dict. "
                      f"Re-run training with instrumentation to get real values.")
            else:
                print(f"[WARN] Using FALLBACK GFLOPs for '{model}' (fwd={fwd_g}, bwd={bwd_g}). "
                      f"Re-run training to get real instrumented values.")
        if model not in model_gflops:
            model_gflops[model] = (float(fwd_g), float(bwd_g))
            
        
    # 3. Create Plots
    # Style setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig_width, fig_height = 8, 5
    
    # Plot A: Burn IoU vs Label Budget
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    
    colors = {
        "unet_scratch":          "#d95f02", 
        "prithvi_seg_lora_r8":   "#1f77b4", 
        "prithvi_seg_lora_r16":  "#17becf", 
        "prithvi_unet_lora_r8":  "#9467bd", 
        "prithvi_unet_lora_r16": "#7f7f7f", 
        "prithvi_fcn_linear_probe":"#c6dbef",
        "prithvi_fcn_lora_r8":   "#6baed6", 
        "prithvi_fcn_lora_r16":  "#2171b5", 
        "prithvi_fcn_full_ft":   "#054a85", 
        "prithvi_fcn_vpt":       "#012242",
        "prithvi_fcn_randomized_linear_probe":"#fcf0d4",
        "prithvi_fcn_randomized_lora_r8":     "#f2d591",
        "prithvi_fcn_randomized_lora_r16":    "#e6c580",
        "prithvi_fcn_randomized_full_ft":     "#b89344",
        "prithvi_fcn_randomized_vpt":         "#735922",
        "dinov2_fcn_linear_probe": "#d9f0d3",
        "dinov2_fcn_lora_r8":      "#a6dba0",
        "dinov2_fcn_lora_r16":     "#5aae61",
        "dinov2_fcn_full_ft":      "#1b7837",
        "dinov2_fcn_vpt":          "#00441b",
    }
    markers = {
        "unet_scratch":          "o", 
        "prithvi_seg_lora_r8":   "s", 
        "prithvi_seg_lora_r16":  ">", 
        "prithvi_unet_lora_r8":  "d", 
        "prithvi_unet_lora_r16": "v", 
        "prithvi_fcn_lora_r8":   "p", 
        "prithvi_fcn_lora_r16":  "<", 
        "prithvi_fcn_full_ft":   "x", 
        "prithvi_fcn_linear_probe":"*", 
        "prithvi_fcn_vpt":       "2",
        "prithvi_fcn_randomized_lora_r8":     "D",
        "prithvi_fcn_randomized_lora_r16":    "8",
        "prithvi_fcn_randomized_full_ft":     "H",
        "prithvi_fcn_randomized_linear_probe":"1",
        "prithvi_fcn_randomized_vpt":         "4",
        "dinov2_fcn_lora_r8":      "^",
        "dinov2_fcn_lora_r16":     "h",
        "dinov2_fcn_full_ft":      "P",
        "dinov2_fcn_linear_probe": "X",
        "dinov2_fcn_vpt":          "3",
    }
    names = {
        "unet_scratch":          "UNet (Baseline)", 
        "prithvi_seg_lora_r8":   "Prithvi + LoRA (r=8) + SegDecoder", 
        "prithvi_seg_lora_r16":  "Prithvi + LoRA (r=16) + SegDecoder", 
        "prithvi_unet_lora_r8":  "Prithvi + LoRA (r=8) + UNet Decoder", 
        "prithvi_unet_lora_r16": "Prithvi + LoRA (r=16) + UNet Decoder", 
        "prithvi_fcn_lora_r8":   "Prithvi + LoRA (r=8) + FCN Decoder", 
        "prithvi_fcn_lora_r16":  "Prithvi + LoRA (r=16) + FCN Decoder", 
        "prithvi_fcn_full_ft":   "Prithvi + Full FT + FCN Decoder", 
        "prithvi_fcn_linear_probe":"Prithvi + Linear Probe + FCN Decoder", 
        "prithvi_fcn_vpt":       "Prithvi + VPT + FCN Decoder",
        "prithvi_fcn_randomized_lora_r8":     "Rand. Prithvi + LoRA (r=8) + FCN",
        "prithvi_fcn_randomized_lora_r16":    "Rand. Prithvi + LoRA (r=16) + FCN",
        "prithvi_fcn_randomized_full_ft":     "Rand. Prithvi + Full FT + FCN",
        "prithvi_fcn_randomized_linear_probe":"Rand. Prithvi + Linear Probe + FCN",
        "prithvi_fcn_randomized_vpt":         "Rand. Prithvi + VPT + FCN",
        "dinov2_fcn_lora_r8":      "DINOv2 + LoRA (r=8) + FCN Decoder",
        "dinov2_fcn_lora_r16":     "DINOv2 + LoRA (r=16) + FCN Decoder",
        "dinov2_fcn_full_ft":      "DINOv2 + Full FT + FCN Decoder",
        "dinov2_fcn_linear_probe": "DINOv2 + Linear Probe + FCN Decoder",
        "dinov2_fcn_vpt":          "DINOv2 + VPT + FCN Decoder",
    }
    
    for model, budgets_dict in data_dict.items():
        sorted_budgets = sorted(budgets_dict.keys())
        
        means_iou = []
        stds_iou = []
        
        for b in sorted_budgets:
            ious = budgets_dict[b]["burn_iou"]
            means_iou.append(np.mean(ious))
            stds_iou.append(np.std(ious) if len(ious) > 1 else 0.0)
            
        color = colors.get(model, "#7f7f7f")
        marker = markers.get(model, "x")
        name = names.get(model, model)
        
        # Map budgets to evenly spaced indices
        x_vals = [BUDGET_MAP[b] for b in sorted_budgets]
        
        plt.errorbar(
            x_vals, means_iou, yerr=stds_iou, 
            fmt=f"-{marker}", color=color, label=name, 
            linewidth=2, elinewidth=1.5, capsize=4, markersize=7
        )
        
        # Add shaded standard deviation band if multiple seeds exist
        if any(std > 0 for std in stds_iou):
            plt.fill_between(
                x_vals, 
                np.array(means_iou) - np.array(stds_iou), 
                np.array(means_iou) + np.array(stds_iou), 
                color=color, alpha=0.15
            )
            
    plt.title("Data Efficiency: Burn-Scar Segmentation Performance vs Label Budget", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Label Budget (% of Training Scenes)", fontsize=10, fontweight="bold")
    plt.ylabel("Validation Burn-Scar IoU", fontsize=10, fontweight="bold")
    plt.xticks(list(range(len(ALL_BUDGETS))), [BUDGET_LABELS[b] for b in ALL_BUDGETS])
    plt.ylim(0.0, 1.0)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="lower right")
    plt.tight_layout()
    
    iou_plot_path = plots_dir / "iou_vs_budget.png"
    plt.savefig(iou_plot_path, bbox_inches="tight")
    print(f"Generated plot: {iou_plot_path}")
    plt.close()

    # Grouped Bar Plot Helper for IoU and Dice
    def plot_grouped_bar_subsets(metric_key, ylabel, title_prefix, fname_prefix, hide_randomized=False, overlay_prithvi_r16=False):
        bar_budgets = [0.01, 0.10, 0.25, 1.00]
        bar_labels = ["1% (5)", "10% (54)", "25% (135)", "100% (540)"]
        
        prithvi_models = [m for m in models_to_plot if (m.startswith("prithvi_") or m == "unet_scratch") and m in data_dict]
        dinov2_models = [m for m in models_to_plot if (m.startswith("dinov2_") or m == "unet_scratch") and m in data_dict]
        
        subsets = {
            "Prithvi-300M & Baseline": (prithvi_models, f"{fname_prefix}_prithvi.png"),
            "DINOv2-Large & Baseline": (dinov2_models, f"{fname_prefix}_dinov2.png")
        }
        
        if overlay_prithvi_r16:
            subsets = {
                "DINOv2-Large & Baseline": (dinov2_models, f"{fname_prefix}_dinov2_prithvi_overlay.png")
            }
        
        for label, (sub_models, fname) in subsets.items():
            if not sub_models:
                continue
            plt.figure(figsize=(11.5, 5.5), dpi=300)
            x = np.arange(len(bar_budgets))
            width = 0.75 / max(1, len(sub_models))
            
            for i, model in enumerate(sub_models):
                offset = (i - (len(sub_models) - 1) / 2) * width
                
                if hide_randomized and "randomized" in model:
                    plt.bar(x + offset, [0]*len(bar_budgets), width, color="none", label="_nolegend_")
                    continue

                means = []
                stds = []
                for b in bar_budgets:
                    if b in data_dict[model] and data_dict[model][b][metric_key]:
                        vals = data_dict[model][b][metric_key]
                        means.append(np.mean(vals))
                        stds.append(np.std(vals) if len(vals) > 1 else 0.0)
                    else:
                        means.append(0.0)
                        stds.append(0.0)
                        
                plt.bar(
                    x + offset, means, width, yerr=stds,
                    label=names.get(model, model), color=colors.get(model, "#7f7f7f"),
                    edgecolor="none", capsize=3, error_kw=dict(elinewidth=1.0)
                )
                
            if overlay_prithvi_r16 and "DINOv2" in label:
                r16_idx = next((i for i, m in enumerate(sub_models) if "lora_r16" in m), -1)
                if r16_idx != -1:
                    offset = (r16_idx - (len(sub_models) - 1) / 2) * width
                    x_pos = x + offset
                    prithvi_m = "prithvi_fcn_lora_r16"
                    prithvi_means = [np.mean(data_dict[prithvi_m][b][metric_key]) if b in data_dict.get(prithvi_m, {}) and data_dict[prithvi_m][b][metric_key] else 0.0 for b in bar_budgets]
                    
                    plt.plot(x_pos, prithvi_means, color="#2171b5", linestyle="--", marker="o", linewidth=2.0, markersize=8, alpha=0.7, label="Prithvi-300M LoRA (r=16)", zorder=10)
                
            plt.title(f"Burn Scar: {title_prefix} ({label})", fontsize=12, fontweight="bold", pad=12)
            plt.xlabel("Label Budget (% of Training Scenes)", fontsize=10, fontweight="bold")
            plt.ylabel(ylabel, fontsize=10, fontweight="bold")
            plt.xticks(x, bar_labels)
            plt.ylim(0.0, 1.0)
            plt.grid(True, axis="y", linestyle="--", alpha=0.7)
            plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8.5, bbox_to_anchor=(1.02, 0), loc="lower left")
            plt.tight_layout()
            out_p = plots_dir / fname
            plt.savefig(out_p, bbox_inches="tight")
            print(f"Generated plot: {out_p}")
            plt.close()

    # Plot A2: Grouped Bar Chart of Burn IoU vs Label Budget (Prithvi & DINOv2)
    models_to_plot = [
        "prithvi_fcn_linear_probe",
        "prithvi_fcn_lora_r8",
        "prithvi_fcn_lora_r16",
        "prithvi_fcn_full_ft",
        "prithvi_fcn_vpt",
        "unet_scratch",
        "prithvi_fcn_randomized_linear_probe",
        "prithvi_fcn_randomized_lora_r8",
        "prithvi_fcn_randomized_lora_r16",
        "prithvi_fcn_randomized_full_ft",
        "prithvi_fcn_randomized_vpt",
        "prithvi_seg_lora_r8",
        "prithvi_seg_lora_r16",
        "prithvi_unet_lora_r8",
        "prithvi_unet_lora_r16",
        "dinov2_fcn_linear_probe",
        "dinov2_fcn_lora_r8",
        "dinov2_fcn_lora_r16",
        "dinov2_fcn_full_ft",
        "dinov2_fcn_vpt",
    ]
    plot_grouped_bar_subsets("burn_iou", "Validation Burn-Scar IoU", "IoU by Label Budget", "burn_scar_iou_bar")
    plot_grouped_bar_subsets("burn_iou", "Validation Burn-Scar IoU", "IoU by Label Budget", "burn_scar_iou_bar_no_rand", hide_randomized=True)

    # Plot B: Dice vs Label Budget
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    
    for model, budgets_dict in data_dict.items():
        sorted_budgets = sorted(budgets_dict.keys())
        
        means_dice = []
        stds_dice = []
        
        for b in sorted_budgets:
            dices = budgets_dict[b]["burn_dice"]
            means_dice.append(np.mean(dices))
            stds_dice.append(np.std(dices) if len(dices) > 1 else 0.0)
            
        color = colors.get(model, "#7f7f7f")
        marker = markers.get(model, "x")
        name = names.get(model, model)
        
        # Map budgets to evenly spaced indices
        x_vals = [BUDGET_MAP[b] for b in sorted_budgets]
        
        plt.errorbar(
            x_vals, means_dice, yerr=stds_dice, 
            fmt=f"-{marker}", color=color, label=name, 
            linewidth=2, elinewidth=1.5, capsize=4, markersize=7
        )
        
        if any(std > 0 for std in stds_dice):
            plt.fill_between(
                x_vals, 
                np.array(means_dice) - np.array(stds_dice), 
                np.array(means_dice) + np.array(stds_dice), 
                color=color, alpha=0.15
            )
            
    plt.title("Burn Scar: Dice/F1 Score vs Label Budget", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Label Budget (% of Training Scenes)", fontsize=10, fontweight="bold")
    plt.ylabel("Validation Burn-Scar Dice Coefficient", fontsize=10, fontweight="bold")
    plt.xticks(list(range(len(ALL_BUDGETS))), [BUDGET_LABELS[b] for b in ALL_BUDGETS])
    plt.ylim(0.0, 1.0)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="lower right")
    plt.tight_layout()
    
    dice_plot_path = plots_dir / "burn_scar_dice_vs_budget.png"
    plt.savefig(dice_plot_path, bbox_inches="tight")
    print(f"Generated plot: {dice_plot_path}")
    plt.close()

    # Plot B2: Grouped Bar Chart of Dice vs Label Budget (Prithvi & DINOv2)
    plot_grouped_bar_subsets("burn_dice", "Validation Burn-Scar Dice Coefficient", "Dice/F1 Score by Label Budget", "burn_scar_dice_bar")
    plot_grouped_bar_subsets("burn_dice", "Validation Burn-Scar Dice Coefficient", "Dice/F1 Score by Label Budget", "burn_scar_dice_bar_no_rand", hide_randomized=True)
    plot_grouped_bar_subsets("burn_dice", "Validation Burn-Scar Dice Coefficient", "Dice/F1 Score by Label Budget", "burn_scar_dice_bar", overlay_prithvi_r16=True)

    # Plot C: Parameter Efficiency vs. Performance
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    
    for model, budgets_dict in data_dict.items():
        # Use the maximum available budget for each model
        best_budget = max(budgets_dict.keys())
        pct_trainable = budgets_dict[best_budget]["pct_trainable"]
        trainable_m = budgets_dict[best_budget]["trainable_m"]
        ious = budgets_dict[best_budget]["burn_iou"]
        mean_iou = np.mean(ious)
        std_iou = np.std(ious) if len(ious) > 1 else 0.0
        
        color = colors.get(model, "#7f7f7f")
        name = names.get(model, model)
        
        # Label showing both parameter count in millions, percentage, and budget used
        label_text = f"{name} ({trainable_m:.2f}M / {pct_trainable:.2f}% params, {best_budget*100.0:.0f}% budget)"
        
        plt.errorbar(
            trainable_m, mean_iou, yerr=std_iou, 
            fmt="o", color=color, markersize=10, 
            label=label_text,
            capsize=5, elinewidth=1.5
        )
            
    # Find the maximum trainable parameter size dynamically
    all_trainable_m = []
    for model, budgets_dict in data_dict.items():
        best_budget = max(budgets_dict.keys())
        all_trainable_m.append(budgets_dict[best_budget]["trainable_m"])
    max_trainable_m = max(all_trainable_m) if all_trainable_m else 306.2
            
    plt.title("Burn Scar: Parameter Efficiency vs. Performance Trade-off (Max Available Budget)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Trainable Parameters (Millions)", fontsize=10, fontweight="bold")
    plt.ylabel("Validation Burn-Scar IoU", fontsize=10, fontweight="bold")
    plt.xscale("log")
    plt.xlim(1.0, 500.0)
    plt.xticks([1.0, 5.0, 10.0, 20.0, 50.0, 100.0, max_trainable_m], ["1M", "5M", "10M", "20M", "50M", "100M", f"{max_trainable_m:.1f}M"])
    plt.ylim(0.0, 1.0)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8.5, loc="lower left")
    plt.tight_layout()
    
    tradeoff_plot_path = plots_dir / "burn_scar_param_efficiency_tradeoff.png"
    plt.savefig(tradeoff_plot_path, bbox_inches="tight")
    print(f"Generated plot: {tradeoff_plot_path}")
    plt.close()
    
    # Plot D: Peak GPU VRAM Memory Allocated (MB) - Split into Prithvi and DINOv2
    if model_vram:
        vram_subsets = {
            "Prithvi-300M & Baseline": (
                [m for m in sorted(model_vram.keys()) if m.startswith("prithvi_") or m == "unet_scratch"],
                "burn_scar_vram_prithvi.png"
            ),
            "DINOv2-Large & Baseline": (
                [m for m in sorted(model_vram.keys()) if m.startswith("dinov2_") or m == "unet_scratch"],
                "burn_scar_vram_dinov2.png"
            )
        }
        for label, (models_sub, fname) in vram_subsets.items():
            if not models_sub:
                continue
            cur_width = 12.0 if len(models_sub) > 7 else 8.0
            plt.figure(figsize=(cur_width, 5.5), dpi=300)
            means_vram = [np.mean(model_vram[m]) for m in models_sub]
            stds_vram = [np.std(model_vram[m]) if len(model_vram[m]) > 1 else 0.0 for m in models_sub]
            plot_colors = [colors.get(m, "#7f7f7f") for m in models_sub]
            plot_names = [names.get(m, m) for m in models_sub]
            
            bars = plt.bar(plot_names, means_vram, yerr=stds_vram, color=plot_colors, alpha=0.85, edgecolor="none", capsize=3, width=0.45)
            for i, bar in enumerate(bars):
                height = bar.get_height()
                err = stds_vram[i]
                plt.annotate(
                    f"{height:,.0f} MB",
                    xy=(bar.get_x() + bar.get_width() / 2, height + err),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.0, fontweight="bold"
                )
                
            plt.title(f"Burn Scar: Peak GPU VRAM Usage ({label})", fontsize=12, fontweight="bold", pad=12)
            plt.xlabel("Model Configuration", fontsize=10, fontweight="bold")
            plt.ylabel("Peak VRAM Allocated (MB)", fontsize=10, fontweight="bold")
            plt.xticks(rotation=22, ha="right", fontsize=8.5)
            plt.grid(True, axis="y", linestyle="--", alpha=0.7)
            plt.ylim(0, (max(means_vram) + max(stds_vram)) * 1.22 if means_vram else 1000)
            plt.tight_layout()
            out_p = plots_dir / fname
            plt.savefig(out_p, bbox_inches="tight")
            print(f"Generated plot: {out_p}")
            plt.close()

    # Plot E: Computational Complexity (GFLOPs Stacked Bar Chart) - Split into Prithvi and DINOv2
    if model_gflops:
        gflops_subsets = {
            "Prithvi-300M & Baseline": (
                [m for m in sorted(model_gflops.keys()) if m.startswith("prithvi_") or m == "unet_scratch"],
                "burn_scar_gflops_prithvi.png"
            ),
            "DINOv2-Large & Baseline": (
                [m for m in sorted(model_gflops.keys()) if m.startswith("dinov2_") or m == "unet_scratch"],
                "burn_scar_gflops_dinov2.png"
            )
        }
        for label, (models_sub, fname) in gflops_subsets.items():
            if not models_sub:
                continue
            cur_width = 12.0 if len(models_sub) > 7 else 8.0
            plt.figure(figsize=(cur_width, 5.5), dpi=300)
            fwd_vals = [model_gflops[m][0] for m in models_sub]
            bwd_vals = [model_gflops[m][1] for m in models_sub]
            plot_names = [names.get(m, m) for m in models_sub]
            
            fwd_colors = []
            bwd_colors = []
            for m in models_sub:
                if m == "unet_scratch":
                    fwd_colors.append("#fdae6b") # Lighter orange
                    bwd_colors.append("#d95f02") # Darker orange
                elif m.startswith("dinov2_"):
                    fwd_colors.append("#a6dba0") # Lighter green
                    bwd_colors.append("#1b7837") # Darker green
                elif "randomized" in m:
                    fwd_colors.append("#f2d591") # Lighter golden
                    bwd_colors.append("#b89344") # Darker golden
                else: # Prithvi
                    fwd_colors.append("#6baed6") # Lighter blue
                    bwd_colors.append("#2171b5") # Darker blue
            
            bars_fwd = plt.bar(plot_names, fwd_vals, color=fwd_colors, alpha=0.9, width=0.45)
            bars_bwd = plt.bar(plot_names, bwd_vals, bottom=fwd_vals, color=bwd_colors, alpha=0.9, width=0.45)
            
            for i, m in enumerate(models_sub):
                total = fwd_vals[i] + bwd_vals[i]
                plt.annotate(
                    f"{total:.1f} GF",
                    xy=(i, total),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.0, fontweight="bold"
                )
                
            plt.title(f"Burn Scar: Computational Complexity ({label})", fontsize=12, fontweight="bold", pad=12)
            plt.xlabel("Model Configuration", fontsize=10, fontweight="bold")
            plt.ylabel("Compute Complexity (GFLOPs)", fontsize=10, fontweight="bold")
            plt.xticks(rotation=22, ha="right", fontsize=8.5)
            plt.grid(True, axis="y", linestyle="--", alpha=0.7)
            plt.ylim(0, max([f + b for f, b in [model_gflops[m] for m in models_sub]]) * 1.25 if models_sub else 100)
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#cccccc', edgecolor='none', label='Forward Pass (Lighter)'),
                Patch(facecolor='#737373', edgecolor='none', label='Backward Pass (Darker)')
            ]
            plt.legend(handles=legend_elements, loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=8.5)
            plt.tight_layout()
            out_p = plots_dir / fname
            plt.savefig(out_p, bbox_inches="tight")
            print(f"Generated plot: {out_p}")
            plt.close()
    
    def generate_summary_table(target_models_list, out_filename, exploded=False):
        rows = []
        if exploded:
            headers = ["Method", "", "IoU\n@ 10%", "IoU\n@ 100%", "Dice\n@ 10%", "Dice\n@ 100%", "", "GFLOPs\n(Total)", "VRAM\n(MB)", "", "Key Findings"]
        else:
            headers = ["Method", "IoU\n@ 10%", "IoU\n@ 100%", "Dice\n@ 10%", "Dice\n@ 100%", "GFLOPs\n(Total)", "VRAM\n(MB)", "Key Findings"]
        
        profiles = {
            "prithvi_fcn_linear_probe": "Cheapest compute;\nlower accuracy.",
            "prithvi_fcn_lora_r8": "Good balance;\nslight variance.",
            "prithvi_fcn_lora_r16": "Best overall\n(Optimal trade-off)",
            "prithvi_fcn_full_ft": "Costliest compute;\noverfits on low data.",
            "prithvi_fcn_vpt": "Prompt tuning;\nfrozen backbone.",
            "prithvi_fcn_randomized_linear_probe": "Random init;\nlinear probe only.",
            "prithvi_fcn_randomized_lora_r8": "Random init;\nLoRA r=8.",
            "prithvi_fcn_randomized_lora_r16": "Random init;\nLoRA r=16.",
            "prithvi_fcn_randomized_full_ft": "Random init;\nfull fine-tune.",
            "prithvi_fcn_randomized_vpt": "Random init;\nprompt tuning.",
            "unet_scratch": "Low VRAM;\nhigh training compute."
        }
        
        for model in target_models_list:
            if model not in data_dict:
                continue
                
            budgets = data_dict[model]
            
            # Get mean IoU and Dice for 10% (0.10) and 100% (1.00)
            iou_10 = np.mean(budgets[0.10]["burn_iou"]) * 100.0 if 0.10 in budgets else 0.0
            iou_100 = np.mean(budgets[1.00]["burn_iou"]) * 100.0 if 1.00 in budgets else 0.0
            dice_10 = np.mean(budgets[0.10]["burn_dice"]) * 100.0 if 0.10 in budgets else 0.0
            dice_100 = np.mean(budgets[1.00]["burn_dice"]) * 100.0 if 1.00 in budgets else 0.0
            
            # GFLOPs
            if model in model_gflops:
                fwd, bwd = model_gflops[model]
                total_gflops = fwd + bwd
                gflops_str = f"{total_gflops:.1f}"
            else:
                gflops_str = "N/A"
                
            # VRAM
            if model in model_vram and len(model_vram[model]) > 0:
                vram_val = np.mean(model_vram[model])
                vram_str = f"{vram_val:,.0f}"
            else:
                vram_str = "N/A"
                
            model_name_map = {
                "prithvi_fcn_linear_probe": "Prithvi Linear Probe",
                "prithvi_fcn_lora_r8": "Prithvi LoRA r=8",
                "prithvi_fcn_lora_r16": "Prithvi LoRA r=16",
                "prithvi_fcn_full_ft": "Prithvi Full FT",
                "prithvi_fcn_vpt": "Prithvi VPT",
                "prithvi_fcn_randomized_linear_probe": "Rand. Prithvi Linear Probe",
                "prithvi_fcn_randomized_lora_r8": "Rand. Prithvi LoRA r=8",
                "prithvi_fcn_randomized_lora_r16": "Rand. Prithvi LoRA r=16",
                "prithvi_fcn_randomized_full_ft": "Rand. Prithvi Full FT",
                "prithvi_fcn_randomized_vpt": "Rand. Prithvi VPT",
                "unet_scratch": "UNet (scratch)"
            }
            model_name_display = model_name_map.get(model, model)
            
            if exploded and model == "unet_scratch":
                rows.append(["", "", "", "", "", "", "", "", "", "", ""])
                
            if exploded:
                rows.append([
                    model_name_display,
                    "",
                    f"~{iou_10/100.0:.2f}" if iou_10 > 0 else "N/A",
                    f"~{iou_100/100.0:.2f}" if iou_100 > 0 else "N/A",
                    f"{dice_10/100.0:.2f}" if dice_10 > 0 else "N/A",
                    f"{dice_100/100.0:.2f}" if dice_100 > 0 else "N/A",
                    "",
                    gflops_str,
                    vram_str,
                    "",
                    profiles.get(model, "")
                ])
            else:
                rows.append([
                    model_name_display,
                    f"~{iou_10/100.0:.2f}" if iou_10 > 0 else "N/A",
                    f"~{iou_100/100.0:.2f}" if iou_100 > 0 else "N/A",
                    f"{dice_10/100.0:.2f}" if dice_10 > 0 else "N/A",
                    f"{dice_100/100.0:.2f}" if dice_100 > 0 else "N/A",
                    gflops_str,
                    vram_str,
                    profiles.get(model, "")
                ])
            
        if not rows:
            return
            
        num_data_rows = sum(1 for r in rows if r[0] != "")
        num_spacer_rows = sum(1 for r in rows if r[0] == "")
        fig_height = 0.375 * (num_data_rows + 1) + (0.1875 * num_spacer_rows if exploded else 0.0) + 0.5
        fig, ax = plt.subplots(figsize=(12.5, fig_height), dpi=300)
        ax.axis('off')
        
        if exploded:
            col_widths = [0.22, 0.015, 0.08, 0.08, 0.08, 0.08, 0.015, 0.09, 0.09, 0.015, 0.22]
        else:
            col_widths = [0.22, 0.09, 0.09, 0.09, 0.09, 0.10, 0.10, 0.22]
            
        tbl = ax.table(
            cellText=rows,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colWidths=col_widths
        )
        
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        
        header_color = '#eb7a34' # Muted orange to match PPT
        row_colors = ['#eaf1f8', '#d6e3f0'] # Alternating blue colors to match PPT
        
        normal_cell_height = 0.81 / fig_height
        spacer_row_height = (0.015 * 12.5) / fig_height  # Matches the physical 0.015 * 12.5 inch column gap
        
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
                if 'r=16' in rows[row-1][0]:
                    cell.set_text_props(weight='bold')
                    
        plt.savefig(out_filename, bbox_inches='tight', dpi=300)
        print(f"Generated plot: {out_filename}")
        plt.close()

    target_models = [
        "prithvi_fcn_linear_probe",
        "prithvi_fcn_lora_r8",
        "prithvi_fcn_lora_r16",
        "prithvi_fcn_full_ft",
        "prithvi_fcn_vpt",
        "prithvi_fcn_randomized_linear_probe",
        "prithvi_fcn_randomized_lora_r8",
        "prithvi_fcn_randomized_lora_r16",
        "prithvi_fcn_randomized_full_ft",
        "prithvi_fcn_randomized_vpt",
        "unet_scratch"
    ]
    
    generate_summary_table(target_models, plots_dir / "burn_scar_summary_table.png")
    generate_summary_table([m for m in target_models if "randomized" not in m], plots_dir / "burn_scar_summary_table_no_rand.png")
    generate_summary_table([m for m in target_models if "randomized" not in m], plots_dir / "burn_scar_summary_table_exploded.png", exploded=True)
    
    print("\nAll evaluation figures successfully generated in results/plots/!")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="burn_scar", choices=["burn_scar", "eurosat"],
                    help="burn_scar = segmentation IoU/Dice plots (default); "
                         "eurosat = classification accuracy/F1 plots.")
    args = ap.parse_args()
    if args.dataset == "eurosat":
        import sys
        sys.path.append(str(Path(__file__).resolve().parent))
        from plot_eurosat import main as eurosat_main
        eurosat_main()
    else:
        main()
