"""
scripts/plot_results.py — Generates analysis plots from evaluation metrics.

This script parses all JSON files in results/metrics/, groups the scores by
model, budget, and seed, and generates high-resolution figures in results/plots/.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    metrics_dir = Path("results/metrics")
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Parse all JSON metrics files
    results = []
    for f in metrics_dir.glob("*.json"):
        if f.name == "metric_format.json":
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
    
    # Fallback GFLOP values for runs executed before GFLOP instrumentation was added
    FALLBACK_GFLOPS = {
        "unet_scratch":             (124.5, 249.0),
        "prithvi_seg_lora_r8":      (384.2, 388.5),
        "prithvi_seg_lora_r16":     (385.1, 390.3),
        "prithvi_unet_lora_r8":     (418.6, 423.8),
        "prithvi_unet_lora_r16":    (419.5, 425.6),
        "prithvi_fcn_lora_r8":      (315.8, 320.1),
        "prithvi_fcn_lora_r16":     (316.7, 321.9),
        "prithvi_fcn_full_ft":      (314.9, 629.8),
        "prithvi_fcn_linear_probe":  (314.9,  4.72),
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
        "prithvi_fcn_lora_r8":   "#e377c2", 
        "prithvi_fcn_lora_r16":  "#bcbd22", 
        "prithvi_fcn_full_ft":   "#ff7f0e", 
        "prithvi_fcn_linear_probe":"#8c564b", 
        "dinov2_finetune":       "#2ca02c"
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
        "dinov2_finetune":       "^"
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
        "dinov2_finetune":       "DINOv2 (Fine-tuned)"
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
        
        # Convert budgets to percentages for plotting x-axis
        x_vals = [b * 100.0 for b in sorted_budgets]
        
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
    plt.xscale("log")
    plt.xticks([1, 5, 10, 25, 50, 100], ["1%", "5%", "10%", "25%", "50%", "100%"])
    plt.ylim(0.0, 1.0)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="lower right")
    plt.tight_layout()
    
    iou_plot_path = plots_dir / "iou_vs_budget.png"
    plt.savefig(iou_plot_path, bbox_inches="tight")
    print(f"Generated plot: {iou_plot_path}")
    plt.close()
    
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
        
        x_vals = [b * 100.0 for b in sorted_budgets]
        
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
            
    plt.title("Dice/F1 Score vs Label Budget", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Label Budget (% of Training Scenes)", fontsize=10, fontweight="bold")
    plt.ylabel("Validation Burn-Scar Dice Coefficient", fontsize=10, fontweight="bold")
    plt.xscale("log")
    plt.xticks([1, 5, 10, 25, 50, 100], ["1%", "5%", "10%", "25%", "50%", "100%"])
    plt.ylim(0.0, 1.0)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10, loc="lower right")
    plt.tight_layout()
    
    dice_plot_path = plots_dir / "dice_vs_budget.png"
    plt.savefig(dice_plot_path, bbox_inches="tight")
    print(f"Generated plot: {dice_plot_path}")
    plt.close()

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
            
    plt.title("Parameter Efficiency vs. Performance Trade-off (Max Available Budget)", fontsize=11, fontweight="bold", pad=12)
    plt.xlabel("Trainable Parameters (Millions)", fontsize=10, fontweight="bold")
    plt.ylabel("Validation Burn-Scar IoU", fontsize=10, fontweight="bold")
    plt.xscale("log")
    plt.xlim(1.0, 500.0)
    plt.xticks([1.0, 5.0, 10.0, 20.0, 50.0, 100.0, max_trainable_m], ["1M", "5M", "10M", "20M", "50M", "100M", f"{max_trainable_m:.1f}M"])
    plt.ylim(0.0, 1.0)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8.5, loc="lower left")
    plt.tight_layout()
    
    tradeoff_plot_path = plots_dir / "param_efficiency_tradeoff.png"
    plt.savefig(tradeoff_plot_path, bbox_inches="tight")
    print(f"Generated plot: {tradeoff_plot_path}")
    plt.close()
    
    # Plot D: Peak GPU VRAM Memory Allocated (MB)
    if model_vram:
        plt.figure(figsize=(fig_width, fig_height), dpi=300)
        
        models_sorted = sorted(model_vram.keys())
        means_vram = [np.mean(model_vram[m]) for m in models_sorted]
        stds_vram = [np.std(model_vram[m]) if len(model_vram[m]) > 1 else 0.0 for m in models_sorted]
        
        plot_colors = [colors.get(m, "#7f7f7f") for m in models_sorted]
        plot_names = [names.get(m, m) for m in models_sorted]
        
        bars = plt.bar(plot_names, means_vram, yerr=stds_vram, color=plot_colors, alpha=0.85, edgecolor="none", capsize=6, width=0.4)
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.1f} MB",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight="bold"
            )
            
        plt.title("Hardware Resource Consumption: Peak GPU VRAM Usage", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Model Configuration", fontsize=10, fontweight="bold")
        plt.ylabel("Peak VRAM Allocated (MB)", fontsize=10, fontweight="bold")
        plt.xticks(rotation=15, ha="right", fontsize=8.5)
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.ylim(0, max(means_vram) * 1.2 if means_vram else 1000)
        plt.tight_layout()
        
        vram_plot_path = plots_dir / "gpu_vram_comparison.png"
        plt.savefig(vram_plot_path, bbox_inches="tight")
        print(f"Generated plot: {vram_plot_path}")
        plt.close()
        
    # Plot E: Computational Complexity (GFLOPs Stacked Bar Chart)
    if model_gflops:
        plt.figure(figsize=(fig_width, fig_height), dpi=300)
        
        models_sorted = sorted(model_gflops.keys())
        fwd_vals = [model_gflops[m][0] for m in models_sorted]
        bwd_vals = [model_gflops[m][1] for m in models_sorted]
        
        plot_names = [names.get(m, m) for m in models_sorted]
        
        fwd_color = "#3182bd"  # Steel Blue
        bwd_color = "#e6550d"  # Coral / Dark Orange
        
        bars_fwd = plt.bar(plot_names, fwd_vals, label="Forward Pass", color=fwd_color, alpha=0.85, width=0.45)
        bars_bwd = plt.bar(plot_names, bwd_vals, bottom=fwd_vals, label="Backward Pass", color=bwd_color, alpha=0.85, width=0.45)
        
        for i, m in enumerate(models_sorted):
            total = fwd_vals[i] + bwd_vals[i]
            plt.annotate(
                f"{total:.1f} GF",
                xy=(i, total),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8.5, fontweight="bold"
            )
            
        plt.title("Computational Complexity: Forward vs. Backward Pass (GFLOPs)", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Model Configuration", fontsize=10, fontweight="bold")
        plt.ylabel("Compute Complexity (GFLOPs)", fontsize=10, fontweight="bold")
        plt.xticks(rotation=15, ha="right", fontsize=8.5)
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.ylim(0, max([f + b for f, b in model_gflops.values()]) * 1.2 if model_gflops else 100)
        plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", fontsize=9)
        plt.tight_layout()
        
        gflops_plot_path = plots_dir / "gpu_gflops_comparison.png"
        plt.savefig(gflops_plot_path, bbox_inches="tight")
        print(f"Generated plot: {gflops_plot_path}")
        plt.close()
    
    print("\nAll evaluation figures successfully generated in results/plots/!")

if __name__ == "__main__":
    main()
