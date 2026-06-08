"""
scripts/plot_results.py — Generates analysis plots from evaluation metrics.

This script parses all JSON files in results/metrics/, groups the scores by
model, budget, and seed, and generates high-resolution figures in results/plots/.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _model_style(model_name: str):
    if model_name == "unet":
        return "#d95f02", "o", "UNet (Baseline)"

    if model_name == "dinov2_finetune":
        return "#2ca02c", "^", "DINOv2 (Fine-tuned)"

    if model_name == "prithvi_lora_r8":
        return "#1f77b4", "s", "Prithvi-EO-2.0 + LoRA r=8"

    if model_name == "prithvi_lora_r16":
        return "#0d3b66", "D", "Prithvi-EO-2.0 + LoRA r=16"

    if model_name.startswith("prithvi_lora_r"):
        rank = model_name.removeprefix("prithvi_lora_r")
        return "#4c78a8", "P", f"Prithvi-EO-2.0 + LoRA r={rank}"

    return "#7f7f7f", "x", model_name


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
    
    for run in results:
        exp = run["experiment"]
        metrics = run["metrics"]
        
        model = exp["model_name"]
        budget = float(exp["label_budget"])
        pct_trainable = float(exp["pct_trainable_params"])
        
        if model not in data_dict:
            data_dict[model] = {}
            
        if budget not in data_dict[model]:
            data_dict[model][budget] = {
                "burn_iou": [],
                "burn_dice": [],
                "pct_trainable": pct_trainable
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
        
    # 3. Create Plots
    # Style setup
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig_width, fig_height = 8, 5
    
    # Plot A: Burn IoU vs Label Budget
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    
    for model, budgets_dict in data_dict.items():
        sorted_budgets = sorted(budgets_dict.keys())
        
        means_iou = []
        stds_iou = []
        
        for b in sorted_budgets:
            ious = budgets_dict[b]["burn_iou"]
            means_iou.append(np.mean(ious))
            stds_iou.append(np.std(ious) if len(ious) > 1 else 0.0)
            
        color, marker, name = _model_style(model)
        
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
            
        color, marker, name = _model_style(model)
        
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
        # Use metrics at 100% budget (1.0) for this tradeoff plot
        if 1.0 in budgets_dict:
            pct_trainable = budgets_dict[1.0]["pct_trainable"]
            ious = budgets_dict[1.0]["burn_iou"]
            mean_iou = np.mean(ious)
            std_iou = np.std(ious) if len(ious) > 1 else 0.0
            
            color, _, name = _model_style(model)
            
            plt.errorbar(
                pct_trainable, mean_iou, yerr=std_iou, 
                fmt="o", color=color, markersize=10, 
                label=f"{name} ({pct_trainable:.2f}% updated params)",
                capsize=5, elinewidth=1.5
            )
            plt.annotate(
                name, (pct_trainable, mean_iou), 
                textcoords="offset points", xytext=(10, -5) if pct_trainable > 5 else (-10, 10),
                ha='left' if pct_trainable > 5 else 'right', fontweight="bold", fontsize=9
            )
            
    plt.title("Parameter Efficiency vs. Performance Trade-off (100% Budget)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Trainable Parameters (% of Total Model Parameters)", fontsize=10, fontweight="bold")
    plt.ylabel("Validation Burn-Scar IoU", fontsize=10, fontweight="bold")
    plt.xscale("log")
    plt.xlim(0.1, 150.0)
    plt.xticks([0.1, 1.0, 10.0, 100.0], ["0.1%", "1.0%", "10.0%", "100%"])
    plt.ylim(0.0, 1.0)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
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
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.ylim(0, max(means_vram) * 1.2 if means_vram else 1000)
        plt.tight_layout()
        
        vram_plot_path = plots_dir / "gpu_vram_comparison.png"
        plt.savefig(vram_plot_path, bbox_inches="tight")
        print(f"Generated plot: {vram_plot_path}")
        plt.close()
    
    print("\nAll evaluation figures successfully generated in results/plots/!")

if __name__ == "__main__":
    main()
