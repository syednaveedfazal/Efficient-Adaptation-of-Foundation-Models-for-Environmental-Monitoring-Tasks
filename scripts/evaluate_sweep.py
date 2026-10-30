"""
scripts/evaluate_sweep.py — Inference sweep over (keep_ratio, share_frac) grid for SCOPE-full.

Sweeps across:
  keep_ratios: [1.0, 0.75, 0.5, 0.3]
  share_fracs: [0.0, 0.25, 0.5]
Total: 12 evaluation points.

For each point, records:
  - burn_iou, mean_iou, burn_dice, burn_precision, burn_recall, loss, pixel_accuracy
  - tokens_attn_mean, tokens_mlp_mean, exit_rate
  - encoder_gflops_inference
"""

import argparse
import json
from pathlib import Path
import sys
import yaml
import torch
import pytorch_lightning as pl

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.registry import build_datamodule
from src.training.module import build_module
from src.utils.flops import count_encoder_flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/scope/s4_full.yaml")
    parser.add_argument("--split_json", default="data/splits/seed_42/split_010pct.json")
    parser.add_argument("--ckpt", required=True, help="Path to best checkpoint")
    parser.add_argument("--out_json", default="results/scope_smoke/sweep_results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["seed"] = 42
    pl.seed_everything(42, workers=True)

    # Build datamodule
    dm = build_datamodule(cfg["data"], split_json=args.split_json)
    dm.setup(stage="validate")
    val_loader = dm.val_dataloader()

    # Load model from checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    lit_model = build_module(cfg)
    ckpt_data = torch.load(args.ckpt, map_location="cpu")
    lit_model.load_state_dict(ckpt_data["state_dict"])
    lit_model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)

    # Grid sweep
    keep_ratios = [1.0, 0.75, 0.5, 0.3]
    share_fracs = [0.0, 0.25, 0.5]

    sweep_results = []
    num_classes = cfg["model"]["params"]["num_classes"]

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        precision=cfg["trainer"].get("precision", "16-mixed"),
    )

    for sf in share_fracs:
        for kr in keep_ratios:
            print(f"\n========================================================")
            print(f" Evaluating point: share_frac = {sf}, keep_ratio = {kr}")
            print(f"========================================================")

            # Configure model inference parameters
            lit_model.model.set_inference_params(
                keep_ratio=kr,
                share_frac=sf,
                exit_enabled=True,
            )

            val_res = trainer.validate(lit_model, dataloaders=val_loader, verbose=False)[0]
            clean_res = {k.replace("val/", ""): float(v) for k, v in val_res.items()}

            attn_counts, mlp_counts, exit_rate = lit_model.model.get_token_counts()
            if attn_counts and mlp_counts:
                tokens_attn_mean = float(torch.tensor(attn_counts, dtype=torch.float32).mean().item())
                tokens_mlp_mean = float(torch.tensor(mlp_counts, dtype=torch.float32).mean().item())
                enc_gflops = count_encoder_flops(attn_counts, mlp_counts)
            else:
                tokens_attn_mean = 1025.0
                tokens_mlp_mean = 1025.0
                enc_gflops = 722.36

            point_data = {
                "share_frac": sf,
                "keep_ratio": kr,
                "metrics": clean_res,
                "tokens_attn_mean": round(tokens_attn_mean, 2),
                "tokens_mlp_mean": round(tokens_mlp_mean, 2),
                "exit_rate": round(float(exit_rate), 4),
                "encoder_gflops": round(float(enc_gflops), 4),
            }
            sweep_results.append(point_data)
            print(f"Result: Burn IoU={clean_res.get('burn_iou', 0.0):.4f}, Mean IoU={clean_res.get('mean_iou', 0.0):.4f}, "
                  f"Exit Rate={exit_rate*100:.2f}%, GFLOPs={enc_gflops:.1f}")

    out_p = Path(args.out_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\n[Sweep] All 12 sweep evaluations exported to: {out_p}\n")


if __name__ == "__main__":
    main()

