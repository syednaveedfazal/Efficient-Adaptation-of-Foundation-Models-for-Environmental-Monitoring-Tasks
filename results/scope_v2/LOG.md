# SCOPE v2 log

## Phase 1: diagnosis (no training)
- Cause of keep 1.0 < keep 0.5 is a gate train/eval mismatch. In training the mask is hard but stochastic, at about 63% keep (token-level stochastic depth). Eval uses a deterministic top-k. Decoder-only IoU: keep 1.0 = 0.335, keep 0.5 = 0.554, train-mode stochastic gate at eval = 0.599. Probes are weaker than the decoder (0.47 / 0.38 vs 0.55).
- FLOPs: 722 = 2×MAC forward of the 24 blocks; 637 = MACs (fvcore, no SDPA) forward + approximate backward, including the head. v2 convention: encoder forward, 2×MAC, including LoRA, patch embed and attention matmuls = 732.03 GFLOPs. Analytical and measured counters agree to 0.00%.
- Early stopping was not premature (both runs plateau by epoch ~20). v1's 254 s is only the resumed segment after a NaN crash at epoch 34. The LoRA 10% checkpoint and its log were gone, so it is being retrained once with the unchanged config (job 282246).

## Phase 2: code fixes
- Gate: `gate_train_mode: topk_st` (exact top-k on logit + float32 Gumbel noise, hard 0/1 forward, sigmoid straight-through backward; no rate loss) and `keep_ratio_train: fixed|uniform`. `prior_mode: off|learned|frozen`, `exit_overwrite: probe|decoder`, per-probe exit thresholds calibrated on val (98% probe pixel accuracy on exited tokens). Warm start `init_from` with `lr_new` / `lr_lora` parameter groups. Frozen-teacher KD (T=2, weight 1, optional `feat_distill`), with per-epoch teacher time logged. `min_epochs` added; the metrics JSON now stores best_ckpt, epochs_run and best_epoch.
- The model-level default stays `gumbel` / `uniform` so the v1 configs keep their behaviour; every v2 config sets `topk_st` / `fixed` explicitly.
- The LoRA r=8 10% warm-start checkpoint was retrained (job 282246, unchanged config): val burn IoU 0.6792 (v1: 0.6727), 82 epochs, best epoch 65, 893 s on A100.

## Phase 3: tests
- 5 new tests in `tests/test_scope.py` (topk_st hard mask with exactly k tokens; train and eval token counts equal; warm-start identity < 1e-4; teacher frozen, no gradients and not checkpointed; measured vs analytical FLOPs within 5%). `pytest -q tests/test_scope.py`: 12 passed.

## Phase 4: Experiment A (10% split)
- Wave 1 (6 runs, about 51 GPU-min), burn IoU at each run's own setting: LoRA 0.6792 (100%), share25 0.6742 (79.2%), share50 0.6574 (59.4%), rand50 0.5643 (71.6%), gate50 0.5393 (71.6%), gate50_prior 0.5750 (71.6%), exit 0.6880 at 100% (calibrated thresholds exit nothing). The share runs peak at Stage-B epoch 1. The gate runs are still rising at epoch 14 of 15.
- The exit calibration first used a "monotone from the top" rule that stopped at the first threshold below 98%. It was changed to the spec's rule: the lowest threshold whose confident tokens (≥ 100) reach 98% probe pixel accuracy. `exit` was re-evaluated. Combos use the learned prior (0.575 > 0.539).
- Wave 2 (combos, learned prior; about 27 GPU-min). Calibrated exits: 98.0–99.4% exited-patch accuracy at the runs' own settings. `combo_strong` at keep 1.0 / share 0.5 = 0.6712 @ 56.9% FLOPs; at its own keep 0.5 / share 0.5 setting = 0.5238 @ 34.1%. `combo_mild` own setting = 0.6097 @ 61.3%.

## Phase 5–6: go/no-go and report
- Go (promising): `combo_strong` (keep 1.0, share 0.5, exit) = 0.6712 at 56.9% of LoRA FLOPs; 0.008 below LoRA, +0.107 over rand50 (71.6%). Caveats: the gate is open at that point, the point was chosen on val, and Stage B alone lifts full compute to about 0.687.
- 100%-label check NOT run: there is no plain LoRA r=8 `split_100pct` checkpoint; asking before training one. Report: REPORT.md; plot: iou_vs_flops.png. About 1.7 GPU-h used in total.
