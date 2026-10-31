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
