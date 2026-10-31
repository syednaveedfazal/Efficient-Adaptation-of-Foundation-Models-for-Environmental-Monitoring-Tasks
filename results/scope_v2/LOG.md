# SCOPE v2 log

## Phase 1: diagnosis (no training)
- Cause of keep 1.0 < keep 0.5 is a gate train/eval mismatch. In training the mask is hard but stochastic, at about 63% keep (token-level stochastic depth). Eval uses a deterministic top-k. Decoder-only IoU: keep 1.0 = 0.335, keep 0.5 = 0.554, train-mode stochastic gate at eval = 0.599. Probes are weaker than the decoder (0.47 / 0.38 vs 0.55).
- FLOPs: 722 = 2×MAC forward of the 24 blocks; 637 = MACs (fvcore, no SDPA) forward + approximate backward, including the head. v2 convention: encoder forward, 2×MAC, including LoRA, patch embed and attention matmuls = 732.03 GFLOPs. Analytical and measured counters agree to 0.00%.
- Early stopping was not premature (both runs plateau by epoch ~20). v1's 254 s is only the resumed segment after a NaN crash at epoch 34. The LoRA 10% checkpoint and its log were gone, so it is being retrained once with the unchanged config (job 282246).
