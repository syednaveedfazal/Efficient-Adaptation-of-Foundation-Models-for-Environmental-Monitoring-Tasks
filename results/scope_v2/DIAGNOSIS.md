# SCOPE v1 diagnosis (Phase 1, no training)

All numbers come from the v1 checkpoint
`results/scope_smoke/checkpoints/prithvi_scope_s4_full_seed_42_split_010pct/epoch=28-val/burn_iou=0.5135.ckpt`,
evaluated on the 264 validation scenes with batch size 4 and fp16 autocast (the same as v1). Metrics are the mean of
per-batch metrics, the same aggregation Lightning used in v1. Raw output: `diagnosis.json`
(`scripts/scope_v2/diagnose.py`, SLURM job 282276, about 4 GPU-minutes on an A100).

## 1. FLOP reconciliation (722 vs 637)

| Counter | What it counts | Plain LoRA r=8 value |
| --- | --- | --- |
| Old `get_model_training_gflops` (`fvcore` missing, so it returns hard-coded constants) | **MACs**, measured earlier with fvcore: encoder + LoRA + patch embed + FCN head, **no attention-score matmuls** (fvcore does not count SDPA). Training value = forward 317.05 + "backward" 319.47 (backward = 1× encoder + 2× head) | 317.05 fwd, **636.5 "train"** |
| v1 `count_encoder_flops` | **FLOPs = 2 × MACs**, 24 transformer blocks only (qkv, QKᵀ, AV, proj, MLP), forward, no LoRA, no patch embed | **722.36** fwd |
| `torch.utils.flop_counter`, math SDPA (new `measure_encoder_gflops`) | FLOPs = 2 × MACs, everything the encoder runs | 732.03 fwd = 619.08 linear + 103.28 attention bmm + 6.45 LoRA + 3.22 patch embed |

So 722 and 637 measure different things in different units. 722 is forward FLOPs of the blocks. 637 is MACs of
forward plus an approximate backward, with the head included and the attention scores left out. They are close only
by coincidence: the ×2 from MACs to FLOPs roughly cancels the forward-plus-backward sum. Input size is not the cause;
both use 512×512 (1025 tokens).

**Convention chosen for v2:** encoder forward GFLOPs, FLOPs = 2 × multiply-adds, one 512×512 image, all matmuls in
the patch embedding and the 24 blocks (including QKᵀ/AV and LoRA r=8), plus SCOPE's gate projections and probes.
Norms, softmax, GELU, patch statistics and the FCN head are not counted. Under this convention:

- analytical `encoder_gflops([1025]*24, [1025]*24)` = **732.03 GFLOPs** (the 100% reference),
- measured `measure_encoder_gflops(plain LoRA)` = **732.03 GFLOPs** (0.00% difference),
- measured `measure_encoder_gflops(SCOPE, everything off)` = 732.05 GFLOPs (+0.003%, from the covariance matmul in
  `patch_stats`).

`count_encoder_flops` keeps its v1 default output (722.36) so the v1 numbers can be reproduced. The legacy function
`get_model_training_gflops` is unchanged and is not used for v2 numbers. A test checks that the two counters agree
within 5%.

## 2. Training curves (v1 logs)

| Run | Epochs actually run | Best epoch (val burn IoU) | Early stop? |
| --- | --- | --- | --- |
| LoRA r=8, 10% (v1 baseline, 0.6727) | unknown: its CSV log and checkpoint no longer exist (only the 1-epoch timing logs remain). The JSON gives 581 s on an A40. | unknown | unknown |
| `s4_full` | 0–34 in job 280862, which **crashed with NaN at epoch 34**. Resumed from the epoch-28 checkpoint in job 280882 and ran 29–38. In total 39 epochs (5 trained twice). | 28 (0.5135) | yes, at epoch 38 (patience 10), 2 epochs short of max 40 |
| `ablation_no_prior` | 0–34 (35 epochs) | 24 (0.5376) | yes, at epoch 34 (patience 10) |

Val burn-IoU curves (every epoch, at the training config's eval setting):

- `s4_full` (share 0.5, keep 0.5, exit on): 0.270 0.333 0.417 0.430 0.453 0.455 0.460 0.458 0.467 0.466 0.486 0.465 0.488 0.490 0.495 0.496 0.441 0.499 0.497 0.502 0.505 0.513 0.468 0.495 0.502 0.512 0.502 0.508 **0.513** 0.513 0.511 0.511 0.513 0.510 | resume: 0.512 0.511 0.498 0.499 0.507 0.511 0.503 0.497 0.505 0.509
- `ablation_no_prior`: 0.305 0.400 0.343 0.404 0.465 0.435 0.478 0.446 0.477 0.453 0.494 0.498 0.508 0.505 0.505 0.512 0.505 0.510 0.491 0.527 0.526 0.535 0.491 0.526 **0.538** 0.527 0.513 0.527 0.522 0.527 0.522 0.522 0.519 0.524 0.529

**Early stopping was not too early.** Both curves plateau around epoch 20. Train burn IoU kept rising to about 0.75
while val stayed flat, so the gap is a generalisation gap, not under-training. The "254 s vs 581 s" in v1 is a
**reporting artefact**: 254 s is only the resumed segment (epochs 29–38) of `s4_full`, and the LoRA time was measured
on an A40, not the A100.

The LoRA r=8 10% checkpoint had been deleted, so it was retrained once for v2 with the unchanged config
(`configs/prithvi_fcn_lora_r8.yaml`, seed 42, job 282246). See LOG.md and REPORT.md for that curve.

## 3. Gate mask audit (train mode, 4 val batches)

- 245,760 mask values: **90,571 exactly 0 and 155,189 exactly 1, none fractional**. The forward mask is hard.
- But the **keep rate in training is about 0.63** (per block 0.55–0.72) and the mask is **stochastic**:
  m = 1[logit + logistic noise > 0], with a per-block random target rate U(0.3, 1.0) in the rate loss. At eval the
  gate instead takes the deterministic top-k by logit at the configured keep ratio (0.5).

## 4. Decomposition of `s4_full` (share 0)

| Setting | Decoder only (exit off) | Probe-8 only | Probe-16 only | Combined (exit on, as v1) | Exit rate (combined) |
| --- | --- | --- | --- | --- | --- |
| keep 1.0 | 0.3346 | 0.2968 | 0.3106 | 0.3344 | 4.1% |
| keep 0.5 | 0.5537 | 0.4681 | 0.3811 | 0.5535 | 63.9% |

Decoder-only keep sweep (exit off): keep 1.0 → 0.335, 0.75 → 0.490, 0.65 → 0.524, **0.5 → 0.554**, 0.3 → 0.339.
**Eval with the stochastic train-mode gate** (about 61% MLP tokens, exit off): **0.599**, better than any
deterministic top-k setting.

## 5. Sanity point (keep 1.0, share 0, exit off)

`nbr_refine` on: 0.3346, off: 0.3222. The refinement head is not the cause; it adds +0.012.

## Conclusion: cause of keep 1.0 < keep 0.5

It is a **train/eval mismatch in the Stage-2 gate**, not the probes or the decoder:

1. In training each block's MLP updated a random ~63% of the tokens. This is token-level stochastic depth with no
   rescaling at eval. At keep 1.0 every token gets every MLP update, so the residual stream drifts away from what
   LoRA, the probes and the head were trained on. Even the probe-8 output degrades (0.47 → 0.30), which shows the
   features are already off-distribution by block 8.
2. At eval the selected set is a deterministic top-k by logit, not the noisy set seen in training. The model scores
   best when evaluated with the same stochastic gate it was trained with (0.599). The keep curve peaks near 0.5–0.65
   only because that is closest to the training rate.
3. The probe path is weaker than the decoder path (0.47/0.38 vs 0.55), and exit changes almost nothing (0.5535 vs
   0.5537). At keep 0.5, 64% of tokens exit, but their overwritten pixels are mostly confident background.

This is fixed in Phase 2 by `gate_train_mode: topk_st` with `keep_ratio_train: fixed`: the same k tokens are routed in
training and at eval, with a hard mask and a straight-through gradient. The design does not change.

### Other v1 problems found along the way

- **Token counts and GFLOPs in v1 come from the last validation batch only** (`get_token_counts()` returns the last
  forward). With exit on they depend on the batch, because a token exits only if it is confident in all 4 images of
  the batch. Example: v1 reported 415 attention tokens per block for share 0.5 / keep 0.5; the mean over all batches is
  457. v2 averages over every batch.
- `b1_random` was trained on the 1% split for 2 epochs (already known).
- The v1 LoRA metrics JSON was produced on an A40; SCOPE runs ran on an A100. Train times were not comparable.
