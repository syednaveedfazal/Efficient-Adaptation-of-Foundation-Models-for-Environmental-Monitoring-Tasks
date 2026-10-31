# SpaCo-LoRA: pilot report (stopped at the Phase 2 gate)

Branch `spaco`, Prithvi-EO-2.0-300M, burn-scar segmentation, `split_010pct` (54 scenes), seed 42, A100.
**The pilot did not pass the continue criterion, so Phases 3–4 were not run** (as the spec requires). One seed:
this is a "not promising at this setting" result, not proof that the method cannot work.
GPU use: about 0.85 GPU-hours (2 pilot runs × ~24 min, LoRA dev/test re-evaluation 2 min, diagnosis 1 min).

## Protocol

- The 264 validation scenes were split once into **dev 132 / test 132** (seed 0, stratified by burn-fraction
  quartile; `splits/val_dev_test.json`; mean burn fraction 0.0916 / 0.0919). Early stopping and checkpoint choice
  for SpaCo runs use dev only (`splits/val_dev.json`; every log confirms 132 scenes). **Test was scored only for
  the existing LoRA r=8 checkpoint, as Phase 1 asks; no SpaCo run has been scored on test.**
- Burn IoU is **pooled over scenes** (Σ TP / Σ (TP + FP + FN)); per-scene counts are saved in `eval/*.json`. The
  Lightning-style batch-mean is also recorded (`burn_iou_batch_mean`).
- Recipe: the LoRA r=8 config unchanged (AdamW 2e-4, wd 1e-2, cosine, max 100 epochs, patience 15, bs 4,
  16-mixed). The only change is the adapter (k=8, static U, one γ per block at 10× lr and no weight decay).

## Pilot result (dev, seed 42)

| Run | Dev burn IoU | Dice | Mean IoU | Δ vs LoRA | Epochs (best) | Train time |
| --- | --- | --- | --- | --- | --- | --- |
| LoRA r=8 (existing checkpoint) | 0.6257 | 0.7697 | 0.7923 | – | 82 (65) | 893 s |
| `spaco_k8_cov` | 0.6251 | 0.7693 | 0.7919 | **−0.0006** | 58 (42) | 999 s |
| `spaco_k8_random` | 0.6298 | 0.7729 | 0.7946 | +0.0041 | 58 (42) | 955 s |

Gate: `cov − LoRA` = **−0.0006** (needs ≥ +0.01) and `cov − random` = **−0.0047** (needs ≥ +0.01). Both fail.

LoRA r=8 on test (for later comparisons): 0.7755 pooled burn IoU. The large dev/test gap (0.626 vs 0.776)
remains despite the stratification: pooled IoU is dominated by a few large scars. One caveat cuts in LoRA's favour:
the existing LoRA checkpoint was early-stopped on all 264 val scenes, dev included. That does not change the
verdict, because cov is also below random.

## Diagnosis (dev only)

`pilot_diagnosis.json`, `gamma_by_block.png`, `mode_maps_dev.png` (`scripts/spaco/diagnose_pilot.py`).

1. **The basis finds burn regions without labels, but the adapter barely uses it.**
   - Mode–scar alignment (max |r| between any of the 8 mode maps and the 32×32 burn fraction): **cov 0.363 vs
     random 0.057**; cov is higher on 132/132 dev scenes.
   - At eval, the cov model loses only 0.0024 when γ is set to 0 (0.6251 → 0.6227), and the same amount when its
     basis is swapped for a random one (0.6229).
   - So the trained model gets about +0.002 from the spatial term, and no more from the covariance basis than
     from a random one.
2. **γ does not encode "use the regions".** Both models learn γ ≈ 0.05–0.19 in blocks 0–14, so the random basis
   learns similar weights. In blocks 16–23 the cov model drives γ to 0 or negative (down to −0.16), while the
   random model stays positive. In deep layers the model *subtracts* the low-pass regional context, if anything.
3. **Explained variance:** the top-8 modes hold 52% of the patch-token energy (p10 40%, p90 64%), so P is a
   strong low-pass filter. The early-block γ ≈ 0.1–0.2 means the adapter sees only a small mixture of it.
4. **Mode maps:** modes 1–3 often follow the scar (rows 1, 4) but are frequently spent on nuisance structure:
   the no-data swath edge (row 3, mode 1), water bodies (row 2) and field patterns. The basis is computed from
   frozen block-0 features, so it reflects appearance, not burn state.

## Answers (pilot only)

- **H1 (beats LoRA r=8 at 10%)?** Not supported at seed 42 on dev: −0.0006.
- **H2 (gain comes from the covariance)?** Not supported: the random basis is +0.0047 better, and swapping the
  basis at eval costs the same as removing the term.
- **H3, H4:** not tested (Phases 3–5 not run).
- **Cost claim:** holds with one implementation fix (below): static k=8 adds 0.23 GFLOPs = 0.031% of the encoder.
  The 24 γ scalars are the only extra parameters (tested). Train time is about 6–12% slower per run.

## Deviations

1. **Projection after A.** `SpaCoLoRALinear` computes A x + γ P (A X)~ instead of A(x + γ P X~). This is
   identical when dropout = 0, which the LoRA config uses (unit test 6). Done in the input space, the overhead would
   be 5.8 GFLOPs (0.8%, mostly fc2's 4096-d input), which contradicts the spec's < 0.1% claim. With dropout > 0 the
   code falls back to the input-space form.
2. **γ scope "block"** shares one γ across all 4 adapted layers of a block (qkv, proj, fc1, fc2), not only q/k/v/proj.
3. **Config layout:** the `spaco:` block lives under `model.params` with `model.name: prithvi_fcn_spaco` and
   `adaptation: spaco_lora` (the registry passes `params` to the model), instead of a top-level `adapter:` key.
4. **Seeds** are training seeds on the same seed-42 label split (`data/splits/seed_42/split_010pct.json`); there are
   no seed-43/44 split files. `train.py` got a `--run_name` flag so runs with different seeds don't collide, and
   it now records the training seed, val_json, git commit, resolved config and learned γ in each metrics JSON.
5. **Main metric is pooled over scenes**, so test bootstrap CIs over scenes are well defined. Batch-mean IoU is kept
   alongside.
6. **AMP test on CPU** uses bfloat16 (no GPU on the login node); the GPU runs use fp16 and had no NaNs.
7. The diagnosis uses dev scenes, not the test scenes the spec's Section 6 mentions, so test stays unused.

## What I would try next (not run; your call)

- Compute U from features that already encode burn state (e.g. a mid-depth block in dynamic mode, or masked
  no-data patches) rather than block-0 appearance features.
- Exclude no-data / swath-edge patches from the SVD.
- A larger γ learning rate or γ init so the regional term is used more strongly. Any of these would need a new
  pilot under the same gate.
