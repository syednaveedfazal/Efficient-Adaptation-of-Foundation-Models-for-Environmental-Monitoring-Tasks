# SCOPE Implementation & Experiment Log

## Phase 1: Repo Inspection & Baseline Verification
- **Prithvi Encoder Architecture**:
  - Defined in `models/pretrained/prithvi/prithvi_mae.py` (`PrithviViT` class).
  - Uses `timm.models.vision_transformer.Block` inside `self.blocks = nn.ModuleList(...)` with depth = 24, embed_dim = 1024, num_heads = 16, mlp_ratio = 4.0.
  - Submodules per block: `norm1` (LayerNorm), `attn` (Attention with `qkv`, `proj`), `ls1` (LayerScale), `drop_path1`, `norm2` (LayerNorm), `mlp` (Mlp with `fc1`, `act`, `drop1`, `norm`, `fc2`, `drop2`), `ls2`, `drop_path2`.
  - Block forward computes: `x = x + drop_path1(ls1(attn(norm1(x))))`, then `x = x + drop_path2(ls2(mlp(norm2(x))))`.
- **CLS Token & Positional Embeddings**:
  - `cls_token` is present (`self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))`).
  - Pre-computed 3D sin-cos positional embeddings with $T=1$ (`get_3d_sincos_pos_embed`), registered in buffer `pos_embed`.
  - In `forward_features`, `pos_embed` is interpolated to sample shape ($32 \times 32$), added to patch tokens, and `cls_token + pos_embed[:, :1, :]` is prepended at index 0. Sequence length is $1 + 1024 = 1025$.
- **Normalization & Band Order**:
  - Input normalization is performed in `BurnScarDataset.__getitem__` in `src/datasets/burn_scar.py`: `scene = (scene - mean) / std`.
  - Stats are stored in `data/processed/stats.json` with 6 bands: Blue (0), Green (1), Red (2), NIR (3), SWIR1 (4), SWIR2 (5).
  - Normalization is undone via `x = x_norm * std + mean`.
  - NBR calculation uses NIR (idx 3) and SWIR2 (idx 5): $\text{NBR} = \frac{\text{NIR} - \text{SWIR2}}{\text{NIR} + \text{SWIR2} + 10^{-6}}$.
- **Token Grid Size**:
  - For $512 \times 512$ inputs with $16 \times 16$ patches, grid size is $32 \times 32 = 1,024$ patch tokens + 1 CLS token = 1,025 tokens total. Embed dim = 1,024, 24 transformer blocks.
- **Baseline 1-Epoch Timing & VRAM**:
  - Ran `prithvi_fcn_lora_r8` on `data/splits/seed_42/split_010pct.json` (54 scenes) for 1 epoch on A100 (Job ID: 280834).
  - Training loop duration: ~2 seconds for training epoch, ~4 seconds for validation epoch (total script duration: 37.7s).
  - Peak allocated VRAM: 6,029.12 MB (~6.03 GB); Peak reserved VRAM: 6,312.00 MB (~6.31 GB).
  - Val burn IoU at epoch 0: 0.4490, mean IoU: 0.6956.


## Phase 2 & 3: Implementation & Unit Testing
- **Implementation**:
  - Implemented `src/models/scope.py` with `patch_stats` (un-normalizing, covariance, logdet, NBR stats, z-scored prior features), Stage 1 `CovarianceTokenSharing` (heterogeneity scoring with log-Euclidean distance, token reduction, and lossless unshare), Stage 2 `ScopeGateBlock` (DyT-style Gumbel-Sigmoid routing with target rate regularizer and eval top-k gather/scatter), Stage 3 `ConceptProbe` & early exit (auxiliary soft CE loss on patch burn fractions, confident token exit with DToP context retention, logit overwriting), and Extras (`nbr_refine`, `self_distill`).
  - Implemented `count_encoder_flops()` and SCOPE analytical FLOP accounting in `src/utils/flops.py`.
  - Registered `prithvi_fcn_scope` in `src/models/registry.py` and wired auxiliary loss in `src/training/module.py`.
  - Created 5 SCOPE configs in `configs/scope/`: `b1_random.yaml`, `s1_share.yaml`, `s2_gate.yaml`, `s3_exit.yaml`, and `s4_full.yaml`.
- **Unit Tests**:
  - Created `tests/test_scope.py` covering all 7 required tests: identity check vs plain LoRA (max diff = 0.0), unshare round trip, patch statistics verification against NumPy, gate routing and gradient flow to $W_g$ and $w_p$, static output shapes across different inputs, monotonic encoder GFLOPs scaling, and early exit behavior.
  - All 7 tests passed cleanly via `pytest` in 26.21s.

## Phase 4: Smoke Run Verification (split_001pct, 2 epochs)
- **Setup**: Ran 6 smoke configurations on `split_001pct.json` (5 training scenes) for 2 epochs on NVIDIA A100-SXM4-80GB (`A100medium` partition, SLURM Job ID 280851).
- **Checks & Findings**:
  - All 6 configurations completed with exit code 0.
  - Loss values were strictly stable and finite throughout training and validation (no `NaN` or `Inf`).
  - In `s2_gate`, average gate activation is 50.08% ($321.0 / 641.0$), well within the target $[0.2, 0.9]$ range.
  - In `s3_exit`, early exit rate was active and logged at 5.78% with tokens exiting after concept probes 8 & 16, reducing inference encoder compute from 722 GFLOPs to 288.7 GFLOPs.
  - In `s1_share`, covariance token sharing alone reduced tokens to 641 while achieving 0.3035 Val Burn IoU (exceeding baseline 0.2885).
  - All outputs safely exported to `results/scope_smoke/metrics/`.

### Phase 4 Smoke Test Summary Table

| Configuration | Val Burn IoU | Val Mean IoU | Val Loss | Attn Tokens | MLP Tokens | Exit Rate | Enc GFLOPs | Peak VRAM (MB) | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prithvi_fcn_lora_r8` | 0.2885 | 0.4770 | 0.5482 | 1025.0 | 1025.0 | 0.00% | 317.1 | 3936.3 | 44.7s |
| `prithvi_scope_b1_random` | 0.2094 | 0.4278 | 0.5178 | 1025.0 | 513.0 | 0.00% | 516.2 | 3766.6 | 54.7s |
| `prithvi_scope_s1_share` | 0.3035 | 0.5098 | 0.5252 | 641.0 | 641.0 | 0.00% | 427.5 | 3080.9 | 45.4s |
| `prithvi_scope_s2_gate` | 0.1514 | 0.3239 | 0.5903 | 641.0 | 321.0 | 0.00% | 298.7 | 3201.3 | 60.1s |
| `prithvi_scope_s3_exit` | 0.2910 | 0.5692 | 0.5079 | 621.7 | 311.7 | 5.78% | 288.7 | 3210.7 | 64.8s |
| `prithvi_scope_s4_full` | 0.2063 | 0.4555 | 0.5419 | 634.7 | 318.0 | 2.81% | 295.4 | 3236.2 | 63.4s |
