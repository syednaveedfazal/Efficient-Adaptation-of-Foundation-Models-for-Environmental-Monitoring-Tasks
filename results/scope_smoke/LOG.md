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
