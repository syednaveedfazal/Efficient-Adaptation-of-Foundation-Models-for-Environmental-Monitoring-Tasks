# SpaCo-LoRA log

## Phase 1: build and test (no training)
- `src/models/spaco_lora.py` (SpaCoLoRALinear, SpaCoContext, SpaCoPrithviFCNSegmentor). Same targets as LoRA r=8 (peft on `attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`, r=8, alpha=8, dropout 0); parameter names mirror peft so LoRA checkpoints load. One gamma per block, shared by the block's 4 adapted layers; gamma at 10x lr, no weight decay. U from block-0 norm1 output (static) via fp32 `svd_lowrank` (q = k+8, niter 2), autocast off.
- Projection is applied after A (A(x + gamma P X~) = A x + gamma P (A X)~, exact with dropout 0, unit-tested), so the static k=8 overhead is 0.23 GFLOPs = 0.031% of 732 GFLOPs. Done literally in input space it would be 5.8 GFLOPs (0.8%), above the spec's < 0.1% claim.
- `splits/val_dev_test.json`: 264 val scenes, seed 0, stratified by burn-fraction quartile → dev 132 / test 132 (mean burn fraction 0.0916 / 0.0919). Configs in `configs/spaco/` = the LoRA r=8 config with `val_json: splits/val_dev.json`. `tests/test_spaco.py`: 9 passed; `tests/test_scope.py`: 12 passed.
