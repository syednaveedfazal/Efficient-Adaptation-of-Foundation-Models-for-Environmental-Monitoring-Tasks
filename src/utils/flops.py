import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union


class _EncoderFeaturesWrapper(nn.Module):
    """Wraps a PrithviViT encoder so that FlopCountAnalysis traces
    forward_features() (the full-token segmentation path) instead of
    forward() (the MAE masking path which drops 75% of tokens)."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder.forward_features(x)


def count_encoder_flops(
    tokens_per_block: List[int],
    mlp_tokens_per_block: List[int],
    D: int = 1024,
    mlp_ratio: float = 4.0,
    lora_rank: int = 0,
    include_patch_embed: bool = False,
    gate_tokens_per_block: Optional[List[int]] = None,
    probe_tokens: Optional[List[int]] = None,
    n_patches: int = 1024,
    patch_dim: int = 6 * 16 * 16,
) -> float:
    """
    Computes total ViT encoder forward GFLOPs analytically according to the SCOPE spec.
    Convention: FLOPs = 2 x multiply-adds (matmuls only; norms, softmax, GELU are ignored).
    Per block:
      attention = 2 * N * D * 3D + 4 * N^2 * D + 2 * N * D * D
      MLP = 2 * 2 * N_mlp * D * (mlp_ratio * D)

    The defaults reproduce the v1 count (blocks only). The SCOPE v2 convention
    (encoder_gflops) also counts LoRA adapters, the patch embedding, the gate
    projections and the concept probes.

    Args:
        tokens_per_block: list of token counts per block (length 24)
        mlp_tokens_per_block: list of MLP token counts per block (length 24)
        D: embedding dimension (default 1024)
        mlp_ratio: MLP expansion ratio (default 4.0)
        lora_rank: LoRA rank on qkv / proj / fc1 / fc2 (0 = not counted)
        include_patch_embed: count the patch-embedding projection
        gate_tokens_per_block: patch tokens scored by the gate Linear(D, 1) per block
        probe_tokens: tokens scored by each concept probe Linear(D, 2)

    Returns:
        Total encoder forward GFLOPs (float, 1e9 FLOPs)
    """
    H = mlp_ratio * D
    total_flops = 0.0
    for N, N_mlp in zip(tokens_per_block, mlp_tokens_per_block):
        attn_flops = (2.0 * N * D * (3 * D)) + (4.0 * (N ** 2) * D) + (2.0 * N * D * D)
        mlp_flops = 2.0 * 2.0 * N_mlp * D * (mlp_ratio * D)
        total_flops += attn_flops + mlp_flops
        if lora_rank:
            total_flops += 2.0 * lora_rank * N * ((D + 3 * D) + (D + D))
            total_flops += 2.0 * lora_rank * N_mlp * ((D + H) + (H + D))
    if include_patch_embed:
        total_flops += 2.0 * n_patches * patch_dim * D
    for n in gate_tokens_per_block or []:
        total_flops += 2.0 * n * D
    for n in probe_tokens or []:
        total_flops += 2.0 * n * D * 2
    return round(float(total_flops / 1e9), 4)


# Plain LoRA r=8 Prithvi encoder at 512x512 under the v2 convention (the 100% FLOP reference)
LORA_ENCODER_GFLOPS = count_encoder_flops([1025] * 24, [1025] * 24, lora_rank=8, include_patch_embed=True)


def encoder_gflops(
    tokens_per_block: List[int],
    mlp_tokens_per_block: List[int],
    gate_tokens_per_block: Optional[List[int]] = None,
    probe_tokens: Optional[List[int]] = None,
    lora_rank: int = 8,
) -> float:
    """SCOPE v2 convention: encoder forward GFLOPs (2 x MACs), 512x512, batch 1, incl. LoRA + patch embed."""
    return count_encoder_flops(
        tokens_per_block, mlp_tokens_per_block, lora_rank=lora_rank, include_patch_embed=True,
        gate_tokens_per_block=gate_tokens_per_block, probe_tokens=probe_tokens,
    )


def measure_encoder_gflops(model: nn.Module, input_size=(1, 6, 512, 512)) -> float:
    """
    Measured encoder forward GFLOPs with torch.utils.flop_counter (FLOPs = 2 x MACs).
    Attention runs on the math SDPA backend so QK^T and AV are counted as bmm.
    For models with a `decoder` (and SCOPE's `refine_net`) the head FLOPs are subtracted.
    """
    from torch.utils.flop_counter import FlopCounterMode
    from torch.nn.attention import sdpa_kernel, SDPBackend

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(*input_size, device=device)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc:
        model(x)
    counts = fc.get_flop_counts()
    total = fc.get_total_flops()
    root = type(model).__name__
    for head in ("decoder", "refine_net"):
        total -= sum(counts.get(f"{root}.{head}", {}).values())
    model.train(was_training)
    return round(float(total / 1e9) / input_size[0], 4)


def get_model_training_gflops(model, adaptation_strategy: str, input_size=(1, 6, 512, 512)):
    """
    Computes both Forward and Backward pass GFLOPs analytically for the model.
    Returns:
        (fwd_gflops, bwd_gflops) as floats.
    """
    # If the model is a SCOPE model, compute GFLOPs analytically
    if hasattr(model, "get_token_counts"):
        attn_tokens, mlp_tokens, _ = model.get_token_counts()
        if not attn_tokens or not mlp_tokens:
            attn_tokens = [1025] * 24
            mlp_tokens = [1025] * 24
        fwd_encoder = count_encoder_flops(attn_tokens, mlp_tokens, D=1024, mlp_ratio=4.0)
        # FCN decoder FLOPs on 512x512
        fwd_decoder = 2.4191
        fwd_total = fwd_encoder + fwd_decoder
        if adaptation_strategy in ("lora", "vpt", "scope"):
            bwd_gflops = (1.0 * fwd_encoder) + (2.0 * fwd_decoder)
        elif adaptation_strategy == "linear_probe":
            bwd_gflops = 2.0 * fwd_decoder
        else:
            bwd_gflops = 2.0 * fwd_total
        return round(float(fwd_total), 4), round(float(bwd_gflops), 4)

    # Standard model path with fvcore if available
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        # Fallback if fvcore is not installed
        # Prithvi baseline: ~317.05 fwd GFLOPs
        if "prithvi" in model.__class__.__name__.lower():
            fwd_total = 317.0508
            bwd_gflops = 319.4699
            return fwd_total, bwd_gflops
        return 0.0, 0.0

    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(*input_size, device=device)

    # Calculate total forward FLOPs
    total_analysis = FlopCountAnalysis(model, x)
    total_analysis.unsupported_ops_warnings(False)
    fwd_total = total_analysis.total() / 1e9

    # Check if the model has a separated backbone encoder
    if hasattr(model, "encoder") and model.encoder is not None:
        encoder_wrapper = _EncoderFeaturesWrapper(model.encoder)
        encoder_analysis = FlopCountAnalysis(encoder_wrapper, x)
        encoder_analysis.unsupported_ops_warnings(False)
        fwd_encoder = encoder_analysis.total() / 1e9
        fwd_decoder = max(0.0, fwd_total - fwd_encoder)
    else:
        fwd_encoder = fwd_total
        fwd_decoder = 0.0

    # Calculate backward pass GFLOPs analytically
    if adaptation_strategy == "linear_probe":
        bwd_gflops = 2.0 * fwd_decoder
    elif adaptation_strategy in ("lora", "vpt", "scope"):
        bwd_gflops = (1.0 * fwd_encoder) + (2.0 * fwd_decoder)
    else:
        bwd_gflops = 2.0 * fwd_total

    return round(float(fwd_total), 4), round(float(bwd_gflops), 4)
