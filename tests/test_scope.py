"""
tests/test_scope.py — Comprehensive unit tests for SCOPE adaptation method.

Required tests:
1. Identity checks: share_frac=0, keep_ratio=1, exit=off, extras=off matches plain LoRA within 1e-4.
2. Unshare round trip: unshare(share(x)) restores 32x32 grid; unshared positions untouched.
3. Stats: nbr_mean matches manual numpy computation; covariance is symmetric PSD.
4. Gate: eval mode with keep_ratio r sends ceil(r*N) tokens through MLP; training mode routes gradients to W_g and w_p.
5. Static shapes: two different inputs produce outputs with same tensor shape.
6. FLOP accounting: encoder GFLOPs decrease monotonically as share_frac rises and keep_ratio falls; SCOPE off matches formula.
7. Early exit: thresholds (0, 1) -> exit rate 0; thresholds (0.5, 0.5) -> all exit except 2k context tokens.
"""

import math
import pytest
import numpy as np
import torch
import torch.nn as nn

from src.models.scope import (
    patch_stats,
    CovarianceTokenSharing,
    ScopeGateBlock,
    ConceptProbe,
    ScopePrithviFCNSegmentor,
)
from src.models.prithvi_fcn import PrithviFCNSegmentor
from src.utils.flops import count_encoder_flops


WEIGHTS_PATH = "models/pretrained/prithvi/Prithvi_EO_V2_300M.pt"


def test_1_identity_check():
    """Test 1: Identity check against plain LoRA model within 1e-4."""
    torch.manual_seed(42)
    base_model = PrithviFCNSegmentor(
        weights_path=WEIGHTS_PATH,
        num_classes=2,
        adaptation="lora",
        lora_rank=8,
        lora_alpha=8,
    )
    base_model.eval()

    scope_model = ScopePrithviFCNSegmentor(
        weights_path=WEIGHTS_PATH,
        num_classes=2,
        lora_rank=8,
        lora_alpha=8,
        share_frac=0.0,
        gate_enabled=False,
        exit_enabled=False,
        nbr_refine=False,
    )
    scope_model.load_state_dict(base_model.state_dict(), strict=False)
    scope_model.eval()

    x = torch.randn(1, 6, 512, 512)
    with torch.no_grad():
        out_base = base_model(x)
        out_scope = scope_model(x)

    max_diff = (out_base - out_scope).abs().max().item()
    assert max_diff < 1e-4, f"Identity check failed with max diff {max_diff}"


def test_2_unshare_round_trip():
    """Test 2: unshare(share(x)) restores 32x32 grid; unshared tokens are untouched."""
    share_frac = 0.5
    cts = CovarianceTokenSharing(share_frac=share_frac, lam=1.0)
    B = 2
    x_norm = torch.randn(B, 6, 512, 512)
    stats = patch_stats(x_norm, torch.zeros(6), torch.ones(6))

    tokens = torch.randn(B, 1024, 64)
    p_tokens = torch.randn(B, 1024, 4)

    reduced_x, reduced_p, share_info = cts.share(tokens, p_tokens, stats)
    expected_g = int(round(share_frac * 256))
    assert reduced_x.shape == (B, 1024 - 3 * expected_g, 64)

    restored_x = cts.unshare(reduced_x, share_info)
    assert restored_x.shape == (B, 1024, 64)

    # Verify unshared token positions match exactly
    gp = cts.group_patches.cpu()
    for b in range(B):
        unsh_groups = share_info["unshared_groups"][b]
        for gid in unsh_groups:
            patches = gp[gid]
            assert torch.allclose(restored_x[b, patches], tokens[b, patches], atol=1e-6)


def test_3_patch_stats():
    """Test 3: nbr_mean matches manual numpy calculation; covariance is symmetric PSD."""
    B, C, H, W = 1, 6, 512, 512
    x_norm = torch.randn(B, C, H, W)
    mean = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    std = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    res = patch_stats(x_norm, mean, std)

    # 1. Check covariance symmetry and positive semi-definiteness
    cov = res["cov"]  # (1, 1024, 6, 6)
    sym_diff = (cov - cov.transpose(-1, -2)).abs().max().item()
    assert sym_diff < 1e-5, f"Covariance is not symmetric: diff {sym_diff}"

    evals = torch.linalg.eigvalsh(cov)
    assert (evals >= -1e-5).all(), "Covariance has negative eigenvalues"

    # 2. Check nbr_mean against manual numpy calculation on patch (0, 0)
    x_unnorm = (x_norm * std.view(1, 6, 1, 1) + mean.view(1, 6, 1, 1))[0].numpy()
    patch0_nir = x_unnorm[3, :16, :16]
    patch0_swir2 = x_unnorm[5, :16, :16]
    nbr_manual = (patch0_nir - patch0_swir2) / (patch0_nir + patch0_swir2 + 1e-6)
    expected_nbr_mean = float(nbr_manual.mean())

    computed_nbr_mean = res["nbr_mean"][0, 0].item()
    assert abs(computed_nbr_mean - expected_nbr_mean) < 1e-5, (
        f"nbr_mean mismatch: computed {computed_nbr_mean} vs manual {expected_nbr_mean}"
    )


def test_4_gate_routing():
    """Test 4: Eval mode routes ceil(r*N) tokens through MLP; training mode routes gradients to W_g and w_p."""
    class MockBlock(nn.Module):
        def __init__(self, d=64):
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.attn = nn.Linear(d, d)
            self.ls1 = nn.Identity()
            self.drop_path1 = nn.Identity()
            self.norm2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            self.ls2 = nn.Identity()
            self.drop_path2 = nn.Identity()

    # 1. Eval mode: hook on mlp
    mock_block = MockBlock(64)
    gate_block = ScopeGateBlock(mock_block, embed_dim=64, a_scale=1.0)
    gate_block.eval()

    mlp_input_counts = []
    def mlp_hook(module, inp, out):
        mlp_input_counts.append(inp[0].shape[1])

    handle = mock_block.mlp.register_forward_hook(mlp_hook)

    N = 100
    r = 0.4
    x = torch.randn(2, N, 64)
    p = torch.randn(2, N, 4)

    gate_block(x, p, keep_ratio=r)
    handle.remove()

    expected_tokens = int(math.ceil(r * N))
    assert mlp_input_counts[-1] == expected_tokens, (
        f"Expected {expected_tokens} MLP tokens, got {mlp_input_counts[-1]}"
    )

    # 2. Training mode: check gradient flow to W_g and w_p
    gate_block.train()
    x_train = torch.randn(2, N, 64, requires_grad=True)
    p_train = torch.randn(2, N, 4)

    out = gate_block(x_train, p_train, r_target=0.5)
    loss = out.sum() + gate_block.last_rate_loss
    loss.backward()

    assert gate_block.W_g.weight.grad is not None, "W_g.weight received no gradient"
    assert gate_block.W_g.bias.grad is not None, "W_g.bias received no gradient"
    assert gate_block.w_p.grad is not None, "w_p received no gradient"


def test_5_static_shapes():
    """Test 5: Two different inputs produce outputs with identical tensor shape."""
    model = ScopePrithviFCNSegmentor(
        weights_path=WEIGHTS_PATH,
        num_classes=2,
        share_frac=0.5,
        gate_enabled=True,
        exit_enabled=True,
        nbr_refine=True,
    )
    model.eval()

    x1 = torch.randn(1, 6, 512, 512)
    x2 = torch.randn(1, 6, 512, 512)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    assert out1.shape == out2.shape == (1, 2, 512, 512), (
        f"Shape mismatch: {out1.shape} vs {out2.shape}"
    )


def test_6_flop_accounting():
    """Test 6: Encoder GFLOPs decrease monotonically as share_frac rises and keep_ratio falls; SCOPE off matches formula."""
    # 1. SCOPE off: 24 blocks, N=1025, N_mlp=1025
    attn_full = [1025] * 24
    mlp_full = [1025] * 24
    flops_off = count_encoder_flops(attn_full, mlp_full, D=1024, mlp_ratio=4.0)

    # Manual analytical calculation
    D = 1024
    N = 1025
    expected_block_flops = (2 * N * D * (3 * D) + 4 * (N ** 2) * D + 2 * N * D * D) + (2 * 2 * N * D * 4 * D)
    expected_total_gflops = round(24 * expected_block_flops / 1e9, 4)
    assert abs(flops_off - expected_total_gflops) < 1e-4, f"Mismatch: {flops_off} vs {expected_total_gflops}"

    # 2. Monotonic decrease as share_frac rises (1024 -> 1024 - 3g patch tokens)
    flops_shares = []
    for share_frac in [0.0, 0.25, 0.5, 0.75]:
        g = int(round(share_frac * 256))
        n_tokens = 1 + (1024 - 3 * g)
        flops = count_encoder_flops([n_tokens] * 24, [n_tokens] * 24)
        flops_shares.append(flops)

    for i in range(len(flops_shares) - 1):
        assert flops_shares[i] > flops_shares[i + 1], (
            f"GFLOPs did not decrease strictly monotonically with share_frac: {flops_shares}"
        )

    # 3. Monotonic decrease as keep_ratio falls
    flops_gates = []
    N = 1025
    for keep_ratio in [1.0, 0.75, 0.5, 0.3]:
        n_mlp = 1 + int(math.ceil(keep_ratio * 1024))
        flops = count_encoder_flops([N] * 24, [n_mlp] * 24)
        flops_gates.append(flops)

    for i in range(len(flops_gates) - 1):
        assert flops_gates[i] > flops_gates[i + 1], (
            f"GFLOPs did not decrease strictly monotonically with keep_ratio: {flops_gates}"
        )


def test_7_early_exit():
    """Test 7: Thresholds (0, 1) -> 0 exit rate; thresholds (0.5, 0.5) -> all exit except 2k context tokens."""
    model = ScopePrithviFCNSegmentor(
        weights_path=WEIGHTS_PATH,
        num_classes=2,
        share_frac=0.0,
        gate_enabled=False,
        exit_enabled=True,
        exit_confident_thresh_high=1.0,
        exit_confident_thresh_low=0.0,
        exit_context_k=16,
    )
    model.eval()

    x = torch.randn(1, 6, 512, 512)

    # 1. No exit when thresholds are impossible (>= 1.0 or <= 0.0)
    with torch.no_grad():
        _ = model(x)
    attn_counts, mlp_counts, exit_rate = model.get_token_counts()
    assert exit_rate == 0.0, f"Expected 0 exit rate, got {exit_rate}"
    assert all(c == 1025 for c in attn_counts), f"Token counts were reduced: {attn_counts}"

    # 2. Maximum exit when thresholds are (0.5, 0.5): all confident, only 2*k=32 tokens remain as context
    model.set_inference_params(exit_enabled=True)
    model.exit_thresh_high = 0.5
    model.exit_thresh_low = 0.5
    with torch.no_grad():
        _ = model(x)
    attn_counts, mlp_counts, exit_rate = model.get_token_counts()

    # Blocks 0..7: 1025 tokens
    assert all(c == 1025 for c in attn_counts[:8]), f"Blocks 0..7 expected 1025, got {attn_counts[:8]}"
    # Blocks 8..23: 1 cls + 2*16 context tokens = 33 tokens
    assert all(c == 33 for c in attn_counts[8:]), f"Blocks 8..23 expected 33 tokens, got {attn_counts[8:]}"
    expected_exit_rate = (1024 - 32) / 1024.0
    assert abs(exit_rate - expected_exit_rate) < 1e-4, (
        f"Expected exit rate {expected_exit_rate}, got {exit_rate}"
    )

