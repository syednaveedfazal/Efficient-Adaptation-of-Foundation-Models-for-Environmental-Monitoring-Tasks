"""
tests/test_spaco.py — Unit tests for SpaCo-LoRA (src/models/spaco_lora.py).

1. gamma = 0 and B = 0: output equals frozen Prithvi (max abs diff < 1e-5).
2. gamma = 0: a trained LoRA r=8 checkpoint loaded into SpaCo-LoRA gives the plain LoRA output (and the same IoU).
3. P = U U^T is a projector and the spatial term is invariant to sign flips / rotations of U's columns.
4. Shapes with batch > 1, CLS token untouched, AMP forward without NaNs; random and box controls run.
5. trainable_params(SpaCo) - trainable_params(LoRA) == number of gamma scalars.
6. Projecting after A (the implementation) equals the spec's input-space formula A(x + gamma P X~).
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.evaluation.metrics import compute_metrics
from src.models.prithvi_fcn import PrithviFCNSegmentor
from src.models.spaco_lora import SpaCoContext, SpaCoLoRALinear, SpaCoPrithviFCNSegmentor

WEIGHTS_PATH = "models/pretrained/prithvi/Prithvi_EO_V2_300M.pt"
LORA_CKPT = "results/scope_v2/checkpoints/lora_r8_seed42_split_010pct.ckpt"


def _spaco(basis="cov", k=8, mode="static"):
    return SpaCoPrithviFCNSegmentor(WEIGHTS_PATH, spaco=dict(k=k, basis=basis, mode=mode))


def _trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_spaco_1_identity_with_frozen_prithvi():
    """gamma = 0, B = 0 at init: SpaCo-LoRA equals the frozen encoder with the same head."""
    torch.manual_seed(0)
    frozen = PrithviFCNSegmentor(WEIGHTS_PATH, adaptation="linear_probe").eval()
    spaco = _spaco().eval()
    spaco.decoder.load_state_dict(frozen.decoder.state_dict())
    x = torch.randn(2, 6, 512, 512)
    with torch.no_grad():
        diff = (frozen(x) - spaco(x)).abs().max().item()
    assert diff < 1e-5, diff


@pytest.mark.skipif(not Path(LORA_CKPT).exists(), reason="trained LoRA r=8 checkpoint not available")
def test_spaco_2_loads_trained_lora():
    """gamma = 0: a trained LoRA r=8 checkpoint gives the plain LoRA output and IoU."""
    sd = torch.load(LORA_CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    lora = PrithviFCNSegmentor(WEIGHTS_PATH, adaptation="lora").eval()
    lora.load_state_dict(sd)
    spaco = _spaco().eval()
    missing, unexpected = spaco.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.endswith("gamma") or k.startswith("spaco_gamma") for k in missing), missing
    torch.manual_seed(1)
    x = torch.randn(2, 6, 512, 512)
    y = (torch.rand(2, 512, 512) > 0.8).long()
    with torch.no_grad():
        a, b = lora(x), spaco(x)
    assert (a - b).abs().max().item() < 1e-4
    assert compute_metrics(a, y, 2)["burn_iou"] == compute_metrics(b, y, 2)["burn_iou"]


def test_spaco_3_projector_and_sign_invariance():
    """P U-projector: P P = P; the spatial term ignores sign flips and rotations within span(U)."""
    torch.manual_seed(0)
    ctx = SpaCoContext(basis="cov", k=8)
    x = torch.randn(2, 1025, 64)
    ctx.set_basis(x)
    U = ctx.U
    P = U @ U.transpose(1, 2)
    assert torch.allclose(P @ P, P, atol=1e-5)
    ref = ctx.spatial_term(x)
    signs = torch.where(torch.rand(1, 1, 8) > 0.5, 1.0, -1.0)
    ctx.U = U * signs
    assert torch.allclose(ctx.spatial_term(x), ref, atol=1e-5)
    R = torch.linalg.qr(torch.randn(8, 8)).Q
    ctx.U = U @ R
    assert torch.allclose(ctx.spatial_term(x), ref, atol=1e-5)


@pytest.mark.parametrize("basis,mode", [("cov", "static"), ("cov", "dynamic"), ("random", "static"), ("box", "static")])
def test_spaco_4_shapes_cls_amp(basis, mode):
    """Batch > 1, CLS untouched by the spatial term, finite AMP forward for every basis / mode."""
    torch.manual_seed(0)
    model = _spaco(basis=basis, mode=mode).eval()
    with torch.no_grad():
        for g in model.spaco_gamma:
            g.fill_(0.5)
        for name, prm in model.named_parameters():
            if "lora_B" in name:
                prm.normal_(0, 0.02)
    x = torch.randn(3, 6, 512, 512)
    tokens = torch.randn(3, 1025, 16)
    model.ctx.set_basis(tokens)
    st = model.ctx.spatial_term(tokens)
    assert st.shape == tokens.shape and st[:, 0].abs().max() == 0
    model.ctx.clear()
    device, dtype = ("cuda", torch.float16) if torch.cuda.is_available() else ("cpu", torch.bfloat16)
    model = model.to(device)
    with torch.no_grad(), torch.autocast(device, dtype=dtype):
        out = model(x.to(device))
    assert out.shape == (3, 2, 512, 512)
    assert torch.isfinite(out.float()).all()
    assert model.ctx.U is None, "context must be cleared after forward"


def test_spaco_5_param_count():
    """SpaCo-LoRA adds exactly one gamma scalar per block over LoRA r=8."""
    lora = PrithviFCNSegmentor(WEIGHTS_PATH, adaptation="lora")
    spaco = _spaco()
    assert _trainable(spaco) - _trainable(lora) == len(spaco.spaco_gamma) == 24


def test_spaco_6_rank_space_projection_exact():
    """A(x + gamma P X~) == A x + gamma P (A X)~ (what SpaCoLoRALinear computes)."""
    torch.manual_seed(0)
    ctx = SpaCoContext(basis="cov", k=8)
    base = nn.Linear(64, 32)
    layer = SpaCoLoRALinear(base, r=8, alpha=8, dropout=0.0, gamma=nn.Parameter(torch.tensor(0.7)), ctx=ctx)
    nn.init.normal_(layer.lora_B["default"].weight, std=0.1)
    x = torch.randn(2, 1025, 64)
    ctx.set_basis(x)
    got = layer(x)
    A, B = layer.lora_A["default"], layer.lora_B["default"]
    ref = base(x) + layer.scaling * B(A(x + 0.7 * ctx.spatial_term(x)))
    assert torch.allclose(got, ref, atol=1e-5)
    got.sum().backward()
    assert layer.gamma.grad is not None and layer.gamma.grad.abs() > 0
