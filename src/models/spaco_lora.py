"""
src/models/spaco_lora.py — SpaCo-LoRA: Spatial-Covariance LoRA for Prithvi-EO-2.0 (FCN head).

Each adapted linear layer computes

    y = W x_i + (alpha / r) * B A dropout( x_i + gamma_l * (P X~)_i ),    P = U U^T

where X~ is the layer input centred over patch tokens (per image) and U (N x k) spans the top-k left singular
vectors of the patch-token matrix (= top-k eigenvectors of the spatial covariance X~ X~^T). U is computed without
gradient, once per image (static mode, from block 0's norm1 output) or per block (dynamic mode). gamma_l is one
scalar per block, shared by the block's adapted layers and initialised to 0, so the model starts as plain LoRA.

Controls (spaco.basis): cov | random (orthonormal U from QR of a Gaussian, fresh per image) |
box (3x3 average pool on the 32x32 token grid instead of P X~) | none (gamma forced to 0, i.e. LoRA).

Parameter names mirror peft (base_layer, lora_A.default, lora_B.default) so plain LoRA checkpoints load directly.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.prithvi_fcn import PrithviFCNSegmentor

TARGET_MODULES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


class SpaCoContext:
    """Per-forward spatial context shared by all SpaCo-LoRA layers of one model (not a module, holds no weights)."""

    def __init__(self, basis: str = "cov", k: int = 8, grid: int = 32, oversample: int = 8, niter: int = 2):
        self.basis = basis
        self.k = k
        self.grid = grid
        self.n_patches = grid * grid
        self.oversample = oversample
        self.niter = niter
        self.U: Optional[torch.Tensor] = None              # (B, N, k) float32
        self.explained_var: Optional[torch.Tensor] = None  # (B,) share of ||X~||_F^2 in the top-k modes

    def clear(self):
        self.U = None

    def patch_slice(self, T: int) -> int:
        """Index of the first patch token: CLS (if present) is token 0."""
        if T == self.n_patches + 1:
            return 1
        if T == self.n_patches:
            return 0
        raise ValueError(f"SpaCo expects {self.n_patches} patch tokens (+ optional CLS), got {T}")

    @torch.no_grad()
    def set_basis(self, x: torch.Tensor):
        """Computes U from a (B, T, D) token tensor (usually a block's norm1 output)."""
        if self.basis not in ("cov", "random"):
            return
        with torch.autocast(device_type=x.device.type, enabled=False):  # fp16 SVD gave NaNs in SCOPE
            self._set_basis(x)

    def _set_basis(self, x: torch.Tensor):
        s = self.patch_slice(x.shape[1])
        xp = x[:, s:].float()
        B, N, _ = xp.shape
        if self.basis == "random":
            g = torch.randn(B, N, self.k, device=x.device, dtype=torch.float32)
            self.U = torch.linalg.qr(g).Q
            self.explained_var = None
            return
        xc = xp - xp.mean(dim=1, keepdim=True)
        U, S, _ = torch.svd_lowrank(xc, q=self.k + self.oversample, niter=self.niter)
        self.U = U[:, :, : self.k].contiguous()
        self.explained_var = (S[:, : self.k] ** 2).sum(-1) / xc.pow(2).sum(dim=(1, 2)).clamp_min(1e-12)

    def spatial_term(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """(P X~)_i for patch tokens, 0 for CLS. Computed in float32, returned in x.dtype. None if no context."""
        if self.basis == "none" or (self.basis in ("cov", "random") and self.U is None):
            return None
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._spatial_term(x)

    def _spatial_term(self, x: torch.Tensor) -> torch.Tensor:
        s = self.patch_slice(x.shape[1])
        xp = x[:, s:].float()
        xc = xp - xp.mean(dim=1, keepdim=True)
        if self.basis == "box":
            B, N, D = xc.shape
            g = xc.transpose(1, 2).reshape(B, D, self.grid, self.grid)
            g = F.avg_pool2d(g, kernel_size=3, stride=1, padding=1, count_include_pad=False)
            px = g.reshape(B, D, N).transpose(1, 2)
        else:
            U = self.U
            px = torch.bmm(U, torch.bmm(U.transpose(1, 2), xc))  # never forms the N x N projector
        px = px.to(x.dtype)
        if s:
            px = torch.cat([torch.zeros_like(x[:, :s]), px], dim=1)
        return px


class SpaCoLoRALinear(nn.Module):
    """LoRA linear whose adapter input is x + gamma * P X~ (see module docstring)."""

    def __init__(self, base_layer: nn.Linear, r: int, alpha: float, dropout: float,
                 gamma: nn.Parameter, ctx: SpaCoContext):
        super().__init__()
        self.base_layer = base_layer
        self.lora_A = nn.ModuleDict({"default": nn.Linear(base_layer.in_features, r, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(r, base_layer.out_features, bias=False)})
        self.lora_dropout = nn.ModuleDict({"default": nn.Dropout(dropout) if dropout > 0 else nn.Identity()})
        self.scaling = alpha / r
        self.gamma = gamma  # shared with the other adapted layers of the same block
        self.ctx = ctx
        # Same initialisation as peft LoRA: A Kaiming-uniform, B zero
        nn.init.kaiming_uniform_(self.lora_A["default"].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B["default"].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base_layer(x)
        A, B = self.lora_A["default"], self.lora_B["default"]
        dropout = self.lora_dropout["default"]
        if isinstance(dropout, nn.Identity):
            # P (token axis) and A (channel axis) commute and centring is linear, so
            # A(x + gamma P X~) = A x + gamma P (A X)~ exactly; projecting in the r-dim space is ~D/r cheaper.
            a = A(x)
            pa = self.ctx.spatial_term(a)
            if pa is not None:
                a = a + self.gamma.to(a.dtype) * pa
        else:
            px = self.ctx.spatial_term(x)
            a = A(dropout(x if px is None else x + self.gamma.to(x.dtype) * px))
        return out + self.scaling * B(a)


class SpaCoPrithviFCNSegmentor(PrithviFCNSegmentor):
    """Prithvi-EO-2.0-300M + SpaCo-LoRA adapters (same targets, rank, alpha, dropout as LoRA r=8) + FCN head."""

    def __init__(
        self,
        weights_path: str,
        num_classes: int = 2,
        adaptation: str = "spaco_lora",
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        spaco: Optional[Dict] = None,
    ):
        # Frozen pretrained encoder + head; adapters are added below
        super().__init__(weights_path=weights_path, num_classes=num_classes, adaptation="linear_probe")
        cfg = dict(k=8, basis="cov", mode="static", gamma_scope="block", gamma_lr_mult=10.0)
        cfg.update(spaco or {})
        if cfg["basis"] not in ("cov", "random", "box", "none"):
            raise ValueError(f"Unknown spaco.basis '{cfg['basis']}'")
        if cfg["mode"] not in ("static", "dynamic"):
            raise ValueError(f"Unknown spaco.mode '{cfg['mode']}'")
        if cfg["gamma_scope"] not in ("block", "layer"):
            raise ValueError(f"Unknown spaco.gamma_scope '{cfg['gamma_scope']}'")
        self.spaco_cfg = cfg
        self.ctx = SpaCoContext(basis=cfg["basis"], k=int(cfg["k"]))

        blocks = self.encoder.blocks
        n_per_block = 1 if cfg["gamma_scope"] == "block" else len(TARGET_MODULES)
        self.spaco_gamma = nn.ParameterList(
            [nn.Parameter(torch.zeros(())) for _ in range(len(blocks) * n_per_block)])
        for b_idx, block in enumerate(blocks):
            for t_idx, path in enumerate(TARGET_MODULES):
                parent_name, child_name = path.split(".")
                parent = getattr(block, parent_name)
                g = self.spaco_gamma[b_idx * n_per_block + (0 if n_per_block == 1 else t_idx)]
                setattr(parent, child_name, SpaCoLoRALinear(
                    getattr(parent, child_name), lora_rank, lora_alpha, lora_dropout, g, self.ctx))
        gamma_ids = {id(g) for g in self.spaco_gamma}
        for name, prm in self.named_parameters():
            prm.requires_grad = ("lora_" in name) or name.startswith("decoder.") or id(prm) in gamma_ids
        if cfg["basis"] == "none":
            for g in self.spaco_gamma:
                g.requires_grad = False

        # U is set before the blocks run: from block 0's norm1 output (static) or from every block's (dynamic)
        hook_blocks = blocks if cfg["mode"] == "dynamic" else blocks[:1]
        for block in hook_blocks:
            block.register_forward_pre_hook(self._set_basis_hook)
        self._print_param_summary()

    def _set_basis_hook(self, block, args):
        self.ctx.set_basis(block.norm1(args[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return super().forward(x)
        finally:
            self.ctx.clear()

    def gammas(self) -> List[float]:
        return [float(g.detach()) for g in self.spaco_gamma]

    def spaco_param_groups(self, lr: float, weight_decay: float) -> List[dict]:
        """LoRA + head at lr; the gamma scalars at gamma_lr_mult x lr with no weight decay."""
        gamma_ids = {id(g) for g in self.spaco_gamma}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in gamma_ids]
        gam = [g for g in self.spaco_gamma if g.requires_grad]
        groups = [{"params": rest, "lr": lr, "weight_decay": weight_decay, "name": "lora_head"}]
        if gam:
            groups.append({"params": gam, "lr": lr * float(self.spaco_cfg["gamma_lr_mult"]),
                           "weight_decay": 0.0, "name": "spaco_gamma"})
        return groups
