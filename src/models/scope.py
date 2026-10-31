"""
src/models/scope.py — Spectral COvariance and Probe-guided Efficient adaptation (SCOPE).

Components:
1. patch_stats(): Training-free extraction of patch-level band mean, covariance,
   NBR stats, SWIR2 std, log-determinant, and normalized prior features.
2. CovarianceTokenSharing (Stage 1): Input token reduction via pairwise heterogeneity
   scoring (Euclidean distance on means + Log-Euclidean distance on covariances + NBR variance).
3. ScopeGateBlock (Stage 2): DyT-style MLP dispatch with learnable / frozen spectral priors
   and Gumbel-Sigmoid routing with target rate loss.
4. LinearProbe & EarlyExit (Stage 3): Intermediate concept probes after blocks 8 & 16 with
   confident token exit and context retention (DToP).
5. Extras: NBR residual refinement (nbr_refine) and Self-Distillation (self_distill).
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model

from src.models.prithvi_fcn import _load_prithvi_mae_module, PrithviFCNDecoder


# ──────────────────────────────────────────────────────────────────────────────
# 2.1 Patch statistics (no network, no gradients)
# ──────────────────────────────────────────────────────────────────────────────

def patch_stats(
    x_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    patch: int = 16,
    prior_stats_dict: Optional[dict] = None,
) -> Dict[str, torch.Tensor]:
    """
    Computes patch-level spectral and covariance statistics.

    Args:
        x_norm: (B, 6, H, W) normalized satellite imagery.
        mean: (6,) tensor or array of per-band means.
        std: (6,) tensor or array of per-band stds.
        patch: patch spatial size (default 16 for 512x512 -> 32x32 grid).
        prior_stats_dict: dict with 'mean' and 'std' for z-scoring the 4 prior features.

    Returns:
        dict containing:
            'mu': (B, N, 6)
            'cov': (B, N, 6, 6)
            'nbr_mean': (B, N)
            'nbr_std': (B, N)
            'swir2_std': (B, N)
            'logdet': (B, N)
            'p': (B, N, 4) z-scored prior features
            'nbr': (B, 1, H, W) full-res NBR map
    """
    B, C, H, W = x_norm.shape
    device = x_norm.device
    dtype = x_norm.dtype

    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=device, dtype=dtype)
    else:
        mean = mean.to(device=device, dtype=dtype)

    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=device, dtype=dtype)
    else:
        std = std.to(device=device, dtype=dtype)

    # 1. Undo normalization to recover reflectance-like values
    x = x_norm * std.view(1, C, 1, 1) + mean.view(1, C, 1, 1)

    # 2. Reshape into patches: (B, N, P*P, C)
    Hp, Wp = H // patch, W // patch
    N = Hp * Wp
    # (B, C, Hp, patch, Wp, patch) -> (B, Hp, Wp, patch, patch, C) -> (B, N, 256, C)
    x_p = (
        x.view(B, C, Hp, patch, Wp, patch)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .view(B, N, patch * patch, C)
    )

    # 3. Patch band mean and covariance
    mu = x_p.mean(dim=2)  # (B, N, 6)
    diff = x_p - mu.unsqueeze(2)  # (B, N, 256, 6)
    cov = torch.matmul(diff.transpose(-1, -2), diff) / float(patch * patch - 1)  # (B, N, 6, 6)

    # 4. NBR per pixel: (NIR - SWIR2) / (NIR + SWIR2 + 1e-6)
    # NIR is band 3, SWIR2 is band 5
    nir = x[:, 3:4, :, :]
    swir2 = x[:, 5:6, :, :]
    nbr = (nir - swir2) / (nir + swir2 + 1e-6)

    nbr_p = (
        nbr.view(B, 1, Hp, patch, Wp, patch)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .view(B, N, patch * patch)
    )
    swir2_p = (
        swir2.view(B, 1, Hp, patch, Wp, patch)
        .permute(0, 2, 4, 3, 5, 1)
        .contiguous()
        .view(B, N, patch * patch)
    )

    nbr_mean = nbr_p.mean(dim=-1)  # (B, N)
    nbr_std = nbr_p.std(dim=-1, unbiased=False)  # (B, N)
    swir2_std = swir2_p.std(dim=-1, unbiased=False)  # (B, N)

    # 5. Log-determinant of regularized covariance
    cov_reg = cov + 1e-4 * torch.eye(C, device=device, dtype=dtype).view(1, 1, C, C)
    # Use slogdet for numerical stability
    _, logdet = torch.linalg.slogdet(cov_reg)  # (B, N)

    # 6. Raw prior features: [-nbr_mean, nbr_std, swir2_std, logdet]
    raw_p = torch.stack([-nbr_mean, nbr_std, swir2_std, logdet], dim=-1)  # (B, N, 4)

    # 7. Z-score prior features
    if prior_stats_dict is not None and "mean" in prior_stats_dict and "std" in prior_stats_dict:
        p_mean = torch.tensor(prior_stats_dict["mean"], device=device, dtype=dtype).view(1, 1, 4)
        p_std = torch.tensor(prior_stats_dict["std"], device=device, dtype=dtype).view(1, 1, 4)
        p_std = torch.where(p_std < 1e-6, torch.ones_like(p_std), p_std)
        p = (raw_p - p_mean) / p_std
    else:
        p = (raw_p - raw_p.mean(dim=(0, 1), keepdim=True)) / (raw_p.std(dim=(0, 1), keepdim=True) + 1e-6)

    return {
        "mu": mu,
        "cov": cov,
        "nbr_mean": nbr_mean,
        "nbr_std": nbr_std,
        "swir2_std": swir2_std,
        "logdet": logdet,
        "p": p,
        "nbr": nbr,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2.2 Stage 1 — Covariance Token Sharing
# ──────────────────────────────────────────────────────────────────────────────

class CovarianceTokenSharing(nn.Module):
    """
    Groups the 32x32 patch grid into 16x16 = 256 non-overlapping 2x2 groups,
    computes pairwise heterogeneity scores, and shares the lowest-heterogeneity groups.
    """

    def __init__(
        self,
        share_frac: float = 0.5,
        lam: float = 1.0,
        nbr_std_threshold: Optional[float] = None,
        s_mu: Optional[float] = None,
    ):
        super().__init__()
        self.share_frac = share_frac
        self.lam = lam
        self.nbr_std_threshold = nbr_std_threshold
        self.s_mu = s_mu if s_mu is not None and s_mu > 1e-6 else 0.0734

        # Precompute the 2x2 group patch indices for a 32x32 grid
        # 16 groups vertically x 16 groups horizontally = 256 groups
        gr = torch.arange(16).view(16, 1).repeat(1, 16).view(-1)
        gc = torch.arange(16).view(1, 16).repeat(16, 1).view(-1)
        p0 = (2 * gr) * 32 + (2 * gc)
        p1 = (2 * gr) * 32 + (2 * gc + 1)
        p2 = (2 * gr + 1) * 32 + (2 * gc)
        p3 = (2 * gr + 1) * 32 + (2 * gc + 1)
        # shape: (256, 4)
        self.register_buffer("group_patches", torch.stack([p0, p1, p2, p3], dim=1), persistent=False)

    def compute_heterogeneity(
        self,
        mu: torch.Tensor,
        cov: torch.Tensor,
        nbr_mean: torch.Tensor,
        nbr_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes heterogeneity score h for each of the 256 groups.
        h = max_pairs ||mu_i - mu_j||_2 / s_mu + lam * max_pairs d_LE(cov_i, cov_j) + nbr_var_across_group
        """
        B = mu.shape[0]
        device = mu.device
        dtype = mu.dtype
        gp = self.group_patches  # (256, 4)

        # 1. mu for groups: (B, 256, 4, 6)
        mu_groups = mu[:, gp]
        # max pairwise Euclidean distance among 4 patches in group
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        d_mu_list = [torch.norm(mu_groups[:, :, i] - mu_groups[:, :, j], dim=-1) for i, j in pairs]
        max_d_mu = torch.stack(d_mu_list, dim=-1).max(dim=-1)[0]  # (B, 256)

        # 2. Log-Euclidean distance on covariances: d_LE(A, B) = ||logm(A + eps*I) - logm(B + eps*I)||_F
        cov_groups = cov[:, gp]  # (B, 256, 4, 6, 6)
        eps = 1e-4
        cov_reg = cov_groups + eps * torch.eye(6, device=device, dtype=dtype).view(1, 1, 1, 6, 6)
        evals, evecs = torch.linalg.eigh(cov_reg)
        evals = torch.clamp(evals, min=1e-6)
        logm = torch.matmul(evecs * torch.log(evals).unsqueeze(-2), evecs.transpose(-1, -2))  # (B, 256, 4, 6, 6)

        d_le_list = [torch.norm(logm[:, :, i] - logm[:, :, j], dim=(-2, -1)) for i, j in pairs]
        max_d_le = torch.stack(d_le_list, dim=-1).max(dim=-1)[0]  # (B, 256)

        # 3. NBR variance across the 4 patches in the group
        nbr_mean_groups = nbr_mean[:, gp]  # (B, 256, 4)
        nbr_var = torch.var(nbr_mean_groups, dim=-1, unbiased=False)  # (B, 256)

        # Total heterogeneity
        h = (max_d_mu / self.s_mu) + self.lam * max_d_le + nbr_var  # (B, 256)

        # 4. Protection threshold: never share group if any patch nbr_std > threshold
        if self.nbr_std_threshold is not None:
            nbr_std_groups = nbr_std[:, gp]  # (B, 256, 4)
            max_nbr_std = nbr_std_groups.max(dim=-1)[0]  # (B, 256)
            protect_mask = max_nbr_std > self.nbr_std_threshold
            h = torch.where(protect_mask, torch.tensor(1e9, device=device, dtype=dtype), h)

        return h

    def share(
        self,
        x_tokens: torch.Tensor,
        p_tokens: torch.Tensor,
        stats: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Reduces tokens from 1024 to (1024 - 3*g).
        """
        B, N, D = x_tokens.shape
        g = int(round(self.share_frac * 256))

        if g <= 0:
            return x_tokens, p_tokens, {"is_shared": False}

        h = self.compute_heterogeneity(stats["mu"], stats["cov"], stats["nbr_mean"], stats["nbr_std"])  # (B, 256)

        # Select the g groups with the lowest heterogeneity for each sample in batch
        _, lowest_idx = torch.topk(-h, g, dim=1, largest=True)  # (B, g)

        gp = self.group_patches  # (256, 4)
        reduced_x_list = []
        reduced_p_list = []

        all_groups_set = set(range(256))
        unshared_groups_batch = []
        shared_groups_batch = []

        for b in range(B):
            sh_g = lowest_idx[b].tolist()
            unsh_g = sorted(list(all_groups_set - set(sh_g)))
            shared_groups_batch.append(sh_g)
            unshared_groups_batch.append(unsh_g)

            # Shared tokens: mean of the 4 patches per shared group
            sh_patches = gp[sh_g]  # (g, 4)
            tokens_sh = x_tokens[b, sh_patches].mean(dim=1)  # (g, D)
            p_sh = p_tokens[b, sh_patches].mean(dim=1)  # (g, 4)

            # Unshared tokens: all 4 patches remain separate
            unsh_patches = gp[unsh_g].view(-1)  # ((256-g)*4,)
            tokens_unsh = x_tokens[b, unsh_patches]  # ((256-g)*4, D)
            p_unsh = p_tokens[b, unsh_patches]  # ((256-g)*4, 4)

            # Concatenate shared tokens first, then unshared tokens
            reduced_x_b = torch.cat([tokens_sh, tokens_unsh], dim=0)  # (1024 - 3*g, D)
            reduced_p_b = torch.cat([p_sh, p_unsh], dim=0)  # (1024 - 3*g, 4)

            reduced_x_list.append(reduced_x_b)
            reduced_p_list.append(reduced_p_b)

        reduced_x = torch.stack(reduced_x_list, dim=0)
        reduced_p = torch.stack(reduced_p_list, dim=0)

        share_info = {
            "is_shared": True,
            "g": g,
            "shared_groups": shared_groups_batch,
            "unshared_groups": unshared_groups_batch,
        }
        return reduced_x, reduced_p, share_info

    def unshare(self, reduced_tokens: torch.Tensor, share_info: dict) -> torch.Tensor:
        """
        Reconstructs the full 32x32 = 1024 grid from reduced tokens.
        """
        if not share_info.get("is_shared", False):
            return reduced_tokens

        B, _, D = reduced_tokens.shape
        g = share_info["g"]
        gp = self.group_patches.cpu()

        full_tokens = torch.zeros(B, 1024, D, device=reduced_tokens.device, dtype=reduced_tokens.dtype)

        for b in range(B):
            sh_g = share_info["shared_groups"][b]
            unsh_g = share_info["unshared_groups"][b]

            # 1. Unpack shared tokens: single token copied to all 4 positions
            tokens_sh = reduced_tokens[b, :g]  # (g, D)
            for i, group_id in enumerate(sh_g):
                p_indices = gp[group_id]
                full_tokens[b, p_indices] = tokens_sh[i]

            # 2. Unpack unshared tokens: tokens copied to exact patch positions
            tokens_unsh = reduced_tokens[b, g:]  # ((256-g)*4, D)
            offset = 0
            for group_id in unsh_g:
                p_indices = gp[group_id]
                full_tokens[b, p_indices] = tokens_unsh[offset : offset + 4]
                offset += 4

        return full_tokens


# ──────────────────────────────────────────────────────────────────────────────
# 2.3 Stage 2 — Spectral-Prior Gate on the MLP
# ──────────────────────────────────────────────────────────────────────────────

def gumbel_sigmoid(logits: torch.Tensor, tau: float = 5.0, hard: bool = True) -> torch.Tensor:
    """
    Gumbel-Sigmoid with straight-through estimator for hard binary dispatch.
    Computes noise in float32 for complete numerical stability in AMP (float16).
    """
    dtype = logits.dtype
    logits_f32 = logits.float()
    u1 = torch.rand_like(logits_f32).clamp(min=1e-7, max=1.0 - 1e-7)
    u2 = torch.rand_like(logits_f32).clamp(min=1e-7, max=1.0 - 1e-7)
    g1 = -torch.log(-torch.log(u1))
    g2 = -torch.log(-torch.log(u2))

    y = torch.sigmoid((logits_f32 + g1 - g2) / tau).to(dtype)
    if hard:
        y_hard = (y > 0.5).to(dtype)
        y = y_hard - y.detach() + y
    return y


def topk_straight_through(logits: torch.Tensor, k: int, noise: bool = True) -> torch.Tensor:
    """
    Hard top-k mask with a straight-through sigmoid gradient (gate_train_mode: topk_st).

    Selects exactly k tokens per sample by top-k on logits + Gumbel noise (float32).
    Forward value is the exact 0/1 mask; the backward pass uses sigmoid(logits).

    Args:
        logits: (B, Np, 1) gate logits.
        k: tokens to keep per sample.
    """
    dtype = logits.dtype
    logits_f32 = logits.float()
    scores = logits_f32.detach().squeeze(-1)
    if noise:
        u = torch.rand_like(scores).clamp(min=1e-7, max=1.0 - 1e-7)
        scores = scores - torch.log(-torch.log(u))
    idx = torch.topk(scores, k, dim=-1, largest=True)[1]
    hard = torch.zeros_like(scores).scatter_(1, idx, 1.0).unsqueeze(-1)
    soft = torch.sigmoid(logits_f32)
    # hard + (soft - soft) keeps the forward value exactly 0/1
    return (hard + (soft - soft.detach())).to(dtype)


class ScopeGateBlock(nn.Module):
    """
    Wraps a transformer block with a DyT-style MLP routing gate.
    All tokens pass through attention; only routed tokens pass through the MLP.
    """

    def __init__(
        self,
        block: nn.Module,
        embed_dim: int = 1024,
        a_scale: float = 1.0,
        prior_frozen: bool = False,
        zero_prior: bool = False,
        is_random: bool = False,
        tau: float = 5.0,
        train_mode: str = "gumbel",
        keep_ratio_train: str = "uniform",
    ):
        super().__init__()
        self.block = block
        self.embed_dim = embed_dim
        self.a_scale = a_scale
        self.is_random = is_random
        self.tau = tau
        # "gumbel" (v1): Gumbel-Sigmoid + rate loss; "topk_st": exact top-k hard mask, ST gradient
        self.train_mode = train_mode
        # "uniform" (v1): keep ratio ~ U(0.3, 1.0) per step; "fixed": the configured keep_ratio
        self.keep_ratio_train = keep_ratio_train
        self.last_mask: Optional[torch.Tensor] = None

        # W_g: Linear(1024 -> 1)
        self.W_g = nn.Linear(embed_dim, 1)

        # w_p: 4 learnable weights, initialized to [1, 1, 0.5, 0.5] or zeros
        if zero_prior:
            init_wp = torch.zeros(4, dtype=torch.float32)
        else:
            init_wp = torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float32)

        if prior_frozen or zero_prior:
            self.register_buffer("w_p", init_wp)
        else:
            self.w_p = nn.Parameter(init_wp)

        # Runtime metrics tracked per forward call
        self.last_attn_tokens: int = 0
        self.last_mlp_tokens: int = 0
        self.last_rate_loss: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        keep_ratio: float = 1.0,
        r_target: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token representations (token 0 may be cls_token).
            p: (B, N_patches, 4) patch prior features.
            keep_ratio: fraction of patch tokens to route through MLP during eval.
            r_target: target rate for regularizer during training.
        """
        B, N, D = x.shape
        self.last_attn_tokens = N

        # 1. Attention path: ALL tokens pass through attention
        attn_out = self.block.drop_path1(self.block.ls1(self.block.attn(self.block.norm1(x))))
        x = x + attn_out

        # If gate is completely off (keep_ratio == 1.0 in eval without gating):
        if not self.training and keep_ratio >= 1.0:
            mlp_out = self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))
            x = x + mlp_out
            self.last_mlp_tokens = N
            self.last_rate_loss = None
            return x

        # 2. Check if x has CLS token prepended
        has_cls = (x.shape[1] == p.shape[1] + 1)
        if has_cls:
            x_patches = x[:, 1:, :]  # (B, N-1, D)
        else:
            x_patches = x  # (B, N, D)
        Np = x_patches.shape[1]

        if self.is_random:
            gate_logits = torch.rand(B, Np, 1, device=x.device, dtype=x.dtype)
        else:
            p_proj = torch.matmul(p, self.w_p).unsqueeze(-1)  # (B, Np, 1)
            gate_logits = self.W_g(x_patches) + self.a_scale * p_proj  # (B, Np, 1)

        # 3. Training dispatch vs. Evaluation top-k routing
        if self.training and self.train_mode == "topk_st":
            if self.keep_ratio_train == "uniform":
                keep_ratio = float(torch.empty(1).uniform_(0.3, 1.0).item())
            k = max(1, min(int(math.ceil(keep_ratio * Np)), Np))
            m_patches = topk_straight_through(gate_logits, k, noise=not self.is_random)  # (B, Np, 1)

            if has_cls:
                m_cls = torch.ones(B, 1, 1, device=x.device, dtype=m_patches.dtype)
                m = torch.cat([m_cls, m_patches], dim=1)  # (B, N, 1)
            else:
                m = m_patches
            self.last_mask = m_patches.detach()

            # MLP on all tokens, masked: unselected tokens still give the gate a ST gradient
            mlp_out = self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))
            x = x + m.to(mlp_out.dtype) * mlp_out
            self.last_mlp_tokens = (1 if has_cls else 0) + k
            self.last_rate_loss = None

        elif self.training:
            m_patches = gumbel_sigmoid(gate_logits, tau=self.tau, hard=True)  # (B, Np, 1)
            self.last_mask = m_patches.detach()

            if has_cls:
                m_cls = torch.ones(B, 1, 1, device=x.device, dtype=x.dtype)
                m = torch.cat([m_cls, m_patches], dim=1)  # (B, N, 1)
            else:
                m = m_patches

            # Compute MLP path for all and mask
            mlp_out = self.block.drop_path2(self.block.ls2(self.block.mlp(self.block.norm2(x))))
            x = x + m * mlp_out

            # Rate loss: (mean(m_patches) - r_target)^2
            if r_target is None:
                r_target = float(torch.empty(1).uniform_(0.3, 1.0).item())
            self.last_rate_loss = (m_patches.mean() - r_target) ** 2
            val = m.sum(dim=1).mean().item()
            self.last_mlp_tokens = int(round(val)) if not math.isnan(val) else N

        else:
            # Inference: gather top-k tokens -> run MLP -> scatter back
            k = int(math.ceil(keep_ratio * Np))
            k = max(1, min(k, Np))

            _, topk_idx = torch.topk(gate_logits.squeeze(-1), k, dim=-1, largest=True)  # (B, k)
            if has_cls:
                topk_idx = topk_idx + 1
                cls_idx = torch.zeros(B, 1, device=x.device, dtype=torch.long)
                active_idx = torch.cat([cls_idx, topk_idx], dim=1)  # (B, 1 + k)
                num_active = 1 + k
            else:
                active_idx = topk_idx
                num_active = k

            self.last_mlp_tokens = num_active

            # Gather active tokens
            gather_idx = active_idx.unsqueeze(-1).expand(-1, -1, D)  # (B, num_active, D)
            x_active = torch.gather(x, 1, gather_idx)

            # Run MLP path only on active tokens
            norm2_active = self.block.norm2(x_active)
            mlp_active = self.block.drop_path2(self.block.ls2(self.block.mlp(norm2_active)))

            # Scatter add back to x
            x = x.scatter_add(1, gather_idx, mlp_active)
            self.last_rate_loss = None

        return x


# ──────────────────────────────────────────────────────────────────────────────
# 2.4 Stage 3 — Concept-Probe Early Exit
# ──────────────────────────────────────────────────────────────────────────────

class ConceptProbe(nn.Module):
    """
    Linear concept probe attached after intermediate blocks (e.g. block 8, 16).
    """

    def __init__(self, in_features: int = 1024, num_classes: int = 2):
        super().__init__()
        self.probe = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.probe(x)


# ──────────────────────────────────────────────────────────────────────────────
# 2.5 Extras: NBR Residual Refinement
# ──────────────────────────────────────────────────────────────────────────────

class NbrRefinement(nn.Module):
    """
    Concatenates full-res NBR map with initial 2-class logits and adds
    Conv3x3(3->16) -> ReLU -> Conv1x1(16->2) residual.
    Zero-initialized so that at step 0 the model equals the baseline.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 2, kernel_size=1)

        # Zero initialize last conv
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, logits: torch.Tensor, nbr: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([logits, nbr], dim=1)
        residual = self.conv2(self.relu(self.conv1(feat)))
        return logits + residual


# ──────────────────────────────────────────────────────────────────────────────
# 2.0 Full SCOPE Prithvi FCN Segmentor
# ──────────────────────────────────────────────────────────────────────────────

class ScopePrithviFCNSegmentor(nn.Module):
    """
    Prithvi-EO-2.0-300M with SCOPE:
      - LoRA r=8 adapters on attention and MLP projections
      - Stage 1: Covariance token sharing
      - Stage 2: Spectral-prior gate on the MLP (DyT dispatch)
      - Stage 3: Concept-probe early exit after blocks 8 & 16
      - Extras: NBR residual refinement & self-distillation
    """

    def __init__(
        self,
        weights_path: str = "models/pretrained/prithvi/Prithvi_EO_V2_300M.pt",
        num_classes: int = 2,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        adaptation: str = "scope",
        stats_path: str = "data/processed/stats.json",
        prior_stats_path: str = "results/scope_smoke/prior_stats.json",
        # Stage 1: Token sharing
        share_frac: float = 0.0,
        sharing_lam: float = 1.0,
        nbr_std_threshold: Optional[float] = None,
        # Stage 2: Gate
        gate_enabled: bool = False,
        gate_is_random: bool = False,
        gate_a_scale: float = 1.0,
        gate_prior_frozen: bool = False,
        gate_zero_prior: bool = False,
        gate_keep_ratio: float = 1.0,
        rate_loss_weight: float = 2.0,
        tau: float = 5.0,
        gate_train_mode: str = "gumbel",
        keep_ratio_train: str = "uniform",
        prior_mode: Optional[str] = None,
        # Stage 3: Early exit
        exit_enabled: bool = False,
        probe_loss_weight: float = 0.2,
        exit_confident_thresh_high: float = 0.95,
        exit_confident_thresh_low: float = 0.05,
        exit_context_k: int = 16,
        exit_overwrite: str = "probe",
        # Extras
        nbr_refine: bool = False,
        self_distill: bool = False,
        # Two-stage training: warm start from a trained plain LoRA r=8 Lightning checkpoint
        init_from: Optional[str] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.adaptation = adaptation
        self.share_frac = share_frac
        self.sharing_lam = sharing_lam
        self.nbr_std_threshold = nbr_std_threshold

        self.gate_enabled = gate_enabled
        self.gate_is_random = gate_is_random
        self.gate_keep_ratio = gate_keep_ratio
        self.rate_loss_weight = rate_loss_weight
        self.tau = tau

        self.exit_enabled = exit_enabled
        self.probe_loss_weight = probe_loss_weight
        self.exit_thresh_high = exit_confident_thresh_high
        self.exit_thresh_low = exit_confident_thresh_low
        # Optional per-probe symmetric thresholds {7: t8, 15: t16}, calibrated on validation
        self.exit_thresh_per_probe: Optional[Dict[int, float]] = None
        self.exit_context_k = exit_context_k
        # "probe" (v1): exited patches take the probe logits; "decoder": the decoder predicts them
        self.exit_overwrite = exit_overwrite
        # Diagnostics: when set, probe logits for all tokens are stored in last_probe_logits
        self.record_probes = False
        self.last_probe_logits: Dict[int, torch.Tensor] = {}
        self.last_spatial_features: Optional[torch.Tensor] = None
        self.last_share_info: dict = {"is_shared": False}
        self.last_exited_probe_logits: Dict[int, torch.Tensor] = {}

        self.nbr_refine = nbr_refine
        self.self_distill = self_distill

        # 1. Load dataset statistics
        self._load_statistics(stats_path, prior_stats_path)

        # 2. Load Prithvi backbone and encoder
        prithvi_mod = _load_prithvi_mae_module()
        full_model = prithvi_mod.PrithviMAE(
            img_size=512,
            num_frames=1,
            patch_size=(1, 16, 16),
            in_chans=6,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4.0,
        )

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        state_dict = {k: v for k, v in state_dict.items() if "pos_embed" not in k}
        full_model.load_state_dict(state_dict, strict=False)
        self.encoder = full_model.encoder

        # 3. Inject LoRA adapters (identical to standard baseline)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            bias="none",
            lora_dropout=0.0,
        )
        inject_adapter_in_model(lora_cfg, self.encoder)

        # Freeze backbone parameters, unfreeze LoRA
        for name, param in self.encoder.named_parameters():
            param.requires_grad = "lora_" in name

        # prior_mode overrides the v1 flags: off -> no prior term, frozen -> fixed w_p, learned -> trainable w_p
        if prior_mode is not None:
            if prior_mode not in ("off", "learned", "frozen"):
                raise ValueError(f"Unknown prior_mode '{prior_mode}'")
            gate_zero_prior = prior_mode == "off"
            gate_prior_frozen = prior_mode != "learned"
        self.prior_mode = prior_mode

        # 4. Wrap transformer blocks with Stage 2 Gate Blocks
        self.blocks = nn.ModuleList()
        for b in self.encoder.blocks:
            if self.gate_enabled or self.gate_is_random:
                g_block = ScopeGateBlock(
                    b,
                    embed_dim=1024,
                    a_scale=gate_a_scale,
                    prior_frozen=gate_prior_frozen,
                    zero_prior=gate_zero_prior,
                    is_random=gate_is_random,
                    tau=tau,
                    train_mode=gate_train_mode,
                    keep_ratio_train=keep_ratio_train,
                )
                self.blocks.append(g_block)
            else:
                self.blocks.append(b)

        # 5. Stage 1: Covariance Token Sharing module
        self.sharing_module = CovarianceTokenSharing(
            share_frac=share_frac,
            lam=sharing_lam,
            nbr_std_threshold=nbr_std_threshold,
            s_mu=self.prior_stats.get("s_mu", 0.0734),
        )

        # 6. Stage 3: Probes after block 8 and block 16 (0-indexed 7 and 15)
        if self.exit_enabled:
            self.probe8 = ConceptProbe(1024, num_classes=2)
            self.probe16 = ConceptProbe(1024, num_classes=2)
        else:
            self.probe8 = None
            self.probe16 = None

        # 7. Extras: NBR refinement residual
        if self.nbr_refine:
            self.refine_net = NbrRefinement()
        else:
            self.refine_net = None

        # 8. FCN Decoder
        self.decoder = PrithviFCNDecoder(in_channels=1024, num_classes=num_classes)

        if init_from:
            self.load_warm_start(init_from)

        # Runtime trackers
        self.last_aux_loss = torch.tensor(0.0)
        self.last_tokens_attn: List[int] = []
        self.last_tokens_mlp: List[int] = []
        self.last_exit_rate: float = 0.0
        self._current_targets: Optional[torch.Tensor] = None

    def load_warm_start(self, ckpt_path: str):
        """Loads encoder, LoRA and FCN head weights from a plain-LoRA Lightning checkpoint."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if k.startswith(("encoder.", "decoder."))}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing_base = [k for k in missing if k.startswith(("encoder.", "decoder.")) and "pos_embed" not in k]
        if missing_base or unexpected:
            raise RuntimeError(f"Warm start mismatch: missing {missing_base[:5]}, unexpected {unexpected[:5]}")
        print(f"[SCOPE] Warm start loaded {len(sd)} tensors from {ckpt_path}")

    def param_groups(self, lr_new: float, lr_lora: float) -> List[dict]:
        """New SCOPE parts (gates, probes, nbr_refine) at lr_new; LoRA and FCN head at lr_lora."""
        new, base, seen = [], [], set()
        for name, prm in self.named_parameters():
            if not prm.requires_grad or id(prm) in seen:
                continue
            seen.add(id(prm))
            (base if name.startswith(("encoder.", "decoder.")) or ".block." in name else new).append(prm)
        return [{"params": new, "lr": lr_new, "name": "new"}, {"params": base, "lr": lr_lora, "name": "lora_head"}]

    def _load_statistics(self, stats_path: str, prior_stats_path: str):
        sp = Path(stats_path)
        if sp.exists():
            with open(sp) as f:
                s = json.load(f)
            self.register_buffer("band_mean", torch.tensor(s["mean"], dtype=torch.float32), persistent=False)
            self.register_buffer("band_std", torch.tensor(s["std"], dtype=torch.float32), persistent=False)
        else:
            self.register_buffer("band_mean", torch.zeros(6), persistent=False)
            self.register_buffer("band_std", torch.ones(6), persistent=False)

        psp = Path(prior_stats_path)
        if psp.exists():
            with open(psp) as f:
                self.prior_stats = json.load(f)
        else:
            self.prior_stats = {
                "mean": [-0.1319, 0.0772, 0.0276, -50.3879],
                "std": [0.2586, 0.0592, 0.0179, 2.0786],
                "s_mu": 0.0734,
            }

    def set_inference_params(
        self,
        keep_ratio: Optional[float] = None,
        share_frac: Optional[float] = None,
        exit_enabled: Optional[bool] = None,
        exit_overwrite: Optional[str] = None,
        exit_thresh_per_probe: Optional[Dict[int, float]] = None,
    ):
        """Allows dynamically overriding evaluation hyperparameters during inference sweeps."""
        if exit_overwrite is not None:
            self.exit_overwrite = exit_overwrite
        if exit_thresh_per_probe is not None:
            self.exit_thresh_per_probe = exit_thresh_per_probe
        if keep_ratio is not None:
            self.gate_keep_ratio = keep_ratio
        if share_frac is not None:
            self.share_frac = share_frac
            self.sharing_module.share_frac = share_frac
        if exit_enabled is not None:
            self.exit_enabled = exit_enabled
            if exit_enabled and self.probe8 is None:
                self.probe8 = ConceptProbe(1024, num_classes=2).to(next(self.parameters()).device)
                self.probe16 = ConceptProbe(1024, num_classes=2).to(next(self.parameters()).device)

    def set_current_targets(self, targets: torch.Tensor):
        self._current_targets = targets

    def get_token_counts(self) -> Tuple[List[int], List[int], float]:
        """Returns (tokens_attn_per_block, tokens_mlp_per_block, exit_rate) from the last forward pass."""
        return self.last_tokens_attn, self.last_tokens_mlp, self.last_exit_rate

    def get_aux_loss(self, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns the accumulated auxiliary loss (rate loss + probe loss + distillation)."""
        loss = self.last_aux_loss
        self.last_aux_loss = torch.tensor(0.0, device=loss.device if isinstance(loss, torch.Tensor) else "cpu")
        return loss

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for SCOPE segmentor.
        """
        B, C, H, W = x.shape
        device = x.device

        # Self-distillation teacher pass if enabled
        teacher_logits = None
        if self.training and self.self_distill:
            with torch.no_grad():
                saved_share_frac = self.share_frac
                saved_keep_ratio = self.gate_keep_ratio
                saved_exit = self.exit_enabled
                self.share_frac = 0.0
                self.gate_keep_ratio = 1.0
                self.exit_enabled = False
                teacher_logits = self._forward_inner(x, None)
                self.share_frac = saved_share_frac
                self.gate_keep_ratio = saved_keep_ratio
                self.exit_enabled = saved_exit

        student_logits = self._forward_inner(x, targets)

        if teacher_logits is not None:
            student_log_probs = F.log_softmax(student_logits, dim=1)
            teacher_probs = F.softmax(teacher_logits, dim=1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            self.last_aux_loss = self.last_aux_loss + 0.5 * kl_loss

        return student_logits

    def _forward_inner(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # 1. Extract patch statistics (no gradient)
        with torch.no_grad():
            stats = patch_stats(x, self.band_mean, self.band_std, patch=16, prior_stats_dict=self.prior_stats)

        # 2. Patch embedding + positional embedding
        if len(x.shape) == 4 and self.encoder.patch_embed.input_size[0] == 1:
            x_in = x.unsqueeze(2)
        else:
            x_in = x
        sample_shape = x_in.shape[-3:]

        patch_tokens = self.encoder.patch_embed(x_in)  # (B, 1024, D)
        pos_embed = self.encoder.interpolate_pos_encoding(sample_shape)
        patch_tokens = patch_tokens + pos_embed[:, 1:, :]  # (B, 1024, D)

        p_tokens = stats["p"]  # (B, 1024, 4)

        # 3. Stage 1: Covariance Token Sharing
        if self.share_frac > 0.0:
            tokens, p_reduced, share_info = self.sharing_module.share(patch_tokens, p_tokens, stats)
        else:
            tokens = patch_tokens
            p_reduced = p_tokens
            share_info = {"is_shared": False}

        N_reduced = tokens.shape[1]

        # Prepend CLS token
        cls_token = self.encoder.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1 + N_reduced, D)

        attn_token_counts = []
        mlp_token_counts = []
        total_rate_loss = torch.tensor(0.0, device=device)
        total_probe_loss = torch.tensor(0.0, device=device)

        # Track active patch tokens across exits
        active_indices = torch.arange(N_reduced, device=device)  # (N_reduced,)
        exited_features = {}       # orig_idx -> tensor (B, D)
        exited_probe_logits = {}   # orig_idx -> tensor (B, 2)
        total_exited_tokens = 0

        # 4. Pass through 24 Transformer Blocks
        for b_idx, block in enumerate(self.blocks):
            # Forward block
            if isinstance(block, ScopeGateBlock):
                tokens = block(
                    tokens,
                    p=p_reduced,
                    keep_ratio=self.gate_keep_ratio,
                )
                attn_token_counts.append(block.last_attn_tokens)
                mlp_token_counts.append(block.last_mlp_tokens)
                if block.last_rate_loss is not None:
                    total_rate_loss = total_rate_loss + self.rate_loss_weight * block.last_rate_loss
            else:
                tokens = block(tokens)
                attn_token_counts.append(tokens.shape[1])
                mlp_token_counts.append(tokens.shape[1])

            if self.record_probes and self.probe8 is not None and b_idx in (7, 15):
                probe = self.probe8 if b_idx == 7 else self.probe16
                self.last_probe_logits[b_idx] = probe(tokens[:, 1:, :]).detach()

            # Stage 3 Probes after block 8 and block 16 (0-indexed 7 and 15)
            if self.exit_enabled and b_idx in (7, 15):
                probe = self.probe8 if b_idx == 7 else self.probe16
                probe_logits = probe(tokens[:, 1:, :])  # (B, N_active, 2)

                if self.training:
                    # Auxiliary loss on token-level targets
                    current_targets = targets if targets is not None else getattr(self, "_current_targets", None)
                    if current_targets is not None:
                        with torch.no_grad():
                            p_targets = (
                                current_targets.float()
                                .view(B, 32, 16, 32, 16)
                                .permute(0, 1, 3, 2, 4)
                                .contiguous()
                                .view(B, 1024, 256)
                                .mean(dim=-1)
                            )  # (B, 1024)
                            if share_info.get("is_shared", False):
                                g = share_info["g"]
                                gp = self.sharing_module.group_patches.to(device)
                                sh_g = share_info["shared_groups"]
                                unsh_g = share_info["unshared_groups"]
                                target_list = []
                                for b in range(B):
                                    sh_p = p_targets[b, gp[sh_g[b]]].mean(dim=1)
                                    unsh_p = p_targets[b, gp[unsh_g[b]].view(-1)]
                                    target_list.append(torch.cat([sh_p, unsh_p], dim=0))
                                reduced_targets = torch.stack(target_list, dim=0)
                            else:
                                reduced_targets = p_targets

                            active_targets = reduced_targets[:, active_indices]

                        log_probs = F.log_softmax(probe_logits, dim=-1)
                        target_dist = torch.stack([1.0 - active_targets, active_targets], dim=-1)
                        p_loss = -(target_dist * log_probs).sum(dim=-1).mean()
                        total_probe_loss = total_probe_loss + self.probe_loss_weight * p_loss

                else:
                    # Inference: confident tokens exit (except top-k context tokens per class)
                    probs = F.softmax(probe_logits, dim=-1)  # (B, N_active, 2)
                    p_burn = probs[:, :, 1]
                    p_unburn = probs[:, :, 0]

                    if self.exit_thresh_per_probe is not None:
                        t = self.exit_thresh_per_probe[b_idx]
                        confident = (p_burn >= t) | (p_burn <= 1.0 - t)
                    else:
                        confident = (p_burn >= self.exit_thresh_high) | (p_burn <= self.exit_thresh_low)

                    k_ctx = min(self.exit_context_k, p_burn.shape[1] // 2)
                    _, top_burn_idx = torch.topk(p_burn, k_ctx, dim=1, largest=True)
                    _, top_unburn_idx = torch.topk(p_unburn, k_ctx, dim=1, largest=True)

                    keep_context = torch.zeros_like(confident)
                    keep_context.scatter_(1, top_burn_idx, True)
                    keep_context.scatter_(1, top_unburn_idx, True)

                    should_exit = confident & (~keep_context)  # (B, N_active)
                    exit_mask = should_exit.all(dim=0)  # (N_active,)
                    num_exited = int(exit_mask.sum().item())
                    if num_exited > 0:
                        total_exited_tokens += num_exited
                        exited_pos = torch.where(exit_mask)[0]
                        for pos in exited_pos:
                            orig_idx = int(active_indices[pos].item())
                            exited_features[orig_idx] = tokens[:, pos + 1, :].clone()
                            exited_probe_logits[orig_idx] = probe_logits[:, pos, :].clone()

                        # Remove exited tokens from the sequence
                        stay_mask = ~exit_mask
                        active_indices = active_indices[stay_mask]
                        tokens = torch.cat([tokens[:, :1, :], tokens[:, 1:, :][:, stay_mask, :]], dim=1)
                        p_reduced = p_reduced[:, stay_mask, :]

        # 5. Final norm on active tokens
        tokens = self.encoder.norm(tokens)

        # Reconstruct full sequence of N_reduced tokens
        reconstructed_tokens = torch.empty(B, N_reduced, 1024, device=device, dtype=tokens.dtype)
        reconstructed_tokens[:, active_indices, :] = tokens[:, 1:, :]
        for orig_idx, feat in exited_features.items():
            reconstructed_tokens[:, orig_idx, :] = self.encoder.norm(feat.unsqueeze(1)).squeeze(1)

        # 6. Unshare back to full 32x32 = 1024 patch grid
        if self.share_frac > 0.0:
            full_patch_features = self.sharing_module.unshare(reconstructed_tokens, share_info)
        else:
            full_patch_features = reconstructed_tokens

        # Reshape to (B, 1024, 32, 32)
        spatial_features = full_patch_features.view(B, 32, 32, 1024).permute(0, 3, 1, 2).contiguous()

        self.last_spatial_features = spatial_features
        self.last_share_info = share_info
        self.last_exited_probe_logits = exited_probe_logits

        # 7. Decode with FCN head
        logits = self.decoder(spatial_features)  # (B, 2, 512, 512)

        # Overwrite exited tokens' pixels with probe logits
        if exited_probe_logits and self.exit_overwrite == "probe":
            gp = self.sharing_module.group_patches.to(device)
            if share_info.get("is_shared", False):
                g = share_info["g"]
                sh_g = share_info["shared_groups"]
                unsh_g = share_info["unshared_groups"]
                for orig_idx, p_logit in exited_probe_logits.items():
                    for b in range(B):
                        if orig_idx < g:
                            group_id = sh_g[b][orig_idx]
                            patch_ids = gp[group_id].tolist()
                        else:
                            unsh_offset = orig_idx - g
                            group_id = unsh_g[b][unsh_offset // 4]
                            patch_ids = [gp[group_id][unsh_offset % 4].item()]
                        for pid in patch_ids:
                            r = pid // 32
                            c = pid % 32
                            logits[b:b+1, :, r*16:(r+1)*16, c*16:(c+1)*16] = p_logit[b:b+1].view(1, 2, 1, 1)
            else:
                for orig_idx, p_logit in exited_probe_logits.items():
                    r = orig_idx // 32
                    c = orig_idx % 32
                    logits[:, :, r*16:(r+1)*16, c*16:(c+1)*16] = p_logit.view(B, 2, 1, 1)

        # 8. Extras: NBR residual refinement
        if self.refine_net is not None:
            logits = self.refine_net(logits, stats["nbr"])

        # Track runtime stats
        self.last_tokens_attn = attn_token_counts
        self.last_tokens_mlp = mlp_token_counts
        self.last_exit_rate = total_exited_tokens / float(N_reduced) if N_reduced > 0 else 0.0
        self.last_aux_loss = total_rate_loss + total_probe_loss

        return logits
