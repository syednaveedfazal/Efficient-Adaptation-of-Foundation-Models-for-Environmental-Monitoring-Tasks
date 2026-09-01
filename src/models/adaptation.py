"""
src/models/adaptation.py — Shared parameter-efficient adaptation logic.

The LoRA / full-FT / linear-probe / VPT strategy is an axis that is *independent*
of the task head (segmentation FCN vs. classification head) and of the backbone
(Prithvi vs. DINOv2). This module centralises it so every model file — seg and
classification, Prithvi and DINOv2 — applies the exact same adaptation, with
one source of truth for the LoRA target modules and VPT wrapper.

Usage:
    from src.models.adaptation import apply_adaptation
    apply_adaptation(self.encoder, adaptation="lora", lora_rank=8, lora_alpha=8)

    # For VPT, the caller must replace self.encoder with the returned wrapper:
    result = apply_adaptation(self.encoder, adaptation="vpt", vpt_num_tokens=10)
    if result is not None:
        self.encoder = result

Note: "randomized" (random-init backbone) is a *weight-initialisation* axis, not
an adaptation strategy — callers handle it before loading weights, then pass the
underlying strategy (usually "full_ft") here.
"""

import math
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model

# Same attention/MLP projections targeted across all backbones (timm & Prithvi
# both expose these names), so LoRA is applied identically everywhere.
LORA_TARGET_MODULES = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]


# ──────────────────────────────────────────────────────────────────────────────
# VPT-Deep wrapper for Prithvi's PrithviViT encoder
# ──────────────────────────────────────────────────────────────────────────────

class PrithviVPTWrapper(nn.Module):
    """
    VPT-Deep wrapper for the Prithvi PrithviViT encoder.

    Prepends *num_tokens* learnable prompt tokens to the sequence at **every**
    transformer block (VPT-Deep). The prompt tokens are stripped before
    returning so that downstream code (decoder / classification head) sees
    exactly the same output shapes as the un-wrapped encoder.

    The wrapper delegates all attribute access to the underlying encoder so
    that `model.encoder.embed_dim`, `model.encoder.patch_embed`, etc. keep
    working transparently.
    """

    def __init__(self, encoder: nn.Module, num_tokens: int = 10):
        super().__init__()
        self.encoder = encoder
        self.num_tokens = num_tokens
        depth = len(encoder.blocks)
        embed_dim = encoder.embed_dim

        # One learnable prompt per layer, shape (1, num_tokens, embed_dim)
        self.prompt_tokens = nn.ParameterList([
            nn.Parameter(torch.empty(1, num_tokens, embed_dim))
            for _ in range(depth)
        ])
        # Xavier-uniform init (fan_in = embed_dim)
        for p in self.prompt_tokens:
            val = math.sqrt(6.0 / (embed_dim + num_tokens))
            nn.init.uniform_(p, -val, val)

    # Transparent attribute delegation to the wrapped encoder
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.encoder, name)

    def forward_features(self, x, temporal_coords=None, location_coords=None):
        """
        Same signature and return type as PrithviViT.forward_features:
        returns list[Tensor] with one entry per block, each of shape
        (B, 1 + N_patches, embed_dim)  — prompts are stripped.
        """
        enc = self.encoder

        if len(x.shape) == 4 and enc.patch_embed.input_size[0] == 1:
            x = x.unsqueeze(2)
        sample_shape = x.shape[-3:]

        # Patch embedding
        x = enc.patch_embed(x)

        pos_embed = enc.interpolate_pos_encoding(sample_shape)
        x = x + pos_embed[:, 1:, :]

        if enc.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // enc.num_frames
            temporal_encoding = enc.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
            x = x + temporal_encoding
        if enc.location_encoding and location_coords is not None:
            location_encoding = enc.location_embed_enc(location_coords)
            x = x + location_encoding

        # Append cls token
        cls_token = enc.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)   # (B, 1+N, D)

        B = x.shape[0]
        out = []
        for i, block in enumerate(enc.blocks):
            # Prepend layer-specific prompt tokens: (B, P+1+N, D)
            prompts = self.prompt_tokens[i].expand(B, -1, -1)
            x = torch.cat((prompts, x), dim=1)

            x = block(x)

            # Strip prompt tokens to restore original sequence length
            x = x[:, self.num_tokens:, :]          # (B, 1+N, D)
            out.append(x.clone())

        x = enc.norm(x)
        out[-1] = x
        return out

    def prepare_features_for_image_model(self, features):
        return self.encoder.prepare_features_for_image_model(features)

    def forward(self, *args, **kwargs):
        return self.forward_features(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# VPT-Deep wrapper for DINOv2 (timm ViT) backbone
# ──────────────────────────────────────────────────────────────────────────────

class DinoV2VPTWrapper(nn.Module):
    """
    VPT-Deep wrapper for the DinoV2Backbone (timm ViT inside).

    Injects learnable prompts into every transformer block of the underlying
    timm ``vit.blocks``.  Prompt tokens are stripped after each block so that
    the output shape of ``forward_features`` remains identical to the
    un-wrapped backbone:  (B, embed_dim, H, W).
    """

    def __init__(self, backbone: nn.Module, num_tokens: int = 10):
        super().__init__()
        self.backbone = backbone
        self.num_tokens = num_tokens
        vit = backbone.vit
        depth = len(vit.blocks)
        embed_dim = vit.embed_dim

        self.prompt_tokens = nn.ParameterList([
            nn.Parameter(torch.empty(1, num_tokens, embed_dim))
            for _ in range(depth)
        ])
        for p in self.prompt_tokens:
            val = math.sqrt(6.0 / (embed_dim + num_tokens))
            nn.init.uniform_(p, -val, val)

    # Transparent attribute delegation to the wrapped backbone
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.backbone, name)

    def forward_features(self, x):
        """
        Same signature and return type as DinoV2Backbone.forward_features:
        returns (B, C, H, W) spatial feature map.
        """
        import torch.nn.functional as F

        bb = self.backbone
        vit = bb.vit

        # Select RGB bands and resize to DINOv2 native resolution
        x = x.index_select(1, bb.rgb_indices)                       # (B, 3, H, W)
        x = F.interpolate(x, size=(bb.IMG_SIZE, bb.IMG_SIZE),
                          mode="bilinear", align_corners=False)

        # timm ViT patch embedding + pos embed + cls token
        x = vit.patch_embed(x)
        x = vit._pos_embed(x)                                       # (B, 1+N, D)
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)

        B = x.shape[0]
        for i, block in enumerate(vit.blocks):
            prompts = self.prompt_tokens[i].expand(B, -1, -1)
            x = torch.cat((prompts, x), dim=1)
            x = block(x)
            x = x[:, self.num_tokens:, :]                            # strip prompts

        x = vit.norm(x)

        # Drop cls / prefix tokens, reshape to spatial grid
        patch = x[:, bb.num_prefix:, :]
        B, N, C = patch.shape
        hw = int(N ** 0.5)
        return patch.transpose(1, 2).reshape(B, C, hw, hw)

    def forward(self, x):
        return self.forward_features(x)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def apply_adaptation(
    encoder,
    adaptation: str = "lora",
    lora_rank:  int = 8,
    lora_alpha: int = 8,
    vpt_num_tokens: int = 10,
):
    """
    Configure `encoder.requires_grad` (and inject LoRA / VPT adapters) in place.

    Args:
        encoder:        the backbone module whose parameters get (un)frozen.
        adaptation:     "lora" | "full_ft" | "linear_probe" | "vpt".
        lora_rank:      LoRA rank r (only used when adaptation == "lora").
        lora_alpha:     LoRA alpha scaling (scale = alpha / rank).
        vpt_num_tokens: number of prompt tokens per layer (VPT only).

    Returns:
        None for lora / full_ft / linear_probe (mutation is in-place).
        For "vpt", returns a VPTWrapper that the caller must assign to
        ``self.encoder`` (because wrapping cannot be done in-place).

    Raises:
        ValueError on an unknown adaptation strategy.
    """
    if adaptation == "lora":
        lora_cfg = LoraConfig(
            r              = lora_rank,
            lora_alpha     = lora_alpha,
            target_modules = LORA_TARGET_MODULES,
            bias           = "none",
            lora_dropout   = 0.0,
        )
        inject_adapter_in_model(lora_cfg, encoder)
        # Freeze backbone; unfreeze only the injected LoRA weights.
        for name, param in encoder.named_parameters():
            param.requires_grad = "lora_" in name

    elif adaptation == "full_ft":
        for param in encoder.parameters():
            param.requires_grad = True

    elif adaptation == "linear_probe":
        for param in encoder.parameters():
            param.requires_grad = False

    elif adaptation == "vpt":
        # Freeze backbone entirely; the wrapper's prompt_tokens are trainable.
        for param in encoder.parameters():
            param.requires_grad = False
        # Return type signals to the caller that wrapping is needed.
        # Caller is responsible for choosing the right wrapper class.
        return "vpt"

    else:
        raise ValueError(f"Unknown adaptation strategy: {adaptation}")
