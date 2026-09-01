"""
src/models/prithvi_cls.py — Prithvi-EO-2.0-300M land-cover classifier.

Classification counterpart of prithvi_fcn.py. Reuses the same Prithvi MAE module
loader and the same encoder feature-extraction path (forward_features -> last
block -> prepare_features_for_image_model), and the shared adaptation axis
(apply_adaptation). Only the head differs: a global-average-pool + linear
classifier instead of the FCN segmentation decoder.

The encoder is instantiated at img_size=224 (Prithvi's pretraining resolution ->
14x14 = 196 patch tokens); EuroSAT's native 64x64 6-band tensor is bilinearly
resized to 224 in forward(). Prithvi's positional embeddings are dropped on load
and rebuilt for the configured img_size, so this "just works".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.prithvi_fcn import _load_prithvi_mae_module
from src.models.adaptation import apply_adaptation, PrithviVPTWrapper
from src.models.heads import ClassificationHead


class PrithviClassifier(nn.Module):
    """
    Prithvi-EO-2.0-300M backbone (6-band) with a selectable adaptation strategy
    and a linear classification head.

    Args:
        weights_path: path to Prithvi_EO_V2_300M.pt.
        num_classes:  number of target classes (10 for EuroSAT).
        adaptation:   "lora" | "full_ft" | "linear_probe" | "randomized".
        randomized:   True -> skip pretrained weights (random init).
        lora_rank:    LoRA rank r (only used when adaptation == "lora").
        lora_alpha:   LoRA alpha scaling (scale = alpha / rank).
        img_size:     encoder input resolution (input is resized to this).
    """

    EMBED_DIM = 1024

    def __init__(
        self,
        weights_path:   str,
        num_classes:    int  = 10,
        adaptation:     str  = "lora",
        randomized:     bool = False,
        lora_rank:      int  = 8,
        lora_alpha:     int  = 8,
        img_size:       int  = 224,
        vpt_num_tokens: int  = 10,
    ):
        super().__init__()
        self.img_size = img_size

        # Legacy alias: adaptation="randomized" ≡ randomized + full_ft
        if adaptation == "randomized":
            randomized = True
            adaptation = "full_ft"

        # ── 1. Build Prithvi backbone at img_size ─────────────────────────
        prithvi_mod = _load_prithvi_mae_module()
        PrithviMAE  = prithvi_mod.PrithviMAE
        full_model = PrithviMAE(
            img_size          = img_size,
            num_frames        = 1,
            patch_size        = (1, 16, 16),
            in_chans          = 6,
            embed_dim         = 1024,
            depth             = 24,
            num_heads         = 16,
            decoder_embed_dim = 512,
            decoder_depth     = 8,
            decoder_num_heads = 16,
            mlp_ratio         = 4.0,
        )

        if not randomized:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            # Drop positional embedding buffers (rebuilt for this img_size)
            state_dict = {k: v for k, v in state_dict.items() if "pos_embed" not in k}
            missing, unexpected = full_model.load_state_dict(state_dict, strict=False)
            print(f"[Prithvi-CLS] Loaded {weights_path}")
            print(f"               Missing keys  : {len(missing)}")
            print(f"               Unexpected keys: {len(unexpected)}")
        else:
            print(f"[Prithvi-CLS] Randomized mode — using randomly initialised weights")

        self.encoder = full_model.encoder

        # ── 2. Adaptation strategy (shared with seg models) ───────────────
        result = apply_adaptation(self.encoder, adaptation, lora_rank=lora_rank,
                                  lora_alpha=lora_alpha, vpt_num_tokens=vpt_num_tokens)
        if result == "vpt":
            self.encoder = PrithviVPTWrapper(self.encoder, num_tokens=vpt_num_tokens)

        # ── 3. Classification head (always trained) ───────────────────────
        self.head = ClassificationHead(self.EMBED_DIM, num_classes)

        self._print_param_summary()

    def _print_param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Prithvi-CLS] Parameters — total: {total:,}  |  trainable: {trainable:,} "
              f"({100 * trainable / total:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) normalized image (HLS band order)
        Returns:
            (B, num_classes) raw logits
        """
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode="bilinear", align_corners=False)
        feats = self.encoder.forward_features(x)
        final_feat = [feats[23]]                                            # last block
        spatial_feat = self.encoder.prepare_features_for_image_model(final_feat)[0]  # (B,1024,14,14)
        return self.head(spatial_feat)
