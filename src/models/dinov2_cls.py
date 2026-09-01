"""
src/models/dinov2_cls.py — DINOv2-Large (generic ViT, ~300M) classifier.

Land-cover classification counterpart of dinov2_fcn.py. It reuses the SAME
DinoV2Backbone (RGB band selection + resize to native 518 + ViT) and the SAME
adaptation axis (apply_adaptation); only the head differs — a global-average
pool + linear classifier instead of the FCN segmentation decoder.

Input is the 6-band HLS-order tensor (EuroSAT emits it in that order); the
backbone selects RGB = [2,1,0] internally, exactly as for burn-scar.
"""

import torch
import torch.nn as nn

from src.models.dinov2_fcn import DinoV2Backbone, DEFAULT_RGB_INDICES
from src.models.adaptation import apply_adaptation, DinoV2VPTWrapper
from src.models.heads import ClassificationHead


class DinoV2Classifier(nn.Module):
    """
    DINOv2-Large backbone (RGB) with a selectable adaptation strategy and a
    linear classification head.

    Args:
        num_classes: number of target classes (10 for EuroSAT).
        adaptation:  "lora" | "full_ft" | "linear_probe" | "randomized".
        lora_rank:   LoRA rank r (only used when adaptation == "lora").
        lora_alpha:  LoRA alpha scaling (scale = alpha / rank).
        rgb_indices: which of the 6 input bands feed DINOv2, as (R, G, B).
    """

    def __init__(
        self,
        num_classes:    int = 10,
        adaptation:     str = "lora",
        lora_rank:      int = 8,
        lora_alpha:     int = 8,
        rgb_indices=DEFAULT_RGB_INDICES,
        vpt_num_tokens: int = 10,
    ):
        super().__init__()

        # ── 1. Backbone (RGB) ─────────────────────────────────────────────
        randomized = (adaptation == "randomized")
        self.encoder = DinoV2Backbone(rgb_indices=rgb_indices, randomized=randomized)

        # ── 2. Adaptation strategy (shared with seg models) ───────────────
        strategy = "full_ft" if randomized else adaptation
        result = apply_adaptation(self.encoder, strategy, lora_rank=lora_rank,
                                  lora_alpha=lora_alpha, vpt_num_tokens=vpt_num_tokens)
        if result == "vpt":
            self.encoder = DinoV2VPTWrapper(self.encoder, num_tokens=vpt_num_tokens)

        # ── 3. Classification head (always trained) ───────────────────────
        self.head = ClassificationHead(DinoV2Backbone.EMBED_DIM, num_classes)

        self._print_param_summary()

    def _print_param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DINOv2-CLS] Parameters — total: {total:,}  |  trainable: {trainable:,} "
              f"({100 * trainable / total:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) normalized image (RGB bands selected internally)
        Returns:
            (B, num_classes) raw logits
        """
        spatial_feat = self.encoder.forward_features(x)   # (B, 1024, 37, 37)
        return self.head(spatial_feat)
