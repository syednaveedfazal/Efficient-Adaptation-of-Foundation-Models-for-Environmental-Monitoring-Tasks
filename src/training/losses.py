"""
src/training/losses.py — Shared loss functions for segmentation.

All models (UNet, Prithvi, DINOv2, ...) use the same losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice loss over all classes.
    Works on raw logits — applies softmax internally.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) raw logits
            targets: (B, H, W)    long tensor with class indices
        """
        num_classes = logits.shape[1]
        probs  = F.softmax(logits, dim=1)
        onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        dims   = (0, 2, 3)   # reduce over batch + spatial
        inter  = (probs * onehot).sum(dims)
        card   = (probs + onehot).sum(dims)
        dice   = (2.0 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of CrossEntropyLoss + DiceLoss.

    Handles class imbalance via:
      - class_weights: per-class weight tensor passed to CrossEntropy
      - dice_weight:   Dice loss is unaffected by class frequency
    """

    def __init__(
        self,
        ce_weight:    float,
        dice_weight:  float,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.ce_w   = ce_weight
        self.dice_w = dice_weight
        self.ce   = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_w * self.ce(logits, targets) + \
               self.dice_w * self.dice(logits, targets)


def build_loss(cfg: dict) -> CombinedLoss:
    """Instantiate loss from the 'loss' section of a config dict."""
    cw = torch.tensor(cfg["class_weights"], dtype=torch.float32)
    return CombinedLoss(
        ce_weight    = cfg["ce_weight"],
        dice_weight  = cfg["dice_weight"],
        class_weights = cw,
    )