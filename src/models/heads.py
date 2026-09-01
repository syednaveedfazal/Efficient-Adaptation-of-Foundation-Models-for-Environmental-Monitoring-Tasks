"""
src/models/heads.py — Task heads shared across backbones.

The segmentation FCN decoders live next to their backbones (prithvi_fcn.py,
dinov2_fcn.py). This module holds the classification head, used by the EuroSAT
land-cover classifiers (prithvi_cls.py, dinov2_cls.py). Both backbones expose a
spatial feature map (B, C, h, w), so a single global-average-pool + linear head
works for either.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Global-average-pool a (B, C, h, w) feature map to (B, C), then a linear
    classifier to (B, num_classes).

    Args:
        in_channels: backbone embedding dim (1024 for Prithvi & DINOv2-Large).
        num_classes: number of target classes (10 for EuroSAT).
        dropout:     dropout applied to the pooled feature before the linear.
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.pool    = nn.AdaptiveAvgPool2d(1)   # (B, C, h, w) -> (B, C, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)   # (B, C)
        x = self.dropout(x)
        return self.fc(x)             # (B, num_classes)
