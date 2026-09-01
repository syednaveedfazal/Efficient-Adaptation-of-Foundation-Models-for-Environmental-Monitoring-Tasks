"""
src/models/cnn.py — From-scratch ResNet-18-style CNN classifier.

The "no-foundation-model" baseline for EuroSAT land-cover classification,
parallel to unet_scratch on burn-scar segmentation: trained from random init
(no pretraining), uses all 6 bands, native 64x64 input (fully convolutional,
no resize). Self-contained in pure torch.nn — no torchvision dependency.

Architecture (input 6x64x64):
    Stem:    Conv(6->64,3x3) + BN + ReLU            -> 64x64x64  (small-image stem)
    Stage 1: 2x BasicBlock(64)                      -> 64x64x64
    Stage 2: 2x BasicBlock(128, stride 2)           -> 32x32x128
    Stage 3: 2x BasicBlock(256, stride 2)           -> 16x16x256
    Stage 4: 2x BasicBlock(512, stride 2)           ->  8x8x512
    Head:    GlobalAvgPool -> Dropout -> Linear(512->num_classes)
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet basic residual block: Conv3x3-BN-ReLU-Conv3x3-BN + skip + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        # Project the skip connection when shape changes (stride>1 or channel change)
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CNNClassifier(nn.Module):
    """
    From-scratch ResNet-18-style CNN classifier for EuroSAT.

    Args:
        in_channels:  Input bands (6 for the 6-band HLS-order EuroSAT tensor).
        num_classes:  Output classes (10 for EuroSAT land cover).
        base_filters: Channels in the first stage; doubles each stage.
        blocks_per_stage: BasicBlocks per stage (2 -> ResNet-18-style, 8 blocks).
        dropout:      Dropout on the pooled feature before the linear head.
    """

    def __init__(
        self,
        in_channels:      int = 6,
        num_classes:      int = 10,
        base_filters:     int = 64,
        blocks_per_stage: int = 2,
        dropout:        float = 0.1,
    ):
        super().__init__()
        f = base_filters

        # Small-image stem (no 7x7/maxpool downsampling — 64x64 is already small)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, f, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(f,     f,     blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(f,     f * 2, blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(f * 2, f * 4, blocks_per_stage, stride=2)
        self.stage4 = self._make_stage(f * 4, f * 8, blocks_per_stage, stride=2)

        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(f * 8, num_classes)

        self._print_param_summary()

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _print_param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"[CNN-scratch] Parameters — total: {total:,} (trained from random init)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) normalized image
        Returns:
            (B, num_classes) raw logits
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)                 # (B, f*8, H/8, W/8)
        x = self.pool(x).flatten(1)        # (B, f*8)
        return self.fc(self.dropout(x))    # (B, num_classes)


if __name__ == "__main__":
    m = CNNClassifier(in_channels=6, num_classes=10)
    dummy = torch.randn(2, 6, 64, 64)
    out = m(dummy)
    print(f"Input : {dummy.shape}  Output: {out.shape}")   # (2, 10)
    print(f"Params: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}")
