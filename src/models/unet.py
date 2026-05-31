"""
src/models/unet.py — UNet from scratch for 6-band satellite image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv2d → BN → ReLU → Conv2d → BN → ReLU"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    """MaxPool → DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.pool_conv(x)


class Up(nn.Module):
    """Upsample → concat skip → DoubleConv"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) \
                    if bilinear else \
                    nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    Standard UNet adapted for multi-spectral satellite imagery.

    Args:
        in_channels:  Input bands (6 for HLS Blue/Green/Red/NIR/SW1/SW2).
        num_classes:  Output classes (2 for binary burn-scar segmentation).
        base_filters: Filters in first encoder block; doubles each level.
        bilinear:     Bilinear upsampling (True) vs transposed convolutions (False).

    Architecture (base_filters=64):
        Encoder: 64 → 128 → 256 → 512 → 512 (bottleneck)
        Decoder: 512 → 256 → 128 → 64 → 64
        Output:  1×1 conv → num_classes logits
    """

    def __init__(
        self,
        in_channels:  int  = 6,
        num_classes:  int  = 2,
        base_filters: int  = 64,
        bilinear:     bool = True,
    ):
        super().__init__()
        f = base_filters

        # Encoder
        self.inc   = DoubleConv(in_channels, f)
        self.down1 = Down(f,     f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 8)   # bottleneck

        # Decoder
        self.up1 = Up(f * 16, f * 4, bilinear)
        self.up2 = Up(f * 8,  f * 2, bilinear)
        self.up3 = Up(f * 4,  f,     bilinear)
        self.up4 = Up(f * 2,  f,     bilinear)

        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)   # (B, num_classes, H, W) raw logits


if __name__ == "__main__":
    model = UNet(in_channels=6, num_classes=2)
    dummy = torch.randn(2, 6, 512, 512)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")    # (2, 2, 512, 512)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")