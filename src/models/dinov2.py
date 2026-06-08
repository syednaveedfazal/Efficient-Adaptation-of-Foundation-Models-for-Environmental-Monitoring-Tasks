"""
src/models/dinov2.py — DINOv2 backbone for RGB-only burn-scar segmentation.

This wrapper expects raw 6-band HLS inputs and performs model-specific preprocessing:
  1. select RGB bands from the 6-channel input,
  2. reorder to Red/Green/Blue,
  3. scale reflectance values to [0, 1],
  4. normalize with ImageNet statistics expected by DINOv2.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Dinov2Config, Dinov2Model


def _validate_pretrained_source(pretrained_model_name_or_path: str):
    path = Path(pretrained_model_name_or_path)
    if not path.exists():
        return

    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"DINOv2 config not found at {config_path}. "
            "Run `python scripts/download_models.py --model dinov2` or point "
            "`pretrained_model_name_or_path` to a valid Hugging Face model ID."
        )


class Dinov2SegDecoder(nn.Module):
    """Lightweight convolutional decoder for dense segmentation."""

    def __init__(self, in_channels: int, num_classes: int, decoder_channels: tuple[int, ...]):
        super().__init__()

        blocks = []
        current_channels = in_channels
        for out_channels in decoder_channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                )
            )
            current_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(current_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = self.blocks(x)
        x = self.head(x)
        return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)


class Dinov2Segmentor(nn.Module):
    """
    DINOv2-Base backbone with a segmentation decoder on top.

    Args:
        pretrained_model_name_or_path: Local snapshot directory or Hugging Face model ID.
        num_classes: Number of output classes.
        rgb_indices: Indices used to convert HLS channels to RGB order.
        raw_scale: Divisor used to map raw HLS reflectance values into [0, 1].
        backbone_input_size: Spatial resolution fed into DINOv2. Should be divisible by patch size 14.
        train_mode: Either "linear_probe" (frozen backbone) or "full_finetune" (unfrozen backbone).
        decoder_channels: Output channels for each decoder stage.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_classes: int = 2,
        rgb_indices: tuple[int, int, int] = (2, 1, 0),
        raw_scale: float = 10000.0,
        backbone_input_size: int = 518,
        train_mode: str = "full_finetune",
        decoder_channels: tuple[int, int, int, int] = (384, 192, 96, 48),
        freeze_backbone: bool | None = None,
    ):
        super().__init__()

        if len(rgb_indices) != 3:
            raise ValueError(f"rgb_indices must contain exactly 3 entries, got {rgb_indices}.")

        _validate_pretrained_source(pretrained_model_name_or_path)

        config = Dinov2Config.from_pretrained(pretrained_model_name_or_path)
        if backbone_input_size % int(config.patch_size) != 0:
            raise ValueError(
                f"backbone_input_size={backbone_input_size} is incompatible with "
                f"DINOv2 patch size {config.patch_size}."
            )
        self.backbone = Dinov2Model.from_pretrained(pretrained_model_name_or_path, config=config)

        if freeze_backbone is not None:
            train_mode = "linear_probe" if freeze_backbone else "full_finetune"
        if train_mode not in {"linear_probe", "full_finetune"}:
            raise ValueError(
                f"Unsupported train_mode '{train_mode}'. "
                "Expected one of {'linear_probe', 'full_finetune'}."
            )

        self.rgb_indices = tuple(rgb_indices)
        self.raw_scale = raw_scale
        self.backbone_input_size = backbone_input_size
        self.train_mode = train_mode
        self.patch_size = int(config.patch_size)
        self.hidden_size = int(config.hidden_size)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.decoder = Dinov2SegDecoder(
            in_channels=self.hidden_size,
            num_classes=num_classes,
            decoder_channels=tuple(decoder_channels),
        )

        if self.train_mode == "linear_probe":
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self._print_param_summary()

    def _print_param_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[DINOv2] Mode: {self.train_mode} | total: {total:,} | "
            f"trainable: {trainable:,} ({100 * trainable / total:.2f}%)"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.train_mode == "linear_probe":
            self.backbone.eval()
        return self

    def _prepare_rgb(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x[:, self.rgb_indices, :, :]
        rgb = rgb / self.raw_scale
        rgb = torch.clamp(rgb, 0.0, 1.0)
        return (rgb - self.imagenet_mean) / self.imagenet_std

    def _forward_backbone(self, pixel_values: torch.Tensor):
        if self.train_mode == "linear_probe":
            with torch.no_grad():
                return self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=True)
        return self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]

        pixel_values = self._prepare_rgb(x)
        pixel_values = F.interpolate(
            pixel_values,
            size=(self.backbone_input_size, self.backbone_input_size),
            mode="bilinear",
            align_corners=False,
        )

        outputs = self._forward_backbone(pixel_values)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        num_patch_tokens = patch_tokens.shape[1]
        grid_size = int(num_patch_tokens ** 0.5)
        if grid_size * grid_size != num_patch_tokens:
            raise RuntimeError(
                f"Cannot reshape {num_patch_tokens} patch tokens into a square feature map."
            )
        feature_map = patch_tokens.transpose(1, 2).reshape(
            x.shape[0],
            self.hidden_size,
            grid_size,
            grid_size,
        )

        return self.decoder(feature_map, output_size=output_size)
