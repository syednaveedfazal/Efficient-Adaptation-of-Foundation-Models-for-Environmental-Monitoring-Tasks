"""
src/models/prithvi.py — Prithvi-EO-2.0-300M backbone with LoRA r=8 for segmentation.

Architecture:
  PrithviViT encoder (295M frozen) + PEFT LoRA adapters (~900K trainable)
  → SegDecoder (4× conv+upsample, ~4.5M trainable)
  → (B, 2, 512, 512) logits

Usage (via registry):
  model.name: prithvi_lora_r8
  model.params:
    weights_path: models/pretrained/prithvi/Prithvi_EO_V2_300M.pt
    num_classes:  2
    lora_rank:    8
    lora_alpha:   8
"""

import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from peft import LoraConfig, inject_adapter_in_model


def _load_prithvi_mae_module():
    """Import PrithviMAE from the local prithvi_mae.py without polluting sys.modules."""
    mae_path = Path(__file__).resolve().parents[2] / "models" / "pretrained" / "prithvi" / "prithvi_mae.py"
    if not mae_path.exists():
        raise FileNotFoundError(
            f"prithvi_mae.py not found at {mae_path}. "
            "Run: python scripts/download_models.py --model prithvi"
        )
    spec = importlib.util.spec_from_file_location("prithvi_mae", mae_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SegDecoder(nn.Module):
    """
    Convolutional decoder that upsamples ViT patch tokens to full resolution.

    Input:  (B, 1024, 32, 32)   — spatial feature map from Prithvi's final block
    Output: (B, num_classes, 512, 512)
    """

    def __init__(self, in_channels: int = 1024, num_classes: int = 2):
        super().__init__()

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            )

        self.stages = nn.Sequential(
            _block(in_channels, 256),   # 32  → 64
            _block(256, 128),           # 64  → 128
            _block(128, 64),            # 128 → 256
            _block(64,  32),            # 256 → 512
        )
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(x))


class PrithviSegmentor(nn.Module):
    """
    Prithvi-EO-2.0-300M backbone with LoRA r=8 + convolutional segmentation decoder.

    Args:
        weights_path: Path to Prithvi_EO_V2_300M.pt
        num_classes:  Number of output classes (2 for binary burn-scar segmentation)
        lora_rank:    LoRA rank r (default 8)
        lora_alpha:   LoRA alpha scaling (default 8, so scale = alpha/r = 1.0)
    """

    def __init__(
        self,
        weights_path: str,
        num_classes:  int   = 2,
        lora_rank:    int   = 8,
        lora_alpha:   int   = 8,
    ):
        super().__init__()

        # ── 1. Load Prithvi backbone ──────────────────────────────────────
        prithvi_mod = _load_prithvi_mae_module()
        PrithviMAE  = prithvi_mod.PrithviMAE

        # Instantiate for T=1 static task, 512×512 input
        full_model = PrithviMAE(
            img_size          = 512,
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

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

        # Drop positional embedding buffers: checkpoint was trained with num_frames=4,
        # so their shape mismatches our T=1 model. They are sincos buffers re-initialised
        # by initialize_weights() and interpolated at runtime by _interpolate_pos_encoding.
        state_dict = {k: v for k, v in state_dict.items() if "pos_embed" not in k}

        missing, unexpected = full_model.load_state_dict(state_dict, strict=False)
        print(f"[Prithvi] Loaded {weights_path}")
        print(f"          Missing keys  : {len(missing)}  (pos_embed sincos buffers excluded — expected)")
        print(f"          Unexpected keys: {len(unexpected)}")

        self.encoder = full_model.encoder   # PrithviViT — we only need the encoder

        # ── 2. Inject LoRA adapters in-place ─────────────────────────────
        lora_cfg = LoraConfig(
            r              = lora_rank,
            lora_alpha     = lora_alpha,
            target_modules = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            bias           = "none",
            lora_dropout   = 0.0,
        )
        inject_adapter_in_model(lora_cfg, self.encoder)

        # Freeze everything; unfreeze only the LoRA A/B matrices
        for name, param in self.encoder.named_parameters():
            param.requires_grad = "lora_" in name

        # ── 3. Segmentation decoder (always trained) ──────────────────────
        # embed_dim=1024 for T=1: prepare_features_for_image_model yields (B, 1024, H/16, W/16)
        self.decoder = SegDecoder(in_channels=1024, num_classes=num_classes)

        self._print_param_summary()

    def _print_param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Prithvi] Parameters — total: {total:,}  |  trainable: {trainable:,} "
              f"({100 * trainable / total:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) normalised satellite image
        Returns:
            (B, num_classes, H, W) raw logits
        """
        # forward_features returns a list of 24 block outputs, each (B, N+1, 1024)
        feats = self.encoder.forward_features(x)

        # Reshape final block tokens to spatial map: (B, 1024, H/16, W/16)
        spatial = self.encoder.prepare_features_for_image_model([feats[-1]])

        return self.decoder(spatial[0])
