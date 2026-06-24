import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from peft import LoraConfig, inject_adapter_in_model


def _load_prithvi_mae_module():
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


class UNetUp(nn.Module):
    """
    Upsample → concat skip → DoubleConv
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))


class PrithviUNetDecoder(nn.Module):
    """
    UNet-style decoder for Prithvi foundation model.
    Utilizes intermediate features from different depths of the ViT encoder
    as skip connections.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Skip connection projections to adjust channels and shapes to fit UNet decoder levels
        # skip3 projection: Block 18 output (32x32) -> (64x64, 512 ch)
        self.skip3_proj = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        
        # skip2 projection: Block 12 output (32x32) -> (128x128, 256 ch)
        self.skip2_proj = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        )
        
        # skip1 projection: Block 6 output (32x32) -> (256x256, 128 ch)
        self.skip1_proj = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        )
        
        # skip0 projection: Input image (512x512, 6 ch) -> (512x512, 64 ch)
        self.skip0_proj = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # UNet Up blocks
        self.up1 = UNetUp(in_ch=1024, skip_ch=512, out_ch=512)
        self.up2 = UNetUp(in_ch=512,  skip_ch=256, out_ch=256)
        self.up3 = UNetUp(in_ch=256,  skip_ch=128, out_ch=128)
        self.up4 = UNetUp(in_ch=128,  skip_ch=64,  out_ch=64)

        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x_input: torch.Tensor, feats_spatial: list[torch.Tensor]) -> torch.Tensor:
        # feats_spatial contains spatial features for chosen block depths in order:
        # [block_6_feat, block_12_feat, block_18_feat, block_24_feat]
        # all of shape (B, 1024, 32, 32)
        feat_block6, feat_block12, feat_block18, feat_block24 = feats_spatial

        # Generate skips
        skip3 = self.skip3_proj(feat_block18)  # (B, 512, 64, 64)
        skip2 = self.skip2_proj(feat_block12)  # (B, 256, 128, 128)
        skip1 = self.skip1_proj(feat_block6)   # (B, 128, 256, 256)
        skip0 = self.skip0_proj(x_input)       # (B, 64, 512, 512)

        # Bottleneck
        x = feat_block24                       # (B, 1024, 32, 32)

        # Decoder steps
        x = self.up1(x, skip3)                 # (B, 512, 64, 64)
        x = self.up2(x, skip2)                 # (B, 256, 128, 128)
        x = self.up3(x, skip1)                 # (B, 128, 256, 256)
        x = self.up4(x, skip0)                 # (B, 64, 512, 512)

        return self.outc(x)


class PrithviUNetSegmentor(nn.Module):
    """
    Prithvi-EO-2.0-300M backbone with LoRA + UNet decoder.
    Uses intermediate layers of Prithvi encoder as skip connections.
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

        # Drop positional embedding buffers: shape mismatch for T=1
        state_dict = {k: v for k, v in state_dict.items() if "pos_embed" not in k}

        missing, unexpected = full_model.load_state_dict(state_dict, strict=False)
        print(f"[Prithvi-UNet] Loaded {weights_path}")
        print(f"               Missing keys  : {len(missing)}")
        print(f"               Unexpected keys: {len(unexpected)}")

        self.encoder = full_model.encoder

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

        # ── 3. UNet decoder (always trained) ──────────────────────────────
        self.decoder = PrithviUNetDecoder(num_classes=num_classes)

        self._print_param_summary()

    def _print_param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Prithvi-UNet] Parameters — total: {total:,}  |  trainable: {trainable:,} "
              f"({100 * trainable / total:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) normalized satellite image
        Returns:
            (B, num_classes, H, W) raw logits
        """
        # forward_features returns list of 24 block outputs, each (B, N+1, 1024)
        feats = self.encoder.forward_features(x)

        # Choose intermediate layers: Block 6, 12, 18, 24 (index 5, 11, 17, 23)
        chosen_indices = [5, 11, 17, 23]
        chosen_feats = [feats[i] for i in chosen_indices]

        # Reshape to spatial representations: list of (B, 1024, 32, 32)
        spatial_feats = self.encoder.prepare_features_for_image_model(chosen_feats)

        return self.decoder(x, spatial_feats)
