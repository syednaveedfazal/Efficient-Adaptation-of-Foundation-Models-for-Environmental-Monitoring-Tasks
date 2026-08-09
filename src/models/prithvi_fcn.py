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


class PrithviFCNDecoder(nn.Module):
    """
    Standard Fully Convolutional Network (FCN) head for segmentation.
    Projects final bottleneck features and directly upsamples to full resolution.
    """
    def __init__(self, in_channels: int = 1024, num_classes: int = 2):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (B, 1024, 32, 32)
        logits = self.head(x)  # (B, num_classes, 32, 32)
        # Direct bilinear upsampling by factor of 16 to (B, num_classes, 512, 512)
        return nn.functional.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=True)


class PrithviFCNSegmentor(nn.Module):
    """
    Prithvi-EO-2.0-300M backbone with selectable adaptation strategy + FCN decoder.

    Weight initialisation and adaptation strategy are independent axes:
      * ``randomized=False`` (default)  → load pretrained weights from *weights_path*
      * ``randomized=True``             → keep random PyTorch init (no checkpoint)

      * ``adaptation="lora"``           → inject LoRA adapters; freeze backbone
      * ``adaptation="full_ft"``        → all backbone params trainable
      * ``adaptation="linear_probe"``   → freeze backbone entirely
    """
    def __init__(
        self,
        weights_path: str,
        num_classes:  int    = 2,
        adaptation:   str    = "lora",   # "lora", "full_ft", "linear_probe"
        randomized:   bool   = False,    # True → skip pretrained weights
        lora_rank:    int    = 8,
        lora_alpha:   int    = 8,
    ):
        super().__init__()

        # ── Backward compatibility ────────────────────────────────────────
        # Legacy configs may pass adaptation="randomized" (≡ randomized + full_ft)
        if adaptation == "randomized":
            randomized = True
            adaptation = "full_ft"

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

        if not randomized:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

            # Drop positional embedding buffers
            state_dict = {k: v for k, v in state_dict.items() if "pos_embed" not in k}

            missing, unexpected = full_model.load_state_dict(state_dict, strict=False)
            print(f"[Prithvi-FCN] Loaded {weights_path}")
            print(f"               Missing keys  : {len(missing)}")
            print(f"               Unexpected keys: {len(unexpected)}")
        else:
            print(f"[Prithvi-FCN] Randomized mode — using randomly initialised weights")

        self.encoder = full_model.encoder

        # ── 2. Handle Adaptation Strategies ──────────────────────────────
        if adaptation == "lora":
            # Inject LoRA adapters
            lora_cfg = LoraConfig(
                r              = lora_rank,
                lora_alpha     = lora_alpha,
                target_modules = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
                bias           = "none",
                lora_dropout   = 0.0,
            )
            inject_adapter_in_model(lora_cfg, self.encoder)

            # Freeze backbone; unfreeze only LoRA weights
            for name, param in self.encoder.named_parameters():
                param.requires_grad = "lora_" in name

        elif adaptation == "full_ft":
            # Full fine-tuning: all backbone layers are fully trainable
            for param in self.encoder.parameters():
                param.requires_grad = True

        elif adaptation == "linear_probe":
            # Linear probing: completely freeze the backbone
            for param in self.encoder.parameters():
                param.requires_grad = False

        else:
            raise ValueError(f"Unknown adaptation strategy: {adaptation}")

        # ── 3. FCN decoder (always trained) ──────────────────────────────
        self.decoder = PrithviFCNDecoder(in_channels=1024, num_classes=num_classes)

        self._print_param_summary()

    def _print_param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Prithvi-FCN] Parameters — total: {total:,}  |  trainable: {trainable:,} "
              f"({100 * trainable / total:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) normalized satellite image
        Returns:
            (B, num_classes, H, W) raw logits
        """
        feats = self.encoder.forward_features(x)
        
        # Take the final block output (index 23)
        final_feat = [feats[23]]

        # Reshape to spatial representation: (B, 1024, 32, 32)
        spatial_feat = self.encoder.prepare_features_for_image_model(final_feat)[0]

        return self.decoder(spatial_feat)
