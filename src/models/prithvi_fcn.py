import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

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
      * ``adaptation="vpt_shallow"``    → learnable visual prompt tokens at input; freeze backbone
      * ``adaptation="vpt_deep"``       → learnable visual prompt tokens at every layer; freeze backbone
    """
    def __init__(
        self,
        weights_path: str,
        num_classes:  int    = 2,
        adaptation:   str    = "lora",   # "lora", "full_ft", "linear_probe", "vpt", "vpt_shallow", "vpt_deep"
        randomized:   bool   = False,    # True → skip pretrained weights
        lora_rank:    int    = 8,
        lora_alpha:   int    = 8,
        num_prompts:  int    = 10,
        prompt_dropout: float = 0.0,
    ):
        super().__init__()
        self.adaptation = adaptation
        self.num_prompts = num_prompts

        # ── Backward compatibility ────────────────────────────────────────
        # Legacy configs may pass adaptation="randomized" (≡ randomized + full_ft)
        if adaptation == "randomized":
            randomized = True
            adaptation = "full_ft"
            self.adaptation = "full_ft"

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

        elif adaptation in ("vpt", "vpt_shallow"):
            # Visual Prompt Tuning (Shallow): Learnable prompts at input level only
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.prompt_embeddings = nn.Parameter(torch.zeros(1, num_prompts, 1024))
            nn.init.normal_(self.prompt_embeddings, std=0.02)
            self.prompt_dropout = nn.Dropout(prompt_dropout)

        elif adaptation == "vpt_deep":
            # Visual Prompt Tuning (Deep): Learnable prompts at every transformer layer
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.prompt_embeddings = nn.Parameter(torch.zeros(len(self.encoder.blocks), num_prompts, 1024))
            nn.init.normal_(self.prompt_embeddings, std=0.02)
            self.prompt_dropout = nn.Dropout(prompt_dropout)

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
        if self.adaptation in ("vpt", "vpt_shallow", "vpt_deep"):
            B = x.shape[0]
            if len(x.shape) == 4 and self.encoder.patch_embed.input_size[0] == 1:
                x_in = x.unsqueeze(2)
            else:
                x_in = x
            sample_shape = x_in.shape[-3:]

            # 1. Patch embeddings
            patch_tokens = self.encoder.patch_embed(x_in)
            pos_embed = self.encoder.interpolate_pos_encoding(sample_shape)
            patch_tokens = patch_tokens + pos_embed[:, 1:, :]

            # 2. CLS token
            cls_token = self.encoder.cls_token + pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)

            # 3. Apply VPT forward pass
            if self.adaptation in ("vpt", "vpt_shallow"):
                prompts = self.prompt_dropout(self.prompt_embeddings).expand(B, -1, -1)
                tokens = torch.cat((cls_tokens, prompts, patch_tokens), dim=1)
                for block in self.encoder.blocks:
                    tokens = block(tokens)
                tokens = self.encoder.norm(tokens)
            else:  # vpt_deep
                tokens = torch.cat((cls_tokens, self.prompt_dropout(self.prompt_embeddings[0]).expand(B, -1, -1), patch_tokens), dim=1)
                for i, block in enumerate(self.encoder.blocks):
                    if i > 0:
                        layer_prompts = self.prompt_dropout(self.prompt_embeddings[i]).expand(B, -1, -1)
                        tokens = torch.cat((tokens[:, :1, :], layer_prompts, tokens[:, (1 + self.num_prompts):, :]), dim=1)
                    tokens = block(tokens)
                tokens = self.encoder.norm(tokens)

            # 4. Extract spatial patch tokens (skipping CLS and prompt tokens)
            spatial_tokens = tokens[:, (1 + self.num_prompts):, :]
            h = int(np.sqrt(spatial_tokens.shape[1]))
            spatial_feat = rearrange(spatial_tokens, "b (h w) c -> b c h w", h=h, w=h)
            return self.decoder(spatial_feat)

        else:
            feats = self.encoder.forward_features(x)
            final_feat = [feats[23]]
            spatial_feat = self.encoder.prepare_features_for_image_model(final_feat)[0]
            return self.decoder(spatial_feat)

