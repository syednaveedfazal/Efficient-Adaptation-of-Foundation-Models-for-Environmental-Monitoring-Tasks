"""
src/models/dinov2_fcn.py — DINOv2-Large (generic ViT, ~300M) backbone + FCN decoder.

DINOv2 is a *generic* self-supervised ViT pretrained by Meta on natural RGB
images (3 bands). Our HLS burn-scar scenes are 6-band multispectral. To keep
DINOv2 in its native modality (and give it a fair, unmodified-backbone shot as
the generic-model baseline against geospatial-native Prithvi), we feed it only
the 3 RGB bands of each scene:

    HLS band order : [Blue, Green, Red, NIR, SWIR1, SWIR2]  (indices 0..5)
    RGB for DINOv2 : [Red, Green, Blue] = indices [2, 1, 0]

The datamodule is unchanged (still loads all 6 z-scored bands); this model
selects the RGB subset internally, so nothing else in the pipeline changes.

Two remaining details:
  * 512×512 / patch16 → 518×518 / patch14 : DINOv2 uses a 14-px patch and was
    trained at 518 (37×37 patches). Input is bilinearly resized to 518 so the
    pretrained positional embeddings apply exactly (no interpolation needed).

The weights ship in HuggingFace format (models/pretrained/dinov2/*.safetensors)
but `transformers.Dinov2Model` is unusable in this env (transformers disables
PyTorch under torch<2.4). We therefore rebuild the identical architecture with
timm (`vit_base_patch14_dinov2`) — which also gives clean LoRA target names
(attn.qkv / attn.proj / mlp.fc1 / mlp.fc2), matching the Prithvi convention —
and load the HF weights via a small key remapper.

Adaptation strategies (selected via the `adaptation` config field), mirroring
src/models/prithvi_fcn.py:
    - "lora"         : freeze backbone, train LoRA adapters + decoder
    - "full_ft"      : train the whole backbone + decoder
    - "linear_probe" : freeze backbone entirely, train decoder only
    - "randomized"   : random-init backbone (no pretrained weights), full train
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file

from peft import LoraConfig, inject_adapter_in_model


DINOV2_DIR = Path(__file__).resolve().parents[2] / "models" / "pretrained" / "dinov2"

# HLS band order is [Blue, Green, Red, NIR, SWIR1, SWIR2]; DINOv2 wants R, G, B.
DEFAULT_RGB_INDICES = (2, 1, 0)


def _remap_hf_to_timm(hf_sd: dict, num_layers: int) -> dict:
    """
    Remap facebook/dinov2-base (HuggingFace Dinov2Model) state_dict keys to the
    timm `vit_base_patch14_dinov2` namespace.
    """
    out = {}
    out["cls_token"] = hf_sd["embeddings.cls_token"]
    out["pos_embed"] = hf_sd["embeddings.position_embeddings"]
    out["patch_embed.proj.weight"] = hf_sd["embeddings.patch_embeddings.projection.weight"]
    out["patch_embed.proj.bias"]   = hf_sd["embeddings.patch_embeddings.projection.bias"]

    for i in range(num_layers):
        h = f"encoder.layer.{i}."
        t = f"blocks.{i}."

        out[t + "norm1.weight"] = hf_sd[h + "norm1.weight"]
        out[t + "norm1.bias"]   = hf_sd[h + "norm1.bias"]

        # HF keeps q, k, v separate; timm fuses them into one (3*dim, dim) matrix.
        q = hf_sd[h + "attention.attention.query.weight"]
        k = hf_sd[h + "attention.attention.key.weight"]
        v = hf_sd[h + "attention.attention.value.weight"]
        out[t + "attn.qkv.weight"] = torch.cat([q, k, v], dim=0)
        qb = hf_sd[h + "attention.attention.query.bias"]
        kb = hf_sd[h + "attention.attention.key.bias"]
        vb = hf_sd[h + "attention.attention.value.bias"]
        out[t + "attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)

        out[t + "attn.proj.weight"] = hf_sd[h + "attention.output.dense.weight"]
        out[t + "attn.proj.bias"]   = hf_sd[h + "attention.output.dense.bias"]

        out[t + "ls1.gamma"] = hf_sd[h + "layer_scale1.lambda1"]
        out[t + "ls2.gamma"] = hf_sd[h + "layer_scale2.lambda1"]

        out[t + "norm2.weight"] = hf_sd[h + "norm2.weight"]
        out[t + "norm2.bias"]   = hf_sd[h + "norm2.bias"]

        out[t + "mlp.fc1.weight"] = hf_sd[h + "mlp.fc1.weight"]
        out[t + "mlp.fc1.bias"]   = hf_sd[h + "mlp.fc1.bias"]
        out[t + "mlp.fc2.weight"] = hf_sd[h + "mlp.fc2.weight"]
        out[t + "mlp.fc2.bias"]   = hf_sd[h + "mlp.fc2.bias"]

    out["norm.weight"] = hf_sd["layernorm.weight"]
    out["norm.bias"]   = hf_sd["layernorm.bias"]
    return out


class DinoV2Backbone(nn.Module):
    """
    DINOv2-Large ViT on RGB input. Exposes forward_features(x) that takes the
    full (B, 6, 512, 512) scene, selects the 3 RGB bands, and returns a spatial
    feature map (B, 1024, 37, 37) — so the rest of the pipeline (and
    src/utils/flops.py) treats it exactly like the Prithvi encoder.

    DINOv2-Large (ViT-L/14, ~300M params) is used so the generic-ViT baseline
    matches Prithvi-EO-2.0-300M in parameter scale for a fair comparison.
    """

    IMG_SIZE   = 518    # 37 * 14 — native DINOv2 resolution, no pos-embed interpolation
    EMBED_DIM  = 1024   # DINOv2-Large hidden size (24 layers, 16 heads)
    TIMM_MODEL = "vit_large_patch14_dinov2"

    def __init__(self, rgb_indices=DEFAULT_RGB_INDICES, randomized: bool = False):
        super().__init__()
        self.register_buffer("rgb_indices", torch.tensor(list(rgb_indices), dtype=torch.long))
        self.vit = timm.create_model(
            self.TIMM_MODEL,
            pretrained  = False,
            num_classes = 0,
            img_size    = self.IMG_SIZE,
            in_chans    = 3,
        )
        self.num_prefix = self.vit.num_prefix_tokens   # 1 (cls token; no registers)

        if not randomized:
            self._load_pretrained()
        else:
            print("[DINOv2] Randomized mode — using randomly initialised weights")

    def _load_pretrained(self):
        weights_path = DINOV2_DIR / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"DINOv2 weights not found at {weights_path}. "
                "Run: python scripts/download_models.py --model dinov2"
            )
        hf_sd   = load_file(str(weights_path))
        timm_sd = _remap_hf_to_timm(hf_sd, num_layers=len(self.vit.blocks))
        missing, unexpected = self.vit.load_state_dict(timm_sd, strict=False)
        print(f"[DINOv2] Loaded pretrained RGB weights from {weights_path.name}")
        print(f"         Missing keys  : {len(missing)}")
        print(f"         Unexpected keys: {len(unexpected)}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 6, H, W) → select RGB → resize to native 518 → ViT → (B, 768, 37, 37)
        x = x.index_select(1, self.rgb_indices)                    # (B, 3, H, W)
        x = F.interpolate(x, size=(self.IMG_SIZE, self.IMG_SIZE),
                          mode="bilinear", align_corners=False)
        tokens = self.vit.forward_features(x)                      # (B, 1 + N, 768)
        patch  = tokens[:, self.num_prefix:, :]                    # (B, N, 768) — drop cls
        B, N, C = patch.shape
        hw = int(N ** 0.5)                                         # 37
        return patch.transpose(1, 2).reshape(B, C, hw, hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class DinoV2FCNDecoder(nn.Module):
    """
    FCN head: project bottleneck features and bilinearly upsample to full res.
    Mirrors src/models/prithvi_fcn.py's PrithviFCNDecoder.

    Input:  (B, 768, 37, 37)
    Output: (B, num_classes, 512, 512)
    """

    def __init__(self, in_channels: int = 768, num_classes: int = 2, out_size: int = 512):
        super().__init__()
        self.out_size = out_size
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)                          # (B, num_classes, 37, 37)
        return F.interpolate(logits, size=(self.out_size, self.out_size),
                             mode="bilinear", align_corners=True)


class DinoV2FCNSegmentor(nn.Module):
    """
    DINOv2-Base backbone (RGB) with a selectable adaptation strategy and an FCN
    segmentation decoder.

    Args:
        num_classes:  Number of output classes (2 for binary burn-scar).
        adaptation:   "lora" | "full_ft" | "linear_probe" | "randomized".
        lora_rank:    LoRA rank r (only used when adaptation == "lora").
        lora_alpha:   LoRA alpha scaling (scale = alpha / rank).
        rgb_indices:  Which of the 6 input bands feed DINOv2, as (R, G, B).
    """

    def __init__(
        self,
        num_classes: int = 2,
        adaptation:  str = "lora",
        lora_rank:   int = 8,
        lora_alpha:  int = 8,
        rgb_indices=DEFAULT_RGB_INDICES,
    ):
        super().__init__()

        # ── 1. Backbone (RGB) ─────────────────────────────────────────────
        randomized = (adaptation == "randomized")
        self.encoder = DinoV2Backbone(rgb_indices=rgb_indices, randomized=randomized)

        # ── 2. Adaptation strategy ────────────────────────────────────────
        if adaptation == "lora":
            lora_cfg = LoraConfig(
                r              = lora_rank,
                lora_alpha     = lora_alpha,
                target_modules = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
                bias           = "none",
                lora_dropout   = 0.0,
            )
            inject_adapter_in_model(lora_cfg, self.encoder)
            for name, param in self.encoder.named_parameters():
                param.requires_grad = "lora_" in name

        elif adaptation in ("full_ft", "randomized"):
            for param in self.encoder.parameters():
                param.requires_grad = True

        elif adaptation == "linear_probe":
            for param in self.encoder.parameters():
                param.requires_grad = False

        else:
            raise ValueError(f"Unknown adaptation strategy: {adaptation}")

        # ── 3. FCN decoder (always trained) ───────────────────────────────
        self.decoder = DinoV2FCNDecoder(
            in_channels=DinoV2Backbone.EMBED_DIM, num_classes=num_classes
        )

        self._print_param_summary()

    def _print_param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[DINOv2-FCN] Parameters — total: {total:,}  |  trainable: {trainable:,} "
              f"({100 * trainable / total:.2f}%)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, H, W) normalized satellite image (RGB bands selected internally)
        Returns:
            (B, num_classes, H, W) raw logits
        """
        spatial_feat = self.encoder.forward_features(x)   # (B, 768, 37, 37)
        return self.decoder(spatial_feat)
