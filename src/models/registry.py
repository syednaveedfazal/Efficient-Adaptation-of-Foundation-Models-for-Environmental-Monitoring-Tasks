"""
src/models/registry.py — Central model registry.

To add a new model, teammates only need to:
  1. Create src/models/<their_model>.py
  2. Import it and add one line to MODEL_REGISTRY below
  3. Set model.name in their config YAML

No changes needed to train.py, module.py, or any shared code.
"""

from src.models.unet import UNet
from src.models.prithvi_seg import PrithviSegmentor
from src.models.prithvi_unet import PrithviUNetSegmentor
from src.models.prithvi_fcn import PrithviFCNSegmentor

MODEL_REGISTRY: dict = {
    "unet_scratch":        UNet,
    "prithvi_seg_lora_r8": PrithviSegmentor,
    "prithvi_seg_lora_r16":PrithviSegmentor,
    "prithvi_unet_lora_r8":PrithviUNetSegmentor,
    "prithvi_unet_lora_r16":PrithviUNetSegmentor,
    "prithvi_fcn_lora_r8": PrithviFCNSegmentor,
    "prithvi_fcn_lora_r16":PrithviFCNSegmentor,
    "prithvi_fcn_full_ft": PrithviFCNSegmentor,
    "prithvi_fcn_linear_probe": PrithviFCNSegmentor,
    "prithvi_fcn_randomized":   PrithviFCNSegmentor,
}


def build_model(cfg: dict):
    """
    Instantiate a model from the 'model' section of a config dict.

    Config shape expected:
        model:
          name:   unet          # key in MODEL_REGISTRY
          params:               # kwargs passed directly to the model __init__
            in_channels: 6
            num_classes: 2
            ...

    Raises:
        KeyError if model.name is not in the registry.
    """
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        registered = list(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{name}' not found in registry. "
            f"Registered models: {registered}"
        )
    model_cls = MODEL_REGISTRY[name]
    return model_cls(**cfg["params"])