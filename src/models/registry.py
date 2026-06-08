"""
src/models/registry.py — Central model registry.

To add a new model, teammates only need to:
  1. Create src/models/<their_model>.py
  2. Import it and add one line to MODEL_REGISTRY below
  3. Set model.name in their config YAML

No changes needed to train.py, module.py, or any shared code.
"""

import re

from src.models.unet import UNet
from src.models.prithvi import PrithviSegmentor

MODEL_REGISTRY: dict = {
    "unet":         UNet,
    "prithvi_lora": PrithviSegmentor,
}


def _resolve_model_class(name: str):
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]

    if re.fullmatch(r"prithvi_lora_r\d+", name):
        return PrithviSegmentor

    registered = list(MODEL_REGISTRY.keys()) + ["prithvi_lora_r<rank>"]
    raise KeyError(
        f"Model '{name}' not found in registry. "
        f"Registered models: {registered}"
    )


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
    model_cls = _resolve_model_class(name)
    return model_cls(**cfg["params"])
