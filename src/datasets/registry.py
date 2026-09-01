"""
src/datasets/registry.py — Central dataset registry.

Mirrors src/models/registry.py. To add a dataset, teammates only need to:
  1. Create src/datasets/<their_dataset>.py with a LightningDataModule that
     accepts a `split_json=` kwarg (for label-budget subsampling).
  2. Import it and add one line to DATASET_REGISTRY below.
  3. Set data.name in their config YAML.

No changes needed to train.py or any shared code.
"""

from src.datasets.burn_scar import BurnScarDataModule
from src.datasets.eurosat import EuroSATDataModule

DATASET_REGISTRY: dict = {
    "burn_scar": BurnScarDataModule,
    "eurosat":   EuroSATDataModule,
}


def build_datamodule(data_cfg: dict, split_json: str = None):
    """
    Instantiate a LightningDataModule from the 'data' section of a config.

    Two accepted config shapes:
        # New (name + params)
        data:
          name: eurosat
          params: {raw_dir: ..., batch_size: ..., ...}

        # Legacy flat block (existing burn-scar configs) — treated as burn_scar
        data:
          raw_dir: ...
          stats_path: ...
          batch_size: ...

    `split_json` (the label-budget JSON from the CLI) is forwarded to the
    datamodule so every dataset shares the same budget-subsampling mechanism.
    """
    name = data_cfg.get("name", "burn_scar")
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f"Dataset '{name}' not found in registry. "
            f"Registered datasets: {list(DATASET_REGISTRY.keys())}"
        )

    if "params" in data_cfg:
        params = dict(data_cfg["params"])
    else:
        # Legacy flat block: pass every key except the (optional) name through.
        params = {k: v for k, v in data_cfg.items() if k != "name"}

    return DATASET_REGISTRY[name](split_json=split_json, **params)
