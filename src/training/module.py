"""
src/training/module.py — Model-agnostic PyTorch Lightning module.

Works with any model registered in src/models/registry.py.
Teammates swap models by changing model.name in their config —
nothing here needs to change.
"""

import torch
import pytorch_lightning as pl

from src.models.registry import build_model
from src.training.losses import build_loss
from src.evaluation.metrics import compute_metrics      # lives in evaluation/


class SegmentationModule(pl.LightningModule):
    """
    Shared Lightning module for all segmentation models.

    Handles:
      - Model instantiation via registry
      - Combined CE + Dice loss
      - mean_iou / burn_iou / bg_iou logging
      - AdamW + CosineAnnealing LR schedule
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        # Model — driven entirely by config, not hardcoded
        self.model = build_model(cfg["model"])

        # Loss — shared across all models
        self.criterion = build_loss(cfg["loss"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        imgs, masks = batch
        logits  = self(imgs)                                              # (B, C, H, W)
        loss    = self.criterion(logits, masks)
        metrics = compute_metrics(logits, masks, self.cfg["model"]["params"]["num_classes"])

        on_step = (stage == "train")
        self.log(f"{stage}/loss",     loss,                on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/mean_iou", metrics["mean_iou"], on_step=False,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}/burn_iou", metrics["burn_iou"], on_step=False,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}/bg_iou",   metrics["bg_iou"],   on_step=False,   on_epoch=True)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = self.cfg["optimizer"]
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr           = opt["lr"],
            weight_decay = opt["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = self.cfg["trainer"]["max_epochs"],
            eta_min = opt["lr"] * 0.01,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}