"""
src/training/module.py — Model-agnostic PyTorch Lightning module.

Works with any model registered in src/models/registry.py.
Teammates swap models by changing model.name in their config —
nothing here needs to change.
"""

import math
import time

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from src.models.registry import build_model
from src.training.losses import build_loss
from src.evaluation.metrics import compute_metrics, classification_metrics  # lives in evaluation/


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

        # Optional self-distillation from a frozen teacher (SCOPE v2): the warm-start LoRA model
        self.teacher = None
        self.distill = bool(cfg.get("distill", False))
        if self.distill:
            self.distill_T = float(cfg.get("distill_T", 2.0))
            self.distill_weight = float(cfg.get("distill_weight", 1.0))
            self.feat_distill = float(cfg.get("feat_distill", 0.0))
            self.teacher = build_teacher(cfg)
            self._teacher_time = 0.0
            self._epoch_start = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()   # BatchNorm in the teacher head must stay frozen
        return self

    def on_save_checkpoint(self, checkpoint):
        # The teacher is a fixed copy of the warm-start checkpoint; don't store it again
        if self.teacher is not None:
            sd = checkpoint["state_dict"]
            for k in [k for k in sd if k.startswith("teacher.")]:
                del sd[k]

    def on_load_checkpoint(self, checkpoint):
        if self.teacher is not None:
            sd = checkpoint["state_dict"]
            for k, v in self.teacher.state_dict().items():
                sd.setdefault(f"teacher.{k}", v)

    def _distill_loss(self, imgs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if imgs.is_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            t_logits, t_feats = teacher_forward(self.teacher, imgs)
        if imgs.is_cuda:
            torch.cuda.synchronize()
        self._teacher_time += time.time() - t0

        T = self.distill_T
        kl = F.kl_div(
            F.log_softmax(logits.float() / T, dim=1),
            F.softmax(t_logits.float() / T, dim=1),
            reduction="none",
        ).sum(dim=1).mean() * (T * T)
        loss = self.distill_weight * kl
        self.log("train/distill_kl", kl, on_step=False, on_epoch=True)
        s_feats = getattr(self.model, "last_spatial_features", None)
        if self.feat_distill > 0 and s_feats is not None:
            mse = F.mse_loss(s_feats.float(), t_feats.float())
            loss = loss + self.feat_distill * mse
            self.log("train/feat_mse", mse, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_start(self):
        if self.teacher is not None:
            self._teacher_time = 0.0
            self._epoch_start = time.time()

    def on_train_epoch_end(self):
        if self.teacher is not None and self._epoch_start is not None:
            self.log("train/teacher_time_s", self._teacher_time)
            self.log("train/epoch_time_s", time.time() - self._epoch_start)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        imgs, masks = batch
        if hasattr(self.model, "set_current_targets") and stage == "train":
            self.model.set_current_targets(masks)
        logits  = self(imgs)                                              # (B, C, H, W)
        loss    = self.criterion(logits, masks)
        if hasattr(self.model, "get_aux_loss") and stage == "train":
            aux_loss = self.model.get_aux_loss(masks=masks)
            loss = loss + aux_loss
            self.log(f"{stage}/aux_loss", aux_loss, on_step=True, on_epoch=True)
        if self.teacher is not None and stage == "train":
            loss = loss + self._distill_loss(imgs, logits)
        metrics = compute_metrics(logits, masks, self.cfg["model"]["params"]["num_classes"])

        on_step = (stage == "train")
        self.log(f"{stage}/loss",           loss,                      on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/mean_iou",       metrics["mean_iou"],       on_step=False,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}/burn_iou",       metrics["burn_iou"],       on_step=False,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}/bg_iou",         metrics["bg_iou"],         on_step=False,   on_epoch=True)
        self.log(f"{stage}/burn_precision", metrics["burn_precision"], on_step=False,   on_epoch=True)
        self.log(f"{stage}/burn_recall",    metrics["burn_recall"],    on_step=False,   on_epoch=True)
        self.log(f"{stage}/burn_dice",      metrics["burn_dice"],      on_step=False,   on_epoch=True)
        self.log(f"{stage}/pixel_accuracy", metrics["pixel_accuracy"], on_step=False,   on_epoch=True)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        opt = self.cfg["optimizer"]
        if "lr_new" in opt and hasattr(self.model, "param_groups"):
            # SCOPE v2 Stage B: new parts at lr_new, LoRA + head at lr_lora; cosine to 1% of each
            optimizer = torch.optim.AdamW(
                self.model.param_groups(opt["lr_new"], opt["lr_lora"]),
                weight_decay = opt["weight_decay"],
            )
            T_max = self.cfg["trainer"]["max_epochs"]
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda e: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(e, T_max) / T_max))
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
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


def build_teacher(cfg: dict) -> torch.nn.Module:
    """Frozen plain LoRA r=8 teacher loaded from the SCOPE warm-start checkpoint."""
    from src.models.prithvi_fcn import PrithviFCNSegmentor
    params = cfg["model"]["params"]
    ckpt_path = params.get("init_from")
    if not ckpt_path:
        raise ValueError("distill: true needs model.params.init_from (the teacher checkpoint)")
    teacher = PrithviFCNSegmentor(
        weights_path = params["weights_path"],
        num_classes  = params["num_classes"],
        adaptation   = "lora",
        lora_rank    = params.get("lora_rank", 8),
        lora_alpha   = params.get("lora_alpha", 8),
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    teacher.load_state_dict(sd, strict=True)
    for prm in teacher.parameters():
        prm.requires_grad = False
    return teacher.eval()


def teacher_forward(teacher: torch.nn.Module, x: torch.Tensor):
    """PrithviFCNSegmentor.forward that also returns the (B, 1024, 32, 32) encoder features."""
    feats = teacher.encoder.forward_features(x)
    spatial = teacher.encoder.prepare_features_for_image_model([feats[23]])[0]
    return teacher.decoder(spatial), spatial


class ClassificationModule(pl.LightningModule):
    """
    Shared Lightning module for single-label image classification (EuroSAT).

    Mirrors SegmentationModule but for (imgs, labels) batches with (B, C) logits:
      - Model instantiation via registry
      - CrossEntropy loss (loss.type == "cross_entropy")
      - accuracy / macro_f1 / macro_precision / macro_recall logging
      - AdamW + CosineAnnealing LR schedule (identical to SegmentationModule)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model     = build_model(cfg["model"])
        self.criterion = build_loss(cfg["loss"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        imgs, labels = batch
        logits  = self(imgs)                                              # (B, C)
        loss    = self.criterion(logits, labels)
        metrics = classification_metrics(logits, labels, self.cfg["model"]["params"]["num_classes"])

        on_step = (stage == "train")
        self.log(f"{stage}/loss",            loss,                       on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/accuracy",        metrics["accuracy"],        on_step=False,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}/macro_f1",        metrics["macro_f1"],        on_step=False,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}/macro_precision", metrics["macro_precision"], on_step=False,   on_epoch=True)
        self.log(f"{stage}/macro_recall",    metrics["macro_recall"],    on_step=False,   on_epoch=True)
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


def build_module(cfg: dict) -> pl.LightningModule:
    """
    Task factory: pick the Lightning module from cfg['task'].
    Defaults to 'segmentation' so existing burn-scar configs are unchanged.
    """
    task = cfg.get("task", "segmentation")
    if task == "classification":
        return ClassificationModule(cfg)
    if task == "segmentation":
        return SegmentationModule(cfg)
    raise ValueError(f"Unknown task '{task}' (expected 'segmentation' or 'classification')")