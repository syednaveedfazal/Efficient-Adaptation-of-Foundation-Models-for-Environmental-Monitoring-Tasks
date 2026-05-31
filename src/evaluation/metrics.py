"""
src/evaluation/metrics.py — Shared segmentation metrics.

Used during training (validation_step) and standalone evaluation.
All models log the same metrics so results are directly comparable.
"""

import torch


def iou_per_class(
    preds:       torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """
    Compute IoU for each class independently.

    Args:
        preds:   (B, H, W) predicted class indices (argmax of logits)
        targets: (B, H, W) ground-truth class indices
    Returns:
        (num_classes,) tensor of per-class IoU values
    """
    ious = []
    for c in range(num_classes):
        pred_c = (preds   == c)
        true_c = (targets == c)
        inter  = (pred_c & true_c).sum().float()
        union  = (pred_c | true_c).sum().float()
        ious.append(inter / (union + 1e-6))
    return torch.stack(ious)


def compute_metrics(
    logits:      torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int = 2,
) -> dict:
    """
    Convenience wrapper: takes raw logits, returns a dict of scalar metrics.

    Returns:
        {
            "mean_iou": float,   # mean IoU across all classes
            "burn_iou": float,   # IoU of burn-scar class (class index 1)
            "bg_iou":   float,   # IoU of background class (class index 0)
        }
    """
    preds = logits.argmax(dim=1)
    ious  = iou_per_class(preds, targets, num_classes)
    return {
        "mean_iou": ious.mean(),
        "burn_iou": ious[1],     # class 1 = burn scar
        "bg_iou":   ious[0],     # class 0 = not burned
    }