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


def precision_per_class(
    preds:       torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """
    Compute Precision for each class independently.
    """
    precisions = []
    for c in range(num_classes):
        pred_c = (preds   == c)
        true_c = (targets == c)
        inter  = (pred_c & true_c).sum().float()
        total_pred = pred_c.sum().float()
        precisions.append(inter / (total_pred + 1e-6))
    return torch.stack(precisions)


def recall_per_class(
    preds:       torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """
    Compute Recall (Sensitivity) for each class independently.
    """
    recalls = []
    for c in range(num_classes):
        pred_c = (preds   == c)
        true_c = (targets == c)
        inter  = (pred_c & true_c).sum().float()
        total_true = true_c.sum().float()
        recalls.append(inter / (total_true + 1e-6))
    return torch.stack(recalls)


def dice_per_class(
    preds:       torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """
    Compute F1-score / Dice coefficient for each class independently.
    """
    dices = []
    for c in range(num_classes):
        pred_c = (preds   == c)
        true_c = (targets == c)
        inter  = (pred_c & true_c).sum().float()
        total = pred_c.sum().float() + true_c.sum().float()
        dices.append((2.0 * inter) / (total + 1e-6))
    return torch.stack(dices)


def compute_metrics(
    logits:      torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int = 2,
) -> dict:
    """
    Convenience wrapper: takes raw logits, returns a dict of scalar metrics.

    Returns:
        {
            "mean_iou": float,         # mean IoU across all classes
            "burn_iou": float,         # IoU of burn-scar class (class index 1)
            "bg_iou":   float,         # IoU of background class (class index 0)
            "burn_precision": float,   # Precision of burn-scar class (class index 1)
            "burn_recall": float,      # Recall of burn-scar class (class index 1)
            "burn_dice": float,        # Dice coefficient of burn-scar class (class index 1)
            "pixel_accuracy": float,   # Overall pixel accuracy
        }
    """
    preds = logits.argmax(dim=1)
    
    ious = iou_per_class(preds, targets, num_classes)
    precisions = precision_per_class(preds, targets, num_classes)
    recalls = recall_per_class(preds, targets, num_classes)
    dices = dice_per_class(preds, targets, num_classes)
    
    pixel_acc = (preds == targets).sum().float() / (targets.numel() + 1e-6)
    
    return {
        "mean_iou": ious.mean(),
        "burn_iou": ious[1],         # class 1 = burn scar
        "bg_iou":   ious[0],         # class 0 = not burned
        "burn_precision": precisions[1],
        "burn_recall": recalls[1],
        "burn_dice": dices[1],
        "pixel_accuracy": pixel_acc,
    }


# ----------------------------------------------------------------------------
# Classification metrics (EuroSAT land-cover, single label per image)
# ----------------------------------------------------------------------------

def classification_metrics(
    logits:      torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int = 10,
) -> dict:
    """
    Image-level classification metrics from raw logits.

    Args:
        logits:  (B, num_classes) raw class scores
        targets: (B,) long tensor of ground-truth class indices
    Returns:
        {
            "accuracy":        overall top-1 accuracy,
            "macro_f1":        unweighted mean per-class F1,
            "macro_precision": unweighted mean per-class precision,
            "macro_recall":    unweighted mean per-class recall,
        }
    All values are class-frequency-agnostic (macro) so a dominant class cannot
    inflate the score — the classification analogue of using IoU/Dice (not
    pixel accuracy) for the imbalanced segmentation task.
    """
    preds = logits.argmax(dim=1)
    accuracy = (preds == targets).float().mean()

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        pred_c = (preds   == c)
        true_c = (targets == c)
        tp = (pred_c & true_c).sum().float()
        fp = (pred_c & ~true_c).sum().float()
        fn = (~pred_c & true_c).sum().float()
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f = 2 * p * r / (p + r + 1e-6)
        precisions.append(p); recalls.append(r); f1s.append(f)

    return {
        "accuracy":        accuracy,
        "macro_f1":        torch.stack(f1s).mean(),
        "macro_precision": torch.stack(precisions).mean(),
        "macro_recall":    torch.stack(recalls).mean(),
    }