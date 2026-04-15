"""
Advanced loss functions for binary segmentation.

These go beyond standard BCE to better handle:
  - Class imbalance (soil >> wheat pixels in some images)
  - Boundary precision (Tversky biases toward recall, useful for thin structures)
  - Combined objectives (Combo loss)

Reference:
    Lin et al. "Focal Loss for Dense Object Detection." ICCV 2017.
    Salehi et al. "Tversky Loss Function for Image Segmentation Using 3D
    Fully Convolutional Deep Networks." MICCAI 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss — directly optimises the F1/Dice overlap metric."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum()
        return 1 - (2 * inter + self.eps) / (pred.sum() + target.sum() + self.eps)


class FocalLoss(nn.Module):
    """
    Focal Loss — down-weights easy negatives, focuses on hard examples.
    Particularly useful when soil/background dominates the image.

    Args:
        alpha: Class balance weight for the positive class (wheat).
        gamma: Focusing parameter (2.0 recommended by Lin et al.).
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t  = torch.exp(-bce)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss — generalisation of Dice that controls the trade-off
    between false positives and false negatives via alpha/beta.
    Setting alpha=0.3, beta=0.7 penalises false negatives more, improving
    recall — useful for segmenting thin/sparse wheat structures.

    Args:
        alpha: Weight for false positives.
        beta:  Weight for false negatives.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        return 1 - (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)


class ComboLoss(nn.Module):
    """
    Combo Loss = weighted BCE + Dice.
    Simple but consistently strong baseline for binary segmentation.

    Args:
        bce_weight: Weight for BCE component (1 - bce_weight goes to Dice).
    """

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + \
               (1 - self.bce_weight) * self.dice(pred, target)


class FocalDiceLoss(nn.Module):
    """
    Focal + Dice combined.
    Addresses class imbalance (Focal) and optimises overlap (Dice) simultaneously.
    Often the strongest choice for agricultural segmentation.

    Args:
        focal_weight: Weight for Focal component.
        alpha:        Focal alpha parameter.
        gamma:        Focal gamma parameter.
    """

    def __init__(self, focal_weight: float = 0.5, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice  = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_weight * self.focal(pred, target) + \
               (1 - self.focal_weight) * self.dice(pred, target)


# Loss factory — select by name
LOSS_REGISTRY = {
    "combo":       ComboLoss,
    "focal_dice":  FocalDiceLoss,
    "tversky":     TverskyLoss,
    "focal":       FocalLoss,
    "dice":        DiceLoss,
    "bce":         nn.BCEWithLogitsLoss,
}


def get_loss(name: str, **kwargs) -> nn.Module:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name](**kwargs)
