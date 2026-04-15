"""
Test-Time Augmentation (TTA) for inference.

Instead of predicting once on the raw image, we predict on several
augmented versions and average the probability maps. This consistently
improves IoU/F1 at no extra training cost.

Augmentations used: original + horizontal flip + vertical flip + 90° rotations.
"""

import torch
import torch.nn.functional as F


def _hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def _vflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-2])


def _rot90(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.rot90(x, k, dims=[-2, -1])


@torch.no_grad()
def tta_predict(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Run TTA inference and return averaged probability map.

    Args:
        model:  Trained segmentation model (returns raw logits).
        images: Batch of input images (B, C, H, W).

    Returns:
        Averaged sigmoid probability map (B, 1, H, W).
    """
    model.eval()
    preds = []

    transforms = [
        (lambda x: x,              lambda x: x),              # original
        (_hflip,                   _hflip),                   # h-flip
        (_vflip,                   _vflip),                   # v-flip
        (lambda x: _rot90(x, 1),   lambda x: _rot90(x, 3)),  # rot 90 → undo
        (lambda x: _rot90(x, 2),   lambda x: _rot90(x, 2)),  # rot 180
        (lambda x: _rot90(x, 3),   lambda x: _rot90(x, 1)),  # rot 270 → undo
    ]

    for fwd, inv in transforms:
        aug_images = fwd(images)
        logits     = model(aug_images)
        prob       = torch.sigmoid(logits)
        preds.append(inv(prob))

    # Average probability maps → convert back to logit space
    avg_prob = torch.stack(preds, dim=0).mean(dim=0)
    return avg_prob   # (B, 1, H, W) — probabilities, not logits
