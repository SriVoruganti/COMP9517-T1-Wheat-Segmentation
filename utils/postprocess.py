"""
Morphological post-processing for predicted segmentation masks.

Motivation (insight from failure analysis in evaluate.py):
    Failure analysis reveals two recurring error patterns:
      1. Small isolated predicted regions far from true wheat areas
         — these inflate false positives and lower precision.
      2. Small holes inside correctly predicted wheat regions
         — these create false negatives and lower recall.

    Both are structural artefacts of the pixel-level prediction:
    the model occasionally fires on isolated soil patches that share
    colour/texture with wheat, and misses small interior regions that
    are obscured or ambiguous.

Post-processing pipeline:
    Step 1 — Remove small connected components (min_area threshold).
             Connected components with area < min_area are discarded.
             Eliminates isolated false-positive specks.

    Step 2 — Morphological closing (dilation then erosion).
             Fills small holes within predicted wheat regions.
             Recovers missed interior pixels (false negatives).

This is applied AFTER thresholding the model's sigmoid output and
operates purely on the binary mask — no retraining required.

The min_area and close_kernel parameters can be tuned on the
validation set before applying to the test set.
"""

import cv2
import numpy as np


def postprocess_mask(
    mask:         np.ndarray,
    min_area:     int = 200,
    close_kernel: int = 5,
) -> np.ndarray:
    """
    Clean a predicted binary mask using morphological operations.

    Args:
        mask:         H×W binary array (0/1 float32 or uint8).
        min_area:     Minimum connected-component area in pixels to keep.
                      Components smaller than this are removed as noise.
                      Tune on validation set (default 200 ≈ ~0.16% of 350×350).
        close_kernel: Side length of the square structuring element for
                      morphological closing. Larger values fill bigger holes
                      but may merge nearby wheat regions.

    Returns:
        Cleaned H×W float32 binary mask (values 0.0 or 1.0).
    """
    mask_u8 = (mask > 0.5).astype(np.uint8)

    # --- Step 1: Remove small connected components ---
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    cleaned = np.zeros_like(mask_u8)
    for i in range(1, n_labels):   # skip label 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    # --- Step 2: Morphological closing — fill small holes ---
    kernel  = np.ones((close_kernel, close_kernel), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned.astype(np.float32)


def postprocess_batch(
    preds:        np.ndarray,
    min_area:     int = 200,
    close_kernel: int = 5,
) -> np.ndarray:
    """
    Apply postprocess_mask to a batch of predictions.

    Args:
        preds: (B, H, W) or (B, 1, H, W) binary array.

    Returns:
        Post-processed array of the same shape.
    """
    squeeze = False
    if preds.ndim == 4:
        preds   = preds[:, 0]   # (B, 1, H, W) → (B, H, W)
        squeeze = True

    result = np.stack([
        postprocess_mask(preds[i], min_area, close_kernel)
        for i in range(len(preds))
    ])
    return result[:, np.newaxis] if squeeze else result


def tune_postprocess(
    model,
    val_dataset,
    device,
    min_areas:     list = None,
    close_kernels: list = None,
) -> dict:
    """
    Grid-search the best (min_area, close_kernel) pair on the validation set.

    Args:
        model:         Trained segmentation model.
        val_dataset:   EWSDataset for the validation split.
        device:        torch.device.
        min_areas:     List of min_area values to try.
        close_kernels: List of close_kernel values to try.

    Returns:
        dict with 'best_min_area', 'best_close_kernel', 'best_iou',
        and 'grid' (full results).
    """
    import torch
    from utils.metrics import iou_score

    if min_areas     is None: min_areas     = [100, 200, 500, 1000]
    if close_kernels is None: close_kernels = [3, 5, 7]

    model.eval()
    best_iou    = -1.0
    best_params = {}
    grid        = {}

    with torch.no_grad():
        for min_area in min_areas:
            for close_kernel in close_kernels:
                ious = []
                for idx in range(len(val_dataset)):
                    image, mask = val_dataset[idx]
                    pred_prob   = torch.sigmoid(
                        model(image.unsqueeze(0).to(device))
                    ).squeeze().cpu().numpy()

                    pred_pp = postprocess_mask(pred_prob, min_area, close_kernel)
                    # Convert postprocessed mask back to pseudo-logit for iou_score
                    pred_t  = torch.tensor(pred_pp).unsqueeze(0).unsqueeze(0)
                    # iou_score expects logits — use large positive/negative values
                    logit   = (pred_t * 20.0) - 10.0
                    ious.append(iou_score(logit, mask.unsqueeze(0)))

                mean_iou = float(np.mean(ious))
                key      = (min_area, close_kernel)
                grid[str(key)] = round(mean_iou, 4)

                if mean_iou > best_iou:
                    best_iou    = mean_iou
                    best_params = {"min_area": min_area, "close_kernel": close_kernel}

    return {
        "best_min_area":     best_params["min_area"],
        "best_close_kernel": best_params["close_kernel"],
        "best_iou":          round(best_iou, 4),
        "grid":              grid,
    }
