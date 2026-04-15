"""
Segmentation evaluation metrics — shared across all methods in the project.
All functions accept raw logits (not sigmoid-applied) for predictions.
"""

import torch
import numpy as np


def _binarise(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(pred) > threshold).float()


def precision(pred, target, eps=1e-6, threshold=0.5):
    pred, target = _binarise(pred, threshold).view(-1), target.view(-1)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return ((tp + eps) / (tp + fp + eps)).item()


def recall(pred, target, eps=1e-6, threshold=0.5):
    pred, target = _binarise(pred, threshold).view(-1), target.view(-1)
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return ((tp + eps) / (tp + fn + eps)).item()


def f1_score(pred, target, eps=1e-6, threshold=0.5):
    p = precision(pred, target, eps, threshold)
    r = recall(pred, target, eps, threshold)
    return (2 * p * r) / (p + r + eps)


def iou_score(pred, target, eps=1e-6, threshold=0.5):
    pred, target = _binarise(pred, threshold).view(-1), target.view(-1)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return ((inter + eps) / (union + eps)).item()


def compute_all_metrics(pred, target, threshold=0.5):
    return {
        "precision": precision(pred, target, threshold=threshold),
        "recall":    recall(pred, target, threshold=threshold),
        "f1":        f1_score(pred, target, threshold=threshold),
        "iou":       iou_score(pred, target, threshold=threshold),
    }


def aggregate_metrics(metric_list: list) -> dict:
    """Average a list of per-batch metric dicts into a single summary dict."""
    keys = metric_list[0].keys()
    return {k: round(float(np.mean([m[k] for m in metric_list])), 4) for k in keys}
