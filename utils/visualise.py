"""
Visualisation utilities:
  - Prediction grids (image / ground truth / prediction)
  - Failure analysis (worst IoU predictions)
  - Training curve plots
  - Per-method comparison bar charts
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor to a displayable numpy array."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * STD + MEAN, 0, 1)


def plot_prediction_grid(
    model,
    dataset,
    device,
    n:         int  = 6,
    threshold: float = 0.5,
    save_path: str  = None,
    title:     str  = "Predictions",
):
    """Save a grid of image | ground-truth | prediction | error map."""
    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 4))
    cols = ["Image", "Ground Truth", "Prediction", "Error Map"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=13, fontweight="bold")

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            image, mask = dataset[idx]
            pred_prob   = torch.sigmoid(model(image.unsqueeze(0).to(device)))
            pred_bin    = (pred_prob.squeeze().cpu().numpy() > threshold).astype(np.uint8)
            mask_np     = mask.squeeze().numpy()

            # Error map: TP=white, FP=red, FN=blue
            error = np.zeros((*mask_np.shape, 3), dtype=np.float32)
            tp = (pred_bin == 1) & (mask_np == 1)
            fp = (pred_bin == 1) & (mask_np == 0)
            fn = (pred_bin == 0) & (mask_np == 1)
            error[tp] = [1, 1, 1]
            error[fp] = [1, 0, 0]
            error[fn] = [0, 0, 1]

            axes[row, 0].imshow(denormalise(image))
            axes[row, 1].imshow(mask_np,  cmap="gray")
            axes[row, 2].imshow(pred_bin, cmap="gray")
            axes[row, 3].imshow(error)
            for ax in axes[row]:
                ax.axis("off")

    # Legend for error map
    legend = [
        mpatches.Patch(color="white",  label="True Positive"),
        mpatches.Patch(color="red",    label="False Positive"),
        mpatches.Patch(color="blue",   label="False Negative"),
        mpatches.Patch(color="black",  label="True Negative"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=11, frameon=True)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_failure_analysis(
    model,
    dataset,
    device,
    n:         int   = 6,
    threshold: float = 0.5,
    save_path: str   = None,
):
    """
    Identify and visualise the N worst predictions by IoU score.
    Useful for understanding where and why the model fails.
    """
    from utils.metrics import iou_score

    scores = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, mask = dataset[idx]
            pred = model(image.unsqueeze(0).to(device))
            iou  = iou_score(pred.cpu(), mask.unsqueeze(0))
            scores.append((iou, idx))

    scores.sort(key=lambda x: x[0])   # ascending → worst first
    worst = scores[:n]

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    fig.suptitle(f"Failure Analysis — {n} Worst Predictions by IoU",
                 fontsize=14, fontweight="bold")
    for ax, col in zip(axes[0], ["Image", "Ground Truth", "Prediction"]):
        ax.set_title(col, fontsize=12, fontweight="bold")

    with torch.no_grad():
        for row, (iou, idx) in enumerate(worst):
            image, mask = dataset[idx]
            pred_bin    = (torch.sigmoid(
                model(image.unsqueeze(0).to(device))
            ).squeeze().cpu().numpy() > threshold).astype(np.uint8)

            axes[row, 0].imshow(denormalise(image))
            axes[row, 1].imshow(mask.squeeze().numpy(), cmap="gray")
            axes[row, 2].imshow(pred_bin, cmap="gray")
            axes[row, 0].set_ylabel(f"IoU = {iou:.3f}\n{dataset.get_filename(idx)}",
                                    fontsize=9)
            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_training_curves(history_path: str, save_path: str = None):
    """Plot train loss, val loss, IoU and F1 from saved JSON history."""
    with open(history_path) as f:
        data    = json.load(f)
    history = data["history"]

    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    iou        = [h["iou"]        for h in history]
    f1         = [h["f1"]         for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, label="Train Loss", color="#2196F3")
    axes[0].plot(epochs, val_loss,   label="Val Loss",   color="#F44336")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, iou, label="IoU",      color="#4CAF50")
    axes[1].plot(epochs, f1,  label="F1-Score", color="#FF9800")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Validation IoU & F1-Score")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("U-Net Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_method_comparison(results: dict, save_path: str = None):
    """
    Bar chart comparing multiple methods across all metrics.

    Args:
        results: {"MethodName": {"precision": .., "recall": .., "f1": .., "iou": ..}, ...}
    """
    metrics = ["precision", "recall", "f1", "iou"]
    labels  = list(results.keys())
    x       = np.arange(len(metrics))
    width   = 0.8 / len(labels)
    colours = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, scores) in enumerate(results.items()):
        vals = [scores[m] for m in metrics]
        bars = ax.bar(x + i * width - (len(labels) - 1) * width / 2, vals,
                      width, label=name, color=colours[i % len(colours)], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Method Comparison — Test Set Metrics", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
