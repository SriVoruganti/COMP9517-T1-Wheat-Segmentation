"""
Label Noise Analysis — Insight to Improvement Experiment.

Motivation (insight from data_scarcity.py):
    Under noisy annotations, standard BCE loss degrades rapidly because
    it penalises every mislabelled pixel independently. Region-overlap
    losses (Dice, Tversky) and imbalance-aware losses (Focal) are more
    robust because they operate on aggregate pixel statistics rather than
    per-pixel cross-entropy terms — so individual noisy pixels have less
    influence on the gradient.

    Specifically, TverskyLoss with alpha=0.3, beta=0.7 penalises false
    negatives more than false positives, which partially counteracts the
    tendency of noisy labels to suppress true positive regions.

This experiment trains models under four label noise levels (0%, 10%,
20%, 30%) with three different loss functions, and evaluates each on a
clean validation set. The expected result is that FocalDice and Tversky
maintain significantly higher IoU/F1 than BCE as noise increases —
demonstrating that the choice of loss function is an effective technique
for improving robustness to annotation noise.

Usage:
    python experiments/label_noise_analysis.py \
        --data_root ./EWS-Dataset \
        --model pretrained \
        --output_dir ./results/label_noise \
        --epochs 30
"""

import os
import sys
import json
import argparse
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import EWSDataset, get_train_transforms, get_val_transforms
from models.unet import UNet
from models.unet_pretrained import PretrainedUNet
from models.losses import get_loss
from utils.metrics import compute_all_metrics, aggregate_metrics


# Noise levels and losses to compare
NOISE_LEVELS = [0.0, 0.10, 0.20, 0.30]

# BCE = baseline (sensitive to noisy labels)
# FocalDice = current best (combines class-imbalance awareness with overlap)
# Tversky = recall-biased overlap loss (robust under label under-annotation)
LOSSES = ["bce", "focal_dice", "tversky"]



# Helpers


def build_model(model_type: str, device):
    if model_type == "pretrained":
        return PretrainedUNet(pretrained=True).to(device)
    return UNet().to(device)


def train_and_eval(model, train_loader, val_loader, criterion, device, epochs, lr):
    """Train model and return best validation metrics (by IoU)."""
    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_iou     = -1.0
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Validate on clean labels
        model.eval()
        batch_metrics = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                batch_metrics.append(compute_all_metrics(model(images), masks))
        val_metrics = aggregate_metrics(batch_metrics)

        if val_metrics["iou"] > best_iou:
            best_iou     = val_metrics["iou"]
            best_metrics = val_metrics

        if epoch % 10 == 0:
            print(f"      Epoch {epoch:03d}: Val IoU = {val_metrics['iou']:.4f}")

    return best_metrics



# Argument parsing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse the effect of label noise on different loss functions."
    )
    parser.add_argument("--data_root",   type=str,   default="./EWS-Dataset")
    parser.add_argument("--model",       type=str,   default="pretrained",
                        choices=["unet", "pretrained"])
    parser.add_argument("--output_dir",  type=str,   default="./results/label_noise")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--image_size",  type=int,   default=350)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--seed",        type=int,   default=42)
    return parser.parse_args()



# Main

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    print(f"Device: {device} | Model: {args.model}")
    print(f"Noise levels: {NOISE_LEVELS} | Losses: {LOSSES}\n")

    # Clean validation set — always evaluate on noise-free labels
    val_ds     = EWSDataset(args.data_root, "val", get_val_transforms(args.image_size))
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # all_results[loss_name][noise_label] = metrics dict
    all_results = {loss: {} for loss in LOSSES}

    for noise in NOISE_LEVELS:
        noise_label = f"{int(noise * 100)}%"
        print(f"{'='*60}")
        print(f"Label Noise: {noise_label}")

        # Training set with injected label noise
        train_ds = EWSDataset(
            args.data_root, "train",
            get_train_transforms(args.image_size),
            label_noise=noise,
            seed=args.seed,
        )
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        print(f"  Training images: {len(train_ds)}")

        for loss_name in LOSSES:
            print(f"  Loss: {loss_name}")
            model     = build_model(args.model, device)
            criterion = get_loss(loss_name)

            t0      = time.time()
            metrics = train_and_eval(model, train_loader, val_loader,
                                     criterion, device, args.epochs, args.lr)
            elapsed = time.time() - t0

            all_results[loss_name][noise_label] = {
                **metrics,
                "time_s": round(elapsed, 1),
            }
            print(
                f"    Best → IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  "
                f"({elapsed / 60:.1f} min)"
            )

    # Save JSON
    out_path = os.path.join(args.output_dir, f"label_noise_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    noise_labels = [f"{int(n * 100)}%" for n in NOISE_LEVELS]
    header = f"{'Loss':<15}" + "".join(f"  {'Noise '+nl:>13}" for nl in noise_labels)
    print("\n" + "="*70)
    print(header)
    print("="*70)
    for loss_name in LOSSES:
        row = f"{loss_name:<15}"
        for nl in noise_labels:
            iou = all_results[loss_name][nl]["iou"]
            row += f"  {iou:>13.4f}"
        print(row)

    # Print IoU degradation relative to 0% noise
    print("\n--- IoU degradation relative to 0% noise ---")
    print(f"{'Loss':<15}" + "".join(f"  {'Noise '+nl:>13}" for nl in noise_labels[1:]))
    print("-"*55)
    for loss_name in LOSSES:
        base = all_results[loss_name]["0%"]["iou"]
        row  = f"{loss_name:<15}"
        for nl in noise_labels[1:]:
            delta = all_results[loss_name][nl]["iou"] - base
            row  += f"  {delta:>+13.4f}"
        print(row)

    # Plot — IoU vs noise level per loss, and F1 vs noise level
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        colours = {
            "bce":        "#EF5350",   # red — baseline
            "focal_dice": "#42A5F5",   # blue — current best
            "tversky":    "#66BB6A",   # green — recall-biased
        }
        markers = {"bce": "o", "focal_dice": "s", "tversky": "^"}
        labels  = {"bce": "BCE (baseline)", "focal_dice": "FocalDice", "tversky": "Tversky"}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(noise_labels))

        for metric, ax, title in [
            ("iou", axes[0], "IoU vs Label Noise Level"),
            ("f1",  axes[1], "F1-Score vs Label Noise Level"),
        ]:
            for loss_name in LOSSES:
                vals = [all_results[loss_name][nl][metric] for nl in noise_labels]
                ax.plot(
                    x, vals,
                    marker=markers[loss_name],
                    linewidth=2,
                    markersize=9,
                    color=colours[loss_name],
                    label=labels[loss_name],
                )
                for xi, v in enumerate(vals):
                    ax.annotate(
                        f"{v:.3f}", (xi, v),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color=colours[loss_name],
                    )
            ax.set_xticks(x)
            ax.set_xticklabels([f"Noise = {nl}" for nl in noise_labels], fontsize=11)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_ylim(0, 1.08)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        fig.suptitle(
            f"Label Noise Robustness: Loss Function Comparison ({args.model})\n"
            "Insight: Overlap-based losses (FocalDice, Tversky) degrade less than BCE under noisy annotations.",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, f"label_noise_{args.model}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {fig_path}")
        plt.close()

        # -- Second plot: IoU degradation bar chart --
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x2    = np.arange(len(noise_labels[1:]))
        width = 0.25
        for i, loss_name in enumerate(LOSSES):
            base   = all_results[loss_name]["0%"]["iou"]
            deltas = [all_results[loss_name][nl]["iou"] - base for nl in noise_labels[1:]]
            ax2.bar(
                x2 + i * width - width,
                deltas, width,
                label=labels[loss_name],
                color=colours[loss_name],
                alpha=0.85,
            )
            for xi, d in enumerate(deltas):
                ax2.text(
                    xi + i * width - width,
                    d + (0.003 if d >= 0 else -0.015),
                    f"{d:+.3f}", ha="center", va="bottom", fontsize=8,
                    color=colours[loss_name],
                )
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xticks(x2)
        ax2.set_xticklabels([f"Noise = {nl}" for nl in noise_labels[1:]], fontsize=11)
        ax2.set_ylabel("IoU Change vs 0% Noise", fontsize=12)
        ax2.set_title(
            f"IoU Degradation Under Label Noise — {args.model}\n"
            "(Less negative = more robust to noisy labels)",
            fontsize=12, fontweight="bold",
        )
        ax2.legend(fontsize=10)
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig2_path = os.path.join(args.output_dir, f"label_noise_degradation_{args.model}.png")
        plt.savefig(fig2_path, dpi=150, bbox_inches="tight")
        print(f"Degradation plot saved to {fig2_path}")
        plt.close()

    except ImportError:
        pass


if __name__ == "__main__":
    main()
