"""
Data Scarcity Experiment.

Trains the model with varying fractions of the training set (25%, 50%, 75%, 100%)
and plots how performance degrades. Also optionally injects label noise to analyse
the effect of annotation quality.

Usage:
    python experiments/data_scarcity.py \
        --data_root ./EWS-Dataset \
        --model pretrained \
        --output_dir ./results/scarcity \
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


def train_and_eval(model, train_loader, val_loader, criterion, device, epochs, lr):
    """Train a model and return best val metrics."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_iou  = -1.0
    best_metrics = {"precision": 0, "recall" : 0, "f1": 0, "iou": 0}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        batch_metrics = []
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                batch_metrics.append(compute_all_metrics(preds, masks))
        val_metrics = aggregate_metrics(batch_metrics)

        if val_metrics["iou"] > best_iou:
            best_iou     = val_metrics["iou"]
            best_metrics = val_metrics

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:03d}: IoU={val_metrics['iou']:.4f}")

    return best_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    type=str,   default="./EWS-Dataset")
    parser.add_argument("--model",        type=str,   default="pretrained",
                        choices=["unet", "pretrained"])
    parser.add_argument("--output_dir",   type=str,   default="./results/scarcity")
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--loss",         type=str,   default="focal_dice")
    parser.add_argument("--label_noise",  type=float, default=0.0,
                        help="Fraction of mask pixels to flip (simulates noisy annotation)")
    parser.add_argument("--image_size",   type=int,   default=350)
    parser.add_argument("--num_workers",  type=int,   default=2)
    parser.add_argument("--seed",         type=int,   default=42)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    fractions  = [0.25, 0.50, 0.75, 1.00]
    criterion  = get_loss(args.loss)
    val_ds     = EWSDataset(args.data_root, "val", get_val_transforms(args.image_size))
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    all_results = {}

    for frac in fractions:
        n_images = int(frac * 100)   # approximate % label
        label = f"{int(frac*100)}%"
        print(f"\n{'='*50}")
        print(f"Training with {label} of data (label_noise={args.label_noise})")

        train_ds = EWSDataset(
            args.data_root, "train",
            get_train_transforms(args.image_size),
            subset_frac=frac,
            label_noise=args.label_noise,
            seed=args.seed,
        )
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        print(f"  Training images: {len(train_ds)}")

        # Re-initialise a fresh model for each run
        if args.model == "pretrained":
            model = PretrainedUNet(pretrained=True).to(device)
        else:
            model = UNet().to(device)

        t0      = time.time()
        metrics = train_and_eval(model, train_loader, val_loader,
                                 criterion, device, args.epochs, args.lr)
        elapsed = time.time() - t0

        all_results[label] = {**metrics, "train_images": len(train_ds), "time_s": round(elapsed, 1)}
        print(f"  Best → IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  ({elapsed/60:.1f} min)")

    # Save
    out_path = os.path.join(args.output_dir, f"scarcity_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        labels = list(all_results.keys())
        ious   = [all_results[l]["iou"] for l in labels]
        f1s    = [all_results[l]["f1"]  for l in labels]
        x      = range(len(labels))

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, ious, "o-", color="#2196F3", linewidth=2, markersize=8, label="IoU")
        ax.plot(x, f1s,  "s-", color="#4CAF50", linewidth=2, markersize=8, label="F1-Score")
        for xi, (iou, f1) in enumerate(zip(ious, f1s)):
            ax.annotate(f"{iou:.3f}", (xi, iou), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10, color="#2196F3")
            ax.annotate(f"{f1:.3f}",  (xi, f1),  textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=10, color="#4CAF50")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_xlabel("Training Data Used", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_ylim(0, 1.05)
        title = f"Data Scarcity Analysis — {args.model}"
        if args.label_noise > 0:
            title += f" (label noise={args.label_noise})"
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, f"scarcity_{args.model}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {fig_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
