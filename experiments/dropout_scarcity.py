"""
Decoder Dropout under Data Scarcity — Model Adjustment Experiment.

Motivation (insight from data_scarcity.py):
    When only 25–50% of training data is available, the pretrained U-Net
    decoder overfits: training loss drops rapidly while validation IoU
    plateaus early. The encoder's pretrained features are strong, but the
    randomly-initialised decoder layers overfit to the small training set.

Model adjustment:
    Adding spatial dropout (Dropout2d) inside the decoder's ConvBnRelu
    blocks prevents co-adaptation of decoder feature maps. This is a
    targeted regularisation that leaves the pretrained encoder unchanged
    and only constrains the part of the network that has no pre-training.

    We compare three dropout levels:
        - 0.0  (baseline — no decoder regularisation)
        - 0.2  (moderate dropout)
        - 0.3  (strong dropout — expected to help most at low data)

    across three training-set fractions: 25%, 50%, 75%.
    (At 100% data, overfitting is less severe so the gain is expected
    to diminish — this also validates that the dropout is doing the
    right thing, not just adding noise.)

Usage:
    python experiments/dropout_scarcity.py \
        --data_root ./EWS-Dataset \
        --output_dir ./results/dropout_scarcity \
        --epochs 40
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
from models.unet_pretrained import PretrainedUNet
from models.losses import get_loss
from utils.metrics import compute_all_metrics, aggregate_metrics


# Fractions and dropout levels to sweep
FRACTIONS      = [0.25, 0.50, 0.75]
DROPOUT_LEVELS = [0.0, 0.2, 0.3]



# Training helper

def train_and_eval(dropout, frac, data_root, val_loader, device, args):
    """Train a PretrainedUNet with given decoder dropout and data fraction."""
    train_ds = EWSDataset(
        data_root, "train",
        get_train_transforms(args.image_size),
        subset_frac=frac,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    model     = PretrainedUNet(pretrained=True, decoder_dropout=dropout).to(device)
    criterion = get_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou     = -1.0
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

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

    return best_metrics, len(train_ds)



# Argument parsing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show that decoder dropout improves performance under data scarcity."
    )
    parser.add_argument("--data_root",   type=str,   default="./EWS-Dataset")
    parser.add_argument("--output_dir",  type=str,   default="./results/dropout_scarcity")
    parser.add_argument("--epochs",      type=int,   default=40)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--loss",        type=str,   default="focal_dice")
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
    print(f"Device: {device}")
    print(f"Fractions: {FRACTIONS} | Dropout levels: {DROPOUT_LEVELS}\n")

    val_ds     = EWSDataset(args.data_root, "val", get_val_transforms(args.image_size))
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # all_results[frac_label][dropout_label] = metrics dict
    all_results = {}

    for frac in FRACTIONS:
        frac_label = f"{int(frac * 100)}%"
        all_results[frac_label] = {}
        print(f"{'='*60}")
        print(f"Training data: {frac_label}")

        for dropout in DROPOUT_LEVELS:
            dropout_label = f"dropout={dropout}"
            print(f"  {dropout_label}")

            t0      = time.time()
            metrics, n_imgs = train_and_eval(
                dropout, frac, args.data_root, val_loader, device, args
            )
            elapsed = time.time() - t0

            all_results[frac_label][dropout_label] = {
                **metrics,
                "n_images": n_imgs,
                "time_s":   round(elapsed, 1),
            }
            print(
                f"    Best → IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}  "
                f"({elapsed / 60:.1f} min)"
            )

    # Save JSON
    out_path = os.path.join(args.output_dir, "dropout_scarcity.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    frac_labels    = [f"{int(f * 100)}%" for f in FRACTIONS]
    dropout_labels = [f"dropout={d}" for d in DROPOUT_LEVELS]
    col_w = 16

    print("\n" + "="*70)
    header = f"{'':12}" + "".join(f"{fl:>{col_w}}" for fl in frac_labels)
    print(header)
    print("-"*70)
    for dl in dropout_labels:
        row = f"{dl:<12}"
        for fl in frac_labels:
            iou = all_results[fl][dl]["iou"]
            row += f"{iou:>{col_w}.4f}"
        print(row)

    # Print gain of best dropout over baseline (dropout=0.0) at each fraction
    print("\n--- IoU gain vs dropout=0.0 baseline ---")
    for fl in frac_labels:
        base = all_results[fl]["dropout=0.0"]["iou"]
        gains = {
            dl: all_results[fl][dl]["iou"] - base
            for dl in dropout_labels[1:]
        }
        best_dl  = max(gains, key=gains.get)
        best_gain = gains[best_dl]
        print(f"  {fl}: best gain = {best_gain:+.4f} ({best_dl})")

    # Plot — IoU vs data fraction, one line per dropout level
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        colours = {
            "dropout=0.0": "#EF5350",
            "dropout=0.2": "#FF9800",
            "dropout=0.3": "#42A5F5",
        }
        markers = {
            "dropout=0.0": "o",
            "dropout=0.2": "s",
            "dropout=0.3": "^",
        }

        x = np.arange(len(frac_labels))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for metric, ax, ylabel in [
            ("iou", axes[0], "IoU Score"),
            ("f1",  axes[1], "F1-Score"),
        ]:
            for dl in dropout_labels:
                vals = [all_results[fl][dl][metric] for fl in frac_labels]
                ax.plot(
                    x, vals,
                    marker=markers[dl], linewidth=2, markersize=9,
                    color=colours[dl], label=dl,
                )
                for xi, v in enumerate(vals):
                    ax.annotate(
                        f"{v:.3f}", (xi, v),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color=colours[dl],
                    )
            ax.set_xticks(x)
            ax.set_xticklabels([f"{fl} data" for fl in frac_labels], fontsize=11)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_ylim(0, 1.08)
            ax.set_title(f"{ylabel} vs Training Set Size", fontsize=12, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        fig.suptitle(
            "Decoder Dropout Improves Pretrained U-Net Under Data Scarcity\n"
            "Insight: Decoder overfits when training data is scarce — "
            "spatial dropout in decoder blocks reduces this.",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, "dropout_scarcity.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {fig_path}")
        plt.close()

    except ImportError:
        pass


if __name__ == "__main__":
    main()
