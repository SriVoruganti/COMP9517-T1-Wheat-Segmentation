"""
Robustness Improvement Experiment.

Demonstrates that training with distortion augmentations significantly
improves model performance under real-world image degradations.

Two models are trained:
  (1) baseline    — minimal augmentation (resize + flips + normalise only)
  (2) hardened    — full augmentation including Gaussian noise, blur,
                    colour jitter, elastic distortion, and CoarseDropout

Both are then evaluated on the test set under each distortion in
data/distortions.py. The comparison quantifies the robustness gain
from augmentation hardening.

Usage:
    python experiments/robustness_improvement.py \
        --data_root ./EWS-Dataset \
        --model pretrained \
        --output_dir ./results/robustness_improvement \
        --epochs 40
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import EWSDataset, get_train_transforms, get_val_transforms
from data.distortions import DISTORTIONS
from models.unet import UNet
from models.unet_pretrained import PretrainedUNet
from models.losses import get_loss
from utils.metrics import compute_all_metrics, aggregate_metrics



# Minimal augmentation — geometric only, no photometric/noise/blur

def get_baseline_transforms(image_size: int = 350) -> A.Compose:
    """
    Baseline augmentation pipeline: flips and rotation only.
    Represents a model trained without any awareness of image distortions.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])



# Dataset that applies a distortion before the normalisation transform


class DistortedEWSDataset(EWSDataset):
    """EWSDataset that applies a synthetic distortion to the image only."""

    def __init__(self, root: str, split: str, distortion_fn, image_size: int = 350):
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        super().__init__(root, split, transform=transform)
        self.distortion_fn = distortion_fn

    def __getitem__(self, idx):
        img_path  = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir,  self.masks[idx])

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        mask  = np.array(Image.open(mask_path).convert("L"),  dtype=np.float32)
        mask  = (mask > 127).astype(np.float32)

        # Apply distortion to image before normalisation (mask stays clean)
        image = self.distortion_fn(image)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
        else:
            mask = torch.tensor(mask).unsqueeze(0)

        return image, mask



# Training and evaluation helpers

def train_model(model, train_loader, val_loader, criterion, device, epochs, lr, tag):
    """Train model and restore best checkpoint by val IoU."""
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_iou   = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
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
        val_iou = aggregate_metrics(batch_metrics)["iou"]

        if val_iou > best_iou:
            best_iou   = val_iou
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  [{tag}] Epoch {epoch:03d}: Val IoU = {val_iou:.4f}")

    model.load_state_dict(best_state)
    print(f"  [{tag}] Best Val IoU: {best_iou:.4f}")
    return model


@torch.no_grad()
def evaluate_under_distortions(model, data_root, split, device, image_size, batch_size):
    """Evaluate model IoU/F1/precision/recall under each registered distortion."""
    results = {}
    model.eval()
    for name, dist_fn in DISTORTIONS.items():
        ds     = DistortedEWSDataset(data_root, split, dist_fn, image_size)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        batch_metrics = []
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            batch_metrics.append(compute_all_metrics(model(images), masks))
        results[name] = aggregate_metrics(batch_metrics)
        print(f"    {name:<30} IoU={results[name]['iou']:.4f}  F1={results[name]['f1']:.4f}")
    return results


def build_model(model_type: str, device):
    if model_type == "pretrained":
        return PretrainedUNet(pretrained=True).to(device)
    return UNet().to(device)



# Argument parsing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare robustness of baseline vs augmentation-hardened model."
    )
    parser.add_argument("--data_root",   type=str,   default="./EWS-Dataset")
    parser.add_argument("--model",       type=str,   default="pretrained",
                        choices=["unet", "pretrained"])
    parser.add_argument("--output_dir",  type=str,   default="./results/robustness_improvement")
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
    print(f"Device: {device} | Model: {args.model} | Loss: {args.loss}")

    criterion  = get_loss(args.loss)
    val_ds     = EWSDataset(args.data_root, "val", get_val_transforms(args.image_size))
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    variants = {
        "baseline_no_aug":   get_baseline_transforms(args.image_size),
        "hardened_full_aug": get_train_transforms(args.image_size),
    }

    all_results = {}

    for variant, transform in variants.items():
        print(f"\n{'='*60}")
        print(f"Variant: {variant}")

        train_ds     = EWSDataset(args.data_root, "train", transform)
        train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        print(f"  Training images: {len(train_ds)}")

        model = build_model(args.model, device)
        t0    = time.time()
        model = train_model(model, train_loader, val_loader, criterion,
                            device, args.epochs, args.lr, variant)
        print(f"  Training time: {(time.time()-t0)/60:.1f} min")

        print(f"  Evaluating under distortions on test set ...")
        results = evaluate_under_distortions(
            model, args.data_root, "test", device, args.image_size, args.batch_size
        )
        all_results[variant] = results

        ckpt_path = os.path.join(args.output_dir, f"{variant}_{args.model}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # Save results JSON
    out_path = os.path.join(args.output_dir, "robustness_improvement.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    distortions = list(DISTORTIONS.keys())
    print("\n" + "="*75)
    print(f"{'Distortion':<30} {'Baseline IoU':>14} {'Hardened IoU':>14} {'Gain':>8}")
    print("="*75)
    for d in distortions:
        base = all_results["baseline_no_aug"][d]["iou"]
        hard = all_results["hardened_full_aug"][d]["iou"]
        gain = hard - base
        sign = "+" if gain >= 0 else ""
        print(f"{d:<30} {base:>14.4f} {hard:>14.4f} {sign}{gain:>7.4f}")

    # Plot grouped bar chart
    try:
        import matplotlib.pyplot as plt

        x         = np.arange(len(distortions))
        width     = 0.35
        base_ious = [all_results["baseline_no_aug"][d]["iou"]   for d in distortions]
        hard_ious = [all_results["hardened_full_aug"][d]["iou"] for d in distortions]

        fig, ax = plt.subplots(figsize=(16, 6))
        bars1 = ax.bar(x - width / 2, base_ious, width,
                       label="Baseline (geometric aug only)", color="#EF5350", alpha=0.85)
        bars2 = ax.bar(x + width / 2, hard_ious, width,
                       label="Hardened (full distortion aug)", color="#42A5F5", alpha=0.85)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(distortions, rotation=30, ha="right", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("IoU Score", fontsize=12)
        ax.set_title(
            f"Robustness Improvement: Baseline vs Augmentation-Hardened ({args.model})",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(args.output_dir, f"robustness_improvement_{args.model}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {fig_path}")
        plt.close()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
