"""
Robustness Evaluation Experiment.

Evaluates how the trained model performs under various realistic image
distortions: noise, blur, low brightness/contrast, occlusion, compression.

This directly addresses the HD-tier requirement:
    "Simulate realistic image distortions and evaluate how your methods
     perform with such images."

Usage:
    python experiments/robustness_eval.py \
        --data_root ./EWS-Dataset \
        --checkpoint ./results/pretrained_focal_dice/best.pth \
        --model pretrained \
        --output_dir ./results/robustness
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.dataset import EWSDataset, get_val_transforms
from data.distortions import DISTORTIONS
from models.unet import UNet
from models.unet_pretrained import PretrainedUNet
from utils.metrics import compute_all_metrics, aggregate_metrics
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Custom dataset that applies a distortion BEFORE the standard transform
# ---------------------------------------------------------------------------

class DistortedEWSDataset(EWSDataset):
    def __init__(self, root, split, distortion_fn, image_size=350):
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
        mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask  = (mask > 127).astype(np.float32)

        # Apply distortion to image only (mask stays clean)
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


@torch.no_grad()
def evaluate_distortion(model, distortion_fn, data_root, split, device, image_size, batch_size):
    ds     = DistortedEWSDataset(data_root, split, distortion_fn, image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    batch_metrics = []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        batch_metrics.append(compute_all_metrics(preds, masks))
    return aggregate_metrics(batch_metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   type=str, default="./EWS-Dataset")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--model",       type=str, default="pretrained",
                        choices=["unet", "pretrained"])
    parser.add_argument("--output_dir",  type=str, default="./results/robustness")
    parser.add_argument("--image_size",  type=int, default=350)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--split",       type=str, default="test")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    if args.model == "pretrained":
        model = PretrainedUNet(pretrained=False).to(device)
    else:
        model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded: {args.checkpoint}")

    # Evaluate under each distortion
    results = {}
    for name, distortion_fn in DISTORTIONS.items():
        print(f"  Evaluating: {name} ...", end=" ", flush=True)
        metrics = evaluate_distortion(
            model, distortion_fn, args.data_root, args.split,
            device, args.image_size, args.batch_size
        )
        results[name] = metrics
        print(f"IoU={metrics['iou']:.4f}  F1={metrics['f1']:.4f}")

    # Save results
    out_path = os.path.join(args.output_dir, f"robustness_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "="*65)
    print(f"{'Distortion':<30} {'IoU':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("="*65)
    for name, m in results.items():
        print(f"{name:<30} {m['iou']:>8.4f} {m['f1']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}")

    # Plot robustness bar chart
    try:
        import matplotlib.pyplot as plt
        names = list(results.keys())
        ious  = [results[n]["iou"] for n in names]
        clean_iou = results.get("clean", {}).get("iou", 1.0)

        colours = ["#4CAF50" if n == "clean" else "#F44336" if iou < clean_iou * 0.85
                   else "#FF9800" for n, iou in zip(names, ious)]

        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(range(len(names)), ious, color=colours, alpha=0.85)
        ax.axhline(clean_iou, color="green", linestyle="--", linewidth=1.5, label=f"Clean IoU ({clean_iou:.3f})")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("IoU Score", fontsize=12)
        ax.set_title(f"Robustness Under Image Distortions — {args.model}", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        for bar, iou in zip(bars, ious):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{iou:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, f"robustness_{args.model}.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {fig_path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
