"""
Training script — supports both vanilla U-Net and pretrained encoder U-Net.

Two-phase training for pretrained model:
  Phase 1: Freeze encoder, train decoder only (fast convergence)
  Phase 2: Unfreeze all, fine-tune end-to-end with lower LR

Usage:
    # Vanilla U-Net (from scratch)
    python train.py --model unet --data_root ./EWS-Dataset --epochs 60

    # Pretrained ResNet-34 encoder
    python train.py --model pretrained --data_root ./EWS-Dataset --epochs 40 --two_phase
"""

import os
import time
import argparse
import json

import torch
from torch.utils.data import DataLoader

from data.dataset import EWSDataset, get_train_transforms, get_val_transforms
from models.unet import UNet
from models.unet_pretrained import PretrainedUNet
from models.losses import get_loss
from utils.metrics import compute_all_metrics, aggregate_metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses, batch_metrics = [], []
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        losses.append(criterion(preds, masks).item())
        batch_metrics.append(compute_all_metrics(preds, masks))
    return sum(losses) / len(losses), aggregate_metrics(batch_metrics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,   default="pretrained",
                        choices=["unet", "pretrained"],
                        help="'unet' = vanilla, 'pretrained' = ResNet-34 encoder")
    parser.add_argument("--data_root",   type=str,   default="./EWS-Dataset")
    parser.add_argument("--output_dir",  type=str,   default="./results")
    parser.add_argument("--image_size",  type=int,   default=350)
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--loss",        type=str,   default="focal_dice",
                        choices=["combo", "focal_dice", "tversky", "focal", "dice", "bce"])
    parser.add_argument("--dropout",     type=float, default=0.1,
                        help="Dropout rate (vanilla U-Net only)")
    parser.add_argument("--two_phase",   action="store_true",
                        help="Use two-phase training for pretrained model")
    parser.add_argument("--phase1_epochs", type=int, default=15,
                        help="Epochs with frozen encoder (phase 1)")
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--seed",        type=int,   default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tag = f"{args.model}_{args.loss}"
    out_dir   = os.path.join(args.output_dir, model_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device} | Model: {args.model} | Loss: {args.loss}")

    # Datasets
    train_ds = EWSDataset(args.data_root, "train", get_train_transforms(args.image_size))
    val_ds   = EWSDataset(args.data_root, "val",   get_val_transforms(args.image_size))
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    if args.model == "pretrained":
        model = PretrainedUNet(pretrained=True,
                               freeze_encoder=args.two_phase).to(device)
    else:
        model = UNet(dropout=args.dropout).to(device)

    criterion = get_loss(args.loss)

    def make_optimizer(lr):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4
        )

    optimizer = make_optimizer(args.lr)
    history   = []
    best_iou  = 0.0
    total_start = time.time()

    # Two-phase training: phase 1 = frozen encoder
    phase_boundary = args.phase1_epochs if args.two_phase else 0

    for epoch in range(1, args.epochs + 1):

        # Unfreeze encoder at phase boundary
        if args.two_phase and epoch == phase_boundary + 1:
            print(f"\n--- Phase 2: Unfreezing encoder at epoch {epoch} ---")
            model.unfreeze_encoder()
            optimizer = make_optimizer(args.lr * 0.1)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - phase_boundary
        )

        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "time_s":     round(time.time() - t0, 2),
            **{k: round(v, 4) for k, v in val_metrics.items()},
        }
        history.append(log)

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1']:.4f} | "
            f"{log['time_s']:.1f}s"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pth"))
            print(f"  ✓ Best saved (IoU: {best_iou:.4f})")

    total_time = time.time() - total_start
    print(f"\nDone in {total_time/60:.1f} min | Best Val IoU: {best_iou:.4f}")

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({
            "args":         vars(args),
            "total_time_s": round(total_time, 2),
            "best_iou":     round(best_iou, 4),
            "history":      history,
        }, f, indent=2)


if __name__ == "__main__":
    main()
