"""
Final test set evaluation with optional TTA and failure analysis.

Usage:
    python evaluate.py \
        --data_root ./EWS-Dataset \
        --checkpoint ./results/pretrained_focal_dice/best.pth \
        --model pretrained \
        --tta \
        --failure_analysis \
        --visualise
"""

import os
import sys
import time
import argparse
import json

import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import EWSDataset, get_val_transforms
from models.unet import UNet
from models.unet_pretrained import PretrainedUNet
from utils.metrics import compute_all_metrics, aggregate_metrics
from utils.tta import tta_predict
from utils.postprocess import postprocess_batch, tune_postprocess
from utils.visualise import (
    plot_prediction_grid,
    plot_failure_analysis,
    plot_training_curves,
)


@torch.no_grad()
def evaluate(model, loader, device, use_tta=False, use_postprocess=False,
             pp_min_area=200, pp_close_kernel=5):
    model.eval()
    batch_metrics   = []
    inference_times = []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        t0 = time.time()

        if use_tta:
            probs = tta_predict(model, images)
            preds = torch.log(probs.clamp(1e-6, 1 - 1e-6) / (1 - probs.clamp(1e-6, 1 - 1e-6)))
        else:
            preds = model(images)

        if use_postprocess:
            # Apply morphological cleanup to binary predictions, then re-encode as logits
            probs_np = torch.sigmoid(preds).cpu().numpy()          # (B, 1, H, W)
            pp_np    = postprocess_batch(probs_np, pp_min_area, pp_close_kernel)
            pp_t     = torch.tensor(pp_np, device=device)
            preds    = (pp_t * 20.0) - 10.0   # binary → pseudo-logit

        inference_times.append((time.time() - t0) / images.size(0))
        batch_metrics.append(compute_all_metrics(preds, masks))

    summary = aggregate_metrics(batch_metrics)
    summary["avg_inference_time_ms"] = round(np.mean(inference_times) * 1000, 2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",        type=str,  default="./EWS-Dataset")
    parser.add_argument("--checkpoint",       type=str,  required=True)
    parser.add_argument("--model",            type=str,  default="pretrained",
                        choices=["unet", "pretrained"])
    parser.add_argument("--output_dir",       type=str,  default="./results")
    parser.add_argument("--image_size",       type=int,  default=350)
    parser.add_argument("--batch_size",       type=int,  default=4)
    parser.add_argument("--num_workers",      type=int,  default=2)
    parser.add_argument("--tta",              action="store_true",
                        help="Enable Test-Time Augmentation")
    parser.add_argument("--postprocess",      action="store_true",
                        help="Apply morphological post-processing (remove small FP regions, fill holes)")
    parser.add_argument("--pp_min_area",      type=int,  default=200,
                        help="Min connected-component area to keep (post-processing)")
    parser.add_argument("--pp_close_kernel",  type=int,  default=5,
                        help="Closing kernel size for hole-filling (post-processing)")
    parser.add_argument("--tune_postprocess", action="store_true",
                        help="Grid-search best post-processing params on val set before test eval")
    parser.add_argument("--visualise",        action="store_true",
                        help="Save prediction grid")
    parser.add_argument("--failure_analysis", action="store_true",
                        help="Identify and visualise worst predictions")
    parser.add_argument("--history_path",     type=str,  default=None,
                        help="Path to training history JSON for curve plots")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Load test dataset
    test_ds     = EWSDataset(args.data_root, "test", get_val_transforms(args.image_size))
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"Test set: {len(test_ds)} images")

    # Load model
    if args.model == "pretrained":
        model = PretrainedUNet(pretrained=False).to(device)
    else:
        model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Checkpoint: {args.checkpoint}")

    # Optionally tune post-processing params on val set first
    pp_min_area    = args.pp_min_area
    pp_close_kernel = args.pp_close_kernel
    if args.tune_postprocess:
        print("\nTuning post-processing parameters on validation set ...")
        val_ds_pp  = EWSDataset(args.data_root, "val", get_val_transforms(args.image_size))
        tune_result = tune_postprocess(model, val_ds_pp, device)
        pp_min_area     = tune_result["best_min_area"]
        pp_close_kernel = tune_result["best_close_kernel"]
        print(f"  Best params: min_area={pp_min_area}, close_kernel={pp_close_kernel}"
              f"  (val IoU={tune_result['best_iou']:.4f})")

    # Evaluate — build result variants to compare
    metrics_no_tta = evaluate(model, test_loader, device, use_tta=False)
    results = {"no_tta": metrics_no_tta}

    if args.tta:
        metrics_tta = evaluate(model, test_loader, device, use_tta=True)
        results["tta"] = metrics_tta

    if args.postprocess:
        metrics_pp = evaluate(model, test_loader, device, use_tta=False,
                              use_postprocess=True,
                              pp_min_area=pp_min_area,
                              pp_close_kernel=pp_close_kernel)
        results["postprocess"] = metrics_pp

    if args.tta and args.postprocess:
        metrics_tta_pp = evaluate(model, test_loader, device, use_tta=True,
                                  use_postprocess=True,
                                  pp_min_area=pp_min_area,
                                  pp_close_kernel=pp_close_kernel)
        results["tta+postprocess"] = metrics_tta_pp

    # Print results
    print("\n" + "="*55)
    print(f"{'':30} {'Precision':>9} {'Recall':>8} {'F1':>8} {'IoU':>8} {'ms/img':>8}")
    print("="*55)
    for tag, m in results.items():
        print(
            f"  {tag:<28} {m['precision']:>9.4f} {m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} {m['iou']:>8.4f} {m['avg_inference_time_ms']:>8.2f}"
        )
    print("="*55)

    # Save
    out_path = os.path.join(args.output_dir, f"test_metrics_{args.model}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    # Visualise predictions
    if args.visualise:
        plot_prediction_grid(
            model, test_ds, device, n=6,
            save_path=os.path.join(fig_dir, f"predictions_{args.model}.png"),
            title=f"Predictions — {args.model}" + (" + TTA" if args.tta else ""),
        )

    # Failure analysis
    if args.failure_analysis:
        plot_failure_analysis(
            model, test_ds, device, n=6,
            save_path=os.path.join(fig_dir, f"failures_{args.model}.png"),
        )

    # Training curves
    if args.history_path and os.path.exists(args.history_path):
        plot_training_curves(
            args.history_path,
            save_path=os.path.join(fig_dir, "training_curves.png"),
        )


if __name__ == "__main__":
    main()
