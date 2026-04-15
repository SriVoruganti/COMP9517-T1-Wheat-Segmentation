"""
Final test set evaluation for Random Forest with optional flip-augmentation
and failure analysis / visualisation.

Usage:
    python evaluate_rf_full.py ^
        --data_root ./data/EWS-Dataset ^
        --model_path ./results/rf_model.pkl ^
        --flip_aug ^
        --visualise ^
        --failure_analysis
"""

import os
import time
import json
import argparse
from pathlib import Path

import numpy as np
import cv2

from models.random_forest import load_model  # expects load_model(path=...) or edit below


# -----------------------------
# Metrics (per-image)
# -----------------------------
def compute_metrics_from_masks(pred01, gt01):
    pred01 = pred01.astype(bool)
    gt01   = gt01.astype(bool)

    tp = np.logical_and(pred01, gt01).sum()
    fp = np.logical_and(pred01, ~gt01).sum()
    fn = np.logical_and(~pred01, gt01).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def aggregate(metrics_list):
    out = {}
    for k in ["precision", "recall", "f1", "iou"]:
        vals = np.array([m[k] for m in metrics_list], dtype=np.float64)
        out[k] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out


# -----------------------------
# EWS mask rule (locked in)
# plant/wheat = 1 where mask_ch0 == 0
# -----------------------------
def load_gt_mask01(mask_path: str) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(mask_path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m == 0).astype(np.uint8)


def load_image_rgb(img_path: str) -> np.ndarray:
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def predict_mask01(clf, img_rgb: np.ndarray) -> np.ndarray:
    H, W, _ = img_rgb.shape
    X = img_rgb.reshape(-1, 3).astype(np.float32)
    y = clf.predict(X).reshape(H, W).astype(np.uint8)
    return y


def predict_mask01_flip_aug(clf, img_rgb: np.ndarray) -> np.ndarray:
    """
    RF 'TTA-like' option: predict on original + horizontal flip,
    then majority vote.
    """
    pred0 = predict_mask01(clf, img_rgb)

    img_flip = np.ascontiguousarray(img_rgb[:, ::-1, :])
    pred1 = predict_mask01(clf, img_flip)[:, ::-1]

    # majority vote (since binary): (pred0 + pred1) >= 1
    return ((pred0 + pred1) >= 1).astype(np.uint8)


def save_error_map(gt01, pred01, save_path: str):
    H, W = gt01.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)

    tp = (pred01 == 1) & (gt01 == 1)
    fp = (pred01 == 1) & (gt01 == 0)
    fn = (pred01 == 0) & (gt01 == 1)

    out[tp] = (255, 255, 255)  # white
    out[fp] = (255, 0, 0)      # red
    out[fn] = (0, 0, 255)      # blue

    cv2.imwrite(save_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def overlay_pred(img_rgb, pred01):
    overlay = img_rgb.copy()
    overlay[pred01 == 1] = (0.5 * overlay[pred01 == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    return overlay


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data/EWS-Dataset")
    p.add_argument("--model_path", type=str, default="./results/rf_model.pkl")
    p.add_argument("--output_dir", type=str, default="./results/rf_full")
    p.add_argument("--flip_aug", action="store_true", help="Enable flip-augmentation voting")
    p.add_argument("--visualise", action="store_true", help="Save a few prediction examples")
    p.add_argument("--failure_analysis", action="store_true", help="Save worst predictions by IoU")
    p.add_argument("--n_vis", type=int, default=6)
    p.add_argument("--n_fail", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    test_root = Path(args.data_root) / "test"
    img_dir = test_root / "images"
    msk_dir = test_root / "masks"

    image_files = sorted([p.name for p in img_dir.glob("*.png")])
    print(f"Test set: {len(image_files)} images")

    # Load model
    # If your load_model() doesn't accept a path, change this line accordingly.
    clf = load_model(args.model_path) if load_model.__code__.co_argcount >= 1 else load_model()
    print(f"Model: {args.model_path}")

    # Evaluate
    per_image = []
    inference_times = []
    saved_examples = []

    for img_file in image_files:
        stem = Path(img_file).stem
        mask_file = stem + "_mask.png"

        img_path = str(img_dir / img_file)
        mask_path = str(msk_dir / mask_file)

        img_rgb = load_image_rgb(img_path)
        gt01 = load_gt_mask01(mask_path)

        t0 = time.perf_counter()
        if args.flip_aug:
            pred01 = predict_mask01_flip_aug(clf, img_rgb)
        else:
            pred01 = predict_mask01(clf, img_rgb)
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000.0)

        m = compute_metrics_from_masks(pred01, gt01)
        m["file"] = img_file
        per_image.append(m)

        # Save a few examples (first n_vis)
        if args.visualise and len(saved_examples) < args.n_vis:
            pred_path = fig_dir / f"{stem}_pred.png"
            gt_path   = fig_dir / f"{stem}_gt.png"
            ov_path   = fig_dir / f"{stem}_overlay.png"
            err_path  = fig_dir / f"{stem}_error.png"

            cv2.imwrite(str(pred_path), pred01 * 255)
            cv2.imwrite(str(gt_path), gt01 * 255)
            cv2.imwrite(str(ov_path), cv2.cvtColor(overlay_pred(img_rgb, pred01), cv2.COLOR_RGB2BGR))
            save_error_map(gt01, pred01, str(err_path))
            saved_examples.append(img_file)

    summary = aggregate(per_image)
    summary["avg_inference_time_ms"] = float(np.mean(inference_times))
    summary["std_inference_time_ms"] = float(np.std(inference_times))

    tag = "flip_aug" if args.flip_aug else "no_aug"

    # Print table (similar to DL)
    print("\n" + "="*60)
    print(f"{'':30} {'Precision':>9} {'Recall':>8} {'F1':>8} {'IoU':>8} {'ms/img':>8}")
    print("="*60)
    print(
        f"  {tag:<28} {summary['precision']:>9.4f} {summary['recall']:>8.4f} "
        f"{summary['f1']:>8.4f} {summary['iou']:>8.4f} {summary['avg_inference_time_ms']:>8.2f}"
    )
    print("="*60)

    # Failure analysis (worst IoU)
    if args.failure_analysis:
        worst = sorted(per_image, key=lambda x: x["iou"])[: args.n_fail]
        fail_dir = fig_dir / "failures"
        fail_dir.mkdir(parents=True, exist_ok=True)

        for item in worst:
            img_file = item["file"]
            stem = Path(img_file).stem
            mask_file = stem + "_mask.png"

            img_rgb = load_image_rgb(str(img_dir / img_file))
            gt01 = load_gt_mask01(str(msk_dir / mask_file))
            pred01 = predict_mask01_flip_aug(clf, img_rgb) if args.flip_aug else predict_mask01(clf, img_rgb)

            ov = overlay_pred(img_rgb, pred01)
            cv2.imwrite(str(fail_dir / f"{stem}_overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            save_error_map(gt01, pred01, str(fail_dir / f"{stem}_error.png"))
            cv2.imwrite(str(fail_dir / f"{stem}_pred.png"), pred01 * 255)
            cv2.imwrite(str(fail_dir / f"{stem}_gt.png"), gt01 * 255)

        print(f"\nSaved failure analysis → {fail_dir}")

    # Save JSON
    out_path = out_dir / f"test_metrics_rf_{tag}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    if args.visualise:
        print(f"Saved prediction examples → {fig_dir}")


if __name__ == "__main__":
    main()
