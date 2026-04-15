"""
Advanced Segmentation Method — HSV Colour Thresholding + Morphological Post-processing
UNSW COMP9517 Group Project 2026 T1

Usage:
    python meanshift_segment.py --data_root ./EWS-Dataset --index 0
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_and_mask(folder_path, index=0):
    """Load an RGB image and its corresponding binary ground-truth mask.

    Args:
        folder_path: Path to a dataset split folder (e.g. EWS-Dataset/train).
        index:       Index of the image to load (sorted alphabetically).

    Returns:
        img_rgb    : H×W×3 uint8 RGB image.
        true_mask  : H×W uint8 binary mask  (1 = soil, 0 = wheat).
        img_name   : Filename of the loaded image.
    """
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if not f.endswith('mask.png') and f.endswith(('.jpg', '.png'))
    ])

    img_name  = image_files[index]
    img_path  = os.path.join(folder_path, img_name)
    mask_name = img_name.rsplit('.', 1)[0] + '_mask.png'
    mask_path = os.path.join(folder_path, mask_name)

    img_rgb   = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, true_mask = cv2.threshold(true_mask, 127, 1, cv2.THRESH_BINARY)

    return img_rgb, true_mask, img_name


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_wheat_meanshift(image_rgb):
    """Segment wheat pixels using HSV colour thresholding + morphological ops.

    Pipeline:
        1. Convert RGB → HSV (more robust to lighting variation than RGB).
        2. Threshold the hue/saturation/value channels to isolate green/yellow
           wheat tones.
        3. Morphological opening  — removes small noise specks in soil regions.
        4. Morphological closing  — fills small holes within wheat leaves.
        5. Invert so that the convention matches ground truth
           (1 = soil/background, 0 = wheat).

    Args:
        image_rgb: H×W×3 uint8 RGB image.

    Returns:
        predicted_mask: H×W uint8 binary mask (1 = soil, 0 = wheat).
    """
    # --- 1. HSV conversion ---
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # --- 2. Colour thresholding (wheat green / yellow tones) ---
    # H (hue), S (saturation), V (brightness)
    lower_green = np.array([25, 35, 35])    # lower bound for wheat green/yellow
    upper_green = np.array([95, 255, 255])  # upper bound
    predicted_mask = cv2.inRange(hsv, lower_green, upper_green)

    # --- 3. Morphological post-processing ---
    kernel = np.ones((3, 3), np.uint8)
    predicted_mask = cv2.morphologyEx(predicted_mask, cv2.MORPH_OPEN,  kernel)  # remove noise
    predicted_mask = cv2.morphologyEx(predicted_mask, cv2.MORPH_CLOSE, kernel)  # fill holes

    # --- 4. Normalise & invert ---
    predicted_mask = (predicted_mask > 0).astype(np.uint8)
    predicted_mask = 1 - predicted_mask  # 1→0 (wheat), 0→1 (background)

    return predicted_mask


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def calculate_iou(mask_true, mask_pred):
    """Compute Intersection-over-Union between two binary masks."""
    intersection = np.logical_and(mask_true, mask_pred)
    union        = np.logical_or(mask_true, mask_pred)
    return float(np.sum(intersection) / np.sum(union)) if np.sum(union) > 0 else 0.0


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise(img_rgb, true_mask, predicted_mask, iou, save_path=None):
    """Plot original image, ground truth, and prediction side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(true_mask * 255, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(predicted_mask * 255, cmap='gray')
    axes[2].set_title(f'Prediction  (IoU: {iou:.4f})')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualisation → {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='HSV Mean-Shift Wheat Segmentation')
    parser.add_argument('--data_root', type=str, default='./EWS-Dataset',
                        help='Root directory of the EWS dataset')
    parser.add_argument('--split',     type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to run on')
    parser.add_argument('--index',     type=int, default=0,
                        help='Index of the image to process')
    parser.add_argument('--save',      type=str, default=None,
                        help='Optional path to save the visualisation (e.g. result.png)')
    args = parser.parse_args()

    folder_path = os.path.join(args.data_root, args.split)

    print(f"Loading image {args.index} from {folder_path} ...")
    img_rgb, true_mask, img_name = load_image_and_mask(folder_path, index=args.index)
    print(f"Loaded: {img_name}")

    print("Running HSV segmentation ...")
    predicted_mask = segment_wheat_meanshift(img_rgb)

    iou = calculate_iou(true_mask, predicted_mask)
    print(f"IoU Score: {iou:.4f}")

    visualise(img_rgb, true_mask, predicted_mask, iou, save_path=args.save)


if __name__ == '__main__':
    main()
