"""
Evaluation script for Random Forest wheat pixel classifier.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path


sys.path.append(os.path.dirname(__file__))

from data.dataset import EWSDatasetRF
from models.random_forest import load_model

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "EWS-Dataset")

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
print("Loading test data...")
test_ds        = EWSDatasetRF(root=DATA_ROOT, split="test")
X_test, y_test = test_ds.load()

# ------------------------------------------------------------------
# Load model and predict
# ------------------------------------------------------------------
print("Loading model...")
clf    = load_model()
y_pred = clf.predict(X_test)

# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

metrics = {
    "precision": precision_score(y_test, y_pred),
    "recall":    recall_score(y_test, y_pred),
    "f1":        f1_score(y_test, y_pred),
    "iou":       jaccard_score(y_test, y_pred),
}

for k, v in metrics.items():
    print(f"  {k:<12}: {v:.4f}")

# ------------------------------------------------------------------
# Qualitative outputs (full-image predictions)
# ------------------------------------------------------------------
out_dir = Path(os.path.dirname(__file__)) / "results" / "rf_qual"
out_dir.mkdir(parents=True, exist_ok=True)

# Use the dataset file lists to locate original images/masks
img_dir = Path(test_ds.image_dir)
msk_dir = Path(test_ds.mask_dir)

# Save a few examples
N = 8
rng = np.random.default_rng(0)
idxs = rng.choice(len(test_ds.image_files), size=min(N, len(test_ds.image_files)), replace=False)

for i in idxs:
    img_file  = test_ds.image_files[i]
    mask_file = test_ds.mask_files[i]

    img_path  = img_dir / img_file
    mask_path = msk_dir / mask_file

    # Load image (RGB)
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    H, W, _ = img_rgb.shape

    # GT mask using the SAME EWS rule as training (channel 0, plant==0)
    gt = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    gt01 = (gt == 0).astype(np.uint8)

    # Predict full image
    X = img_rgb.reshape(-1, 3).astype(np.float32)
    pred01 = clf.predict(X).reshape(H, W).astype(np.uint8)

    # Save masks
    stem = Path(img_file).stem
    cv2.imwrite(str(out_dir / f"{stem}_pred.png"), pred01 * 255)
    cv2.imwrite(str(out_dir / f"{stem}_gt.png"),   gt01 * 255)

    # Overlay (pred in red)
    overlay = img_rgb.copy()
    overlay[pred01 == 1] = (0.5 * overlay[pred01 == 1] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(f"\nSaved qualitative predictions → {out_dir}")

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
results_path = os.path.join(os.path.dirname(__file__), "results", "test_metrics_rf.json")
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved → {results_path}")