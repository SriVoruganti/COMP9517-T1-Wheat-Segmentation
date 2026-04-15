"""
Evaluation script for Random Forest wheat pixel classifier.
"""

import os
import sys
import json

sys.path.append(os.path.dirname(__file__))

from data.dataset import EWSDataset
from models.random_forest import load_model

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "EWS-Dataset")

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
print("Loading test data...")
test_ds        = EWSDataset(root=DATA_ROOT, split="test")
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
# Save
# ------------------------------------------------------------------
results_path = os.path.join(os.path.dirname(__file__), "results", "test_metrics_rf.json")
os.makedirs(os.path.dirname(results_path), exist_ok=True)

with open(results_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved → {results_path}")