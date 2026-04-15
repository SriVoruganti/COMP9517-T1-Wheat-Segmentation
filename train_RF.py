"""
Training script for Random Forest wheat pixel classifier.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from data.dataset import EWSDatasetRF


from models.random_forest import build_model, save_model

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "EWS-Dataset")

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
print("Loading training data...")
train_ds = EWSDatasetRF(root=DATA_ROOT, split="train", max_pixels_per_image=5000)
X_train, y_train = train_ds.load()


print("Loading validation data...")
val_ds = EWSDatasetRF(root=DATA_ROOT, split="val", max_pixels_per_image=5000)
X_val, y_val = val_ds.load()

print(f"  X_train : {X_train.shape}")
print(f"  X_val   : {X_val.shape}")
print(f"  Class balance (train): {y_train.mean():.2%} wheat pixels")

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
print("\nTraining Random Forest...")
clf = build_model()
clf.fit(X_train, y_train)

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------
save_model(clf)