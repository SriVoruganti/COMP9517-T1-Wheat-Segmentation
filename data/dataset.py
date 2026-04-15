"""
EWS Dataset loader with support for:
  - Deep Learning (PyTorch) — U-Net training and evaluation
  - Classical ML (Random Forest) — pixel-level feature extraction
  - Standard train/val/test loading
  - Automatic folder preprocessing
  - Subset sampling for data scarcity experiments
  - Label noise injection for robustness analysis
"""

import os
import random
import shutil
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ===========================================================================
# Shared Utility — Folder Preprocessing
# ===========================================================================

def preprocess_folders(root: str, split: str):
    """
    Creates images/ and masks/ subdirectories and moves files into them
    if they don't already exist. Safe to re-run.

    Also handles the 'validation' → 'val' rename automatically.
    """
    # Handle validation → val rename
    source_dir = os.path.join(root, "validation")
    target_dir = os.path.join(root, "val")
    if os.path.exists(source_dir) and not os.path.exists(target_dir):
        os.rename(source_dir, target_dir)

    split_dir  = os.path.join(root, split)
    images_dir = os.path.join(split_dir, "images")
    masks_dir  = os.path.join(split_dir, "masks")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir,  exist_ok=True)

    for file in sorted(os.listdir(split_dir)):
        file_path = os.path.join(split_dir, file)

        if os.path.isdir(file_path):
            continue
        if not file.lower().endswith(".png"):
            continue

        dest = os.path.join(
            masks_dir if "_mask.png" in file else images_dir,
            file
        )
        if not os.path.exists(dest):
            shutil.move(file_path, dest)


def _load_file_lists(root, split, subset_frac, seed):
    """Shared helper to load and optionally subset image/mask file lists."""
    image_dir = os.path.join(root, split, "images")
    mask_dir  = os.path.join(root, split, "masks")

    images = sorted(os.listdir(image_dir))
    masks  = sorted(os.listdir(mask_dir))
    assert len(images) == len(masks), (
        f"Mismatch: {len(images)} images vs {len(masks)} masks"
    )

    if subset_frac < 1.0:
        rng = random.Random(seed)
        n   = max(1, int(len(images) * subset_frac))
        idx = rng.sample(range(len(images)), n)
        images = [images[i] for i in sorted(idx)]
        masks  = [masks[i]  for i in sorted(idx)]

    return image_dir, mask_dir, images, masks


# ===========================================================================
# Deep Learning Dataset — PyTorch / U-Net
# ===========================================================================

class EWSDataset(Dataset):
    """
    PyTorch Dataset for U-Net training and evaluation.

    Returns whole images as normalised tensors with albumentations
    augmentations applied. Use this with train.py and evaluate.py.

    Directory structure expected:
        root/
            train/images/*.png   train/masks/*.png
            val/images/*.png     val/masks/*.png
            test/images/*.png    test/masks/*.png

    Args:
        root:         Path to EWS dataset root.
        split:        'train', 'val', or 'test'.
        transform:    Albumentations pipeline (use get_train_transforms
                      or get_val_transforms below).
        subset_frac:  Float in (0, 1] — use only this fraction of the
                      split. Useful for data scarcity experiments.
        label_noise:  Float in [0, 1) — randomly flip this fraction of
                      mask pixels to simulate noisy annotation.
        seed:         Random seed for reproducibility.
    """

    def __init__(
        self,
        root:        str,
        split:       str   = "train",
        transform          = None,
        subset_frac: float = 1.0,
        label_noise: float = 0.0,
        seed:        int   = 42,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: '{split}'"
        assert 0 < subset_frac <= 1.0, "subset_frac must be in (0, 1]"
        assert 0 <= label_noise < 1.0, "label_noise must be in [0, 1)"

        self.transform   = transform
        self.label_noise = label_noise

        self.image_dir, self.mask_dir, self.images, self.masks = (
            _load_file_lists(root, split, subset_frac, seed)
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir,  self.masks[idx])

        image = np.array(Image.open(img_path).convert("RGB"),  dtype=np.float32)
        mask  = np.array(Image.open(mask_path).convert("L"),   dtype=np.float32)
        mask  = (mask > 127).astype(np.float32)

        # Inject label noise
        if self.label_noise > 0:
            noise_map = np.random.rand(*mask.shape) < self.label_noise
            mask      = np.where(noise_map, 1.0 - mask, mask)

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]

        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
        else:
            mask = torch.tensor(mask).unsqueeze(0)

        return image, mask

    def get_filename(self, idx: int) -> str:
        return self.images[idx]


# ===========================================================================
# Classical ML Dataset — Random Forest / Pixel Features
# ===========================================================================

class EWSDatasetRF:
    """
    Classical ML Dataset for Random Forest training and evaluation.

    Flattens images into a pixel-level RGB feature matrix (n_pixels, 3)
    suitable for sklearn classifiers. Handles raw EWS folder structure
    automatically via preprocessing.

    Directory structure after preprocessing:
        root/
            train/images/*.png   train/masks/*.png
            val/images/*.png     val/masks/*.png
            test/images/*.png    test/masks/*.png

    Args:
        root:                 Path to EWS dataset root.
        split:                'train', 'val', or 'test'.
        max_pixels_per_image: Max pixels to sample per image.
                              Keeps memory manageable for RF training.
        subset_frac:          Float in (0, 1] — use only this fraction
                              of the split. For data scarcity experiments.
        label_noise:          Float in [0, 1) — randomly flip this fraction
                              of mask pixels. For robustness analysis.
        seed:                 Random seed for reproducibility.
    """

    def __init__(
        self,
        root:                 str,
        split:                str   = "train",
        max_pixels_per_image: int   = 5000,
        subset_frac:          float = 1.0,
        label_noise:          float = 0.0,
        seed:                 int   = 42,
    ):
        assert split in ("train", "val", "test"), f"Invalid split: '{split}'"
        assert 0 < subset_frac <= 1.0,  "subset_frac must be in (0, 1]"
        assert 0 <= label_noise < 1.0,  "label_noise must be in [0, 1)"

        self.max_pixels_per_image = max_pixels_per_image
        self.label_noise          = label_noise
        self.seed                 = seed

        # Auto-preprocess raw EWS folder structure
        preprocess_folders(root, split)

        self.image_dir, self.mask_dir, self.image_files, self.mask_files = (
            _load_file_lists(root, split, subset_frac, seed)
        )

    def load(self):
        """
        Loads all images and masks into a flat numpy feature matrix.

        Returns:
            X: (n_pixels, 3)  — RGB features per pixel
            y: (n_pixels,)    — binary label per pixel (0=background, 1=wheat)
        """
        np.random.seed(self.seed)

        X_list = []
        y_list = []

        for img_file, mask_file in zip(self.image_files, self.mask_files):
            img_path  = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir,  mask_file)

            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(np.float32)
            mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask  = (mask > 127).astype(np.float32)

            # Inject label noise
            if self.label_noise > 0:
                noise_map = np.random.rand(*mask.shape) < self.label_noise
                mask      = np.where(noise_map, 1.0 - mask, mask)

            # Flatten and subsample pixels
            H, W, C = image.shape
            pixels  = image.reshape(-1, C)
            labels  = mask.reshape(-1)

            idx = np.random.choice(
                len(pixels),
                size=min(self.max_pixels_per_image, len(pixels)),
                replace=False
            )
            X_list.append(pixels[idx])
            y_list.append(labels[idx])

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        return X, y

    def get_filename(self, idx: int) -> str:
        return self.image_files[idx]

    def __len__(self):
        return len(self.image_files)


# ===========================================================================
# Augmentation Pipelines (for EWSDataset / U-Net)
# ===========================================================================

def get_train_transforms(image_size: int = 350) -> A.Compose:
    """Comprehensive augmentation pipeline for U-Net training."""
    return A.Compose([
        A.Resize(image_size, image_size),
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.3),
        # Colour / photometric
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.6),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        # Noise & blur
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.ISONoise(p=0.2),
        # Occlusion
        A.CoarseDropout(max_holes=8, max_height=30, max_width=30, p=0.3),
        # Normalise to ImageNet stats
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 350) -> A.Compose:
    """No augmentation for validation and testing."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
