"""
EWS Dataset loader with support for:
  - Standard train/val/test loading
  - Subset sampling for data scarcity experiments
  - Label noise injection for robustness analysis
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EWSDataset(Dataset):
    """
    Eschikon Wheat Segmentation (EWS) Dataset.

    Directory structure expected:
        root/
            train/images/*.png   train/masks/*.png
            val/images/*.png     val/masks/*.png
            test/images/*.png    test/masks/*.png

    Args:
        root:         Path to EWS dataset root.
        split:        'train', 'val', or 'test'.
        transform:    Albumentations pipeline.
        subset_frac:  Float in (0, 1] — use only this fraction of the split.
                      Useful for data scarcity experiments.
        label_noise:  Float in [0, 1) — randomly flip this fraction of mask
                      pixels to simulate noisy annotation.
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

        self.image_dir   = os.path.join(root, split, "images")
        self.mask_dir    = os.path.join(root, split, "masks")
        self.transform   = transform
        self.label_noise = label_noise

        images = sorted(os.listdir(self.image_dir))
        masks  = sorted(os.listdir(self.mask_dir))
        assert len(images) == len(masks), (
            f"Mismatch: {len(images)} images vs {len(masks)} masks"
        )

        # Subset sampling
        if subset_frac < 1.0:
            rng = random.Random(seed)
            n   = max(1, int(len(images) * subset_frac))
            idx = rng.sample(range(len(images)), n)
            images = [images[i] for i in sorted(idx)]
            masks  = [masks[i]  for i in sorted(idx)]

        self.images = images
        self.masks  = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir,  self.masks[idx])

        image = np.array(Image.open(img_path).convert("RGB"),  dtype=np.float32)
        mask  = np.array(Image.open(mask_path).convert("L"),   dtype=np.float32)
        mask  = (mask > 127).astype(np.float32)

        # Inject label noise (flip random pixels)
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


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 350) -> A.Compose:
    """Comprehensive augmentation for training on a small dataset."""
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
        # Normalise
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
