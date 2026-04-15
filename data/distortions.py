"""
Synthetic image distortions for robustness evaluation.

Each distortion takes a numpy RGB image (H, W, 3) float32 [0-255]
and returns a distorted version of the same shape.
Used to evaluate how well trained models generalise to degraded inputs.
"""

import cv2
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Individual distortion functions
# ---------------------------------------------------------------------------

def apply_gaussian_noise(image: np.ndarray, var: float = 500.0) -> np.ndarray:
    """Add Gaussian noise to simulate sensor noise."""
    noise  = np.random.normal(0, var ** 0.5, image.shape).astype(np.float32)
    return np.clip(image + noise, 0, 255).astype(np.float32)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 11) -> np.ndarray:
    """Blur to simulate camera defocus or motion."""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), 0).astype(np.float32)


def apply_low_brightness(image: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """Darken image to simulate poor lighting or overcast conditions."""
    return np.clip(image * factor, 0, 255).astype(np.float32)


def apply_low_contrast(image: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """Reduce contrast — simulate flat / foggy conditions."""
    mean = image.mean()
    return np.clip(mean + (image - mean) * factor, 0, 255).astype(np.float32)


def apply_partial_occlusion(
    image:       np.ndarray,
    n_patches:   int = 5,
    patch_size:  int = 60,
) -> np.ndarray:
    """Randomly black out rectangular patches to simulate partial occlusion."""
    out = image.copy()
    H, W = out.shape[:2]
    for _ in range(n_patches):
        y = np.random.randint(0, H - patch_size)
        x = np.random.randint(0, W - patch_size)
        out[y:y + patch_size, x:x + patch_size] = 0
    return out.astype(np.float32)


def apply_jpeg_compression(image: np.ndarray, quality: int = 10) -> np.ndarray:
    """Simulate low-quality JPEG compression artefacts."""
    img_u8  = np.clip(image, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc  = cv2.imencode(".jpg", cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR), encode_param)
    dec     = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype(np.float32)


# ---------------------------------------------------------------------------
# Distortion registry — used by robustness_eval.py
# ---------------------------------------------------------------------------

DISTORTIONS = {
    "clean": lambda img: img.copy(),
    "gaussian_noise_mild":   lambda img: apply_gaussian_noise(img, var=200),
    "gaussian_noise_strong": lambda img: apply_gaussian_noise(img, var=800),
    "blur_mild":             lambda img: apply_gaussian_blur(img, kernel_size=7),
    "blur_strong":           lambda img: apply_gaussian_blur(img, kernel_size=15),
    "low_brightness":        lambda img: apply_low_brightness(img, factor=0.4),
    "low_contrast":          lambda img: apply_low_contrast(img, factor=0.4),
    "occlusion":             lambda img: apply_partial_occlusion(img, n_patches=5, patch_size=60),
    "jpeg_compression":      lambda img: apply_jpeg_compression(img, quality=10),
}
