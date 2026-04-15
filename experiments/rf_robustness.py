import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json, time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.random_forest import load_model


# ---------- EWS helpers ----------
def load_image_rgb(p):
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def load_gt_mask01(p):
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m == 0).astype(np.uint8)

def predict_mask01(clf, img_rgb):
    H, W, _ = img_rgb.shape
    X = img_rgb.reshape(-1, 3).astype(np.float32)
    return clf.predict(X).reshape(H, W).astype(np.uint8)

def iou(gt01, pred01):
    inter = np.logical_and(gt01 == 1, pred01 == 1).sum()
    union = np.logical_or(gt01 == 1, pred01 == 1).sum()
    return float(inter / (union + 1e-8))

# ---------- Distortions (OpenCV) ----------
def low_contrast(img, alpha=0.6):
    return np.clip(128 + alpha * (img - 128), 0, 255).astype(np.uint8)

def low_brightness(img, beta=-40):
    return np.clip(img.astype(np.int16) + beta, 0, 255).astype(np.uint8)

def gaussian_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def blur(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)

def occlusion(img, n=8, size=30):
    out = img.copy()
    H, W, _ = out.shape
    rng = np.random.default_rng(0)
    for _ in range(n):
        y = int(rng.integers(0, max(1, H - size)))
        x = int(rng.integers(0, max(1, W - size)))
        out[y:y+size, x:x+size] = 0
    return out

def jpeg_compress(img, quality=20):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

DISTORTIONS = [
    ("clean", lambda x: x),
    ("low_contrast", lambda x: low_contrast(x, alpha=0.6)),
    ("low_brightness", lambda x: low_brightness(x, beta=-40)),
    ("gaussian_noise_mild", lambda x: gaussian_noise(x, sigma=10)),
    ("gaussian_noise_strong", lambda x: gaussian_noise(x, sigma=25)),
    ("occlusion", lambda x: occlusion(x, n=8, size=30)),
    ("blur_mild", lambda x: blur(x, k=5)),
    ("blur_strong", lambda x: blur(x, k=11)),
    ("jpeg_compression", lambda x: jpeg_compress(x, quality=20)),
]

def main():
    data_root = Path("data/EWS-Dataset")
    img_dir = data_root / "test" / "images"
    msk_dir = data_root / "test" / "masks"
    image_files = sorted([p.name for p in img_dir.glob("*.png")])

    out_dir = Path("results/rf_full")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    clf = load_model("results/rf_model.pkl") if load_model.__code__.co_argcount >= 1 else load_model()

    results = []
    for name, fn in DISTORTIONS:
        ious = []
        t0 = time.perf_counter()
        for img_file in image_files:
            stem = Path(img_file).stem
            img = load_image_rgb(img_dir / img_file)
            gt  = load_gt_mask01(msk_dir / f"{stem}_mask.png")

            img_d = fn(img)
            pred = predict_mask01(clf, img_d)
            ious.append(iou(gt, pred))
        t1 = time.perf_counter()

        results.append({
            "distortion": name,
            "iou": float(np.mean(ious)),
            "std": float(np.std(ious)),
            "ms_img": float((t1 - t0) * 1000 / len(image_files)),
        })
        print(name, "IoU", results[-1]["iou"])

    # Save JSON
    out_path = out_dir / "rf_robustness.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved:", out_path)

    # Plot (bar + dashed clean line)
    clean_iou = [r["iou"] for r in results if r["distortion"] == "clean"][0]
    labels = [r["distortion"] for r in results]
    vals   = [r["iou"] for r in results]

    colors = []
    for lab, v in zip(labels, vals):
        if lab == "clean":
            colors.append("#54A24B")  # green
        elif v >= clean_iou:
            colors.append("#F58518")  # orange
        else:
            colors.append("#E45756")  # red

    plt.figure(figsize=(10,4.5))
    plt.bar(range(len(vals)), vals, color=colors)
    plt.axhline(clean_iou, linestyle="--", color="black", linewidth=1, label=f"Clean IoU ({clean_iou:.3f})")
    plt.ylim(0, 1)
    plt.ylabel("IoU")
    plt.title("Robustness Under Image Distortions — Random Forest")
    plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "rf_robustness.png", dpi=200)
    print("Saved:", fig_dir / "rf_robustness.png")

if __name__ == "__main__":
    main()
