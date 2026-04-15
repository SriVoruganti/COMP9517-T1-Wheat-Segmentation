import json, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from data.dataset import EWSDatasetRF
from models.random_forest import train_model, save_model, load_model

# If your random_forest.py uses different function names, paste it and I’ll align.

def main():
    out_dir = Path("results/rf_full")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fracs = [0.25, 0.50, 0.75, 1.00]
    rows = []

    for frac in fracs:
        print("\n=== frac", frac, "===")

        # Train
        train_ds = EWSDatasetRF(root="data/EWS-Dataset", split="train", subset_frac=frac, max_pixels_per_image=5000, seed=42)
        X_train, y_train = train_ds.load()

        t0 = time.perf_counter()
        clf = train_model(X_train, y_train)  # must exist in your models/random_forest.py
        t1 = time.perf_counter()

        model_path = out_dir / f"rf_model_frac_{int(frac*100)}.pkl"
        save_model(clf, str(model_path))

        # Evaluate full-image using your evaluate_rf_full JSON (reuse by importing would be nicer, but keep simple)
        # We'll do a quick full-image eval here:
        test_root = Path("data/EWS-Dataset/test")
        img_dir = test_root / "images"
        msk_dir = test_root / "masks"
        image_files = sorted([p.name for p in img_dir.glob("*.png")])

        def load_image_rgb(p):
            import cv2
            bgr = cv2.imread(str(p))
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        def load_gt_mask01(p):
            import cv2
            m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if m.ndim == 3:
                m = m[:, :, 0]
            return (m == 0).astype(np.uint8)

        def predict_mask01(img_rgb):
            H, W, _ = img_rgb.shape
            X = img_rgb.reshape(-1, 3).astype(np.float32)
            return clf.predict(X).reshape(H, W).astype(np.uint8)

        def f1_iou(gt01, pred01):
            gt = gt01.astype(bool)
            pr = pred01.astype(bool)
            tp = np.logical_and(gt, pr).sum()
            fp = np.logical_and(~gt, pr).sum()
            fn = np.logical_and(gt, ~pr).sum()
            f1 = (2*tp) / (2*tp + fp + fn + 1e-8)
            iou = tp / (tp + fp + fn + 1e-8)
            return float(f1), float(iou)

        f1s, ious = [], []
        for img_file in image_files:
            stem = Path(img_file).stem
            img = load_image_rgb(img_dir / img_file)
            gt  = load_gt_mask01(msk_dir / f"{stem}_mask.png")
            pred = predict_mask01(img)
            f1, iou = f1_iou(gt, pred)
            f1s.append(f1); ious.append(iou)

        rows.append({
            "frac": frac,
            "images": len(train_ds.image_files),
            "f1": float(np.mean(f1s)),
            "iou": float(np.mean(ious)),
            "train_time_s": float(t1 - t0),
        })

        print("images", rows[-1]["images"], "IoU", rows[-1]["iou"], "F1", rows[-1]["f1"])

    # Save JSON
    out_path = out_dir / "rf_data_scarcity.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print("Saved:", out_path)

    # Plot (line chart like your DL)
    xs = [int(r["frac"]*100) for r in rows]
    ious = [r["iou"] for r in rows]
    f1s  = [r["f1"] for r in rows]

    plt.figure(figsize=(7,4))
    plt.plot(xs, ious, marker="o", label="IoU")
    plt.plot(xs, f1s, marker="s", label="F1-Score")
    plt.ylim(0, 1)
    plt.xlabel("Training Data Used (%)")
    plt.ylabel("Score")
    plt.title("Data Scarcity Analysis — Random Forest")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "rf_data_scarcity.png", dpi=200)
    print("Saved:", fig_dir / "rf_data_scarcity.png")

if __name__ == "__main__":
    main()
