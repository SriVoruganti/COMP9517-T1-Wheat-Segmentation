import json
from pathlib import Path
import matplotlib.pyplot as plt

def load_summary(p):
    with open(p, "r") as f:
        return json.load(f)["summary"]

def main():
    out_dir = Path("results/rf_full/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {
        "RF (no aug)": Path("results/rf_full/test_metrics_rf_no_aug.json"),
        "RF (+ flip aug)": Path("results/rf_full/test_metrics_rf_flip_aug.json"),
    }

    rows = []
    for name, path in runs.items():
        if not path.exists():
            continue
        s = load_summary(path)
        rows.append((name, s["precision"], s["recall"], s["f1"], s["iou"], s["avg_inference_time_ms"]))

    # Print table to console
    print(f"{'Model':25} {'P':>7} {'R':>7} {'F1':>7} {'IoU':>7} {'ms/img':>10}")
    for r in rows:
        print(f"{r[0]:25} {r[1]:7.4f} {r[2]:7.4f} {r[3]:7.4f} {r[4]:7.4f} {r[5]:10.2f}")

    # Simple bar chart for IoU
    labels = [r[0] for r in rows]
    ious   = [r[4] for r in rows]

    plt.figure(figsize=(7,4))
    plt.bar(labels, ious, color=["#4C78A8", "#F58518"])
    plt.ylim(0, 1)
    plt.ylabel("IoU")
    plt.title("Test Set IoU — Random Forest")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "rf_test_iou.png", dpi=200)
    print("Saved:", out_dir / "rf_test_iou.png")

if __name__ == "__main__":
    main()
    