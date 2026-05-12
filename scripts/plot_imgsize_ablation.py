"""Plot input image size ablation: val curves + acc / epoch-time vs img_size.

Reads:
    runs/ablation_imgsize_128/
    runs/ablation_imgsize_192/
    runs/ablation_imgsize_224/
    runs/resnet18_pretrained/   (256, baseline reused)
    runs/ablation_imgsize_384/

Writes:
    runs/ablation_imgsize.png        curves + summary chart
    runs/ablation_imgsize_table.csv  numeric summary used in report.md
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ABLATION = [
    (128, "runs/ablation_imgsize_128"),
    (192, "runs/ablation_imgsize_192"),
    (224, "runs/ablation_imgsize_224"),
    (256, "runs/resnet18_pretrained"),     # baseline reused
    (384, "runs/ablation_imgsize_384"),
]


def read_log(run_dir: Path) -> list[dict]:
    log = run_dir / "log.csv"
    if not log.exists():
        return []
    return list(csv.DictReader(log.open(encoding="utf-8")))


def read_eval(run_dir: Path) -> dict | None:
    ev = run_dir / "eval.json"
    if not ev.exists():
        return None
    return json.loads(ev.read_text(encoding="utf-8"))


def main() -> None:
    rows_table: list[dict] = []
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    # (1) Val_acc per epoch — one line per size
    for size, path in ABLATION:
        rows = read_log(Path(path))
        if not rows:
            print(f"[skip] no log.csv in {path}")
            continue
        epochs = [int(r["epoch"]) for r in rows]
        va = [float(r["val_acc"]) for r in rows]
        axes[0].plot(epochs, va, "-o", markersize=3, label=f"{size}px")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val_acc")
    axes[0].set_title("Img-size ablation — val_acc per epoch")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # (2) Final val/test acc vs img_size
    sizes: list[int] = []
    val_vals: list[float] = []
    test_vals: list[float] = []
    epoch_times: list[float] = []
    for size, path in ABLATION:
        ev = read_eval(Path(path))
        if ev is None:
            continue
        sizes.append(size)
        val_vals.append(float(ev["best_val_acc"]))
        test_vals.append(float(ev["test_acc"]))
        rows = read_log(Path(path))
        avg_t = (sum(float(r["epoch_time_sec"]) for r in rows[1:]) / max(1, len(rows) - 1)
                 if rows else float("nan"))  # skip epoch-1 warmup
        epoch_times.append(avg_t)
        rows_table.append({
            "img_size": size,
            "run_dir": path,
            "best_epoch": ev.get("best_epoch"),
            "best_val_acc": f"{ev['best_val_acc']:.4f}",
            "test_acc": f"{ev['test_acc']:.4f}",
            "avg_epoch_time_sec": f"{avg_t:.1f}",
        })

    axes[1].plot(sizes, val_vals, "-o", label="best val_acc", color="steelblue")
    axes[1].plot(sizes, test_vals, "-s", label="test_acc", color="darkorange")
    axes[1].set_xlabel("input size (px)")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Img-size ablation — final acc")
    axes[1].legend(loc="lower right")
    axes[1].grid(alpha=0.3)
    for x, v in zip(sizes, val_vals):
        axes[1].text(x, v + 0.0005, f"{v:.3f}", ha="center", fontsize=8,
                     color="steelblue")
    for x, v in zip(sizes, test_vals):
        axes[1].text(x, v - 0.001, f"{v:.3f}", ha="center", va="top",
                     fontsize=8, color="darkorange")

    # (3) Avg epoch time vs img_size (cost side of trade-off)
    axes[2].plot(sizes, epoch_times, "-D", color="firebrick")
    axes[2].set_xlabel("input size (px)")
    axes[2].set_ylabel("avg epoch time (s, excl. warmup)")
    axes[2].set_title("Img-size ablation — training cost")
    axes[2].grid(alpha=0.3)
    for x, t in zip(sizes, epoch_times):
        axes[2].text(x, t + 0.3, f"{t:.1f}s", ha="center", fontsize=8)

    fig.tight_layout()
    out_png = Path("runs/ablation_imgsize.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[wrote] {out_png}")

    out_csv = Path("runs/ablation_imgsize_table.csv")
    if rows_table:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_table[0].keys()))
            writer.writeheader()
            writer.writerows(rows_table)
        print(f"[wrote] {out_csv}")


if __name__ == "__main__":
    main()
