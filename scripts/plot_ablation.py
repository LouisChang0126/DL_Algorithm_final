"""Plot augmentation ablation: val curves + a bar chart of final val/test acc.

Reads:
    runs/ablation_aug_none/      (resize only)
    runs/ablation_aug_hflip/     (+ hflip)
    runs/ablation_aug_hflip_rot/ (+ rotation)
    runs/resnet18_pretrained/    (full — hflip+rot+color_jitter, reused baseline)

Writes:
    runs/ablation_aug.png         curves + bar chart
    runs/ablation_aug_table.csv   numeric summary used in report.md
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ABLATION = [
    ("none",      "runs/ablation_aug_none"),
    ("hflip",     "runs/ablation_aug_hflip"),
    ("hflip_rot", "runs/ablation_aug_hflip_rot"),
    ("full",      "runs/resnet18_pretrained"),  # baseline already in main runs
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: val_acc curves
    for level, path in ABLATION:
        rows = read_log(Path(path))
        if not rows:
            print(f"[skip] no log.csv in {path}")
            continue
        epochs = [int(r["epoch"]) for r in rows]
        va = [float(r["val_acc"]) for r in rows]
        axes[0].plot(epochs, va, "-o", markersize=3, label=level)

    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("val_acc")
    axes[0].set_title("Augmentation ablation — val_acc per epoch")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Right: bar chart of best_val_acc & test_acc
    levels: list[str] = []
    val_vals: list[float] = []
    test_vals: list[float] = []
    for level, path in ABLATION:
        ev = read_eval(Path(path))
        if ev is None:
            print(f"[skip] no eval.json in {path}")
            continue
        levels.append(level)
        val_vals.append(float(ev["best_val_acc"]))
        test_vals.append(float(ev["test_acc"]))
        rows_table.append({
            "aug_level": level,
            "run_dir": path,
            "best_epoch": ev.get("best_epoch"),
            "best_val_acc": f"{ev['best_val_acc']:.4f}",
            "test_acc": f"{ev['test_acc']:.4f}",
        })

    x = range(len(levels))
    w = 0.38
    axes[1].bar([i - w / 2 for i in x], val_vals, width=w, label="best val_acc",
                color="steelblue")
    axes[1].bar([i + w / 2 for i in x], test_vals, width=w, label="test_acc",
                color="darkorange")
    axes[1].set_xticks(list(x), levels)
    axes[1].set_xlabel("augmentation level")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Augmentation ablation — final acc")
    axes[1].legend(loc="lower right")
    axes[1].set_ylim(min(val_vals + test_vals) - 0.01,
                     max(val_vals + test_vals) + 0.01)
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(val_vals):
        axes[1].text(i - w / 2, v + 0.0005, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(test_vals):
        axes[1].text(i + w / 2, v + 0.0005, f"{v:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    out_png = Path("runs/ablation_aug.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"[wrote] {out_png}")

    out_csv = Path("runs/ablation_aug_table.csv")
    if rows_table:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_table[0].keys()))
            writer.writeheader()
            writer.writerows(rows_table)
        print(f"[wrote] {out_csv}")


if __name__ == "__main__":
    main()
