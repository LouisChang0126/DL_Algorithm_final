"""Plot per-run train/val curves and an across-run comparison figure."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_one(run_dir: Path) -> None:
    log = run_dir / "log.csv"
    if not log.exists():
        print(f"[skip] no log.csv in {run_dir}")
        return
    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    if not rows:
        return
    epochs = [int(r["epoch"]) for r in rows]
    tl = [float(r["train_loss"]) for r in rows]
    vl = [float(r["val_loss"]) for r in rows]
    ta = [float(r["train_acc"]) for r in rows]
    va = [float(r["val_acc"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, tl, "-o", label="train")
    axes[0].plot(epochs, vl, "-s", label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title(f"{run_dir.name} loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, ta, "-o", label="train")
    axes[1].plot(epochs, va, "-s", label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title(f"{run_dir.name} acc"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    out = run_dir / "curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[wrote] {out}")


def plot_comparison(runs_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for run_dir in sorted(runs_root.iterdir()):
        log = run_dir / "log.csv"
        if not log.exists():
            continue
        rows = list(csv.DictReader(log.open(encoding="utf-8")))
        if not rows:
            continue
        epochs = [int(r["epoch"]) for r in rows]
        va = [float(r["val_acc"]) for r in rows]
        ax.plot(epochs, va, label=run_dir.name)
        plotted += 1

    ax.set_xlabel("epoch"); ax.set_ylabel("val_acc")
    ax.set_title("val_acc — all runs")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = runs_root / "comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[wrote] {out} ({plotted} runs)")

    # Bar chart for final test_acc across all runs (incl. pca_dt which has no log.csv)
    bars = []
    for run_dir in sorted(runs_root.iterdir()):
        ev = run_dir / "eval.json"
        if not ev.exists():
            continue
        d = json.loads(ev.read_text(encoding="utf-8"))
        if "best" in d:
            test_acc = d["best"].get("test_acc")
        else:
            test_acc = d.get("test_acc")
        if test_acc is None:
            continue
        bars.append((run_dir.name, float(test_acc)))

    if bars:
        names = [b[0] for b in bars]
        vals = [b[1] for b in bars]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(names)), vals, color="steelblue")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_ylabel("test_acc")
        ax.set_title("test_acc — all runs")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0, max(1.0, max(vals) + 0.1))
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = runs_root / "test_acc_bar.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[wrote] {out} ({len(bars)} runs)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default=None, help="single run dir")
    p.add_argument("--runs_root", type=str, default="runs")
    p.add_argument("--comparison", action="store_true",
                   help="also produce comparison.png across all runs")
    p.add_argument("--all", action="store_true",
                   help="produce per-run curves for every dir in runs_root")
    args = p.parse_args()

    runs_root = Path(args.runs_root)
    if args.run:
        plot_one(Path(args.run))
    if args.all:
        for d in sorted(runs_root.iterdir()):
            if d.is_dir():
                plot_one(d)
    if args.comparison:
        plot_comparison(runs_root)


if __name__ == "__main__":
    main()
