"""PCA + Decision Tree baseline.

Replaces the original 'CNN + DT' model from note.md. No epochs, no checkpoints.
Pipeline:
    PIL image -> grayscale 64x64 -> flatten (4096-d)
              -> PCA(n_components) on train fit, transform all splits
              -> DecisionTreeClassifier(max_depth=d) per (n, d) cell
              -> pick (n, d) with best val acc, evaluate on test set
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import random_split
import torch
from torchvision.datasets import ImageFolder

from src.utils import set_seed, write_json


GRAY_SIZE = 64
N_COMPONENTS_GRID = [64, 128, 256]
MAX_DEPTH_GRID = [3, 5, 10, 20, 30, 50]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--name", type=str, default="pca_dt")
    p.add_argument("--quick", action="store_true",
                   help="run only (n=64, depth=10) for smoke test")
    return p.parse_args()


def load_split_as_arrays(folder: Path, indices: np.ndarray | None = None,
                         dataset: ImageFolder | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load images from an ImageFolder (optionally restricted to `indices`),
    resize to 64x64 grayscale, flatten to (N, 4096) float32 in [0, 1]."""
    if dataset is None:
        dataset = ImageFolder(str(folder))
    samples = dataset.samples if indices is None else [dataset.samples[i] for i in indices]
    n = len(samples)
    X = np.empty((n, GRAY_SIZE * GRAY_SIZE), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    for i, (path, label) in enumerate(samples):
        with Image.open(path) as im:
            im = im.convert("L").resize((GRAY_SIZE, GRAY_SIZE), Image.BILINEAR)
            X[i] = np.asarray(im, dtype=np.float32).reshape(-1) / 255.0
        y[i] = label
        if (i + 1) % 1000 == 0:
            print(f"  loaded {i + 1}/{n}")
    return X, y


def plot_grid_heatmap(grid_csv: Path, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(grid_csv.open(encoding="utf-8")))
    ns = sorted({int(r["n_components"]) for r in rows})
    ds = sorted({int(r["max_depth"]) for r in rows})
    mat = np.full((len(ns), len(ds)), np.nan)
    for r in rows:
        i = ns.index(int(r["n_components"]))
        j = ds.index(int(r["max_depth"]))
        mat[i, j] = float(r["val_acc"])

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(ds)), [str(d) for d in ds])
    ax.set_yticks(range(len(ns)), [str(n) for n in ns])
    ax.set_xlabel("max_depth")
    ax.set_ylabel("n_components")
    ax.set_title("PCA+DT val_acc grid")
    for i in range(len(ns)):
        for j in range(len(ds)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="white" if v < mat.max() * 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="val_acc")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def plot_depth_curve(grid_csv: Path, best_n: int, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    rows = [r for r in csv.DictReader(grid_csv.open(encoding="utf-8"))
            if int(r["n_components"]) == best_n]
    rows.sort(key=lambda r: int(r["max_depth"]))
    depths = [int(r["max_depth"]) for r in rows]
    train_acc = [float(r["train_acc"]) for r in rows]
    val_acc = [float(r["val_acc"]) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(depths, train_acc, "-o", label="train_acc")
    ax.plot(depths, val_acc, "-s", label="val_acc")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("accuracy")
    ax.set_title(f"PCA+DT depth ablation (n_components={best_n})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(cfg["data"]["root"])
    train_dir = data_root / cfg["data"]["train_subdir"]
    test_dir = data_root / cfg["data"]["test_subdir"]

    print(f"[data] reading {train_dir}")
    base = ImageFolder(str(train_dir))
    n_total = len(base)
    n_val = int(round(n_total * cfg["val_split"]))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(cfg["seed"])
    train_sub, val_sub = random_split(base, [n_train, n_val], generator=g)

    print(f"[data] loading train ({n_train}) ...")
    X_train, y_train = load_split_as_arrays(train_dir, np.array(train_sub.indices), base)
    print(f"[data] loading val ({n_val}) ...")
    X_val, y_val = load_split_as_arrays(train_dir, np.array(val_sub.indices), base)
    print(f"[data] loading test from {test_dir}")
    X_test, y_test = load_split_as_arrays(test_dir)
    print(f"[data] num_classes={len(base.classes)} classes={base.classes}")

    n_grid = [64] if args.quick else N_COMPONENTS_GRID
    d_grid = [10] if args.quick else MAX_DEPTH_GRID

    grid_csv = run_dir / "grid.csv"
    with grid_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n_components", "max_depth", "train_acc", "val_acc",
        ])
        writer.writeheader()

        cache_pca: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
        for n in n_grid:
            print(f"[pca] fitting n_components={n}")
            pca = PCA(n_components=n, random_state=cfg["seed"])
            Xtr = pca.fit_transform(X_train)
            Xva = pca.transform(X_val)
            Xte = pca.transform(X_test)
            evr = float(pca.explained_variance_ratio_.sum())
            cache_pca[n] = (Xtr, Xva, Xte, evr)
            print(f"  explained variance ratio sum = {evr:.4f}")

            for d in d_grid:
                clf = DecisionTreeClassifier(max_depth=d, random_state=cfg["seed"])
                clf.fit(Xtr, y_train)
                tr = float((clf.predict(Xtr) == y_train).mean())
                va = float((clf.predict(Xva) == y_val).mean())
                writer.writerow({
                    "n_components": n, "max_depth": d,
                    "train_acc": f"{tr:.4f}", "val_acc": f"{va:.4f}",
                })
                print(f"  n={n} d={d}: train={tr:.4f} val={va:.4f}")

    # Pick best (n, d) by val_acc
    rows = list(csv.DictReader(grid_csv.open(encoding="utf-8")))
    best = max(rows, key=lambda r: float(r["val_acc"]))
    best_n = int(best["n_components"])
    best_d = int(best["max_depth"])
    Xtr, _, Xte, evr = cache_pca[best_n]

    print(f"[best] n_components={best_n} max_depth={best_d} "
          f"val_acc={best['val_acc']}")
    final_clf = DecisionTreeClassifier(max_depth=best_d, random_state=cfg["seed"])
    final_clf.fit(Xtr, y_train)
    test_acc = float((final_clf.predict(Xte) == y_test).mean())
    print(f"[test] acc={test_acc:.4f}")

    write_json({
        "best": {
            "n_components": best_n,
            "max_depth": best_d,
            "val_acc": float(best["val_acc"]),
            "train_acc": float(best["train_acc"]),
            "test_acc": test_acc,
            "explained_variance_ratio_sum": evr,
        },
        "grid": {
            "n_components": n_grid,
            "max_depth": d_grid,
        },
        "model": "pca_dt",
        "n_train": n_train, "n_val": n_val, "n_test": int(len(y_test)),
    }, run_dir / "eval.json")

    if not args.quick:
        plot_grid_heatmap(grid_csv, run_dir / "grid.png")
        plot_depth_curve(grid_csv, best_n, run_dir / "depth_curve.png")

    # Save effective config
    eff = dict(cfg)
    eff.update({
        "model": "pca_dt", "experiment_name": args.name,
        "gray_size": GRAY_SIZE,
        "n_components_grid": n_grid, "max_depth_grid": d_grid,
    })
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(eff, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
