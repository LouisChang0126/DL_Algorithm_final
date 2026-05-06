"""Generate Grad-CAM overlays for one test image per class for each run.

Per run, picks the first correctly classified test image of every class (so
visualizations focus on what the model is *correctly* attending to). Saves a
2×5 grid (10 classes) at runs/<exp>/gradcam.png.

Skips runs/pca_dt (no model). For ViT runs: timm's fused attention path makes
classic Grad-CAM target the input to the last block's MHA (norm1), with a
reshape that drops the CLS token and forms a 16×16 patch-token grid.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib import cm

from src.data import build_loaders
from src.gradcam import GradCAM, get_target_layer
from src.models import build_model


def overlay_heatmap(img_chw: torch.Tensor, cam: torch.Tensor) -> np.ndarray:
    """img_chw: (C,H,W) in [0,1]; cam: (H,W) in [0,1]. Returns RGB uint8 (H,W,3)."""
    img = img_chw.permute(1, 2, 0).cpu().numpy()  # H,W,C
    heat = cm.jet(cam.numpy())[..., :3]  # H,W,3, drop alpha
    blended = 0.55 * img + 0.45 * heat
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)


def pick_one_per_class(model, test_loader, device, num_classes: int) -> dict[int, dict]:
    """Walk the test loader, return {class_idx: {'img': tensor(C,H,W), 'pred': int, 'conf': float}}
    for the first correctly predicted image of each class."""
    chosen: dict[int, dict] = {}
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu()
            for img, true, pred in zip(imgs.cpu(), labels, preds):
                t = int(true.item())
                if t not in chosen and int(pred.item()) == t:
                    chosen[t] = {"img": img, "pred": int(pred.item())}
            if len(chosen) == num_classes:
                break
    return chosen


def run_one(run_dir: Path, out_path: Path) -> None:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_loaders(
        data_root=cfg["data"]["root"],
        train_subdir=cfg["data"]["train_subdir"],
        test_subdir=cfg["data"]["test_subdir"],
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        num_workers=0,
        val_split=cfg["val_split"],
        seed=cfg["seed"],
        augment_cfg=cfg["augment"],
    )
    class_names = data["class_names"]
    num_classes = data["num_classes"]

    model = build_model(
        cfg["model"], num_classes=num_classes,
        pretrained=cfg["pretrained"], img_size=cfg["img_size"],
    ).to(device)
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])

    chosen = pick_one_per_class(model, data["test_loader"], device, num_classes)

    target_layer, reshape_transform = get_target_layer(model, cfg["model"])
    cam_extractor = GradCAM(model, target_layer, reshape_transform)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7.2))
    axes = axes.ravel()
    for cls_idx in range(num_classes):
        ax = axes[cls_idx]
        if cls_idx not in chosen:
            ax.text(0.5, 0.5, f"no correct\nsample for\n{class_names[cls_idx]}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        img = chosen[cls_idx]["img"].to(device).unsqueeze(0)
        cam, pred_idx, conf = cam_extractor(img, class_idx=cls_idx)
        overlay = overlay_heatmap(chosen[cls_idx]["img"], cam)
        ax.imshow(overlay)
        ax.set_title(f"{class_names[cls_idx]}\nconf={conf:.2f}", fontsize=9)
        ax.axis("off")

    cam_extractor.remove_hooks()

    fig.suptitle(f"Grad-CAM — {run_dir.name}  (best epoch {state.get('epoch')})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", type=str, default="runs")
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict to these run dir names; default = all eligible")
    args = p.parse_args()

    runs_root = Path(args.runs_root)
    skip = {"pca_dt"}  # no torch model
    candidates = [d for d in sorted(runs_root.iterdir())
                  if d.is_dir() and (d / "best.pt").exists()
                  and (d / "config.yaml").exists() and d.name not in skip]
    if args.only:
        wanted = set(args.only)
        candidates = [d for d in candidates if d.name in wanted]

    for run_dir in candidates:
        out = run_dir / "gradcam.png"
        try:
            run_one(run_dir, out)
        except Exception as e:
            print(f"[fail] {run_dir.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
