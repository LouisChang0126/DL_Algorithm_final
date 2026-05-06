"""Standalone evaluator: load runs/<name>/best.pt and evaluate on the test set.

train.py already runs this at the end of training; this script is for re-running
the test eval against an already-trained run (e.g. after dataset changes)."""
import argparse
from pathlib import Path

import torch
import yaml
from torch import nn

from src.data import build_loaders
from src.models import build_model
from src.train import evaluate
from src.utils import write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True, help="path to runs/<name>/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_loaders(
        data_root=cfg["data"]["root"],
        train_subdir=cfg["data"]["train_subdir"],
        test_subdir=cfg["data"]["test_subdir"],
        img_size=cfg["img_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        val_split=cfg["val_split"],
        seed=cfg["seed"],
        augment_cfg=cfg["augment"],
    )

    model = build_model(
        cfg["model"], num_classes=data["num_classes"],
        pretrained=cfg["pretrained"], img_size=cfg["img_size"],
    ).to(device)

    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, data["test_loader"], criterion, device,
                                   amp=cfg.get("amp", False))
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}")

    write_json({
        "best_epoch": state.get("epoch"),
        "best_val_acc": state.get("val_acc"),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "model": cfg["model"],
        "pretrained": cfg["pretrained"],
        "epochs": cfg["epochs"],
    }, run_dir / "eval.json")


if __name__ == "__main__":
    main()
