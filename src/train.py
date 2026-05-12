import argparse
import math
import time
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from src.data import build_loaders
from src.models import build_model
from src.utils import (
    AverageMeter,
    CSVLogger,
    copy_as_best,
    count_params,
    save_ckpt,
    set_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--name", type=str, required=True, help="experiment name (run dir)")
    p.add_argument("--model", type=str, required=True,
                   choices=["resnet18", "resnet50", "vit_base"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--aug", type=str, default=None,
                   choices=["none", "hflip", "hflip_rot", "full"],
                   help="override augmentation level (for ablation). "
                        "none = resize only; hflip = + horizontal flip; "
                        "hflip_rot = + rotation; full = + color_jitter (yaml default).")
    p.add_argument("--save_every_epoch", action="store_true",
                   help="save ckpt for every epoch (note.md original ask). "
                        "Off by default — only best.pt + last.pt are kept, "
                        "because per-epoch ckpts for 7 runs require ~130 GB.")
    return p.parse_args()


def apply_aug_override(augment_cfg: dict, level: str) -> dict:
    """Return a new augment_cfg with progressive augmentation levels.

    Used by the augmentation-ablation experiments so the YAML config
    stays the 'full' baseline and CLI overrides peel layers off.
    """
    cfg = dict(augment_cfg)
    if level == "none":
        cfg["hflip"] = False
        cfg["rotation"] = 0
        cfg["color_jitter"] = None
    elif level == "hflip":
        cfg["hflip"] = True
        cfg["rotation"] = 0
        cfg["color_jitter"] = None
    elif level == "hflip_rot":
        cfg["hflip"] = True
        cfg["rotation"] = augment_cfg.get("rotation", 15) or 15
        cfg["color_jitter"] = None
    elif level == "full":
        pass
    return cfg


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_scheduler(optimizer, total_epochs: int, warmup_epochs: int, steps_per_epoch: int):
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None,
              scaler=None, amp: bool = False) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_m = AverageMeter()
    acc_m = AverageMeter()

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        ctx = torch.amp.autocast("cuda", enabled=amp and device.type == "cuda")
        with ctx:
            logits = model(imgs)
            loss = criterion(logits, targets)

        if is_train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

        bs = imgs.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(accuracy(logits.detach(), targets), bs)

    return loss_m.avg, acc_m.avg


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp: bool = False) -> tuple[float, float]:
    return run_epoch(model, loader, criterion, device, amp=amp)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["optimizer"]["lr"] = args.lr
    if args.no_amp:
        cfg["amp"] = False
    if args.aug is not None:
        cfg["augment"] = apply_aug_override(cfg["augment"], args.aug)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Freeze the effective config for this run
    effective = dict(cfg)
    effective.update({
        "model": args.model,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "experiment_name": args.name,
        "aug_level": args.aug or "full",
    })
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(effective, f, sort_keys=False, allow_unicode=True)

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
    print(f"[data] num_classes={data['num_classes']} "
          f"train={data['n_train']} val={data['n_val']} test={data['n_test']}")
    print(f"[data] classes: {data['class_names']}")

    model = build_model(
        args.model,
        num_classes=data["num_classes"],
        pretrained=args.pretrained,
        img_size=cfg["img_size"],
    ).to(device)
    print(f"[model] {args.model} pretrained={args.pretrained} "
          f"params={count_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
    )
    scheduler = make_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=cfg["scheduler"].get("warmup_epochs", 0),
        steps_per_epoch=len(data["train_loader"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")

    logger = CSVLogger(
        run_dir / "log.csv",
        ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc",
         "epoch_time_sec"],
    )

    best_val = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, data["train_loader"], criterion, device,
            optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, amp=cfg["amp"],
        )
        val_loss, val_acc = evaluate(
            model, data["val_loader"], criterion, device, amp=cfg["amp"]
        )
        dt = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "args": vars(args),
        }
        if args.save_every_epoch:
            ckpt_path = run_dir / f"ckpt_epoch_{epoch:02d}.pt"
            save_ckpt(state, ckpt_path)
        else:
            ckpt_path = run_dir / "last.pt"
            save_ckpt(state, ckpt_path)

        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            copy_as_best(ckpt_path, run_dir / "best.pt")

        logger.append({
            "epoch": epoch, "lr": f"{cur_lr:.6e}",
            "train_loss": f"{train_loss:.4f}", "train_acc": f"{train_acc:.4f}",
            "val_loss": f"{val_loss:.4f}", "val_acc": f"{val_acc:.4f}",
            "epoch_time_sec": f"{dt:.1f}",
        })
        print(f"[epoch {epoch:02d}/{args.epochs}] "
              f"lr={cur_lr:.2e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"({dt:.1f}s) {'*' if epoch == best_epoch else ''}")

    # Final test eval with best ckpt
    print(f"[best] epoch={best_epoch} val_acc={best_val:.4f}")
    best_state = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    test_loss, test_acc = evaluate(
        model, data["test_loader"], criterion, device, amp=cfg["amp"]
    )
    print(f"[test] loss={test_loss:.4f} acc={test_acc:.4f}")

    write_json({
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "model": args.model,
        "pretrained": args.pretrained,
        "epochs": args.epochs,
        "aug_level": args.aug or "full",
    }, run_dir / "eval.json")


if __name__ == "__main__":
    main()
