from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class TransformSubset(Dataset):
    """Wraps a Subset and applies its own transform — needed because
    `random_split` shares the underlying dataset (and thus its transform)
    between train and val splits."""

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        img, label = self.subset.dataset.samples[self.subset.indices[idx]]
        from PIL import Image
        with Image.open(img) as im:
            im = im.convert("RGB")
            return self.transform(im), label


def build_train_transform(img_size: int, augment_cfg: dict) -> transforms.Compose:
    ops: list = [transforms.Resize((img_size, img_size))]
    if augment_cfg.get("hflip", True):
        ops.append(transforms.RandomHorizontalFlip())
    rot = augment_cfg.get("rotation", 0)
    if rot:
        ops.append(transforms.RandomRotation(rot))
    cj = augment_cfg.get("color_jitter")
    if cj:
        ops.append(transforms.ColorJitter(*cj))
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def build_loaders(
    data_root: str | Path,
    train_subdir: str,
    test_subdir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
    augment_cfg: dict,
) -> dict[str, Any]:
    data_root = Path(data_root)
    train_dir = data_root / train_subdir
    test_dir = data_root / test_subdir
    if not train_dir.exists():
        raise FileNotFoundError(f"train dir not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"test dir not found: {test_dir}")

    base = ImageFolder(str(train_dir))
    n_total = len(base)
    n_val = int(round(n_total * val_split))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(base, [n_train, n_val], generator=g)

    train_tf = build_train_transform(img_size, augment_cfg)
    eval_tf = build_eval_transform(img_size)

    train_ds = TransformSubset(train_sub, train_tf)
    val_ds = TransformSubset(val_sub, eval_tf)
    test_ds = ImageFolder(str(test_dir), transform=eval_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_classes": len(base.classes),
        "class_names": base.classes,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": len(test_ds),
    }
