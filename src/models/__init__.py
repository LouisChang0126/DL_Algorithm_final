from torch import nn

from .resnet import build_resnet18, build_resnet50
from .vit import build_vit_base


def build_model(name: str, num_classes: int, pretrained: bool, img_size: int = 256) -> nn.Module:
    name = name.lower()
    if name in {"resnet18", "rn18"}:
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    if name in {"resnet50", "rn50"}:
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)
    if name in {"vit_base", "vit_b", "vit"}:
        return build_vit_base(num_classes=num_classes, pretrained=pretrained, img_size=img_size)
    raise ValueError(f"Unknown model name: {name}")
