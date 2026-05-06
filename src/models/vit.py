import timm
from torch import nn


def build_vit_base(num_classes: int, pretrained: bool, img_size: int = 256) -> nn.Module:
    # vit_base_patch16_224 with img_size=256: timm interpolates the position
    # embedding so the same architecture/weights work at non-native resolutions.
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        img_size=img_size,
        num_classes=num_classes,
    )
