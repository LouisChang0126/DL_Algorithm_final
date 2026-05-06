"""Grad-CAM for both CNNs (ResNet) and ViT.

For CNN: target the last conv block's output. Gradients are 4-D feature maps,
weights are the channel-wise mean of gradients (vanilla Grad-CAM).

For ViT: target the input to the last transformer block's MHA (norm1). The
hooked tensor shape is (B, N+1, C) with one CLS token at index 0, so we drop
CLS and reshape the remaining patch tokens back to (B, C, H, W) before applying
the same Grad-CAM math.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        reshape_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, out):
        self.activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[torch.Tensor, int, float]:
        self.model.eval()
        self.model.zero_grad()
        # input_tensor: (1, C, H, W)
        with torch.enable_grad():
            input_tensor = input_tensor.requires_grad_(True)
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            confidence = float(probs[0, class_idx].item())
            score = logits[0, class_idx]
            score.backward()

        acts = self.activations
        grads = self.gradients
        assert acts is not None and grads is not None, "hooks did not fire"

        if self.reshape_transform is not None:
            acts = self.reshape_transform(acts)
            grads = self.reshape_transform(grads)

        # acts/grads: (1, C, H, W). vanilla grad-cam: weights = mean over H,W of grads.
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze(0).squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu(), class_idx, confidence


def vit_reshape_transform(tokens: torch.Tensor) -> torch.Tensor:
    """timm ViT hook output is (B, N+1, C) — drop CLS, reshape to (B, C, H, W).

    For vit_base_patch16 at img_size=256, N = (256/16)^2 = 256 → H=W=16.
    """
    b, n_plus_1, c = tokens.shape
    n = n_plus_1 - 1  # drop CLS
    side = int(round(n ** 0.5))
    assert side * side == n, f"non-square token grid: n={n}"
    patch_tokens = tokens[:, 1:, :]  # (B, N, C)
    grid = patch_tokens.reshape(b, side, side, c).permute(0, 3, 1, 2).contiguous()
    return grid


def get_target_layer(model: nn.Module, model_name: str) -> tuple[nn.Module, Callable | None]:
    """Return (target_layer, reshape_transform) for a model."""
    if model_name in ("resnet18", "resnet50"):
        return model.layer4, None
    if model_name == "vit_base":
        # timm ViT: use the LayerNorm before MHA in the last block.
        return model.blocks[-1].norm1, vit_reshape_transform
    raise ValueError(f"unknown model for grad-cam: {model_name}")
