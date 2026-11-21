"""Feature encoder used both to cloak an image and to score the cloak.

The encoder is the SCRAM ViT-B/16 backbone with its classification head removed,
so it returns the 768-d CLS feature for any [0, 1] RGB tensor.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FeatureEncoder(nn.Module):
    """Resize to the backbone's input size, normalize, and return features.

    Accepts a float tensor of shape (N, 3, H, W) with values in [0, 1]. Resizing
    is differentiable, so gradients reach the perturbation at full resolution.
    """

    def __init__(self, backbone: nn.Module, input_size: int = 224) -> None:
        super().__init__()
        self.backbone = backbone
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return self.backbone((x - self.mean) / self.std)


def load_scram_encoder(checkpoint_path: str | Path, device: str = "cpu") -> FeatureEncoder:
    """Load a SCRAM checkpoint (vision_transformer/models.py layout) as a feature encoder.

    Expects the torchvision ViT-B/16 layout with the custom head
    (heads.head.0 / .3 / .6). Accepts a raw state dict or a dict holding one under
    ``model_state_dict``, ``state_dict`` or ``model``.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    state = {
        k.removeprefix("module.").removeprefix("model."): v for k, v in state.items()
    }

    if "heads.head.6.weight" not in state:
        raise ValueError(
            "Checkpoint does not match the SCRAM ViT head layout (heads.head.6.weight missing)"
        )
    num_classes = state["heads.head.6.weight"].shape[0]

    model = vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Linear(in_features, 512), nn.GELU(), nn.Dropout(0.4),
        nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    model.load_state_dict(state, strict=True)
    model.heads.head = nn.Identity()  # keep the 768-d features, drop the classifier

    return FeatureEncoder(model).to(device).eval()
