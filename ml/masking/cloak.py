"""Feature-space cloaking: a perturbation that is small in pixels but large in features.

Each pixel moves by at most ``epsilon`` (L-infinity), so the image looks the same to a
person. The perturbation is optimized with signed gradient steps (PGD) so the encoder's
features move away from the image's own features (untargeted), or toward a decoy image's
features (targeted). The result is quantized to 8 bits and checked against an SSIM floor.

Note: the optimization is white-box against the encoder you pass in. How well the cloak
transfers to scrapers that use other models is an empirical question; protection_health()
measures robustness to resizing, JPEG and denoising, not transfer to unseen models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity


@dataclass(frozen=True)
class CloakConfig:
    epsilon: float = 8 / 255      # max per-pixel change, in [0, 1] units
    steps: int = 100
    step_size: float = 1 / 255
    min_ssim: float = 0.95        # visual-fidelity floor on the 8-bit output; 0 disables
    robust: bool = True           # optimize through random resize + noise (EOT), so the cloak
                                  # survives common scraper preprocessing
    min_scale: float = 0.5        # smallest random downscale used during robust optimization
    max_noise: float = 0.02       # largest Gaussian noise std (in [0, 1] units)

    def __post_init__(self) -> None:
        if not 0.0 <= self.epsilon <= 0.25:
            raise ValueError("epsilon must be in [0, 0.25]")
        if self.steps < 0:
            raise ValueError("steps must be >= 0")
        if self.step_size <= 0:
            raise ValueError("step_size must be > 0")
        if not 0.1 <= self.min_scale <= 1.0:
            raise ValueError("min_scale must be in [0.1, 1]")
        if not 0.0 <= self.max_noise <= 0.1:
            raise ValueError("max_noise must be in [0, 0.1]")
        if not 0.0 <= self.min_ssim <= 1.0:
            raise ValueError("min_ssim must be in [0, 1]; 0 disables the check")


def _to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = (tensor[0].clamp(0, 1).permute(1, 2, 0) * 255.0).round().to(torch.uint8)
    return Image.fromarray(arr.cpu().numpy(), mode="RGB")


def _augment(x: torch.Tensor, config: CloakConfig) -> torch.Tensor:
    """Random downscale-upscale plus Gaussian noise, all differentiable.

    Simulates the resizing and noise that platforms and scrapers apply, so the
    optimized perturbation has to survive them.
    """
    _, _, h, w = x.shape
    scale = config.min_scale + (1.0 - config.min_scale) * float(torch.rand(1))
    small = (max(32, int(h * scale)), max(32, int(w * scale)))
    y = F.interpolate(x, size=small, mode="bilinear", align_corners=False, antialias=True)
    y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
    sigma = config.max_noise * float(torch.rand(1))
    return (y + sigma * torch.randn_like(y)).clamp(0, 1)


def image_ssim(first: Image.Image, second: Image.Image) -> float:
    """RGB SSIM between two same-sized images, on 8-bit values."""
    if first.size != second.size:
        raise ValueError("Images must have the same size")
    a = np.asarray(first.convert("RGB"), dtype=np.uint8)
    b = np.asarray(second.convert("RGB"), dtype=np.uint8)
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


def cloak(
    image: Image.Image,
    encoder: nn.Module,
    config: CloakConfig = CloakConfig(),
    decoy: Image.Image | None = None,
    device: str = "cpu",
) -> Image.Image:
    """Return a cloaked copy of ``image`` that looks the same but whose features are shifted.

    Args:
        image: The artwork to protect (any size).
        encoder: A FeatureEncoder (see load_scram_encoder). Used as-is; it is not modified.
        config: Budget and optimization settings.
        decoy: Optional image whose features the cloak should move toward. Without a
            decoy the cloak moves away from the image's own features.
        device: Torch device string.

    Raises:
        ValueError: If the cloaked image falls below ``config.min_ssim``.
    """
    encoder = encoder.to(device).eval()
    x = _to_tensor(image, device)

    with torch.no_grad():
        anchor = encoder(_to_tensor(decoy, device) if decoy is not None else x)

    # Minimizing this loss moves the features away from the anchor (untargeted)
    # or toward it (targeted, since the anchor is a decoy).
    direction = -1.0 if decoy is not None else 1.0

    delta = torch.zeros_like(x)
    for _ in range(config.steps):
        delta.requires_grad_(True)
        view = (x + delta).clamp(0, 1)
        if config.robust:
            view = _augment(view, config)
        features = encoder(view)
        loss = direction * F.cosine_similarity(features, anchor).mean()
        (grad,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = (delta - config.step_size * grad.sign()).clamp(
                -config.epsilon, config.epsilon
            )

    cloaked = _to_pil((x + delta.detach()).clamp(0, 1))
    score = image_ssim(image, cloaked)
    if score < config.min_ssim:
        raise ValueError(
            f"Cloaked image SSIM {score:.4f} is below the floor {config.min_ssim}; "
            "lower epsilon or raise min_ssim"
        )
    return cloaked
