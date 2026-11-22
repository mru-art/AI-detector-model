"""Scraper-degradation health score for a cloaked image.

For each simulated scraper, we ask how much of the cloak's feature shift survives the
scraper's preprocessing. The shift is measured in the encoder's feature space, not in
pixels. A scraper that trains on the image sees features that are still displaced,
so its ability to use the image is degraded.

Score = 100 * mean over the four preprocessing attackers of the retained fraction (Identity is
reported but not averaged, since it always keeps the whole shift), where for each attacker

    retained = (disp(attacked cloak) - disp(attacked clean)) / disp(cloak)

and disp(v) = 1 - cos(f(v), f(original)), clipped to [0, 1].
  100 = every scored attack keeps the full shift; 0 = every attack removes it.

Limits: the attackers are fixed preprocessors (download, screenshot/JPEG, blur+JPEG,
median denoise, downscale-upscale), not learned scrapers. This module has no dependency on
backend/services/scraper_eval. The shift is measured with the same encoder that produced the
cloak, so the score is a white-box upper bound on protection against unseen models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import io

import numpy as np
from PIL import Image, ImageFilter

from .cloak import _to_tensor


class _Identity:
    """Scraper that downloads the file unchanged."""

    def attack(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB").copy()


class _Screenshot:
    """Render at a 1280x720 viewport, then re-encode as JPEG q85 at the original size."""

    def attack(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        scale = min(1280 / img.width, 720 / img.height, 1.0)
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        canvas = img.resize(size, Image.Resampling.LANCZOS).resize(img.size, Image.Resampling.BICUBIC)
        return _jpeg(canvas, 85)


class _Recompress:
    """Blur, downscale by 0.75, restore, JPEG q70."""

    def attack(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=0.6))
        small = img.resize((max(1, round(img.width * 0.75)), max(1, round(img.height * 0.75))),
                           Image.Resampling.LANCZOS)
        return _jpeg(small.resize(img.size, Image.Resampling.BICUBIC), 70)


class _Denoise:
    """Median filter, a scraper's standard noise-removal step."""

    def attack(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB").filter(ImageFilter.MedianFilter(size=3))


class _Reconstruct:
    """Downscale to 25% and upscale: a stand-in for an autoencoder bottleneck."""

    def attack(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGB")
        small = img.resize((max(1, round(img.width * 0.25)), max(1, round(img.height * 0.25))),
                           Image.Resampling.BOX)
        return small.resize(img.size, Image.Resampling.BICUBIC)


def _jpeg(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB").copy()


DEFAULT_ATTACKERS = (
    ("Identity", _Identity),
    ("Screenshot", _Screenshot),
    ("Recompress", _Recompress),
    ("Denoise", _Denoise),
    ("Reconstruct", _Reconstruct),
)


@dataclass(frozen=True)
class HealthReport:
    score: float                       # 0-100, higher = scraper more degraded
    per_attacker: dict[str, float] = field(default_factory=dict)
    cloak_shift: float = 0.0           # 1 - cos of cloaked vs original, before attacks

    @property
    def worst_attacker(self) -> tuple[str, float]:
        name = min(self.per_attacker, key=self.per_attacker.get)
        return name, self.per_attacker[name]


def _features(encoder: nn.Module, image: Image.Image, device: str) -> torch.Tensor:
    with torch.no_grad():
        return encoder(_to_tensor(image, device))


def _displacement(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(1.0 - F.cosine_similarity(a, b).mean())


def protection_health(
    original: Image.Image,
    cloaked: Image.Image,
    encoder: nn.Module,
    device: str = "cpu",
) -> HealthReport:
    """Score how much the cloak's feature shift survives each simulated scraper."""
    if original.size != cloaked.size:
        raise ValueError("original and cloaked must have the same size")

    encoder = encoder.to(device).eval()
    f_orig = _features(encoder, original, device)
    f_cloak = _features(encoder, cloaked, device)
    nominal = _displacement(f_cloak, f_orig)

    if nominal < 1e-6:
        return HealthReport(score=0.0, per_attacker={}, cloak_shift=0.0)

    per_attacker: dict[str, float] = {}
    for name, attacker_cls in DEFAULT_ATTACKERS:
        attacker = attacker_cls()
        f_attacked_cloak = _features(encoder, attacker.attack(cloaked), device)
        f_attacked_clean = _features(encoder, attacker.attack(original), device)
        shift_kept = _displacement(f_attacked_cloak, f_orig) - _displacement(
            f_attacked_clean, f_orig
        )
        per_attacker[name] = round(100.0 * min(max(shift_kept / nominal, 0.0), 1.0), 2)

    # Identity keeps the full shift by definition, so it is reported but not averaged in.
    scored = [v for name, v in per_attacker.items() if name != "Identity"]
    score = round(sum(scored) / len(scored), 2) if scored else 0.0
    return HealthReport(score=score, per_attacker=per_attacker, cloak_shift=nominal)
