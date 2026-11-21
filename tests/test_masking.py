import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ml.masking.cloak import CloakConfig, _to_tensor, cloak, image_ssim
from ml.masking.encoder import FeatureEncoder
from ml.masking.health import protection_health


class TinyEncoder(nn.Module):
    """Stand-in backbone: 2 convs + pooling. Mechanics only, not the SCRAM model."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


@pytest.fixture(scope="module")
def encoder():
    torch.manual_seed(0)
    return FeatureEncoder(TinyEncoder(), input_size=64).eval()


@pytest.fixture
def artwork():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _cos(encoder, a, b):
    with torch.no_grad():
        fa = encoder(_to_tensor(a, "cpu"))
        fb = encoder(_to_tensor(b, "cpu"))
        return float(F.cosine_similarity(fa, fb).mean())


def test_budget_is_respected(encoder, artwork):
    eps = 8 / 255
    out = cloak(artwork, encoder, CloakConfig(epsilon=eps, steps=20, min_ssim=0.0))
    diff = np.abs(np.asarray(out, np.float32) - np.asarray(artwork, np.float32)) / 255.0
    assert diff.max() <= eps + 1 / 255 + 1e-6


def test_untargeted_cloak_moves_features_away(encoder, artwork):
    out = cloak(artwork, encoder, CloakConfig(epsilon=8 / 255, steps=40, min_ssim=0.0))
    assert _cos(encoder, out, artwork) < 0.999


def test_targeted_cloak_moves_toward_decoy(encoder, artwork):
    torch.manual_seed(0)
    rng = np.random.default_rng(7)
    decoy = Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8))
    before = _cos(encoder, artwork, decoy)
    out = cloak(artwork, encoder, CloakConfig(epsilon=8 / 255, steps=80, min_ssim=0.0, robust=False), decoy=decoy)
    assert _cos(encoder, out, decoy) > before


def test_zero_budget_is_identity(encoder, artwork):
    out = cloak(artwork, encoder, CloakConfig(epsilon=0.0, steps=10, min_ssim=0.5))
    assert np.array_equal(np.asarray(out), np.asarray(artwork))


def test_ssim_floor_is_enforced(encoder, artwork):
    with pytest.raises(ValueError, match="below the floor"):
        cloak(artwork, encoder, CloakConfig(epsilon=0.25, steps=10, min_ssim=0.999))


def test_health_is_zero_without_perturbation(encoder, artwork):
    report = protection_health(artwork, artwork, encoder)
    assert report.score == 0.0


def test_health_report_is_bounded(encoder, artwork):
    out = cloak(artwork, encoder, CloakConfig(epsilon=16 / 255, steps=40, min_ssim=0.0))
    report = protection_health(artwork, out, encoder)
    assert 0.0 <= report.score <= 100.0
    assert len(report.per_attacker) == 5
    assert all(0.0 <= v <= 100.0 for v in report.per_attacker.values())
    assert report.cloak_shift > 0


def test_config_validation():
    with pytest.raises(ValueError):
        CloakConfig(epsilon=0.5)
    with pytest.raises(ValueError):
        CloakConfig(min_ssim=1.5)


def test_robust_and_plain_both_respect_budget(encoder, artwork):
    eps = 8 / 255
    for robust in (True, False):
        out = cloak(artwork, encoder, CloakConfig(epsilon=eps, steps=15, min_ssim=0.0, robust=robust))
        diff = np.abs(np.asarray(out, np.float32) - np.asarray(artwork, np.float32)) / 255.0
        assert diff.max() <= eps + 1 / 255 + 1e-6
