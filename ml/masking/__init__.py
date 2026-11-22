"""Feature-space cloaking for artworks (Glaze-style) and a scraper-degradation health score."""

from .cloak import CloakConfig, cloak, image_ssim
from .encoder import FeatureEncoder, load_scram_encoder
from .health import HealthReport, protection_health

__all__ = [
    "CloakConfig",
    "FeatureEncoder",
    "HealthReport",
    "cloak",
    "image_ssim",
    "load_scram_encoder",
    "protection_health",
]
