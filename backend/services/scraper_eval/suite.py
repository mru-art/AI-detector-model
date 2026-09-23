"""Mock scraper attacks for evaluating artwork protection transformations."""

from __future__ import annotations

from io import BytesIO
from typing import Protocol

import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity


class ScraperAttacker(Protocol):
    """Interface implemented by each mock scraper."""

    def attack(self, image: Image.Image) -> Image.Image:
        """Return an image recovered by the simulated attacker."""
        ...


def _rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def _as_array(image: Image.Image) -> np.ndarray:
    return np.asarray(_rgb(image), dtype=np.uint8)


class AttackerA_HTTPDownloader:
    """Directly downloads the supplied image without modification."""

    def attack(self, image: Image.Image) -> Image.Image:
        return _rgb(image).copy()


class AttackerB_ScreenshotScraper:
    """Simulate browser viewport rendering, cropping, and canvas re-encoding.

    The image is scaled to fit within a mock viewport, centered, then saved
    and reopened as JPEG to simulate a canvas export.
    """

    def __init__(
        self,
        viewport_size: tuple[int, int] = (1280, 720),
        jpeg_quality: int = 85,
    ) -> None:
        self.viewport_size = viewport_size
        self.jpeg_quality = jpeg_quality

    def attack(self, image: Image.Image) -> Image.Image:
        source = _rgb(image)
        viewport_width, viewport_height = self.viewport_size

        # Simulate a browser displaying the full image inside its viewport.
        scale = min(
            viewport_width / source.width,
            viewport_height / source.height,
            1.0,
        )
        rendered_size = (
            max(1, round(source.width * scale)),
            max(1, round(source.height * scale)),
        )
        rendered = source.resize(rendered_size, Image.Resampling.LANCZOS)

        # Canvas export re-encodes the rendered image. Restore its original
        # dimensions so similarity comparisons use matching image sizes.
        canvas = rendered.resize(source.size, Image.Resampling.BICUBIC)
        buffer = BytesIO()
        canvas.save(buffer, format="JPEG", quality=self.jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as recovered:
            return recovered.convert("RGB").copy()


class AttackerC_PreprocessingScraper:
    """Apply blur, resize round-trip, and JPEG compression at quality 70."""

    def __init__(self, jpeg_quality: int = 70, downscale: float = 0.75) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not 0.0 < downscale <= 1.0:
            raise ValueError("downscale must be in (0, 1]")
        self.jpeg_quality = jpeg_quality
        self.downscale = downscale

    def attack(self, image: Image.Image) -> Image.Image:
        source = _rgb(image)
        blurred = source.filter(ImageFilter.GaussianBlur(radius=0.6))

        smaller_size = (
            max(1, round(source.width * self.downscale)),
            max(1, round(source.height * self.downscale)),
        )
        resized = blurred.resize(smaller_size, Image.Resampling.LANCZOS)
        restored = resized.resize(source.size, Image.Resampling.BICUBIC)

        buffer = BytesIO()
        restored.save(buffer, format="JPEG", quality=self.jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as recovered:
            return recovered.convert("RGB").copy()


class AttackerD_DenoisingScraper:
    """Denoise the image to reduce high-frequency perturbations.

    Uses scikit-image total-variation denoising when available. If that
    import is unavailable, applies a median filter instead.
    """

    def __init__(self, weight: float = 0.12, median_size: int = 3) -> None:
        self.weight = weight
        self.median_size = median_size

    def attack(self, image: Image.Image) -> Image.Image:
        source = _rgb(image)
        try:
            from skimage.restoration import denoise_tv_chambolle
        except ImportError:
            return source.filter(ImageFilter.MedianFilter(size=self.median_size))

        pixels = np.asarray(source, dtype=np.float32) / 255.0
        denoised = denoise_tv_chambolle(
            pixels,
            weight=self.weight,
            channel_axis=2,
        )
        denoised_uint8 = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(denoised_uint8, mode="RGB")


class AttackerE_MLReconstructionScraper:
    """Mock autoencoder/super-resolution reconstruction attack.

    This model-free approximation removes fine detail by reducing the image
    to a low-resolution latent-like representation and upscaling it smoothly.
    Replace it with a trained reconstruction or super-resolution model for a
    deployment-specific evaluation.
    """

    def __init__(self, latent_scale: float = 0.25) -> None:
        if not 0.0 < latent_scale <= 1.0:
            raise ValueError("latent_scale must be in (0, 1]")
        self.latent_scale = latent_scale

    def attack(self, image: Image.Image) -> Image.Image:
        source = _rgb(image)
        latent_size = (
            max(1, round(source.width * self.latent_scale)),
            max(1, round(source.height * self.latent_scale)),
        )
        latent = source.resize(latent_size, Image.Resampling.BOX)
        return latent.resize(source.size, Image.Resampling.BICUBIC)


def _ssim_similarity(original: Image.Image, recovered: Image.Image) -> float:
    """Return SSIM after resizing the recovery to the original dimensions."""
    original_rgb = _rgb(original)
    recovered_rgb = _rgb(recovered)

    if recovered_rgb.size != original_rgb.size:
        recovered_rgb = recovered_rgb.resize(
            original_rgb.size,
            Image.Resampling.LANCZOS,
        )

    original_array = _as_array(original_rgb)
    recovered_array = _as_array(recovered_rgb)

    # SSIM needs a window smaller than the image dimensions.
    smallest_dimension = min(original_rgb.width, original_rgb.height)
    if smallest_dimension < 3:
        mse = float(
            np.mean(
                (original_array.astype(np.float32) - recovered_array.astype(np.float32))
                ** 2
            )
        )
        return 1.0 if mse == 0.0 else 0.0

    win_size = min(7, smallest_dimension)
    if win_size % 2 == 0:
        win_size -= 1

    score = structural_similarity(
        original_array,
        recovered_array,
        channel_axis=2,
        data_range=255,
        win_size=win_size,
    )
    return float(np.clip(score, 0.0, 1.0))


def evaluate_protection_resistance(
    original_img: Image.Image,
    protected_img: Image.Image,
) -> dict[str, float]:
    """Run mock scraper attacks and return resistance scores from 0 to 100.

    Resistance is computed as ``(1 - SSIM(original, recovered)) * 100``.
    A score near 0 means the attacker recovered an image structurally similar
    to the original; a higher score means its result is less similar.

    Args:
        original_img: Original artwork as a PIL image.
        protected_img: Protected artwork as a PIL image.

    Returns:
        Mapping from attacker tier to resistance percentage.
    """
    attackers: tuple[tuple[str, ScraperAttacker], ...] = (
        ("AttackerA_HTTPDownloader", AttackerA_HTTPDownloader()),
        ("AttackerB_ScreenshotScraper", AttackerB_ScreenshotScraper()),
        ("AttackerC_PreprocessingScraper", AttackerC_PreprocessingScraper()),
        ("AttackerD_DenoisingScraper", AttackerD_DenoisingScraper()),
        ("AttackerE_MLReconstructionScraper", AttackerE_MLReconstructionScraper()),
    )

    original = _rgb(original_img)
    protected = _rgb(protected_img)
    scores: dict[str, float] = {}

    for name, attacker in attackers:
        recovered = attacker.attack(protected)
        similarity = _ssim_similarity(original, recovered)
        scores[name] = round((1.0 - similarity) * 100.0, 2)

    return scores
