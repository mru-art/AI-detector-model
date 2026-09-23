"""Subtle high-frequency artwork protection and image-quality metrics.

Requires Pillow, NumPy, scikit-image, and the project's async storage client.
"""

from __future__ import annotations

import argparse
import asyncio
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from backend.storage import ObjectStorage, storage

PROTECTED_BUCKET = "protected"
MINIMUM_SSIM = 0.92


def _as_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def calculate_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """Calculate RGB structural similarity between two PIL images."""
    first = _as_rgb_array(img1)
    second = _as_rgb_array(img2)
    if first.shape != second.shape:
        raise ValueError("Images must have the same dimensions for SSIM")
    return float(
        structural_similarity(
            first,
            second,
            channel_axis=2,
            data_range=255.0,
        )
    )


def calculate_psnr(img1: Image.Image, img2: Image.Image) -> float:
    """Calculate RGB peak signal-to-noise ratio in decibels."""
    first = _as_rgb_array(img1)
    second = _as_rgb_array(img2)
    if first.shape != second.shape:
        raise ValueError("Images must have the same dimensions for PSNR")
    return float(peak_signal_noise_ratio(first, second, data_range=255.0))


class ArtworkProtector:
    """Create a visually constrained, high-frequency protected image.

    Args:
        object_storage: Storage adapter compatible with the project's
            ``ObjectStorage.put_object`` method.

    After a successful call, ``last_object_uri`` contains the uploaded S3 URI.
    The method also returns the protected image as a PIL image.
    """

    def __init__(self, object_storage: ObjectStorage = storage) -> None:
        self.object_storage = object_storage
        self.last_object_uri: str | None = None

    async def apply_protection(
        self,
        image_path: str,
        intensity: float = 0.05,
    ) -> Image.Image:
        """Perturb high-frequency image detail, enforce SSIM, and upload it.

        ``intensity`` is a normalized amplitude in [0, 1]. The candidate
        perturbation is scaled by this value and repeatedly reduced if needed
        to meet the SSIM floor. Very low values or highly sensitive images may
        therefore receive only a minimal perturbation.

        Returns:
            Protected RGB image. Its SSIM against the source is greater than
            ``MINIMUM_SSIM`` or the method raises an error.
        """
        if not 0.0 <= intensity <= 1.0:
            raise ValueError("intensity must be between 0 and 1")

        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(path) as source_file:
            original = source_file.convert("RGB")

        source = _as_rgb_array(original)
        height, width, _ = source.shape

        # A deterministic, zero-mean checker/noise pattern concentrates energy
        # in high spatial frequencies and keeps CLI runs reproducible.
        rng = np.random.default_rng(0)
        random_pattern = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32),
            size=(height, width, 1),
        )
        checker = ((np.indices((height, width)).sum(axis=0) % 2) * 2 - 1).astype(
            np.float32
        )[..., None]
        pattern = 0.65 * checker + 0.35 * random_pattern
        pattern -= pattern.mean(axis=(0, 1), keepdims=True)

        amplitude = intensity * 12.0
        protected: Image.Image | None = None
        score = 0.0

        # Enforce the requested visual similarity empirically. In pathological
        # cases the loop can reduce perturbation to zero; that still preserves
        # fidelity but may provide no meaningful protective effect.
        for _ in range(24):
            candidate = np.clip(source + amplitude * pattern, 0, 255).astype(np.uint8)
            protected = Image.fromarray(candidate, mode="RGB")
            score = calculate_ssim(original, protected)
            if score > MINIMUM_SSIM:
                break
            amplitude *= 0.5
        else:
            protected = original.copy()
            score = calculate_ssim(original, protected)

        if score <= MINIMUM_SSIM:
            raise RuntimeError(
                f"Unable to satisfy SSIM floor; measured SSIM={score:.5f}"
            )

        output = BytesIO()
        protected.save(output, format="PNG")
        key = f"{uuid4().hex}.png"

        await self.object_storage.put_object(
            bucket=PROTECTED_BUCKET,
            key=key,
            body=output.getvalue(),
            content_type="image/png",
        )
        self.last_object_uri = f"s3://{PROTECTED_BUCKET}/{key}"
        return protected


async def _run_cli(image_path: str, intensity: float) -> None:
    protector = ArtworkProtector()
    protected = await protector.apply_protection(image_path, intensity)

    with Image.open(image_path) as source_file:
        original = source_file.convert("RGB")

    output_path = Path(image_path).with_name(f"{Path(image_path).stem}.protected.png")
    protected.save(output_path, format="PNG")

    print(f"Protected copy: {output_path}")
    print(f"Object URI: {protector.last_object_uri}")
    print(f"SSIM: {calculate_ssim(original, protected):.6f}")
    print(f"PSNR: {calculate_psnr(original, protected):.2f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create and evaluate a protected artwork copy."
    )
    parser.add_argument("image_path", help="Path to the source image")
    parser.add_argument(
        "--intensity",
        type=float,
        default=0.05,
        help="Perturbation intensity from 0 to 1 (default: 0.05)",
    )
    args = parser.parse_args()
    asyncio.run(_run_cli(args.image_path, args.intensity))
