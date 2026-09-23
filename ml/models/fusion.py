"""Multi-branch forensic detector and calibrated inference wrapper."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .encoders import FrequencyEncoder, RGBEncoder, ResidualEncoder

DEFAULT_GENERATORS: tuple[str, ...] = (
    "SDXL",
    "FLUX",
    "DALL-E",
    "Midjourney",
    "Unknown",
)


class ForensicDetector(nn.Module):
    """Fuse RGB, frequency, and residual features for forensic prediction.

    Args:
        num_generators: Number of generator classes. Must match the generator
            label mapping used during training.
        generator_names: Ordered generator labels. If supplied, its length
            must match ``num_generators``.
        backbone_name: timm backbone name used by :class:`RGBEncoder`.
        pretrained: Whether to initialize the RGB backbone with pretrained
            weights.

    Input:
        RGB image batch with shape ``[B, 3, H, W]``.

    Output:
        A dictionary containing:
        - ``binary_logits``: ``[B, 2]``, ordered as ``[real, AI]``.
        - ``generator_logits``: ``[B, num_generators]``.
        - ``fused_embedding``: ``[B, 512]``.
    """

    def __init__(
        self,
        num_generators: int = len(DEFAULT_GENERATORS),
        generator_names: tuple[str, ...] = DEFAULT_GENERATORS,
        backbone_name: str = "convnext_small",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if num_generators <= 0:
            raise ValueError("num_generators must be positive")
        if len(generator_names) != num_generators:
            raise ValueError("generator_names length must equal num_generators")

        self.generator_names = generator_names

        self.rgb_encoder = RGBEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            embedding_dim=512,
        )
        self.frequency_encoder = FrequencyEncoder(embedding_dim=256)
        self.residual_encoder = ResidualEncoder(embedding_dim=256)

        self.fusion = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
        )
        self.binary_head = nn.Linear(512, 2)
        self.generator_head = nn.Linear(512, num_generators)

    def forward(self, image_tensor: Tensor) -> dict[str, Tensor]:
        """Encode and fuse images: [B, 3, H, W] -> logits and [B, 512]."""
        rgb_embedding = self.rgb_encoder(image_tensor)  # [B, 512]
        frequency_embedding = self.frequency_encoder(image_tensor)  # [B, 256]
        residual_embedding = self.residual_encoder(image_tensor)  # [B, 256]

        concatenated = torch.cat(
            (rgb_embedding, frequency_embedding, residual_embedding),
            dim=1,
        )  # [B, 1024]

        fused_embedding = self.fusion(concatenated)  # [B, 512]
        return {
            "binary_logits": self.binary_head(fused_embedding),  # [B, 2]
            "generator_logits": self.generator_head(fused_embedding),
            "fused_embedding": fused_embedding,
        }


class CalibratedInferenceWrapper(nn.Module):
    """Apply temperature scaling and convert detector outputs to predictions.

    The scalar temperature is learned during calibration, typically by
    minimizing validation negative log-likelihood. It is constrained to be
    positive by exponentiating the stored log-temperature parameter.

    Args:
        model: Trained forensic detector.
        unknown_threshold: Generator probability below which the label is
            returned as ``"Unknown"``.
        initial_temperature: Positive starting temperature. Values greater
            than 1 soften predicted probabilities; values below 1 sharpen them.

    ``predict`` returns binary probabilities, a confidence score, the
    predicted generator label, and generator probabilities. Confidence is the
    maximum calibrated binary class probability.
    """

    def __init__(
        self,
        model: ForensicDetector,
        unknown_threshold: float = 0.40,
        initial_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= unknown_threshold <= 1.0:
            raise ValueError("unknown_threshold must be between 0 and 1")
        if initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive")

        self.model = model
        self.unknown_threshold = unknown_threshold
        self.log_temperature = nn.Parameter(
            torch.tensor(float(initial_temperature)).log()
        )

    @property
    def temperature(self) -> Tensor:
        """Positive learned temperature scalar."""
        return self.log_temperature.exp()

    @torch.inference_mode()
    def predict(self, image_tensor: Tensor) -> dict[str, Any]:
        """Return calibrated predictions for an image batch.

        Args:
            image_tensor: RGB tensor of shape ``[B, 3, H, W]``.

        Returns:
            A dictionary with probability tensors and per-item labels.
            ``binary_probabilities`` has shape ``[B, 2]`` and
            ``generator_probabilities`` has shape ``[B, N_generators]``.
        """
        self.eval()
        outputs = self.model(image_tensor)
        temperature = self.temperature.clamp_min(1e-6)

        binary_probabilities = torch.softmax(
            outputs["binary_logits"] / temperature,
            dim=-1,
        )
        generator_probabilities = torch.softmax(
            outputs["generator_logits"] / temperature,
            dim=-1,
        )

        generator_confidence, generator_indices = generator_probabilities.max(dim=-1)
        predicted_generators = [
            (
                self.model.generator_names[index]
                if confidence >= self.unknown_threshold
                else "Unknown"
            )
            for index, confidence in zip(
                generator_indices.tolist(),
                generator_confidence.tolist(),
                strict=True,
            )
        ]

        return {
            "binary_probabilities": binary_probabilities,
            "confidence": binary_probabilities.max(dim=-1).values,
            "generator_probabilities": generator_probabilities,
            "generator": predicted_generators,
            "fused_embedding": outputs["fused_embedding"],
        }


if __name__ == "__main__":
    detector = ForensicDetector(pretrained=False)
    detector.eval()

    dummy_images = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        result = detector(dummy_images)

    assert result["binary_logits"].shape == (1, 2)
    assert result["generator_logits"].shape == (1, len(DEFAULT_GENERATORS))
    assert result["fused_embedding"].shape == (1, 512)

    wrapper = CalibratedInferenceWrapper(detector)
    prediction = wrapper.predict(dummy_images)

    assert prediction["binary_probabilities"].shape == (1, 2)
    assert prediction["generator_probabilities"].shape == (
        1,
        len(DEFAULT_GENERATORS),
    )
    print("ForensicDetector output shapes are valid.")
