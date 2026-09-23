"""Feature encoders for multi-branch AI-art forensics models.

The encoders accept RGB image tensors in NCHW layout. Input normalization and
image resizing should be handled by the dataset or preprocessing pipeline.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RGBEncoder(nn.Module):
    """Encode RGB images with a pretrained timm image backbone.

    The backbone's classification head is removed. Its pooled feature vector
    is projected to a fixed 512-dimensional embedding.

    Args:
        backbone_name: timm model name, such as ``convnext_small`` or
            ``vit_base_patch16_224``.
        pretrained: Whether to load pretrained backbone weights.
        embedding_dim: Output embedding size. Defaults to 512.

    Input:
        RGB image tensor with shape ``[B, 3, H, W]``.

    Output:
        Image embedding with shape ``[B, embedding_dim]``.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_small",
        pretrained: bool = True,
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "RGBEncoder requires timm. Install it with `pip install timm`."
            ) from exc

        # num_classes=0 removes the classifier and returns pooled features.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )
        feature_dim = int(self.backbone.num_features)
        self.projection = nn.Linear(feature_dim, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode an image batch: [B, 3, H, W] -> [B, 512]."""
        features = self.backbone(x)  # [B, backbone_feature_dim]
        return self.projection(features)  # [B, embedding_dim]


class FrequencyEncoder(nn.Module):
    """Encode channel-wise Fourier magnitude spectra with a compact CNN.

    The encoder computes a 2D FFT over image height and width, shifts the
    zero-frequency component to the spectrum center, and applies ``log1p`` to
    the magnitude. The resulting three-channel spectrum is passed through
    three convolutional blocks and adaptive pooling.

    Args:
        embedding_dim: Output embedding size. Defaults to 256.

    Input:
        RGB image tensor with shape ``[B, 3, H, W]``.

    Output:
        Frequency embedding with shape ``[B, embedding_dim]``.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Transform RGB images to FFT embeddings: [B, 3, H, W] -> [B, 256]."""
        spectrum = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        magnitude = torch.fft.fftshift(spectrum.abs(), dim=(-2, -1)).clamp_min_(0.0)
        log_spectrum = torch.log1p(magnitude)  # [B, 3, H, W]

        features = self.features(log_spectrum).flatten(start_dim=1)
        return self.projection(features)  # [B, embedding_dim]


class ResidualEncoder(nn.Module):
    """Encode high-frequency image residuals using fixed SRM-style filters.

    A small bank of fixed 5x5 high-pass filters is applied independently to
    each RGB channel. The nine resulting residual maps are processed by a
    lightweight CNN.

    This filter bank is a compact SRM-inspired feature extractor, not the full
    published SRM filter set. Inputs should generally be RGB values scaled to
    a consistent range, such as ``[0, 1]`` or ``[-1, 1]``.

    Args:
        embedding_dim: Output embedding size. Defaults to 256.

    Input:
        RGB image tensor with shape ``[B, 3, H, W]``.

    Output:
        Residual embedding with shape ``[B, embedding_dim]``.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()

        # Three zero-sum high-pass kernels. Each is applied to each RGB channel.
        filters = torch.zeros(3, 1, 5, 5, dtype=torch.float32)

        # 5x5 Laplacian-like filter.
        filters[0, 0, 1:4, 1:4] = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        )

        # Horizontal second derivative.
        filters[1, 0, 2, :] = torch.tensor([-1.0, 2.0, 0.0, 2.0, -1.0])

        # Vertical second derivative.
        filters[2, 0, :, 2] = torch.tensor([-1.0, 2.0, 0.0, 2.0, -1.0])

        self.register_buffer("srm_filters", filters, persistent=True)

        self.features = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Extract residual features: [B, 3, H, W] -> [B, 256]."""
        # Repeat each filter for each input channel, yielding 3 * 3 maps.
        kernels = self.srm_filters.repeat(3, 1, 1, 1)  # [9, 1, 5, 5]
        residuals = F.conv2d(
            x,
            kernels,
            padding=2,
            groups=3,
        )  # [B, 9, H, W]

        features = self.features(residuals).flatten(start_dim=1)
        return self.projection(features)  # [B, embedding_dim]
