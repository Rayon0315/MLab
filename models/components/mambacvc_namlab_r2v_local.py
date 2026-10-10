from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.mambacvc_blocks import ConvBNReLU
from models.components.mambacvc_namlab_local_cross import (
    PixelLayerNorm,
    RegionMeanPreservingEncoder,
)


class LocalRegionToVisualInteraction(nn.Module):
    """One-way local NAM region -> visual cross attention.

    Visual features only produce queries. Encoded NAM region features only
    produce keys/values. There is deliberately no Visual -> Region branch.

    To keep the direction ablation controlled against the existing full
    bidirectional block, the same local window and the same raw-M60 structural
    bias are retained:

        A(p,q) = softmax(Q_visual(p) K_region(q)^T / sqrt(d) + B_M60(p,q))

    The result reconstructs the visual stream from [visual, region, R->V].
    The region encoder remains trainable; "one-way" describes the forward
    interaction topology, not gradient detachment.
    """

    def __init__(
        self,
        channels: int = 32,
        inner_channels: int = 16,
        num_heads: int = 4,
        window_size: int = 3,
    ) -> None:
        super().__init__()

        if inner_channels % num_heads != 0:
            raise ValueError("inner_channels must be divisible by num_heads")
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")

        self.channels = int(channels)
        self.inner_channels = int(inner_channels)
        self.num_heads = int(num_heads)
        self.head_channels = inner_channels // num_heads
        self.window_size = int(window_size)
        self.window_area = window_size * window_size
        self.padding = window_size // 2
        self.scale = self.head_channels ** -0.5

        self.visual_norm = PixelLayerNorm(channels)
        self.region_norm = PixelLayerNorm(channels)

        # Strictly one-way topology: Q comes from visual; K/V come from region.
        self.visual_query = nn.Conv2d(
            channels, inner_channels, kernel_size=1, bias=False
        )
        self.region_key = nn.Conv2d(
            channels, inner_channels, kernel_size=1, bias=False
        )
        self.region_value = nn.Conv2d(
            channels, inner_channels, kernel_size=1, bias=False
        )

        self.visual_from_region_projection = ConvBNReLU(
            inner_channels, channels, 1
        )

        # Preserve the current LocalCross structural prior for a clean
        # direction-only mechanism ablation. exp() keeps lambda > 0.
        self.region_bias_log_scale = nn.Parameter(torch.tensor(0.0))

        # No explicit scalar bypass/gate. The guidance participates directly
        # in the GC -> LC transition, matching the current full block policy.
        self.reconstruction = nn.Sequential(
            ConvBNReLU(channels * 3, channels, 1),
            ConvBNReLU(channels, channels, 3),
        )

    def _reshape_tokens(self, feature: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feature.shape
        return feature.reshape(
            b, self.num_heads, self.head_channels, h * w
        )

    def _extract_patches(self, feature: torch.Tensor) -> torch.Tensor:
        b, _, h, w = feature.shape
        patches = F.unfold(
            feature,
            kernel_size=self.window_size,
            padding=self.padding,
        )
        return patches.reshape(
            b,
            self.num_heads,
            self.head_channels,
            self.window_area,
            h * w,
        )

    def _valid_window_mask(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ones = torch.ones(
            batch_size,
            1,
            height,
            width,
            device=device,
            dtype=dtype,
        )
        valid = F.unfold(
            ones,
            kernel_size=self.window_size,
            padding=self.padding,
        )
        return valid.reshape(
            batch_size,
            1,
            self.window_area,
            height * width,
        ) > 0.5

    def _region_similarity_bias(
        self,
        mean_60: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        mean = F.interpolate(
            mean_60,
            size=target_size,
            mode="nearest",
        )
        b, c, h, w = mean.shape
        if c != 3:
            raise ValueError("mean_60 must contain RGB region means")

        center = mean.reshape(b, c, 1, h * w)
        patches = F.unfold(
            mean,
            kernel_size=self.window_size,
            padding=self.padding,
        ).reshape(
            b,
            c,
            self.window_area,
            h * w,
        )

        difference = (
            patches - center
        ).abs().mean(dim=1, keepdim=True)
        strength = torch.exp(self.region_bias_log_scale)
        return -strength * difference

    def compute_region_to_visual(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        if visual_feature.shape != region_feature.shape:
            raise ValueError("visual and region features must have equal shape")

        visual = self.visual_norm(visual_feature)
        region = self.region_norm(region_feature)

        query = self.visual_query(visual)
        key = self.region_key(region)
        value = self.region_value(region)

        b, _, h, w = query.shape
        q = self._reshape_tokens(query)
        k = self._extract_patches(key)
        v = self._extract_patches(value)

        scores = (
            q.unsqueeze(3) * k
        ).sum(dim=2) * self.scale
        scores = scores + self._region_similarity_bias(
            mean_60,
            target_size=(h, w),
        )

        valid = self._valid_window_mask(
            b,
            h,
            w,
            device=query.device,
            dtype=query.dtype,
        )
        scores = scores.masked_fill(
            ~valid,
            torch.finfo(scores.dtype).min,
        )
        attention = torch.softmax(scores, dim=2)

        update = (
            attention.unsqueeze(2) * v
        ).sum(dim=3)
        update = update.reshape(
            b,
            self.inner_channels,
            h,
            w,
        )
        return self.visual_from_region_projection(update)

    def forward(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        visual_from_region = self.compute_region_to_visual(
            visual_feature=visual_feature,
            region_feature=region_feature,
            mean_60=mean_60,
        )

        fused = torch.cat(
            [
                visual_feature,
                region_feature,
                visual_from_region,
            ],
            dim=1,
        )
        return self.reconstruction(fused)


class RegionMeanToVisualLocalGuidance(nn.Module):
    """Unified one-way NAM M60 guidance used at fn1/fn2/fn3."""

    def __init__(
        self,
        channels: int = 32,
        inner_channels: int = 16,
        num_heads: int = 4,
        window_size: int = 3,
    ) -> None:
        super().__init__()
        self.region_encoder = RegionMeanPreservingEncoder(
            channels=channels
        )
        self.interaction = LocalRegionToVisualInteraction(
            channels=channels,
            inner_channels=inner_channels,
            num_heads=num_heads,
            window_size=window_size,
        )

    def forward(
        self,
        feature: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        region = self.region_encoder(
            mean_60,
            target_size=feature.shape[-2:],
        )
        return self.interaction(
            visual_feature=feature,
            region_feature=region,
            mean_60=mean_60,
        )
