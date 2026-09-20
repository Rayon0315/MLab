from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
)


class AsymmetricForegroundSupport(nn.Module):
    """
    Positive-only foreground evidence module.

    Design principle:
        absence of foreground evidence != presence of background evidence.

    The module therefore builds only a foreground prototype and produces an
    independent foreground support map. There is no background prototype and
    no FG/BG softmax.

    Pipeline:
        high_feature -> high projection -> weighted FG prototype
        low_feature  -> low projection  -> cosine with FG prototype
        cosine -> calibrated sigmoid -> foreground support
        low_feature * support -> zero-init residual -> refined low feature

    The guide is detached, matching the successful DAD-v2 behavior: the
    auxiliary guide should select prototype evidence, not reshape itself just
    to make prototype construction easier.
    """

    def __init__(
        self,
        channels: int,
        metric_channels: int = 64,
        initial_support_scale: float = 4.0,
        eps: float = 1e-6,
        detach_guide: bool = True,
    ) -> None:
        super().__init__()

        if metric_channels % 8 != 0:
            raise ValueError(
                "metric_channels must be divisible by 8 for GroupNorm."
            )

        if initial_support_scale <= 0.0:
            raise ValueError(
                "initial_support_scale must be > 0."
            )

        self.eps = float(eps)
        self.detach_guide = bool(
            detach_guide
        )

        # Learned cross-level metric space.
        # No activation after GroupNorm: signed coordinates are useful for
        # cosine matching and avoid forcing all projected features positive.
        self.high_projection = nn.Sequential(
            nn.Conv2d(
                channels,
                metric_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=metric_channels,
            ),
        )

        self.low_projection = nn.Sequential(
            nn.Conv2d(
                channels,
                metric_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=metric_channels,
            ),
        )

        # Positive scale through softplus:
        # support = sigmoid(scale * cosine + bias)
        initial_raw = math.log(
            math.expm1(
                initial_support_scale
            )
        )

        self.support_scale_raw = nn.Parameter(
            torch.tensor(
                initial_raw,
                dtype=torch.float32,
            )
        )

        self.support_bias = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

        # Only inject evidence as a residual correction. The final convolution
        # is zero-initialized so the module starts exactly as identity:
        # refined_feature == low_feature.
        self.foreground_residual = nn.Sequential(
            ConvNormAct(
                channels,
                channels,
                kernel_size=3,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        )

        nn.init.zeros_(
            self.foreground_residual[-1].weight
        )

    def _activation_magnitude(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        magnitude = torch.linalg.vector_norm(
            feature,
            ord=2,
            dim=1,
            keepdim=True,
        )

        maximum = magnitude.amax(
            dim=(-2, -1),
            keepdim=True,
        )

        return magnitude / (
            maximum + self.eps
        )

    def _weighted_prototype(
        self,
        projected_feature: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        numerator = (
            projected_feature
            * weight
        ).sum(
            dim=(-2, -1),
            keepdim=True,
        )

        denominator = weight.sum(
            dim=(-2, -1),
            keepdim=True,
        )

        return numerator / (
            denominator + self.eps
        )

    def forward(
        self,
        high_feature: torch.Tensor,
        low_feature: torch.Tensor,
        guide_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        guide_probability = torch.sigmoid(
            guide_logits
        )

        if self.detach_guide:
            guide_probability = (
                guide_probability.detach()
            )

        guide_probability_high = F.interpolate(
            guide_probability,
            size=high_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        activation = self._activation_magnitude(
            high_feature
        )

        fg_weight = (
            activation
            * guide_probability_high
        )

        high_metric = self.high_projection(
            high_feature
        )

        low_metric = self.low_projection(
            low_feature
        )

        fg_prototype = self._weighted_prototype(
            projected_feature=high_metric,
            weight=fg_weight,
        )

        fg_similarity = F.cosine_similarity(
            low_metric,
            fg_prototype,
            dim=1,
            eps=self.eps,
        ).unsqueeze(1)

        support_scale = F.softplus(
            self.support_scale_raw
        )

        fg_support = torch.sigmoid(
            support_scale
            * fg_similarity
            + self.support_bias
        )

        supported_feature = (
            low_feature
            * fg_support
        )

        residual = self.foreground_residual(
            supported_feature
        )

        refined_feature = (
            low_feature
            + residual
        )

        return {
            "refined_feature": refined_feature,
            "fg_support": fg_support,
            "fg_similarity": fg_similarity,
            "fg_prototype": fg_prototype,
            "fg_weight": fg_weight,
            "activation": activation,
            "guide_foreground_probability": (
                guide_probability_high
            ),
            "support_scale": support_scale,
        }
