from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
)


class AsymmetricForegroundSupportV2(nn.Module):
    """
    Semantically anchored positive foreground support.

    v1 already removed the symmetric FG/BG prototype assumption. v2 keeps
    that asymmetric topology but makes the internal foreground evidence
    directly supervisable.

    Important separation of roles:
        fg_similarity:
            raw cross-level metric relation, range [-1, 1].

        fg_support_logits:
            learned semantic calibration of the metric relation.

        fg_support:
            sigmoid(fg_support_logits), used as positive evidence for
            feature modulation and later as conditioning for BG rejection.

    There is still:
        - no background prototype;
        - no FG/BG softmax;
        - no requirement that lack of FG evidence means BG evidence.

    Residual injection:
        refined = low + residual_scale * Psi(low * fg_support)

    residual_scale starts at exactly zero, making this block an identity at
    initialization. The residual branch itself is NOT zero-initialized:
    otherwise both the residual branch and residual_scale would receive zero
    gradient at the first step (a dead branch).
    """

    def __init__(
        self,
        channels: int,
        metric_channels: int = 64,
        initial_support_scale: float = 4.0,
        initial_residual_scale: float = 0.0,
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

        # Learned common metric space.
        # Signed coordinates are preserved for cosine matching.
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

        # Positive semantic calibration slope.
        # softplus(raw) == initial_support_scale.
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

        # Do NOT zero-initialize foreground_residual[-1].
        # With residual_scale == 0 that would make both sides of the product
        # zero-gradient at initialization.
        self.residual_scale = nn.Parameter(
            torch.tensor(
                float(initial_residual_scale),
                dtype=torch.float32,
            )
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

        # Keep the activation weighting that fixed the signed / batch-global
        # issue in DAD-v2.
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

        fg_support_logits = (
            support_scale
            * fg_similarity
            + self.support_bias
        )

        fg_support = torch.sigmoid(
            fg_support_logits
        )

        supported_feature = (
            low_feature
            * fg_support
        )

        foreground_residual = (
            self.foreground_residual(
                supported_feature
            )
        )

        refined_feature = (
            low_feature
            + self.residual_scale
            * foreground_residual
        )

        return {
            "refined_feature": refined_feature,
            "fg_similarity": fg_similarity,
            "fg_support_logits": fg_support_logits,
            "fg_support": fg_support,
            "fg_prototype": fg_prototype,
            "fg_weight": fg_weight,
            "activation": activation,
            "guide_foreground_probability": (
                guide_probability_high
            ),
            "support_scale": support_scale,
            "residual_scale": self.residual_scale,
        }
