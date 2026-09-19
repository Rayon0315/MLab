from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferenceAwarePrototypeGenerationV2(nn.Module):
    """
    DAD-style FG/BG prototype evidence, adapted to MLab feature statistics.

    Differences from the literal DAD v1 port:
        1. Feature intensity uses per-pixel L2 channel norm instead of
           signed channel sum.
        2. Foreground/background guides are strictly complementary:
               p_fg = sigmoid(C)
               p_bg = 1 - p_fg
        3. Soft weights are used directly; no hard 0.5 threshold.
        4. The guide is detached so the prototype path does not reshape
           the aux prediction only to make prototype generation easier.
        5. FG/BG cosine scores are converted into a 2-way relative
           probability by softmax.
        6. The two gated low-level branches are complementary:
               F_fg = q_fg * F_low
               F_bg = q_bg * F_low
           and q_fg + q_bg = 1 at every pixel.

    This component deliberately keeps the outer DAD-v1 topology unchanged
    so that the experiment isolates the internal evidence expression.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        eps: float = 1e-6,
        detach_guide: bool = True,
    ) -> None:
        super().__init__()

        if temperature <= 0.0:
            raise ValueError(
                "temperature must be > 0."
            )

        self.temperature = float(
            temperature
        )
        self.eps = float(
            eps
        )
        self.detach_guide = bool(
            detach_guide
        )

    def _activation_magnitude(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Positive per-pixel activation magnitude:
            a(x) = ||F(x)||_2

        Normalize independently for each sample so one image cannot set
        the scale of another image in the batch.
        """
        magnitude = torch.linalg.vector_norm(
            features,
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
        features: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted global prototype:
            P = sum_x w(x) F(x) / sum_x w(x)
        """
        numerator = (
            features * weights
        ).sum(
            dim=(-2, -1),
            keepdim=True,
        )

        denominator = weights.sum(
            dim=(-2, -1),
            keepdim=True,
        )

        return numerator / (
            denominator + self.eps
        )

    def _cosine_similarity_map(
        self,
        features: torch.Tensor,
        prototype: torch.Tensor,
    ) -> torch.Tensor:
        similarity = F.cosine_similarity(
            features,
            prototype,
            dim=1,
            eps=self.eps,
        )

        return similarity.unsqueeze(1)

    def forward(
        self,
        high_feature: torch.Tensor,
        low_feature: torch.Tensor,
        guide_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if (
            high_feature.shape[1]
            != low_feature.shape[1]
        ):
            raise ValueError(
                "DAD prototype generation requires matching "
                "high/low feature channel dimensions."
            )

        foreground_probability = torch.sigmoid(
            guide_logits
        )

        if self.detach_guide:
            foreground_probability = (
                foreground_probability.detach()
            )

        foreground_probability_high = F.interpolate(
            foreground_probability,
            size=high_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        background_probability_high = (
            1.0 - foreground_probability_high
        )

        activation = self._activation_magnitude(
            high_feature
        )

        foreground_weight = (
            activation
            * foreground_probability_high
        )

        background_weight = (
            activation
            * background_probability_high
        )

        foreground_prototype = (
            self._weighted_prototype(
                high_feature,
                foreground_weight,
            )
        )

        background_prototype = (
            self._weighted_prototype(
                high_feature,
                background_weight,
            )
        )

        foreground_similarity = (
            self._cosine_similarity_map(
                low_feature,
                foreground_prototype,
            )
        )

        background_similarity = (
            self._cosine_similarity_map(
                low_feature,
                background_prototype,
            )
        )

        similarity_pair = torch.cat(
            [
                foreground_similarity,
                background_similarity,
            ],
            dim=1,
        )

        evidence_probability = torch.softmax(
            similarity_pair
            / self.temperature,
            dim=1,
        )

        foreground_evidence = (
            evidence_probability[:, 0:1]
        )
        background_evidence = (
            evidence_probability[:, 1:2]
        )

        semantic_margin = (
            foreground_evidence
            - background_evidence
        )

        foreground_feature = (
            low_feature
            * foreground_evidence
        )

        background_feature = (
            low_feature
            * background_evidence
        )

        return {
            "foreground_feature": foreground_feature,
            "background_feature": background_feature,
            "foreground_similarity": foreground_similarity,
            "background_similarity": background_similarity,
            "foreground_evidence": foreground_evidence,
            "background_evidence": background_evidence,
            "semantic_margin": semantic_margin,
            "foreground_prototype": foreground_prototype,
            "background_prototype": background_prototype,
            "foreground_weight": foreground_weight,
            "background_weight": background_weight,
            "activation": activation,
            "guide_foreground_probability": (
                foreground_probability_high
            ),
            "guide_background_probability": (
                background_probability_high
            ),
        }
