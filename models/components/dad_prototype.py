from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferenceAwarePrototypeGeneration(nn.Module):
    """
    FG/BG prototype generation adapted from the official DAD code.

    Official DAD behavior retained here:
        intensity = sum_c(Fh) / max(sum_c(Fh))
        foreground guide = sigmoid(C0)
        background guide = sigmoid(1 - C0)
        binary guide = interpolated_guide > 0.5
        prototype = weighted average of Fh
        similarity = cosine(Fl, prototype)
        prototype-aware feature = Fl * similarity

    The original DAD then feeds the two prototype-aware branches into
    its cross-level semantic guidance module. This MLab ablation stops
    at the prototype-aware branches so that the added evidence source
    can be tested independently.
    """

    def __init__(
        self,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.eps = eps

    def _dynamic_weights(
        self,
        features: torch.Tensor,
        guide_probability: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        guide_probability = F.interpolate(
            guide_probability,
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        mask = (
            guide_probability > 0.5
        ).to(
            dtype=features.dtype
        )

        intensity = features.sum(
            dim=1,
            keepdim=True,
        )

        maximum = torch.max(
            intensity
        )

        intensity = intensity / (
            maximum + self.eps
        )

        weights = (
            intensity * mask
        )

        return (
            weights,
            mask,
            intensity,
        )

    def _prototype(
        self,
        features: torch.Tensor,
        guide_probability: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            weights,
            mask,
            intensity,
        ) = self._dynamic_weights(
            features=features,
            guide_probability=guide_probability,
        )

        prototype = (
            features * weights
        ).sum(
            dim=(-2, -1)
        ) / (
            weights.sum(
                dim=(-2, -1)
            )
            + self.eps
        )

        prototype = prototype.unsqueeze(
            -1
        ).unsqueeze(
            -1
        )

        return (
            prototype,
            weights,
            mask,
            intensity,
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
        if high_feature.shape[1] != low_feature.shape[1]:
            raise ValueError(
                "DAD prototype generation requires matching "
                "high/low feature channel dimensions."
            )

        foreground_guide = torch.sigmoid(
            guide_logits
        )

        # Keep the official DAD implementation literally here:
        # sigmoid(1 - Guide_Map), rather than 1 - sigmoid(Guide_Map).
        background_guide = torch.sigmoid(
            1.0 - guide_logits
        )

        (
            foreground_prototype,
            foreground_weight,
            foreground_mask,
            intensity,
        ) = self._prototype(
            features=high_feature,
            guide_probability=foreground_guide,
        )

        (
            background_prototype,
            background_weight,
            background_mask,
            _,
        ) = self._prototype(
            features=high_feature,
            guide_probability=background_guide,
        )

        foreground_similarity = (
            self._cosine_similarity_map(
                features=low_feature,
                prototype=foreground_prototype,
            )
        )

        background_similarity = (
            self._cosine_similarity_map(
                features=low_feature,
                prototype=background_prototype,
            )
        )

        foreground_feature = (
            low_feature
            * foreground_similarity
        )

        background_feature = (
            low_feature
            * background_similarity
        )

        return {
            "foreground_feature": foreground_feature,
            "background_feature": background_feature,
            "foreground_similarity": foreground_similarity,
            "background_similarity": background_similarity,
            "semantic_margin": (
                foreground_similarity
                - background_similarity
            ),
            "foreground_prototype": foreground_prototype,
            "background_prototype": background_prototype,
            "foreground_weight": foreground_weight,
            "background_weight": background_weight,
            "foreground_mask": foreground_mask,
            "background_mask": background_mask,
            "intensity": intensity,
        }
