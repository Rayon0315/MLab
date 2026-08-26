# losses/small_foreground_loss.py

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from losses.sod_loss import SODLoss


class SmallForegroundAwareSODLoss(SODLoss):
    """
    Standard SOD loss + a small-foreground-aware regularizer on the final map.

    Baseline terms are kept unchanged:
        final: BCE + soft IoU
        aux:   BCE + soft IoU

    The extra SFA term does two things:
    1. Gives harder foreground pixels more weight when foreground occupies
       only a small fraction of the image.
    2. Uses focal-style hardness so easy background pixels do not dominate
       the extra optimization term, while hard background is still penalized.

    No fixed "small object threshold" is used. Foreground weighting changes
    continuously with the foreground area ratio.
    """

    def __init__(
        self,
        aux_weight: float = 0.4,
        edge_weight: float = 0.1,
        region_weight: float = 0.0,
        smooth: float = 1.0,
        region_color_epsilon: float = 1e-4,
        sfa_weight: float = 0.5,
        sfa_beta: float = 0.5,
        sfa_gamma: float = 2.0,
        sfa_max_fg_scale: float = 8.0,
        sfa_epsilon: float = 1e-6,
    ) -> None:
        super().__init__(
            aux_weight=aux_weight,
            edge_weight=edge_weight,
            region_weight=region_weight,
            smooth=smooth,
            region_color_epsilon=region_color_epsilon,
        )

        self.sfa_weight = sfa_weight
        self.sfa_beta = sfa_beta
        self.sfa_gamma = sfa_gamma
        self.sfa_max_fg_scale = sfa_max_fg_scale
        self.sfa_epsilon = sfa_epsilon

    def forward(
        self,
        outputs: dict[str, Any],
        target: Tensor,
        nam_target: Tensor | None = None,
        region_target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        loss_dict = super().forward(
            outputs=outputs,
            target=target,
            nam_target=nam_target,
            region_target=region_target,
        )

        base_main_loss = loss_dict["loss_main"]

        (
            sfa_loss,
            foreground_ratio,
            foreground_scale,
        ) = self._small_foreground_aware_loss(
            logits=outputs["pred"],
            target=target,
        )

        weighted_sfa_loss = (
            self.sfa_weight
            * sfa_loss
        )

        loss_dict["loss_main_base"] = base_main_loss
        loss_dict["loss_sfa"] = sfa_loss
        loss_dict["stat_fg_ratio"] = foreground_ratio
        loss_dict["stat_fg_scale"] = foreground_scale

        loss_dict["loss_main"] = (
            base_main_loss
            + weighted_sfa_loss
        )

        loss_dict["loss"] = (
            loss_dict["loss"]
            + weighted_sfa_loss
        )

        return loss_dict

    def _small_foreground_aware_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits = logits.float()
        target = (target.float() > 0.5).float()

        probability = torch.sigmoid(logits)

        foreground_ratio = target.mean(
            dim=(1, 2, 3),
            keepdim=True,
        )

        has_foreground = (
            foreground_ratio
            > self.sfa_epsilon
        )

        foreground_odds = (
            (1.0 - foreground_ratio + self.sfa_epsilon)
            / (foreground_ratio + self.sfa_epsilon)
        )

        foreground_scale = foreground_odds.pow(
            self.sfa_beta
        ).clamp(
            min=1.0,
            max=self.sfa_max_fg_scale,
        )

        foreground_scale = torch.where(
            has_foreground,
            foreground_scale,
            torch.ones_like(foreground_scale),
        )

        # Re-distribute pixel importance inside each image while keeping
        # the average class weight equal to 1. This avoids changing the
        # overall loss magnitude merely because an image has a tiny object.
        class_weight = (
            1.0
            + target
            * (foreground_scale - 1.0)
        )

        class_weight = (
            class_weight
            / class_weight.mean(
                dim=(1, 2, 3),
                keepdim=True,
            ).clamp_min(
                self.sfa_epsilon
            )
        )

        # |p - y| is high for missed foreground and false-positive
        # background. The focal exponent suppresses already-easy pixels.
        hardness = (
            probability
            - target
        ).abs().pow(
            self.sfa_gamma
        )

        binary_cross_entropy = (
            F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
            )
        )

        sfa_loss = (
            class_weight
            * hardness
            * binary_cross_entropy
        ).mean()

        return (
            sfa_loss,
            foreground_ratio.mean().detach(),
            foreground_scale.mean().detach(),
        )
