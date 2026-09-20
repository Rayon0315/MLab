from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from losses.fp_verifier_loss import (
    FalsePositiveAwareSODLoss,
)


class AsymmetricEvidenceSODLoss(
    FalsePositiveAwareSODLoss
):
    """
    SOD + asymmetric evidence supervision.

    Inherits:
        main / aux saliency supervision
        optional region / edge losses
        false-positive risk supervision

    Adds:
        direct foreground-support supervision

        L_fg = BCEWithLogits(fg_support_logits, GT)
             + SoftIoU(fg_support_logits, GT)

    This anchors fg_support to actual foreground semantics before the support
    signal is allowed to condition the detached BG rejection branch.
    """

    def __init__(
        self,
        aux_weight: float = 0.4,
        edge_weight: float = 0.0,
        region_weight: float = 0.0,
        smooth: float = 1.0,
        region_color_epsilon: float = 1e-4,
        fp_risk_weight: float = 0.2,
        fp_risk_gamma: float = 2.0,
        easy_background_weight: float = 0.05,
        fg_support_weight: float = 0.1,
    ) -> None:
        super().__init__(
            aux_weight=aux_weight,
            edge_weight=edge_weight,
            region_weight=region_weight,
            smooth=smooth,
            region_color_epsilon=(
                region_color_epsilon
            ),
            fp_risk_weight=fp_risk_weight,
            fp_risk_gamma=fp_risk_gamma,
            easy_background_weight=(
                easy_background_weight
            ),
        )

        self.fg_support_weight = float(
            fg_support_weight
        )

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

        support_logits = outputs.get(
            "fg_support_logits"
        )

        if support_logits is None:
            return loss_dict

        support_logits = (
            support_logits.float()
        )

        support_target = target.float()

        if (
            support_logits.shape[-2:]
            != support_target.shape[-2:]
        ):
            support_logits = F.interpolate(
                support_logits,
                size=support_target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fg_support_loss = (
            self._saliency_loss(
                logits=support_logits,
                target=support_target,
            )
        )

        loss_dict[
            "loss_fg_support"
        ] = fg_support_loss

        loss_dict["loss"] = (
            loss_dict["loss"]
            + self.fg_support_weight
            * fg_support_loss
        )

        return loss_dict
