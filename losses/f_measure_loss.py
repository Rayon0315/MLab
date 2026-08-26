# losses/f_measure_loss.py

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from losses.sod_loss import SODLoss


class FMeasureSODLoss(SODLoss):
    """
    Standard SOD loss + relaxed differentiable F-measure loss.

    Final prediction:
        BCE + soft IoU + lambda_f * F-loss

    Auxiliary predictions:
        BCE + soft IoU

    Soft F-beta:
        TP = sum(p * y)
        FP = sum(p * (1 - y))
        FN = sum((1 - p) * y)

        F_beta =
            (1 + beta^2) * TP
            ---------------------------------------
            (1 + beta^2) * TP + beta^2 * FN + FP
    """

    def __init__(
        self,
        aux_weight: float = 0.4,
        edge_weight: float = 0.1,
        region_weight: float = 0.0,
        smooth: float = 1.0,
        region_color_epsilon: float = 1e-4,
        f_weight: float = 0.2,
        f_beta2: float = 0.3,
        f_epsilon: float = 1e-6,
    ) -> None:
        super().__init__(
            aux_weight=aux_weight,
            edge_weight=edge_weight,
            region_weight=region_weight,
            smooth=smooth,
            region_color_epsilon=region_color_epsilon,
        )

        self.f_weight = f_weight
        self.f_beta2 = f_beta2
        self.f_epsilon = f_epsilon

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

        f_measure_loss = self._f_measure_loss(
            logits=outputs["pred"],
            target=target,
        )

        weighted_f_loss = (
            self.f_weight
            * f_measure_loss
        )

        loss_dict["loss_main_base"] = base_main_loss
        loss_dict["loss_f"] = f_measure_loss

        loss_dict["loss_main"] = (
            base_main_loss
            + weighted_f_loss
        )

        loss_dict["loss"] = (
            loss_dict["loss"]
            + weighted_f_loss
        )

        return loss_dict

    def _f_measure_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        logits = logits.float()
        target = target.float()

        probability = torch.sigmoid(
            logits
        ).flatten(
            start_dim=1
        )

        target = target.flatten(
            start_dim=1
        )

        true_positive = (
            probability
            * target
        ).sum(
            dim=1
        )

        false_positive = (
            probability
            * (1.0 - target)
        ).sum(
            dim=1
        )

        false_negative = (
            (1.0 - probability)
            * target
        ).sum(
            dim=1
        )

        numerator = (
            (1.0 + self.f_beta2)
            * true_positive
            + self.f_epsilon
        )

        denominator = (
            (1.0 + self.f_beta2)
            * true_positive
            + self.f_beta2
            * false_negative
            + false_positive
            + self.f_epsilon
        )

        f_measure = (
            numerator
            / denominator
        )

        return (
            1.0
            - f_measure.mean()
        )