from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from losses.sod_loss import SODLoss


class FalsePositiveAwareSODLoss(SODLoss):
    """
    Original SODLoss + explicit false-positive risk supervision.

    fp_risk target:
        (1 - GT) * stopgrad(sigmoid(coarse_pred)) ** gamma

    Therefore:
        - true foreground is always a negative FP-risk sample;
        - easy background receives little weight;
        - background already predicted as foreground receives
          strong positive FP-risk supervision.

    The main saliency loss is still applied to outputs["pred"],
    i.e. after one-way suppression.
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
    ) -> None:
        super().__init__(
            aux_weight=aux_weight,
            edge_weight=edge_weight,
            region_weight=region_weight,
            smooth=smooth,
            region_color_epsilon=region_color_epsilon,
        )

        self.fp_risk_weight = fp_risk_weight
        self.fp_risk_gamma = fp_risk_gamma
        self.easy_background_weight = (
            easy_background_weight
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

        fp_risk_logits = outputs.get(
            "fp_risk"
        )
        coarse_logits = outputs.get(
            "coarse_pred"
        )

        if (
            fp_risk_logits is None
            or coarse_logits is None
        ):
            return loss_dict

        fp_risk_logits = fp_risk_logits.float()

        if target.shape[-2:] != fp_risk_logits.shape[-2:]:
            fp_target_mask = F.interpolate(
                target.float(),
                size=fp_risk_logits.shape[-2:],
                mode="nearest",
            )
        else:
            fp_target_mask = target.float()

        if coarse_logits.shape[-2:] != fp_risk_logits.shape[-2:]:
            coarse_logits = F.interpolate(
                coarse_logits,
                size=fp_risk_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        with torch.no_grad():
            coarse_probability = torch.sigmoid(
                coarse_logits.detach().float()
            )

            # Soft false-positive risk target.
            fp_risk_target = (
                (1.0 - fp_target_mask)
                * coarse_probability.pow(
                    self.fp_risk_gamma
                )
            )

            # Strong negatives:
            # true foreground must NOT be suppressed.
            #
            # Easy background remains a weak negative so the head
            # does not freely fire everywhere.
            negative_weight = (
                fp_target_mask
                + self.easy_background_weight
                * (1.0 - fp_target_mask)
                * (1.0 - fp_risk_target)
            )

        # Positive term:
        # high-confidence background should predict q_fp -> 1.
        positive_loss = (
            F.softplus(
                -fp_risk_logits
            )
            * fp_risk_target
        ).sum() / (
            fp_risk_target.sum()
            + 1.0
        )

        # Negative term:
        # true foreground, plus a small amount of easy background,
        # should predict q_fp -> 0.
        negative_loss = (
            F.softplus(
                fp_risk_logits
            )
            * negative_weight
        ).sum() / (
            negative_weight.sum()
            + 1.0
        )

        fp_risk_loss = 0.5 * (
            positive_loss
            + negative_loss
        )

        loss_dict["loss_fp_risk_positive"] = (
            positive_loss
        )
        loss_dict["loss_fp_risk_negative"] = (
            negative_loss
        )
        loss_dict["loss_fp_risk"] = (
            fp_risk_loss
        )

        loss_dict["loss"] = (
            loss_dict["loss"]
            + self.fp_risk_weight
            * fp_risk_loss
        )

        return loss_dict
