from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from losses.fp_verifier_loss import (
    FalsePositiveAwareSODLoss,
)


class RelativeCandidateAwareSODLoss(
    FalsePositiveAwareSODLoss
):
    """
    Existing SOD + FP-risk loss
    + candidate-level relative saliency supervision.

    Candidate target:
        soft foreground fraction inside each detached
        semantic candidate region.

    Candidate loss:
        1. BCE with soft candidate target
        2. pairwise relative-ranking loss

    The pairwise term only becomes strong when two candidates
    have clearly different GT overlap, because it is weighted by
    |target_i - target_j|.
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
        candidate_weight: float = 0.1,
        candidate_rank_weight: float = 0.5,
    ) -> None:
        super().__init__(
            aux_weight=aux_weight,
            edge_weight=edge_weight,
            region_weight=region_weight,
            smooth=smooth,
            region_color_epsilon=region_color_epsilon,
            fp_risk_weight=fp_risk_weight,
            fp_risk_gamma=fp_risk_gamma,
            easy_background_weight=(
                easy_background_weight
            ),
        )

        self.candidate_weight = (
            candidate_weight
        )
        self.candidate_rank_weight = (
            candidate_rank_weight
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

        candidate_logits = outputs.get(
            "candidate_logits"
        )
        candidate_regions = outputs.get(
            "candidate_regions"
        )

        if (
            candidate_logits is None
            or candidate_regions is None
        ):
            return loss_dict

        candidate_logits = (
            candidate_logits.float()
        )

        candidate_regions = (
            candidate_regions
            .detach()
            .float()
        )

        target_small = F.interpolate(
            target.float(),
            size=(
                candidate_regions.shape[-2],
                candidate_regions.shape[-1],
            ),
            mode="nearest",
        )

        numerator = (
            candidate_regions
            * target_small
        ).sum(
            dim=(-2, -1)
        )

        denominator = (
            candidate_regions.sum(
                dim=(-2, -1)
            ).clamp_min(
                1e-6
            )
        )

        candidate_target = (
            numerator
            / denominator
        ).clamp(
            0.0,
            1.0,
        )

        candidate_bce = (
            F.binary_cross_entropy_with_logits(
                candidate_logits,
                candidate_target,
            )
        )

        # -------------------------------------------------
        # Pairwise relative saliency ranking.
        # -------------------------------------------------
        target_difference = (
            candidate_target.unsqueeze(2)
            - candidate_target.unsqueeze(1)
        )

        logit_difference = (
            candidate_logits.unsqueeze(2)
            - candidate_logits.unsqueeze(1)
        )

        target_sign = torch.sign(
            target_difference
        )

        pair_weight = (
            target_difference.abs()
        )

        pairwise_loss = (
            F.softplus(
                -target_sign
                * logit_difference
            )
            * pair_weight
        )

        num_candidates = (
            candidate_logits.shape[1]
        )

        upper_triangle = torch.triu(
            torch.ones(
                num_candidates,
                num_candidates,
                device=candidate_logits.device,
                dtype=candidate_logits.dtype,
            ),
            diagonal=1,
        ).unsqueeze(0)

        pairwise_loss = (
            pairwise_loss
            * upper_triangle
        )

        pair_weight = (
            pair_weight
            * upper_triangle
        )

        candidate_rank_loss = (
            pairwise_loss.sum()
            / (
                pair_weight.sum()
                + 1.0
            )
        )

        candidate_loss = (
            candidate_bce
            + self.candidate_rank_weight
            * candidate_rank_loss
        )

        loss_dict[
            "loss_candidate_bce"
        ] = candidate_bce

        loss_dict[
            "loss_candidate_rank"
        ] = candidate_rank_loss

        loss_dict[
            "loss_candidate"
        ] = candidate_loss

        loss_dict["loss"] = (
            loss_dict["loss"]
            + self.candidate_weight
            * candidate_loss
        )

        return loss_dict
