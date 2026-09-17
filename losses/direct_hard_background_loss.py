from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from losses.sod_loss import SODLoss


class DirectHardBackgroundSODLoss(SODLoss):
    """
    Standard SOD loss + direct hard-background penalty on the main saliency logit.

    No CFPS verifier/head is required.

    For each pixel:
        p = sigmoid(pred)
        w_fp = (1 - y) * stopgrad(p ** gamma)

    The added term is:
        sum(w_fp * softplus(pred)) / (sum(w_fp) + 1)

    Therefore only GT-background pixels are affected, and background pixels that
    the model currently predicts as foreground receive the strongest penalty.
    """

    def __init__(
        self,
        aux_weight: float = 0.4,
        edge_weight: float = 0.0,
        region_weight: float = 0.0,
        smooth: float = 1.0,
        region_color_epsilon: float = 1e-4,
        hard_background_weight: float = 0.2,
        hard_background_gamma: float = 2.0,
    ) -> None:
        super().__init__(
            aux_weight=aux_weight,
            edge_weight=edge_weight,
            region_weight=region_weight,
            smooth=smooth,
            region_color_epsilon=region_color_epsilon,
        )

        self.hard_background_weight = hard_background_weight
        self.hard_background_gamma = hard_background_gamma

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

        logits = outputs["pred"].float()

        if target.shape[-2:] != logits.shape[-2:]:
            target_for_fp = F.interpolate(
                target.float(),
                size=logits.shape[-2:],
                mode="nearest",
            )
        else:
            target_for_fp = target.float()

        with torch.no_grad():
            probability = torch.sigmoid(
                logits.detach()
            )

            hard_background_map = (
                (1.0 - target_for_fp)
                * probability.pow(
                    self.hard_background_gamma
                )
            )

        hard_background_loss = (
            F.softplus(logits)
            * hard_background_map
        ).sum() / (
            hard_background_map.sum()
            + 1.0
        )

        loss_dict["loss_hard_background"] = (
            hard_background_loss
        )

        loss_dict["loss"] = (
            loss_dict["loss"]
            + self.hard_background_weight
            * hard_background_loss
        )

        return loss_dict
