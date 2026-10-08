from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ScribbleSODLoss(nn.Module):
    """Partial BCE on annotated scribble pixels only."""

    def __init__(
        self,
        aux_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.aux_weight = aux_weight

    def forward(
        self,
        outputs: dict[str, Any],
        target: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        pred = outputs["pred"]

        main_loss = self._masked_bce(
            logits=pred,
            target=target,
            valid_mask=valid_mask,
        )

        total_loss = main_loss
        loss_dict: dict[str, Tensor] = {
            "loss_main": main_loss,
        }

        auxiliary_outputs = outputs.get("aux")

        if auxiliary_outputs:
            auxiliary_losses: list[Tensor] = []

            for auxiliary_prediction in auxiliary_outputs:
                auxiliary_prediction = F.interpolate(
                    auxiliary_prediction,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                auxiliary_losses.append(
                    self._masked_bce(
                        logits=auxiliary_prediction,
                        target=target,
                        valid_mask=valid_mask,
                    )
                )

            auxiliary_loss = torch.stack(
                auxiliary_losses
            ).mean()

            total_loss = (
                total_loss
                + self.aux_weight * auxiliary_loss
            )

            loss_dict["loss_aux"] = auxiliary_loss

        valid_mask_float = valid_mask.detach().float()
        target_float = target.detach().float()

        valid_count = valid_mask_float.sum().clamp_min(1.0)

        loss_dict["stat_valid_ratio"] = (
            valid_mask_float.mean()
        )
        loss_dict["stat_fg_ratio_on_valid"] = (
            (target_float * valid_mask_float).sum()
            / valid_count
        )
        loss_dict["loss"] = total_loss

        return loss_dict

    @staticmethod
    def _masked_bce(
        logits: Tensor,
        target: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        logits = logits.float()
        target = target.float()
        valid_mask = valid_mask.float()

        loss_map = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )

        valid_count = valid_mask.sum().clamp_min(1.0)

        return (
            loss_map * valid_mask
        ).sum() / valid_count
