from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from losses.pafr_loss import PAFRWeakLoss


def nam_region_consistency_loss(
    logits: Tensor,
    region_partition: Tensor,
) -> Tensor:
    """Penalize prediction variance inside each NAMLab region.

    The partition supplies grouping only. It never assigns foreground or
    background labels. The loss is computed on sigmoid probabilities from the
    refined/native PAFR prediction.
    """
    probability = torch.sigmoid(logits.float())[:, 0]

    if region_partition.ndim == 4:
        region_partition = region_partition[:, 0]
    region_partition = region_partition.long()

    if probability.shape != region_partition.shape:
        raise ValueError(
            "Prediction/partition shape mismatch: "
            f"probability={tuple(probability.shape)} "
            f"partition={tuple(region_partition.shape)}"
        )

    batch_size = probability.shape[0]
    region_stride = int(region_partition.max().item()) + 1
    offsets = (
        torch.arange(
            batch_size,
            device=region_partition.device,
            dtype=region_partition.dtype,
        )
        * region_stride
    ).view(batch_size, 1, 1)

    global_region_ids = (
        region_partition + offsets
    ).reshape(-1)
    values = probability.reshape(-1)

    num_groups = batch_size * region_stride
    counts = torch.zeros(
        num_groups,
        device=values.device,
        dtype=values.dtype,
    )
    sums = torch.zeros_like(counts)

    counts.scatter_add_(
        0,
        global_region_ids,
        torch.ones_like(values),
    )
    sums.scatter_add_(
        0,
        global_region_ids,
        values,
    )

    means = sums / counts.clamp_min(1.0)
    centered = values - means[global_region_ids]
    return centered.square().mean()


class PAFRRegionConsistencyLoss(PAFRWeakLoss):
    """PAFR weak objective + soft NAMLab region-consistency regularization."""

    def __init__(
        self,
        region_consistency_weight: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.region_consistency_weight = float(region_consistency_weight)

    def forward(
        self,
        outputs: dict[str, Any],
        scaled_outputs: dict[str, Any],
        image: Tensor,
        target: Tensor,
        valid_mask: Tensor,
        region_partition: Tensor,
    ) -> dict[str, Tensor]:
        loss_dict = super().forward(
            outputs=outputs,
            scaled_outputs=scaled_outputs,
            image=image,
            target=target,
            valid_mask=valid_mask,
        )

        region_loss = nam_region_consistency_loss(
            logits=outputs["pred"],
            region_partition=region_partition,
        )
        weighted_region_loss = (
            self.region_consistency_weight * region_loss
        )

        loss_dict["loss"] = (
            loss_dict["loss"] + weighted_region_loss
        )
        loss_dict["loss_main"] = (
            loss_dict["loss_main"] + weighted_region_loss
        )
        loss_dict["loss_region_consistency"] = region_loss
        return loss_dict
