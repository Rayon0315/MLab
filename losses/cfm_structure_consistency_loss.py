from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from losses.sod_loss import SODLoss


class CFMStructureConsistencyLoss(SODLoss):
    """
    Saliency loss + structure supervision + prediction/structure consistency.

    structure supervision:
        Stage1 SSRM/LDFM structure head -> GT-derived boundary

    consistency:
        differentiable edge(prediction) -> detached structure probability
    """

    def __init__(
        self,
        aux_weight: float = 0.4,
        structure_weight: float = 0.10,
        consistency_weight: float = 0.10,
        edge_kernel: int = 5,
    ) -> None:
        super().__init__(
            aux_weight=aux_weight,
            edge_weight=0.0,
            region_weight=0.0,
        )

        if edge_kernel % 2 == 0:
            raise ValueError("edge_kernel must be odd.")

        self.structure_weight = structure_weight
        self.consistency_weight = consistency_weight
        self.edge_kernel = edge_kernel

    def _soft_morphological_edge(
        self,
        probability: Tensor,
    ) -> Tensor:
        pad = self.edge_kernel // 2

        dilation = F.max_pool2d(
            probability,
            kernel_size=self.edge_kernel,
            stride=1,
            padding=pad,
        )

        erosion = -F.max_pool2d(
            -probability,
            kernel_size=self.edge_kernel,
            stride=1,
            padding=pad,
        )

        return (dilation - erosion).clamp(0.0, 1.0)

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

        structure_logits = outputs.get("structure")
        if structure_logits is None:
            raise KeyError(
                "Structure consistency training requires outputs['structure']."
            )

        structure_logits = F.interpolate(
            structure_logits,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).float()

        target_edge = self._soft_morphological_edge(
            target.float()
        ).detach()

        structure_loss = self._structure_loss(
            logits=structure_logits,
            target=target_edge,
        )

        pred_probability = torch.sigmoid(
            outputs["pred"].float()
        )

        pred_edge = self._soft_morphological_edge(
            pred_probability
        )

        structure_probability = torch.sigmoid(
            structure_logits
        ).detach()

        weight = 1.0 + 4.0 * target_edge

        consistency = (
            weight
            * F.smooth_l1_loss(
                pred_edge,
                structure_probability,
                reduction="none",
            )
        ).sum() / weight.sum().clamp_min(1.0)

        total = (
            loss_dict["loss"]
            + self.structure_weight * structure_loss
            + self.consistency_weight * consistency
        )

        loss_dict["loss_structure"] = structure_loss
        loss_dict["loss_structure_consistency"] = consistency
        loss_dict["loss"] = total

        return loss_dict
