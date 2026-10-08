"""Canonical SOD loss + soft NRTR region-occupancy auxiliary supervision."""
from __future__ import annotations
from typing import Any
import torch
from torch import Tensor
from torch.nn import functional as F
from losses.sod_loss import SODLoss

class NRTRSODLoss(SODLoss):
    def __init__(
        self,
        aux_weight=0.4,
        edge_weight=0.0,
        region_weight=0.0,
        token_weight=0.1,
        **kwargs,
    ):
        super().__init__(
            aux_weight=aux_weight,
            edge_weight=edge_weight,
            region_weight=region_weight,
            **kwargs,
        )
        self.token_weight = token_weight

    def forward(
        self,
        outputs: dict[str, Any],
        target: Tensor,
        nam_target: Tensor | None = None,
        region_target: Tensor | None = None,
    ):
        loss_dict = super().forward(
            outputs=outputs,
            target=target,
            nam_target=nam_target,
            region_target=region_target,
        )
        region_logits = outputs.get('region_logits')
        association = outputs.get('region_assoc')
        valid = outputs.get('region_valid')
        region_hw = outputs.get('region_hw')
        if region_logits is None or association is None or valid is None or region_hw is None:
            return loss_dict

        target_small = F.interpolate(
            target.float(), size=tuple(region_hw), mode='area'
        )[:, 0].flatten(1)
        a = association.float()
        mass = a.sum(dim=1).clamp_min(1e-6)
        occupancy = torch.einsum('bn,bnk->bk', target_small, a) / mass
        raw = F.binary_cross_entropy_with_logits(
            region_logits.float(), occupancy, reduction='none'
        )
        valid_f = valid.float()
        token_loss = (raw * valid_f).sum() / valid_f.sum().clamp_min(1.0)
        loss_dict['loss_region_token'] = token_loss
        loss_dict['loss'] = loss_dict['loss'] + self.token_weight * token_loss
        return loss_dict
