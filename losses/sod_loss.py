# losses/sod_loss.py
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SODLoss(nn.Module):
    """
    Saliency loss:
        BCEWithLogitsLoss + Soft IoU Loss

    Optional edge loss:
        BCEWithLogitsLoss + Soft Dice Loss

    Expected outputs:
        {
            "pred": Tensor,
            "aux": list[Tensor],
            "edge": Tensor,
        }
    """

    def __init__(
        self,
        aux_weight: float = 0.4,
        edge_weight: float = 0.2,
        edge_kernel_size: int = 3,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.aux_weight = aux_weight
        self.edge_weight = edge_weight
        self.edge_kernel_size = edge_kernel_size
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: dict[str, Any],
        target: Tensor,
    ) -> dict[str, Tensor]:
        pred = outputs["pred"]
        main_loss = self._saliency_loss(
            logits=pred,
            target=target,
        )

        total_loss = main_loss
        loss_dict = {
            "loss": total_loss,
            "loss_main": main_loss,
        }

        aux_outputs = outputs.get("aux")
        if aux_outputs:
            aux_losses = []
            for aux_pred in aux_outputs:
                aux_pred = F.interpolate(
                    aux_pred,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                aux_losses.append(
                    self._saliency_loss(
                        logits=aux_pred,
                        target=target,
                    )
                )

            auxiliary_loss = torch.stack(aux_losses).mean()
            total_loss = total_loss + self.aux_weight * auxiliary_loss
            loss_dict["loss_aux"] = auxiliary_loss

        edge_pred = outputs.get("edge")
        if edge_pred is not None:
            if edge_pred.shape[-2:] != target.shape[-2:]:
                edge_pred = F.interpolate(
                    edge_pred,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            edge_target = self._build_edge_target(target)
            edge_loss = self._edge_loss(
                logits=edge_pred,
                target=edge_target,
            )
            total_loss = total_loss + self.edge_weight * edge_loss
            loss_dict["loss_edge"] = edge_loss

        loss_dict["loss"] = total_loss
        return loss_dict

    def _saliency_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        return self.bce(logits, target) + self._soft_iou_loss(
            logits,
            target,
        )

    def _edge_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        return self.bce(logits, target) + self._soft_dice_loss(
            logits,
            target,
        )

    def _soft_iou_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        probability = torch.sigmoid(logits).flatten(start_dim=1)
        target = target.flatten(start_dim=1)

        intersection = (probability * target).sum(dim=1)
        union = (
            probability.sum(dim=1)
            + target.sum(dim=1)
            - intersection
        )
        iou = (intersection + self.smooth) / (
            union + self.smooth
        )
        return 1.0 - iou.mean()

    def _soft_dice_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        probability = torch.sigmoid(logits).flatten(start_dim=1)
        target = target.flatten(start_dim=1)

        intersection = (probability * target).sum(dim=1)
        denominator = probability.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        return 1.0 - dice.mean()

    def _build_edge_target(
        self,
        target: Tensor,
    ) -> Tensor:
        padding = self.edge_kernel_size // 2
        dilated = F.max_pool2d(
            target,
            kernel_size=self.edge_kernel_size,
            stride=1,
            padding=padding,
        )
        eroded = 1.0 - F.max_pool2d(
            1.0 - target,
            kernel_size=self.edge_kernel_size,
            stride=1,
            padding=padding,
        )
        return (dilated - eroded).clamp(0.0, 1.0)
