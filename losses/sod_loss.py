# losses/sod_loss.py
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SODLoss(nn.Module):
    def __init__(
        self,
        aux_weight: float = 0.4,
        edge_weight: float = 0.1,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        self.aux_weight = aux_weight
        self.edge_weight = edge_weight
        self.smooth = smooth

        self.bce = nn.BCEWithLogitsLoss()

        sobel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer(
            "sobel_x",
            sobel_x,
            persistent=False,
        )

        self.register_buffer(
            "sobel_y",
            sobel_y,
            persistent=False,
        )

    def forward(
        self,
        outputs: dict[str, Any],
        target: Tensor,
        nam_target: Tensor | None = None,
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

        aux_outputs = outputs.get(
            "aux"
        )

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

            auxiliary_loss = torch.stack(
                aux_losses
            ).mean()

            total_loss = (
                total_loss
                + self.aux_weight
                * auxiliary_loss
            )

            loss_dict[
                "loss_aux"
            ] = auxiliary_loss

        edge_pred = outputs.get(
            "edge"
        )

        if edge_pred is not None:
            edge_pred = F.interpolate(
                edge_pred,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            prediction_edge_target = (
                self._label_edge_prediction(
                    torch.sigmoid(
                        pred
                    ).detach()
                )
            )

            edge_consistency_loss = (
                self._structure_loss(
                    logits=edge_pred,
                    target=prediction_edge_target,
                )
            )

            edge_loss = (
                edge_consistency_loss
            )

            loss_dict[
                "loss_edge_consistency"
            ] = edge_consistency_loss

            if nam_target is not None:
                nam_target = F.interpolate(
                    nam_target,
                    size=target.shape[-2:],
                    mode="nearest",
                )

                nam_target = (
                    nam_target > 0.5
                ).float()

                nam_edge_loss = (
                    self._ness_dice_loss(
                        logits=edge_pred,
                        target=nam_target,
                    )
                )

                edge_loss = (
                    edge_loss
                    + nam_edge_loss
                )

                loss_dict[
                    "loss_edge_nam"
                ] = nam_edge_loss

            total_loss = (
                total_loss
                + self.edge_weight
                * edge_loss
            )

            loss_dict[
                "loss_edge"
            ] = edge_loss

        loss_dict["loss"] = (
            total_loss
        )

        return loss_dict

    def _saliency_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        return (
            self.bce(
                logits,
                target,
            )
            + self._soft_iou_loss(
                logits,
                target,
            )
        )

    def _soft_iou_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        probability = torch.sigmoid(
            logits
        ).flatten(start_dim=1)

        target = target.flatten(
            start_dim=1
        )

        intersection = (
            probability * target
        ).sum(dim=1)

        union = (
            probability.sum(dim=1)
            + target.sum(dim=1)
            - intersection
        )

        iou = (
            intersection + self.smooth
        ) / (
            union + self.smooth
        )

        return 1.0 - iou.mean()

    def _structure_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        weight = (
            1.0
            + 5.0
            * torch.abs(
                F.avg_pool2d(
                    target,
                    kernel_size=31,
                    stride=1,
                    padding=15,
                )
                - target
            )
        )

        epsilon = 0.001

        smoothed_target = (
            (1.0 - epsilon)
            * target
            + epsilon / 2.0
        )

        weighted_bce = (
            F.binary_cross_entropy_with_logits(
                logits,
                smoothed_target,
                reduction="none",
            )
        )

        weighted_bce = (
            (
                weight
                * weighted_bce
            ).sum(dim=(2, 3))
            / weight.sum(dim=(2, 3))
        )

        probability = torch.sigmoid(
            logits
        )

        intersection = (
            probability
            * target
            * weight
        ).sum(dim=(2, 3))

        union = (
            (
                probability
                + target
            )
            * weight
        ).sum(dim=(2, 3))

        weighted_iou = (
            1.0
            - (
                intersection + 1.0
            )
            / (
                union
                - intersection
                + 1.0
            )
        )

        return (
            weighted_bce
            + weighted_iou
        ).mean()

    def _ness_dice_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        probability = torch.sigmoid(
            logits
        ).flatten(start_dim=1)

        target = target.flatten(
            start_dim=1
        )

        numerator = (
            2.0
            * (
                probability
                * target
            ).sum(dim=1)
            + self.smooth
        )

        denominator = (
            probability.pow(2).sum(dim=1)
            + target.pow(2).sum(dim=1)
            + self.smooth
        )

        return (
            1.0
            - numerator
            / denominator
        ).mean()

    def _label_edge_prediction(
        self,
        target: Tensor,
    ) -> Tensor:
        target = (
            target > 0.5
        ).float()

        target = F.pad(
            target,
            pad=(1, 1, 1, 1),
            mode="replicate",
        )

        gradient_x = F.conv2d(
            target,
            self.sobel_x,
        )

        gradient_y = F.conv2d(
            target,
            self.sobel_y,
        )

        gradient = torch.sqrt(
            gradient_x.pow(2)
            + gradient_y.pow(2)
        )

        return (
            gradient > 1.5
        ).float()