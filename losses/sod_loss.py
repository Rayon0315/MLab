# losses/sod_loss.py

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SODLoss(nn.Module):
    def __init__(
        self,
        aux_weight: float = 0.4,
        edge_weight: float = 0.1,
        region_weight: float = 0.0,
        smooth: float = 1.0,
        region_color_epsilon: float = 1e-4,
    ) -> None:
        super().__init__()

        self.aux_weight = aux_weight
        self.edge_weight = edge_weight
        self.region_weight = region_weight
        self.smooth = smooth
        self.region_color_epsilon = (
            region_color_epsilon
        )

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
        region_target: Tensor | None = None,
    ) -> dict[str, Tensor]:
        pred = outputs["pred"]

        main_loss = self._saliency_loss(
            logits=pred,
            target=target,
        )

        total_loss = main_loss

        loss_dict: dict[
            str,
            Tensor,
        ] = {
            "loss_main": main_loss,
        }

        auxiliary_outputs = outputs.get(
            "aux"
        )

        if auxiliary_outputs:
            auxiliary_losses: list[
                Tensor
            ] = []

            for auxiliary_prediction in (
                auxiliary_outputs
            ):
                auxiliary_prediction = (
                    F.interpolate(
                        auxiliary_prediction,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                )

                auxiliary_losses.append(
                    self._saliency_loss(
                        logits=(
                            auxiliary_prediction
                        ),
                        target=target,
                    )
                )

            auxiliary_loss = torch.stack(
                auxiliary_losses
            ).mean()

            total_loss = (
                total_loss
                + self.aux_weight
                * auxiliary_loss
            )

            loss_dict[
                "loss_aux"
            ] = auxiliary_loss

        if (
            region_target is not None
            and self.region_weight > 0.0
        ):
            region_loss = (
                self._region_consistency_loss(
                    logits=pred,
                    region_map=region_target,
                )
            )

            total_loss = (
                total_loss
                + self.region_weight
                * region_loss
            )

            loss_dict[
                "loss_region"
            ] = region_loss

        edge_prediction = outputs.get(
            "edge"
        )

        if edge_prediction is not None:
            edge_prediction = F.interpolate(
                edge_prediction,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            edge_prediction_float = (
                edge_prediction.float()
            )

            saliency_probability = (
                torch.sigmoid(
                    pred.detach().float()
                )
            )

            prediction_edge_target = (
                self._label_edge_prediction(
                    saliency_probability
                )
            )

            edge_consistency_loss = (
                self._structure_loss(
                    logits=(
                        edge_prediction_float
                    ),
                    target=(
                        prediction_edge_target
                    ),
                )
            )

            edge_loss = (
                edge_consistency_loss
            )

            loss_dict[
                "loss_edge_consistency"
            ] = edge_consistency_loss

            if nam_target is not None:
                nam_target_float = (
                    nam_target
                    .detach()
                    .float()
                )

                nam_target_float = (
                    F.interpolate(
                        nam_target_float,
                        size=target.shape[-2:],
                        mode="nearest",
                    )
                )

                nam_target_float = (
                    nam_target_float
                    > 0.5
                ).float()

                nam_edge_loss = (
                    self._ness_dice_loss(
                        logits=(
                            edge_prediction_float
                        ),
                        target=(
                            nam_target_float
                        ),
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

        loss_dict[
            "loss"
        ] = total_loss

        return loss_dict

    def _saliency_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        binary_cross_entropy = (
            self.bce(
                logits,
                target,
            )
        )

        soft_iou = (
            self._soft_iou_loss(
                logits=logits,
                target=target,
            )
        )

        return (
            binary_cross_entropy
            + soft_iou
        )

    def _soft_iou_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        probability = torch.sigmoid(
            logits
        ).flatten(
            start_dim=1
        )

        flattened_target = (
            target.flatten(
                start_dim=1
            )
        )

        intersection = (
            probability
            * flattened_target
        ).sum(
            dim=1
        )

        union = (
            probability.sum(
                dim=1
            )
            + flattened_target.sum(
                dim=1
            )
            - intersection
        )

        iou = (
            intersection
            + self.smooth
        ) / (
            union
            + self.smooth
        )

        return (
            1.0
            - iou.mean()
        )

    def _region_consistency_loss(
        self,
        logits: Tensor,
        region_map: Tensor,
    ) -> Tensor:
        logits = logits.float()

        region_map = (
            region_map
            .detach()
            .float()
        )

        if (
            region_map.shape[-2:]
            != logits.shape[-2:]
        ):
            region_map = (
                F.interpolate(
                    region_map,
                    size=logits.shape[-2:],
                    mode="nearest",
                )
            )

        probability = torch.sigmoid(
            logits
        )

        horizontal_region_difference = (
            region_map[
                :,
                :,
                :,
                1:,
            ]
            - region_map[
                :,
                :,
                :,
                :-1,
            ]
        ).abs()

        horizontal_same_region = (
            horizontal_region_difference
            .amax(
                dim=1,
                keepdim=True,
            )
            <= self.region_color_epsilon
        ).float()

        horizontal_prediction_difference = (
            probability[
                :,
                :,
                :,
                1:,
            ]
            - probability[
                :,
                :,
                :,
                :-1,
            ]
        ).abs()

        vertical_region_difference = (
            region_map[
                :,
                :,
                1:,
                :,
            ]
            - region_map[
                :,
                :,
                :-1,
                :,
            ]
        ).abs()

        vertical_same_region = (
            vertical_region_difference
            .amax(
                dim=1,
                keepdim=True,
            )
            <= self.region_color_epsilon
        ).float()

        vertical_prediction_difference = (
            probability[
                :,
                :,
                1:,
                :,
            ]
            - probability[
                :,
                :,
                :-1,
                :,
            ]
        ).abs()

        numerator = (
            (
                horizontal_prediction_difference
                * horizontal_same_region
            ).sum()
            + (
                vertical_prediction_difference
                * vertical_same_region
            ).sum()
        )

        denominator = (
            horizontal_same_region.sum()
            + vertical_same_region.sum()
        ).clamp_min(
            1.0
        )

        return (
            numerator
            / denominator
        )

    def _structure_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        logits = logits.float()
        target = target.float()

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
            ).sum(
                dim=(2, 3)
            )
            / weight.sum(
                dim=(2, 3)
            )
        )

        probability = torch.sigmoid(
            logits
        )

        intersection = (
            probability
            * target
            * weight
        ).sum(
            dim=(2, 3)
        )

        union = (
            (
                probability
                + target
            )
            * weight
        ).sum(
            dim=(2, 3)
        )

        weighted_iou = (
            1.0
            - (
                intersection
                + 1.0
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
            logits.float()
        ).flatten(
            start_dim=1
        )

        flattened_target = (
            target.float().flatten(
                start_dim=1
            )
        )

        numerator = (
            2.0
            * (
                probability
                * flattened_target
            ).sum(
                dim=1
            )
            + self.smooth
        )

        denominator = (
            probability.pow(2).sum(
                dim=1
            )
            + flattened_target.pow(2).sum(
                dim=1
            )
            + self.smooth
        )

        dice = (
            numerator
            / denominator
        )

        return (
            1.0
            - dice
        ).mean()

    def _label_edge_prediction(
        self,
        target: Tensor,
    ) -> Tensor:
        device_type = (
            target.device.type
        )

        with torch.autocast(
            device_type=device_type,
            enabled=False,
        ):
            binary_target = (
                target
                .detach()
                .float()
                > 0.5
            ).float()

            binary_target = F.pad(
                binary_target,
                pad=(1, 1, 1, 1),
                mode="replicate",
            )

            sobel_x = self.sobel_x.to(
                device=binary_target.device,
                dtype=torch.float32,
            )

            sobel_y = self.sobel_y.to(
                device=binary_target.device,
                dtype=torch.float32,
            )

            gradient_x = F.conv2d(
                binary_target,
                sobel_x,
            )

            gradient_y = F.conv2d(
                binary_target,
                sobel_y,
            )

            gradient = torch.sqrt(
                gradient_x.square()
                + gradient_y.square()
            )

            edge_target = (
                gradient > 1.5
            ).float()

        return edge_target