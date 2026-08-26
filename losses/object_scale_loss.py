# losses/object_scale_loss.py

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from losses.sod_loss import SODLoss


class ObjectBalancedSODLoss(SODLoss):
    """
    Standard SOD loss + cosine-decayed object-balanced foreground supervision.

    Final prediction:
        BCE + soft IoU + lambda_obj(epoch) * object-balanced loss

    Auxiliary predictions:
        BCE + soft IoU

    Object-balanced term:
        1. Average positive BCE inside each connected salient object.
        2. Average objects inside each image.
        3. Average images inside the batch.

    lambda_obj uses cosine decay from object_weight at epoch 1
    to 0 at the final epoch.
    """

    def __init__(
        self,
        aux_weight: float = 0.4,
        edge_weight: float = 0.1,
        smooth: float = 1.0,
        object_weight: float = 0.1,
    ) -> None:
        super().__init__(
            aux_weight=aux_weight,
            edge_weight=edge_weight,
            region_weight=0.0,
            smooth=smooth,
        )

        self.object_weight = float(
            object_weight
        )
        self.current_object_weight = float(
            object_weight
        )

    def set_epoch(
        self,
        epoch: int,
        total_epochs: int,
    ) -> None:
        if total_epochs <= 1:
            progress = 1.0
        else:
            progress = (
                (epoch - 1)
                / (total_epochs - 1)
            )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        cosine_factor = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi * progress
                )
            )
        )

        self.current_object_weight = (
            self.object_weight
            * cosine_factor
        )

    def forward(
        self,
        outputs: dict[str, Any],
        target: Tensor,
        nam_target: Tensor | None = None,
        object_labels: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if object_labels is None:
            raise ValueError(
                "object_labels are required "
                "for ObjectBalancedSODLoss."
            )

        loss_dict = super().forward(
            outputs=outputs,
            target=target,
            nam_target=nam_target,
            region_target=None,
        )

        base_main_loss = (
            loss_dict["loss_main"]
        )

        (
            object_loss,
            objects_per_image,
            mean_object_area_ratio,
        ) = self._object_balanced_loss(
            logits=outputs["pred"],
            object_labels=object_labels,
        )

        weighted_object_loss = (
            self.current_object_weight
            * object_loss
        )

        loss_dict[
            "loss_main_base"
        ] = base_main_loss

        loss_dict[
            "loss_object"
        ] = object_loss

        loss_dict[
            "stat_object_weight"
        ] = object_loss.new_tensor(
            self.current_object_weight
        ).detach()

        loss_dict[
            "stat_objects_per_image"
        ] = objects_per_image

        loss_dict[
            "stat_object_area_ratio"
        ] = mean_object_area_ratio

        loss_dict[
            "loss_main"
        ] = (
            base_main_loss
            + weighted_object_loss
        )

        loss_dict[
            "loss"
        ] = (
            loss_dict["loss"]
            + weighted_object_loss
        )

        return loss_dict

    def _object_balanced_loss(
        self,
        logits: Tensor,
        object_labels: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
    ]:
        logits = logits.float()

        labels = (
            object_labels
            .detach()
            .long()
        )

        if (
            labels.shape[-2:]
            != logits.shape[-2:]
        ):
            labels = F.interpolate(
                labels.float(),
                size=logits.shape[-2:],
                mode="nearest",
            ).long()

        labels = labels.squeeze(1)

        positive_bce = F.softplus(
            -logits.squeeze(1)
        )

        batch_size = labels.shape[0]

        max_label = int(
            labels.max()
            .detach()
            .item()
        )

        if max_label == 0:
            zero = logits.sum() * 0.0

            return (
                zero,
                zero.detach(),
                zero.detach(),
            )

        stride = max_label + 1

        image_offsets = (
            torch.arange(
                batch_size,
                device=labels.device,
                dtype=labels.dtype,
            )
            * stride
        ).view(
            batch_size,
            1,
            1,
        )

        global_labels = (
            labels + image_offsets
        )

        foreground_mask = (
            labels > 0
        )

        component_indices = (
            global_labels[
                foreground_mask
            ]
        )

        component_values = (
            positive_bce[
                foreground_mask
            ]
        )

        slot_count = (
            batch_size * stride
        )

        component_loss_sum = torch.zeros(
            slot_count,
            device=logits.device,
            dtype=logits.dtype,
        )

        component_pixel_count = torch.zeros(
            slot_count,
            device=logits.device,
            dtype=logits.dtype,
        )

        component_loss_sum.scatter_add_(
            dim=0,
            index=component_indices,
            src=component_values,
        )

        component_pixel_count.scatter_add_(
            dim=0,
            index=component_indices,
            src=torch.ones_like(
                component_values
            ),
        )

        valid_components = (
            component_pixel_count > 0
        )

        valid_component_ids = torch.nonzero(
            valid_components,
            as_tuple=False,
        ).squeeze(1)

        component_losses = (
            component_loss_sum[
                valid_components
            ]
            / component_pixel_count[
                valid_components
            ]
        )

        component_image_ids = torch.div(
            valid_component_ids,
            stride,
            rounding_mode="floor",
        )

        image_loss_sum = torch.zeros(
            batch_size,
            device=logits.device,
            dtype=logits.dtype,
        )

        image_object_count = torch.zeros(
            batch_size,
            device=logits.device,
            dtype=logits.dtype,
        )

        image_loss_sum.scatter_add_(
            dim=0,
            index=component_image_ids,
            src=component_losses,
        )

        image_object_count.scatter_add_(
            dim=0,
            index=component_image_ids,
            src=torch.ones_like(
                component_losses
            ),
        )

        valid_images = (
            image_object_count > 0
        )

        per_image_object_loss = (
            image_loss_sum[
                valid_images
            ]
            / image_object_count[
                valid_images
            ]
        )

        object_loss = (
            per_image_object_loss.mean()
        )

        objects_per_image = (
            image_object_count
            .mean()
            .detach()
        )

        image_area = float(
            labels.shape[-2]
            * labels.shape[-1]
        )

        mean_object_area_ratio = (
            component_pixel_count[
                valid_components
            ].mean()
            / image_area
        ).detach()

        return (
            object_loss,
            objects_per_image,
            mean_object_area_ratio,
        )
