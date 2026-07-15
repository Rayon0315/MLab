# losses/sod_loss.py

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SODLoss(nn.Module):
    """
    Default loss for binary RGB salient object detection.

    The loss for one prediction is:

        BCEWithLogitsLoss + Soft IoU Loss

    Expected model outputs:

        {
            "pred": Tensor[B, 1, H, W],
            "aux": list[Tensor]  # optional
        }

    Returned dictionary:

        {
            "loss": total_loss,
            "loss_main": main_loss,
            "loss_aux": auxiliary_loss,  # only when aux exists
        }
    """

    def __init__(
        self,
        aux_weight: float = 0.4,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()

        if aux_weight < 0:
            raise ValueError(
                f"aux_weight must be non-negative, got {aux_weight}."
            )

        if smooth <= 0:
            raise ValueError(
                f"smooth must be positive, got {smooth}."
            )

        self.aux_weight = aux_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        outputs: dict[str, Any],
        target: Tensor,
    ) -> dict[str, Tensor]:
        self._check_target(target)

        if "pred" not in outputs:
            raise KeyError(
                'Model outputs must contain the key "pred".'
            )

        pred = outputs["pred"]

        if not isinstance(pred, Tensor):
            raise TypeError(
                'outputs["pred"] must be a torch.Tensor.'
            )

        self._check_prediction(
            prediction=pred,
            target=target,
            name='outputs["pred"]',
            require_same_size=True,
        )

        main_loss = self._prediction_loss(
            logits=pred,
            target=target,
        )

        loss_dict = {
            "loss": main_loss,
            "loss_main": main_loss,
        }

        aux_outputs = outputs.get("aux")

        if aux_outputs is None:
            return loss_dict

        if not isinstance(aux_outputs, list):
            raise TypeError(
                'outputs["aux"] must be a list of tensors.'
            )

        if len(aux_outputs) == 0:
            return loss_dict

        aux_losses: list[Tensor] = []

        for index, aux_pred in enumerate(aux_outputs):
            if not isinstance(aux_pred, Tensor):
                raise TypeError(
                    f'outputs["aux"][{index}] must be a torch.Tensor.'
                )

            self._check_prediction(
                prediction=aux_pred,
                target=target,
                name=f'outputs["aux"][{index}]',
                require_same_size=False,
            )

            if aux_pred.shape[-2:] != target.shape[-2:]:
                aux_pred = F.interpolate(
                    aux_pred,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            aux_losses.append(
                self._prediction_loss(
                    logits=aux_pred,
                    target=target,
                )
            )

        auxiliary_loss = torch.stack(aux_losses).mean()
        total_loss = main_loss + self.aux_weight * auxiliary_loss

        return {
            "loss": total_loss,
            "loss_main": main_loss,
            "loss_aux": auxiliary_loss,
        }

    def _prediction_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        bce_loss = self.bce(logits, target)
        iou_loss = self._soft_iou_loss(logits, target)

        return bce_loss + iou_loss

    def _soft_iou_loss(
        self,
        logits: Tensor,
        target: Tensor,
    ) -> Tensor:
        probability = torch.sigmoid(logits)

        probability = probability.flatten(start_dim=1)
        target = target.flatten(start_dim=1)

        intersection = (probability * target).sum(dim=1)

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

    @staticmethod
    def _check_target(target: Tensor) -> None:
        if not isinstance(target, Tensor):
            raise TypeError("target must be a torch.Tensor.")

        if target.ndim != 4:
            raise ValueError(
                "target must have shape [B, 1, H, W], "
                f"but got {tuple(target.shape)}."
            )

        if target.shape[1] != 1:
            raise ValueError(
                "target must have exactly one channel, "
                f"but got {target.shape[1]}."
            )

        if not target.is_floating_point():
            raise TypeError(
                "target must be a floating-point tensor."
            )

    @staticmethod
    def _check_prediction(
        prediction: Tensor,
        target: Tensor,
        name: str,
        require_same_size: bool,
    ) -> None:
        if prediction.ndim != 4:
            raise ValueError(
                f"{name} must have shape [B, 1, H, W], "
                f"but got {tuple(prediction.shape)}."
            )

        if prediction.shape[1] != 1:
            raise ValueError(
                f"{name} must have exactly one channel, "
                f"but got {prediction.shape[1]}."
            )

        if prediction.shape[0] != target.shape[0]:
            raise ValueError(
                f"{name} and target have different batch sizes: "
                f"{prediction.shape[0]} and {target.shape[0]}."
            )

        if not prediction.is_floating_point():
            raise TypeError(
                f"{name} must be a floating-point tensor."
            )

        if (
            require_same_size
            and prediction.shape[-2:] != target.shape[-2:]
        ):
            raise ValueError(
                f"{name} must match the target spatial size. "
                f"Got prediction={tuple(prediction.shape[-2:])}, "
                f"target={tuple(target.shape[-2:])}."
            )