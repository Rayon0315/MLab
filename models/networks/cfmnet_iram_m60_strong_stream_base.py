"""Canonical CFMNet + IRAM + M60-aware SACA task stream + UNetFormer."""
from __future__ import annotations

import torch
from torch import nn

from models.components.cfm_iram_saca_m60_region_relation import (
    IRAMM60StrongNeck,
)
from models.networks.cfmnet_unetformer_sod import (
    CFMNetUNetFormerSOD,
)


class CFMNetIRAMM60StrongStreamSOD(
    CFMNetUNetFormerSOD
):
    """Only the task-stream interface gains mean_60.

    Backbone, IRAM, outer ThreeStreamFusion and UNetFormer remain unchanged.
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path,
        x_stream: nn.Module,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            decode_channels=64,
            window_size=8,
        )

        self.neck = IRAMM60StrongNeck(
            x_stream
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ):
        features, coarse = self.neck(
            self.backbone(image),
            mean_60=mean_60,
        )

        prediction, auxiliary = self.decoder(
            features=features,
            output_size=image.shape[-2:],
        )

        outputs = {
            "pred": prediction,
        }

        if self.training:
            outputs["aux"] = [
                auxiliary,
                coarse,
            ]

        return outputs
