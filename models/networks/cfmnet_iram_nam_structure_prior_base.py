"""Canonical CFMNet + IRAM + NAM-structure-prior SACA + UNetFormer."""
from __future__ import annotations

import torch
from torch import nn

from models.components.cfm_iram_saca_nam_structure_prior import (
    IRAMNAMStructurePriorNeck,
)
from models.networks.cfmnet_unetformer_sod import (
    CFMNetUNetFormerSOD,
)


class CFMNetIRAMNAMStructurePriorSOD(
    CFMNetUNetFormerSOD
):
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

        self.neck = (
            IRAMNAMStructurePriorNeck(
                x_stream=x_stream,
            )
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ):
        features, coarse = self.neck(
            features=self.backbone(image),
            image=image,
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
