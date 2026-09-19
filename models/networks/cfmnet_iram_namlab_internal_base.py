"""Canonical CFMNet + unchanged UNetFormer with internal-NAM saliency neck."""
from __future__ import annotations

import torch

from models.components.cfm_iram_namlab_internal_adapter import (
    IRAMNAMLabInternalNeck,
)
from models.networks.cfmnet_unetformer_sod import (
    CFMNetUNetFormerSOD,
)


class CFMNetIRAMNAMLabInternalSOD(CFMNetUNetFormerSOD):
    input_keys = ("image", "mean_60")

    def __init__(
        self,
        pretrained_path,
        mode: str,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            decode_channels=64,
            window_size=8,
        )
        self.neck = IRAMNAMLabInternalNeck(
            mode=mode,
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
            # Preserve the current successful supervision protocol:
            # canonical decoder auxiliary + saliency-task coarse output.
            outputs["aux"] = [
                auxiliary,
                coarse,
            ]

        return outputs
