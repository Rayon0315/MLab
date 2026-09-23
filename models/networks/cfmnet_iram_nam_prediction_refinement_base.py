"""Canonical CFMNet + IRAM + SACA + UNetFormer, then NAM prediction refinement."""
from __future__ import annotations

import torch
from torch import nn

from models.components.cfm_iram_saca_nam_structural_evidence import (
    NAMPredictionStructureRefiner,
)
from models.components.cfm_iram_strong_common import IRAMStrongNeck
from models.networks.cfmnet_unetformer_sod import CFMNetUNetFormerSOD


class CFMNetIRAMNAMPredictionRefinementSOD(CFMNetUNetFormerSOD):
    input_keys = ("image", "mean_60")

    def __init__(self, pretrained_path, x_stream: nn.Module) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            decode_channels=64,
            window_size=8,
        )
        self.neck = IRAMStrongNeck(x_stream=x_stream)
        self.prediction_refiner = NAMPredictionStructureRefiner()

    def forward(self, image: torch.Tensor, mean_60: torch.Tensor):
        features, coarse = self.neck(self.backbone(image))
        base_prediction, auxiliary = self.decoder(
            features=features,
            output_size=image.shape[-2:],
        )
        prediction = self.prediction_refiner(
            logits=base_prediction,
            mean_60=mean_60,
        )

        outputs = {"pred": prediction}
        if self.training:
            outputs["aux"] = [auxiliary, coarse]
        return outputs
