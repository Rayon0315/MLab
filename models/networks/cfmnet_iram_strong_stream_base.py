"""Canonical CFMNet and UNetFormer, with a supervised independent-stream neck."""
from __future__ import annotations

import torch

from models.components.cfm_iram_strong_common import IRAMStrongNeck
from models.networks.cfmnet_unetformer_sod import CFMNetUNetFormerSOD


class CFMNetIRAMStrongStreamSOD(CFMNetUNetFormerSOD):
    def __init__(self, pretrained_path, x_stream):
        super().__init__(pretrained_path=pretrained_path, decode_channels=64, window_size=8)
        self.neck = IRAMStrongNeck(x_stream)

    def forward(self, image: torch.Tensor):
        # Only the four fused backbone outputs cross the neck interface.
        features, coarse = self.neck(self.backbone(image))
        prediction, auxiliary = self.decoder(features=features, output_size=image.shape[-2:])
        outputs = {"pred": prediction}
        if self.training:
            # Canonical decoder produces one aux; SODLoss averages these two.
            outputs["aux"] = [auxiliary, coarse]
        return outputs
