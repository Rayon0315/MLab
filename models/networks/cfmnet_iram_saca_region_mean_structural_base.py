"""Shared base for SACA-v1/v2 x Region-Mean structural field."""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from models.components.cfm_iram_saca_region_mean_structural import (
    F1StructuralReconstruction,
    LateStructuralCorrection,
    RegionMeanStructuralEncoder,
)
from models.components.cfm_iram_strong_common import IRAMStrongNeck
from models.networks.cfmnet_unetformer_sod import CFMNetUNetFormerSOD


RegionMeanMode = Literal["f1", "late"]


class CFMNetIRAMSACARegionMeanStructuralSOD(CFMNetUNetFormerSOD):
    input_keys = ("image", "mean_60")

    def __init__(
        self,
        pretrained_path,
        x_stream: nn.Module,
        mode: RegionMeanMode,
        structural_channels: int = 96,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            decode_channels=64,
            window_size=8,
        )

        if mode not in {"f1", "late"}:
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        self.neck = IRAMStrongNeck(x_stream=x_stream)
        self.structural_encoder = RegionMeanStructuralEncoder(
            channels=structural_channels
        )

        if mode == "f1":
            self.f1_reconstruction = F1StructuralReconstruction(
                f1_channels=96,
                structural_channels=structural_channels,
            )
        else:
            self.late2 = LateStructuralCorrection(
                host_channels=192,
                structural_channels=structural_channels,
            )
            self.late3 = LateStructuralCorrection(
                host_channels=384,
                structural_channels=structural_channels,
            )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ):
        original_features = list(self.backbone(image))

        # SACA/IRAM always see only the original CFMNet features.
        host_features, coarse = self.neck(original_features)

        if self.mode == "f1":
            structural1 = self.structural_encoder(
                mean_60=mean_60,
                target_size=original_features[0].shape[-2:],
            )
            host_features[0] = self.f1_reconstruction(
                f1=original_features[0],
                structural=structural1,
            )
        else:
            structural2 = self.structural_encoder(
                mean_60=mean_60,
                target_size=host_features[1].shape[-2:],
            )
            structural3 = self.structural_encoder(
                mean_60=mean_60,
                target_size=host_features[2].shape[-2:],
            )

            host_features[1] = self.late2(
                host_feature=host_features[1],
                structural=structural2,
            )
            host_features[2] = self.late3(
                host_feature=host_features[2],
                structural=structural3,
            )

        prediction, auxiliary = self.decoder(
            features=host_features,
            output_size=image.shape[-2:],
        )

        outputs = {"pred": prediction}

        if self.training:
            outputs["aux"] = [auxiliary, coarse]

        return outputs
