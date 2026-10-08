"""Shared CFMNet + IRAM + SACA + NRTR base.

This version consumes the official NAMLab MAT -> NPZ partition through the
`region_partition` model input. SACA internals remain unchanged.
"""
from __future__ import annotations

import torch
from torch import nn

from models.components.cfm_iram_saca_nrtr import (
    NAMRegionTokenReasoning,
)
from models.components.cfm_iram_strong_common import (
    IRAMStrongNeck,
)
from models.networks.cfmnet_unetformer_sod import (
    CFMNetUNetFormerSOD,
)


class CFMNetIRAMSACANRTRSOD(
    CFMNetUNetFormerSOD
):
    input_keys = (
        "image",
        "mean_60",
        "region_partition",
    )

    def __init__(
        self,
        pretrained_path,
        x_stream: nn.Module,
        token_dim=128,
        max_regions=96,
    ):
        super().__init__(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=64,
            window_size=8,
        )

        self.neck = IRAMStrongNeck(
            x_stream=x_stream
        )

        self.nrtr = (
            NAMRegionTokenReasoning(
                feature_channels=192,
                token_dim=token_dim,
                max_regions=max_regions,
                depth=2,
                num_heads=4,
            )
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
        region_partition: torch.Tensor,
    ):
        original_features = list(
            self.backbone(
                image
            )
        )

        # Complete validated IRAM + SACA host first.
        host_features, coarse = (
            self.neck(
                original_features
            )
        )

        # Region tokens are gathered from ORIGINAL F2.
        # The COMPLETE host F2 acts as the pixel query during readout.
        refined_f2, region_aux = (
            self.nrtr(
                original_f2=(
                    original_features[1]
                ),
                host_f2=(
                    host_features[1]
                ),
                mean_60=mean_60,
                region_partition=(
                    region_partition
                ),
                f1_size=(
                    original_features[0]
                    .shape[-2:]
                ),
            )
        )

        host_features[1] = (
            refined_f2
        )

        prediction, auxiliary = (
            self.decoder(
                features=host_features,
                output_size=(
                    image.shape[-2:]
                ),
            )
        )

        outputs = {
            "pred": prediction,
            **region_aux,
            "region_hw": (
                refined_f2.shape[-2],
                refined_f2.shape[-1],
            ),
        }

        if self.training:
            outputs["aux"] = [
                auxiliary,
                coarse,
            ]

        return outputs
