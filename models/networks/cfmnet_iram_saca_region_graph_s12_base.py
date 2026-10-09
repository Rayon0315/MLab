"""CFMNet + Stage1/Stage2 RegionGraph + canonical IRAM/SACA host.

RegionGraph is applied before the validated IRAM + SACA neck:

    CFMNet -> [F1, F2, F3, F4]
              |    |
              v    v
             RG1  RG2
              |    |
              v    v
           [F1G, F2G, F3, F4]
                    -> unchanged IRAM + SACA
                    -> unchanged UNetFormer

The graph block is task-agnostic. SACA/IRAM internals are not modified.
"""
from __future__ import annotations

import torch
from torch import nn

from models.components.cfm_region_graph import RegionGraphBlock
from models.components.cfm_iram_strong_common import IRAMStrongNeck
from models.networks.cfmnet_unetformer_sod import CFMNetUNetFormerSOD


class CFMNetIRAMSACARegionGraphS12SOD(CFMNetUNetFormerSOD):
    input_keys = (
        "image",
        "region_partition",
    )

    def __init__(
        self,
        pretrained_path,
        x_stream: nn.Module,
        graph_depth: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            decode_channels=64,
            window_size=8,
        )

        # CFMNet output channels are (96, 192, 384, 768).
        # Stage1 and Stage2 use the same RegionGraph mechanism but retain
        # independent parameters because their channel spaces are different.
        self.region_graph1 = RegionGraphBlock(
            channels=96,
            graph_depth=graph_depth,
            num_heads=num_heads,
        )
        self.region_graph2 = RegionGraphBlock(
            channels=192,
            graph_depth=graph_depth,
            num_heads=num_heads,
        )

        # Canonical validated IRAM + task stream host.
        self.neck = IRAMStrongNeck(
            x_stream=x_stream
        )

    def forward(
        self,
        image: torch.Tensor,
        region_partition: torch.Tensor,
    ):
        features = list(
            self.backbone(image)
        )

        # Region relationship reasoning happens BEFORE IRAM and SACA.
        features[0] = self.region_graph1(
            feature=features[0],
            region_partition=region_partition,
        )
        features[1] = self.region_graph2(
            feature=features[1],
            region_partition=region_partition,
        )

        # Everything below remains the validated IRAM + SACA v1 host path.
        features, coarse = self.neck(features)

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
