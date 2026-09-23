from __future__ import annotations

import torch
import torch.nn as nn

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
)


class RegionVisualUnifiedFusion(nn.Module):
    """
    Simple unified region-visual fusion for Stage2/3/4.

    Input:
        visual_feature: [B, C, H, W]
        region_feature: [B, C, H, W]

    Output:
        visual_feature + residual reconstruction
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.fusion = nn.Sequential(
            ConvNormAct(
                channels * 2,
                channels,
                kernel_size=1,
                padding=0,
            ),
            ResidualConvBlock(
                channels
            ),
        )

    def forward(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
    ) -> torch.Tensor:
        fused = torch.cat(
            [
                visual_feature,
                region_feature,
            ],
            dim=1,
        )

        reconstruction = self.fusion(
            fused
        )

        return (
            visual_feature
            + reconstruction
        )
