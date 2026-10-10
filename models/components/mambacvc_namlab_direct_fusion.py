from __future__ import annotations

import torch
import torch.nn as nn

from models.components.mambacvc_blocks import ConvBNReLU
from models.components.mambacvc_namlab_local_cross import (
    RegionMeanPreservingEncoder,
)


class RegionMeanDirectFusion(nn.Module):
    """Directly fuse encoded NAM region information into a visual feature.

    This ablation deliberately removes all relation modeling:
      - no Q/K/V projections;
      - no local attention;
      - no raw-M60 structural bias.

    The only NAM path is:
        mean_60 -> RegionMeanPreservingEncoder -> region feature
        [visual, region] -> 1x1 Conv-BN-ReLU -> 3x3 Conv-BN-ReLU
    """

    def __init__(
        self,
        channels: int = 32,
    ) -> None:
        super().__init__()

        self.region_encoder = RegionMeanPreservingEncoder(
            channels=channels,
        )
        self.reconstruction = nn.Sequential(
            ConvBNReLU(channels * 2, channels, 1),
            ConvBNReLU(channels, channels, 3),
        )

    def forward(
        self,
        feature: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        region = self.region_encoder(
            mean_60,
            target_size=feature.shape[-2:],
        )
        return self.reconstruction(
            torch.cat([feature, region], dim=1)
        )
