from __future__ import annotations

import torch
import torch.nn as nn

from models.components.mambacvc_blocks import ConvBNReLU
from models.components.mambacvc_namlab_local_cross import (
    RegionMeanPreservingEncoder,
)
from models.components.mambacvc_namlab_r2v_local import (
    LocalRegionToVisualInteraction,
)


class PureLocalRegionToVisualInteraction(LocalRegionToVisualInteraction):
    """R->V local attention with no direct raw-region reconstruction path.

    Attention itself is unchanged from ``LocalRegionToVisualInteraction``:
    visual provides Q, encoded NAM region provides K/V, and B_M60 remains in
    the local attention logits. The only mechanism change is reconstruction:

        old: [F, R, R->F]
        new: [F, R->F]

    Therefore encoded region information can enter the visual stream only
    through the selected R->F evidence.
    """

    def __init__(
        self,
        channels: int = 32,
        inner_channels: int = 16,
        num_heads: int = 4,
        window_size: int = 3,
    ) -> None:
        super().__init__(
            channels=channels,
            inner_channels=inner_channels,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.reconstruction = nn.Sequential(
            ConvBNReLU(channels * 2, channels, 1),
            ConvBNReLU(channels, channels, 3),
        )

    def forward(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        visual_from_region = self.compute_region_to_visual(
            visual_feature=visual_feature,
            region_feature=region_feature,
            mean_60=mean_60,
        )

        fused = torch.cat(
            [
                visual_feature,
                visual_from_region,
            ],
            dim=1,
        )
        return self.reconstruction(fused)


class RegionMeanToVisualPureLocalGuidance(nn.Module):
    """Encode M60, then expose it to visual features only through R->V."""

    def __init__(
        self,
        channels: int = 32,
        inner_channels: int = 16,
        num_heads: int = 4,
        window_size: int = 3,
    ) -> None:
        super().__init__()
        self.region_encoder = RegionMeanPreservingEncoder(
            channels=channels
        )
        self.interaction = PureLocalRegionToVisualInteraction(
            channels=channels,
            inner_channels=inner_channels,
            num_heads=num_heads,
            window_size=window_size,
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
        return self.interaction(
            visual_feature=feature,
            region_feature=region,
            mean_60=mean_60,
        )
