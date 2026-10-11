from __future__ import annotations

import torch
import torch.nn as nn

from models.components.mambacvc_blocks import ConvBNReLU
from models.components.mambacvc_namlab_local_cross import (
    RegionMeanPreservingEncoder,
)
from models.components.mambacvc_namlab_pure_r2v_local import (
    PureLocalRegionToVisualInteraction,
)


class ScaleSPADELocalRegionToVisualInteraction(
    PureLocalRegionToVisualInteraction
):
    """SPADE-inspired scale-only modulation for selected R->V evidence.

    The validated pure R->V attention is kept unchanged:
      - visual feature provides Q;
      - encoded NAM region feature provides K/V;
      - the 3x3 local window is unchanged;
      - B_M60 remains in the attention logits.

    Raw encoded region feature R never enters reconstruction as content.
    Instead R predicts a spatial-and-channel scale for the already selected
    R->V evidence G:

        scale = 1 + tanh(Psi(R))       in (0, 2)
        G_mod = scale * G
        out = reconstruction([F, G_mod])

    There is deliberately no beta/shift branch and no scalar bypass gate.
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

        self.region_scale_controller = nn.Sequential(
            ConvBNReLU(channels, channels, 3),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True,
            ),
        )

    def compute_region_scale(
        self,
        region_feature: torch.Tensor,
    ) -> torch.Tensor:
        gamma = self.region_scale_controller(
            region_feature
        )
        return 1.0 + torch.tanh(gamma)

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

        region_scale = self.compute_region_scale(
            region_feature
        )
        modulated_evidence = (
            region_scale
            * visual_from_region
        )

        fused = torch.cat(
            [
                visual_feature,
                modulated_evidence,
            ],
            dim=1,
        )
        return self.reconstruction(fused)


class RegionMeanToVisualScaleSPADEGuidance(nn.Module):
    """Encode M60, select R->V evidence, then scale-modulate it with R."""

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
        self.interaction = ScaleSPADELocalRegionToVisualInteraction(
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
