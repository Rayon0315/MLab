from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
)


class ThreeWayFPNFusion(nn.Module):
    """
    Simple three-source FPN fusion.

    Inputs:
        low_feature:
            Current lateral feature from the encoder.
        high_feature:
            Top-down decoder feature from the deeper level.
        global_feature:
            Persistent Stage4 global semantic descriptor.

    All three inputs use the same channel width.

    Fusion:
        [low, upsample(high), broadcast(global)]
            -> concat
            -> 1x1 projection
            -> residual refinement
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.fusion = nn.Sequential(
            ConvNormAct(
                channels * 3,
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
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        target_size = low_feature.shape[-2:]

        high_feature = F.interpolate(
            high_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        global_feature = F.interpolate(
            global_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat(
            [
                low_feature,
                high_feature,
                global_feature,
            ],
            dim=1,
        )

        return self.fusion(
            fused
        )


class ThreeWayAttentionFPNFusion(nn.Module):
    """
    Competitive spatial branch attention for three FPN sources.

    At every spatial location, the module predicts three weights:

        A_local + A_topdown + A_global = 1

    The three sources therefore compete instead of being independently
    amplified by sigmoid gates.

    Attention shape:
        [B, 3, H, W]

    Weighted fusion:
        F = A_local * L
          + A_topdown * H
          + A_global * G
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.attention = nn.Sequential(
            ConvNormAct(
                channels * 3,
                channels,
                kernel_size=1,
                padding=0,
            ),
            nn.Conv2d(
                channels,
                3,
                kernel_size=1,
            ),
        )

        self.refine = ResidualConvBlock(
            channels
        )

    def forward(
        self,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        target_size = low_feature.shape[-2:]

        high_feature = F.interpolate(
            high_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        global_feature = F.interpolate(
            global_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        joint_feature = torch.cat(
            [
                low_feature,
                high_feature,
                global_feature,
            ],
            dim=1,
        )

        attention = torch.softmax(
            self.attention(
                joint_feature
            ),
            dim=1,
        )

        local_weight = attention[:, 0:1]
        topdown_weight = attention[:, 1:2]
        global_weight = attention[:, 2:3]

        fused = (
            local_weight * low_feature
            + topdown_weight * high_feature
            + global_weight * global_feature
        )

        return self.refine(
            fused
        )
