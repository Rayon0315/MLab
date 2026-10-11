from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GroupedActivation5x5(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = math.gcd(in_channels, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                padding=2,
                groups=groups,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpsampleConv3x3(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.relu(self.conv(x))


class FeatureConcatFusion(nn.Module):
    """Fuse an original feature and its NAM-adjusted counterpart.

    The module intentionally operates only in feature space. It does not
    receive the raw NAMLab mean map.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        original: torch.Tensor,
        adjusted: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion(
            torch.cat((original, adjusted), dim=1)
        )


class FeatureECM(nn.Module):
    """Five-level PAFR ECM that consumes learned feature maps only.

    Each level is projected to a common width with a grouped 5x5 activation.
    A top-down cascade accumulates the five levels and returns a structural
    feature at stage-1 resolution. No raw NAMLab representation is appended
    at the ECM output.
    """

    def __init__(
        self,
        input_channels: tuple[int, int, int, int, int],
        edge_channels: int = 96,
    ) -> None:
        super().__init__()
        self.stage_activations = nn.ModuleList(
            [
                _GroupedActivation5x5(channels, edge_channels)
                for channels in input_channels
            ]
        )
        self.top_down_steps = nn.ModuleList(
            [_UpsampleConv3x3(edge_channels) for _ in range(4)]
        )
        self.final_structure = nn.Sequential(
            nn.Conv2d(
                edge_channels,
                edge_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        features: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        activated = [
            block(feature)
            for block, feature in zip(self.stage_activations, features)
        ]

        top_down = self.top_down_steps[0](
            activated[4],
            activated[3].shape[-2:],
        )
        top_down = activated[3] + top_down

        top_down = self.top_down_steps[1](
            top_down,
            activated[2].shape[-2:],
        )
        top_down = activated[2] + top_down

        top_down = self.top_down_steps[2](
            top_down,
            activated[1].shape[-2:],
        )
        top_down = activated[1] + top_down

        top_down = self.top_down_steps[3](
            top_down,
            activated[0].shape[-2:],
        )

        return self.final_structure(
            activated[0] + top_down
        )
