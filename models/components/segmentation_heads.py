# models/components/segmentation_heads.py

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )


class SegFormerHead(nn.Module):
    """
    Standalone SegFormer-style decode head.

    Architecture follows the standard SegFormer decoder:
        C1/C2/C3/C4
        -> per-level linear/1x1 embedding
        -> resize all to C1 resolution
        -> concatenate
        -> 1x1 fusion
        -> dropout
        -> 1x1 classifier

    Adaptations for MLab:
        - binary output channel = 1
        - plain PyTorch implementation, no mmseg/mmcv dependency
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        embed_dim: int = 256,
        dropout_ratio: float = 0.1,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError(
                "SegFormerHead expects four feature levels."
            )

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    embed_dim,
                    kernel_size=1,
                    bias=True,
                )
                for channels
                in in_channels
            ]
        )

        self.linear_fuse = (
            ConvBNReLU(
                embed_dim * 4,
                embed_dim,
                kernel_size=1,
            )
        )

        self.dropout = nn.Dropout2d(
            dropout_ratio
        )

        self.linear_pred = nn.Conv2d(
            embed_dim,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(
                "SegFormerHead expects four feature levels."
            )

        target_size = (
            features[0].shape[-2:]
        )

        projected = []

        for (
            projection,
            feature,
        ) in zip(
            self.projections,
            features,
        ):
            feature = projection(
                feature
            )

            if (
                feature.shape[-2:]
                != target_size
            ):
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            projected.append(
                feature
            )

        # Same high-to-low concatenation order used by the common
        # SegFormer implementation.
        fused = torch.cat(
            [
                projected[3],
                projected[2],
                projected[1],
                projected[0],
            ],
            dim=1,
        )

        fused = self.linear_fuse(
            fused
        )

        fused = self.dropout(
            fused
        )

        return self.linear_pred(
            fused
        )


class PSPModule(nn.Module):
    """
    Pyramid Pooling Module used by UPerNet.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        pool_scales: Sequence[int] = (
            1,
            2,
            3,
            6,
        ),
    ) -> None:
        super().__init__()

        self.pool_scales = tuple(
            pool_scales
        )

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(
                        output_size=scale
                    ),
                    ConvBNReLU(
                        in_channels,
                        channels,
                        kernel_size=1,
                    ),
                )
                for scale
                in self.pool_scales
            ]
        )

        self.bottleneck = (
            ConvBNReLU(
                in_channels
                + len(
                    self.pool_scales
                )
                * channels,
                channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        target_size = (
            x.shape[-2:]
        )

        outputs = [
            x
        ]

        for branch in self.branches:
            pooled = branch(
                x
            )

            pooled = F.interpolate(
                pooled,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

            outputs.append(
                pooled
            )

        return self.bottleneck(
            torch.cat(
                outputs,
                dim=1,
            )
        )


class UPerNetHead(nn.Module):
    """
    Standalone UPerNet decode head.

    Standard path:
        C4 -> PPM
        C1/C2/C3 -> lateral 1x1 convs
        top-down FPN fusion
        per-level 3x3 FPN conv
        resize all to C1
        concatenate
        3x3 bottleneck
        dropout
        classifier

    Adaptations for MLab:
        - binary output channel = 1
        - plain PyTorch implementation, no mmseg/mmcv dependency
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        channels: int = 256,
        pool_scales: Sequence[int] = (
            1,
            2,
            3,
            6,
        ),
        dropout_ratio: float = 0.1,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        if len(in_channels) != 4:
            raise ValueError(
                "UPerNetHead expects four feature levels."
            )

        self.lateral_convs = (
            nn.ModuleList(
                [
                    ConvBNReLU(
                        in_channels[index],
                        channels,
                        kernel_size=1,
                    )
                    for index
                    in range(3)
                ]
            )
        )

        self.psp = PSPModule(
            in_channels=in_channels[3],
            channels=channels,
            pool_scales=pool_scales,
        )

        self.fpn_convs = nn.ModuleList(
            [
                ConvBNReLU(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                )
                for _ in range(3)
            ]
        )

        self.fpn_bottleneck = (
            ConvBNReLU(
                channels * 4,
                channels,
                kernel_size=3,
                padding=1,
            )
        )

        self.dropout = nn.Dropout2d(
            dropout_ratio
        )

        self.classifier = nn.Conv2d(
            channels,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError(
                "UPerNetHead expects four feature levels."
            )

        laterals = [
            lateral_conv(
                features[index]
            )
            for index, lateral_conv
            in enumerate(
                self.lateral_convs
            )
        ]

        laterals.append(
            self.psp(
                features[3]
            )
        )

        for index in range(
            len(laterals) - 1,
            0,
            -1,
        ):
            previous_size = (
                laterals[
                    index - 1
                ].shape[-2:]
            )

            laterals[
                index - 1
            ] = (
                laterals[
                    index - 1
                ]
                + F.interpolate(
                    laterals[
                        index
                    ],
                    size=previous_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        fpn_outputs = [
            self.fpn_convs[
                index
            ](
                laterals[
                    index
                ]
            )
            for index
            in range(3)
        ]

        fpn_outputs.append(
            laterals[3]
        )

        target_size = (
            fpn_outputs[0]
            .shape[-2:]
        )

        for index in range(
            1,
            len(
                fpn_outputs
            ),
        ):
            fpn_outputs[index] = (
                F.interpolate(
                    fpn_outputs[
                        index
                    ],
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        output = torch.cat(
            fpn_outputs,
            dim=1,
        )

        output = (
            self.fpn_bottleneck(
                output
            )
        )

        output = self.dropout(
            output
        )

        return self.classifier(
            output
        )


__all__ = [
    "SegFormerHead",
    "UPerNetHead",
]
