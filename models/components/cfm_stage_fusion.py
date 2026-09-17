# models/components/cfm_stage_fusion.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


BRANCH_ORDER = (
    "tcrm",
    "ssrm",
    "ldfm",
    "egpcm",
)


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
    ) -> None:
        padding = kernel_size // 2

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
            nn.ReLU6(
                inplace=True
            ),
        )


class FourStageProjectConcatHead(
    nn.Module
):
    """
    Minimal four-stage direct decoder.

    S1 / S2 / S3 / S4:
        project each fused CFMNet stage feature to the same width,
        resize all stages to Stage-1 resolution,
        concatenate them once,
        apply one lightweight fusion block,
        predict saliency.

    No branch-level interpretation, attention, top-down recurrence,
    or extra refinement is introduced.
    """

    def __init__(
        self,
        encoder_channels: tuple[
            int,
            int,
            int,
            int,
        ],
        decode_channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.projectors = (
            nn.ModuleList(
                [
                    ConvBNReLU(
                        in_channels,
                        decode_channels,
                        kernel_size=1,
                    )
                    for in_channels
                    in encoder_channels
                ]
            )
        )

        self.fuse = (
            ConvBNReLU(
                decode_channels * 4,
                decode_channels,
                kernel_size=3,
            )
        )

        self.head = nn.Sequential(
            nn.Dropout2d(
                p=dropout,
                inplace=False,
            ),
            nn.Conv2d(
                decode_channels,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        features: list[
            torch.Tensor
        ]
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        output_size: tuple[
            int,
            int,
        ],
    ) -> torch.Tensor:
        target_size = (
            features[0]
            .shape[-2:]
        )

        projected = []

        for (
            feature,
            projector,
        ) in zip(
            features,
            self.projectors,
        ):
            feature = (
                projector(
                    feature
                )
            )

            if (
                feature.shape[-2:]
                != target_size
            ):
                feature = (
                    F.interpolate(
                        feature,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                )

            projected.append(
                feature
            )

        fused = torch.cat(
            projected,
            dim=1,
        )

        prediction = self.head(
            self.fuse(
                fused
            )
        )

        return F.interpolate(
            prediction,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


def concat_stage_branches(
    branches: dict[
        str,
        torch.Tensor,
    ],
) -> torch.Tensor:
    """
    Fuse the four raw outputs from one CFMBlock only by concatenation.

    Each branch owns one quarter of the stage channels, therefore:
        C/4 + C/4 + C/4 + C/4 = C

    The returned tensor can directly replace the normal CFMNet stage
    feature for a decoder expecting the canonical stage channel width.
    """

    return torch.cat(
        [
            branches[name]
            for name
            in BRANCH_ORDER
        ],
        dim=1,
    )
