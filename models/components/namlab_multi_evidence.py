from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.namlab_hybrid import (
    HybridHier60RegionEncoder,
    HybridRegionScaleEncoder,
    NAMLabHybrid,
    PixelLayerNorm,
)


class DepthwiseCenterDifference(nn.Module):
    """
    Pure depthwise center-difference operator.

    For each channel c:

        y_c(p) = sum_delta w_c(delta)
                 * [x_c(p + delta) - x_c(p)]

    The center weight cancels automatically. Replicate padding is used so a
    spatially constant feature remains zero even at the image boundary.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.channels = int(channels)

        self.weight = nn.Parameter(
            torch.empty(
                channels,
                1,
                3,
                3,
            )
        )

        nn.init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        padded = F.pad(
            feature,
            pad=(1, 1, 1, 1),
            mode="replicate",
        )

        neighbor_response = F.conv2d(
            padded,
            self.weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.channels,
        )

        weight_sum = self.weight.sum(
            dim=(2, 3),
            keepdim=True,
        ).view(
            1,
            self.channels,
            1,
            1,
        )

        return (
            neighbor_response
            - feature * weight_sum
        )


class DepthwiseOppositeAngularDifference(nn.Module):
    """
    Depthwise opposite-pair angular difference (OAD).

    Four learnable directional relations are used per channel:

        N  - S
        E  - W
        NE - SW
        NW - SE

    This is intentionally named OAD rather than ADC so it is not confused
    with other published "angular difference convolution" definitions.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                channels,
                4,
            )
        )

        nn.init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        padded = F.pad(
            feature,
            pad=(1, 1, 1, 1),
            mode="replicate",
        )

        north = padded[
            :,
            :,
            0:-2,
            1:-1,
        ]
        south = padded[
            :,
            :,
            2:,
            1:-1,
        ]
        east = padded[
            :,
            :,
            1:-1,
            2:,
        ]
        west = padded[
            :,
            :,
            1:-1,
            0:-2,
        ]

        north_east = padded[
            :,
            :,
            0:-2,
            2:,
        ]
        south_west = padded[
            :,
            :,
            2:,
            0:-2,
        ]
        north_west = padded[
            :,
            :,
            0:-2,
            0:-2,
        ]
        south_east = padded[
            :,
            :,
            2:,
            2:,
        ]

        weight = self.weight.view(
            1,
            self.weight.shape[0],
            4,
            1,
            1,
        )

        return (
            weight[:, :, 0]
            * (north - south)
            + weight[:, :, 1]
            * (east - west)
            + weight[:, :, 2]
            * (north_east - south_west)
            + weight[:, :, 3]
            * (north_west - south_east)
        )


class MultiEvidenceRegionScaleEncoder(
    HybridRegionScaleEncoder
):
    """
    Validated NAMLab Hybrid encoder plus optional relation-defined evidence.

    The original branch is kept exactly:

        nearest resize
        -> 1x1 projection
        -> channel-only residual MLP
        -> preserve

        preserve
        -> DWConv 3x3
        -> PixelLayerNorm
        -> GELU
        -> 1x1 projection
        -> context

        output = preserve + context_scale * context

    Optional branches read the SAME `preserve` feature:

        CDC(preserve)
        OAD(preserve)

    Each branch has its own independent learnable residual scale.
    No softmax competition is used.
    """

    def __init__(
        self,
        out_channels: int,
        bottleneck_ratio: float = 0.5,
        initial_context_scale: float = 0.1,
        use_cdc: bool = False,
        use_oad: bool = False,
        initial_cdc_scale: float = 0.05,
        initial_oad_scale: float = 0.05,
    ) -> None:
        super().__init__(
            out_channels=out_channels,
            bottleneck_ratio=bottleneck_ratio,
            initial_context_scale=initial_context_scale,
        )

        self.use_cdc = bool(use_cdc)
        self.use_oad = bool(use_oad)

        if self.use_cdc:
            self.cdc_depthwise = (
                DepthwiseCenterDifference(
                    out_channels
                )
            )
            self.cdc_norm = PixelLayerNorm(
                out_channels
            )
            self.cdc_activation = nn.GELU()
            self.cdc_projection = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            )
            self.cdc_scale = nn.Parameter(
                torch.tensor(
                    float(initial_cdc_scale)
                )
            )

        if self.use_oad:
            self.oad_depthwise = (
                DepthwiseOppositeAngularDifference(
                    out_channels
                )
            )
            self.oad_norm = PixelLayerNorm(
                out_channels
            )
            self.oad_activation = nn.GELU()
            self.oad_projection = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            )
            self.oad_scale = nn.Parameter(
                torch.tensor(
                    float(initial_oad_scale)
                )
            )

    def forward(
        self,
        region: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        # Keep the validated Hybrid path exactly.
        region = F.interpolate(
            region,
            size=target_size,
            mode="nearest",
        )

        base = self.input_projection(
            region
        )

        preserve = (
            base
            + self.channel_mlp(
                self.preserve_norm(
                    base
                )
            )
        )

        context = self.context_depthwise(
            preserve
        )
        context = self.context_norm(
            context
        )
        context = self.context_activation(
            context
        )
        context = self.context_projection(
            context
        )

        output = (
            preserve
            + self.context_scale
            * context
        )

        if self.use_cdc:
            cdc = self.cdc_depthwise(
                preserve
            )
            cdc = self.cdc_norm(
                cdc
            )
            cdc = self.cdc_activation(
                cdc
            )
            cdc = self.cdc_projection(
                cdc
            )

            output = (
                output
                + self.cdc_scale
                * cdc
            )

        if self.use_oad:
            oad = self.oad_depthwise(
                preserve
            )
            oad = self.oad_norm(
                oad
            )
            oad = self.oad_activation(
                oad
            )
            oad = self.oad_projection(
                oad
            )

            output = (
                output
                + self.oad_scale
                * oad
            )

        return output


class MultiEvidenceHier60RegionEncoder(
    HybridHier60RegionEncoder
):
    """
    Keep Stage1 unchanged; replace only Stage2/3/4 region encoders.

        Stage1: RGB - M60 -> original RegionScaleEncoder
        Stage2: M60 -> MultiEvidenceRegionScaleEncoder
        Stage3: M60 -> MultiEvidenceRegionScaleEncoder
        Stage4: M60 -> MultiEvidenceRegionScaleEncoder
    """

    def __init__(
        self,
        stage_channels: tuple[
            int,
            int,
            int,
            int,
        ],
        initial_context_scale: float = 0.1,
        use_cdc: bool = False,
        use_oad: bool = False,
        initial_cdc_scale: float = 0.05,
        initial_oad_scale: float = 0.05,
    ) -> None:
        super().__init__(
            stage_channels=stage_channels,
            initial_context_scale=initial_context_scale,
        )

        (
            _,
            stage2_channels,
            stage3_channels,
            stage4_channels,
        ) = stage_channels

        def build_encoder(
            channels: int,
        ) -> MultiEvidenceRegionScaleEncoder:
            return MultiEvidenceRegionScaleEncoder(
                out_channels=channels,
                initial_context_scale=initial_context_scale,
                use_cdc=use_cdc,
                use_oad=use_oad,
                initial_cdc_scale=initial_cdc_scale,
                initial_oad_scale=initial_oad_scale,
            )

        self.stage2_encoder = build_encoder(
            stage2_channels
        )
        self.stage3_encoder = build_encoder(
            stage3_channels
        )
        self.stage4_encoder = build_encoder(
            stage4_channels
        )


class NAMLabMultiEvidenceHybrid(
    NAMLabHybrid
):
    """
    Drop-in replacement for NAMLabHybrid.

    Stage1 reconstruction, Stage2 local reconstruction and Stage3/4
    bidirectional interactions are inherited unchanged. The only controlled
    change is how M60 is encoded at Stage2/3/4.
    """

    def __init__(
        self,
        stage_channels: tuple[
            int,
            int,
            int,
            int,
        ],
        initial_context_scale: float = 0.1,
        use_cdc: bool = False,
        use_oad: bool = False,
        initial_cdc_scale: float = 0.05,
        initial_oad_scale: float = 0.05,
    ) -> None:
        super().__init__(
            stage_channels=stage_channels,
            initial_context_scale=initial_context_scale,
        )

        self.region_encoder = (
            MultiEvidenceHier60RegionEncoder(
                stage_channels=stage_channels,
                initial_context_scale=initial_context_scale,
                use_cdc=use_cdc,
                use_oad=use_oad,
                initial_cdc_scale=initial_cdc_scale,
                initial_oad_scale=initial_oad_scale,
            )
        )
