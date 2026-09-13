from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
)


FeaturePyramid = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class PixelLayerNorm(nn.Module):
    """LayerNorm over channels independently at each spatial position."""

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(channels)
        )
        self.bias = nn.Parameter(
            torch.zeros(channels)
        )
        self.eps = eps

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        feature = feature.permute(
            0,
            2,
            3,
            1,
        )

        feature = F.layer_norm(
            feature,
            (feature.shape[-1],),
            self.weight,
            self.bias,
            self.eps,
        )

        return feature.permute(
            0,
            3,
            1,
            2,
        ).contiguous()


class Hier60RegionInput(nn.Module):
    """
    Build the validated Hier60 region inputs.

        Stage1 <- RGB - M60
        Stage2 <- M60
        Stage3 <- M60
        Stage4 <- M60

    `image` is already ImageNet-normalized by SODDataset.
    `mean_60` is loaded in [0, 1], so it is normalized into
    the same space before the residual is computed.
    """

    def __init__(self) -> None:
        super().__init__()

        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(
                1,
                3,
                1,
                1,
            ),
            persistent=False,
        )

        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(
                1,
                3,
                1,
                1,
            ),
            persistent=False,
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        mean_60 = (
            mean_60
            - self.image_mean
        ) / self.image_std

        detail_region = (
            image
            - mean_60
        )

        return (
            detail_region,
            mean_60,
        )


class RegionScaleEncoder(nn.Module):
    """
    Original Stage1 detail encoder from the Direct Region model.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            ConvNormAct(
                3,
                hidden_channels,
                kernel_size=3,
            ),
            ResidualConvBlock(
                hidden_channels
            ),
            ConvNormAct(
                hidden_channels,
                out_channels,
                kernel_size=1,
                padding=0,
            ),
            ResidualConvBlock(
                out_channels
            ),
        )

    def forward(
        self,
        region: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        region = F.interpolate(
            region,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return self.encoder(
            region
        )


class HybridRegionScaleEncoder(nn.Module):
    """
    Region-preserving main branch + lightweight local context branch.

    Main:
        nearest resize
        -> 1x1 projection
        -> channel-only residual MLP

    Context:
        depthwise 3x3
        -> pixel LayerNorm
        -> GELU
        -> 1x1 projection

    Output:
        preserve + alpha * context
    """

    def __init__(
        self,
        out_channels: int,
        bottleneck_ratio: float = 0.5,
        initial_context_scale: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_channels = max(
            32,
            int(
                out_channels
                * bottleneck_ratio
            ),
        )

        self.input_projection = nn.Conv2d(
            3,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.preserve_norm = PixelLayerNorm(
            out_channels
        )

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(
                out_channels,
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        # Preserve the initialization used by the validated hybrid model.
        nn.init.zeros_(
            self.channel_mlp[-1].weight
        )
        nn.init.zeros_(
            self.channel_mlp[-1].bias
        )

        self.context_depthwise = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False,
        )

        self.context_norm = PixelLayerNorm(
            out_channels
        )

        self.context_activation = nn.GELU()

        self.context_projection = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.context_scale = nn.Parameter(
            torch.tensor(
                float(
                    initial_context_scale
                )
            )
        )

    def forward(
        self,
        region: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
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

        return (
            preserve
            + self.context_scale
            * context
        )


class HybridHier60RegionEncoder(nn.Module):
    """
    Encode the Hier60 region prior in the native backbone channels.

        Stage1: RGB - M60 -> original detail encoder
        Stage2: M60       -> hybrid region encoder
        Stage3: M60       -> hybrid region encoder
        Stage4: M60       -> hybrid region encoder
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
    ) -> None:
        super().__init__()

        (
            stage1_channels,
            stage2_channels,
            stage3_channels,
            stage4_channels,
        ) = stage_channels

        self.stage1_encoder = (
            RegionScaleEncoder(
                out_channels=stage1_channels,
            )
        )

        self.stage2_encoder = (
            HybridRegionScaleEncoder(
                out_channels=stage2_channels,
                initial_context_scale=initial_context_scale,
            )
        )

        self.stage3_encoder = (
            HybridRegionScaleEncoder(
                out_channels=stage3_channels,
                initial_context_scale=initial_context_scale,
            )
        )

        self.stage4_encoder = (
            HybridRegionScaleEncoder(
                out_channels=stage4_channels,
                initial_context_scale=initial_context_scale,
            )
        )

    def forward(
        self,
        detail_region: torch.Tensor,
        mean_60: torch.Tensor,
        feature_sizes: tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
        ],
    ) -> FeaturePyramid:
        (
            stage1_size,
            stage2_size,
            stage3_size,
            stage4_size,
        ) = feature_sizes

        region1 = self.stage1_encoder(
            detail_region,
            target_size=stage1_size,
        )

        region2 = self.stage2_encoder(
            mean_60,
            target_size=stage2_size,
        )

        region3 = self.stage3_encoder(
            mean_60,
            target_size=stage3_size,
        )

        region4 = self.stage4_encoder(
            mean_60,
            target_size=stage4_size,
        )

        return (
            region1,
            region2,
            region3,
            region4,
        )


class BidirectionalRegionVisualInteraction(nn.Module):
    """
    Bidirectional cross-attention between visual and region features.

        Visual <- Region
        Region <- Visual

    Both directions are fused back into the visual representation.
    """

    def __init__(
        self,
        channels: int,
        inner_channels: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        if (
            inner_channels
            % num_heads
            != 0
        ):
            raise ValueError(
                "inner_channels must be divisible by num_heads."
            )

        self.inner_channels = (
            inner_channels
        )
        self.num_heads = num_heads
        self.head_channels = (
            inner_channels
            // num_heads
        )
        self.scale = (
            self.head_channels
            ** -0.5
        )

        self.visual_norm = nn.GroupNorm(
            8,
            channels,
        )
        self.region_norm = nn.GroupNorm(
            8,
            channels,
        )

        self.visual_query = nn.Conv2d(
            channels,
            inner_channels,
            kernel_size=1,
            bias=False,
        )
        self.visual_key = nn.Conv2d(
            channels,
            inner_channels,
            kernel_size=1,
            bias=False,
        )
        self.visual_value = nn.Conv2d(
            channels,
            inner_channels,
            kernel_size=1,
            bias=False,
        )

        self.region_query = nn.Conv2d(
            channels,
            inner_channels,
            kernel_size=1,
            bias=False,
        )
        self.region_key = nn.Conv2d(
            channels,
            inner_channels,
            kernel_size=1,
            bias=False,
        )
        self.region_value = nn.Conv2d(
            channels,
            inner_channels,
            kernel_size=1,
            bias=False,
        )

        self.visual_from_region_projection = (
            ConvNormAct(
                inner_channels,
                channels,
                kernel_size=1,
                padding=0,
            )
        )

        self.region_from_visual_projection = (
            ConvNormAct(
                inner_channels,
                channels,
                kernel_size=1,
                padding=0,
            )
        )

        self.fusion = nn.Sequential(
            ConvNormAct(
                channels * 4,
                channels,
                kernel_size=1,
                padding=0,
            ),
            ResidualConvBlock(
                channels
            ),
        )

    def _to_tokens(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        (
            batch_size,
            _,
            height,
            width,
        ) = feature.shape

        feature = feature.reshape(
            batch_size,
            self.num_heads,
            self.head_channels,
            height * width,
        )

        return feature.transpose(
            2,
            3,
        )

    def _to_feature_map(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size = (
            tokens.shape[0]
        )

        tokens = tokens.transpose(
            2,
            3,
        ).contiguous()

        return tokens.reshape(
            batch_size,
            self.inner_channels,
            height,
            width,
        )

    def forward(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
    ) -> torch.Tensor:
        (
            height,
            width,
        ) = (
            visual_feature.shape[-2:]
        )

        visual_normalized = (
            self.visual_norm(
                visual_feature
            )
        )

        region_normalized = (
            self.region_norm(
                region_feature
            )
        )

        visual_query = self._to_tokens(
            self.visual_query(
                visual_normalized
            )
        )
        visual_key = self._to_tokens(
            self.visual_key(
                visual_normalized
            )
        )
        visual_value = self._to_tokens(
            self.visual_value(
                visual_normalized
            )
        )

        region_query = self._to_tokens(
            self.region_query(
                region_normalized
            )
        )
        region_key = self._to_tokens(
            self.region_key(
                region_normalized
            )
        )
        region_value = self._to_tokens(
            self.region_value(
                region_normalized
            )
        )

        visual_region_attention = (
            torch.matmul(
                visual_query,
                region_key.transpose(
                    -2,
                    -1,
                ),
            )
            * self.scale
        )

        visual_region_attention = (
            torch.softmax(
                visual_region_attention,
                dim=-1,
            )
        )

        visual_from_region = torch.matmul(
            visual_region_attention,
            region_value,
        )

        region_visual_attention = (
            torch.matmul(
                region_query,
                visual_key.transpose(
                    -2,
                    -1,
                ),
            )
            * self.scale
        )

        region_visual_attention = (
            torch.softmax(
                region_visual_attention,
                dim=-1,
            )
        )

        region_from_visual = torch.matmul(
            region_visual_attention,
            visual_value,
        )

        visual_from_region = (
            self._to_feature_map(
                visual_from_region,
                height=height,
                width=width,
            )
        )

        region_from_visual = (
            self._to_feature_map(
                region_from_visual,
                height=height,
                width=width,
            )
        )

        visual_from_region = (
            self.visual_from_region_projection(
                visual_from_region
            )
        )

        region_from_visual = (
            self.region_from_visual_projection(
                region_from_visual
            )
        )

        fused = torch.cat(
            [
                visual_feature,
                region_feature,
                visual_from_region,
                region_from_visual,
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


class DepthwiseDilatedBranch(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                channels,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(
            feature
        )


class RegionConditionedLocalReconstruction(nn.Module):
    """
    Stage2 reconstruction.

    The region feature controls:
        - spatial routing between dilation 1/2/3 branches
        - channel-wise affine modulation
        - spatial-channel region gating
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.visual_refine = (
            ResidualConvBlock(
                channels
            )
        )

        self.region_refine = (
            ResidualConvBlock(
                channels
            )
        )

        self.branches = nn.ModuleList(
            [
                DepthwiseDilatedBranch(
                    channels,
                    dilation=1,
                ),
                DepthwiseDilatedBranch(
                    channels,
                    dilation=2,
                ),
                DepthwiseDilatedBranch(
                    channels,
                    dilation=3,
                ),
            ]
        )

        self.routing = nn.Sequential(
            ConvNormAct(
                channels,
                channels // 2,
                kernel_size=1,
                padding=0,
            ),
            nn.Conv2d(
                channels // 2,
                3,
                kernel_size=1,
            ),
        )

        self.channel_condition = nn.Sequential(
            nn.AdaptiveAvgPool2d(
                output_size=1
            ),
            nn.Conv2d(
                channels,
                channels * 2,
                kernel_size=1,
            ),
        )

        self.mixed_projection = (
            ConvNormAct(
                channels,
                channels,
                kernel_size=1,
                padding=0,
            )
        )

        self.region_gate = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

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
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.visual_refine(
            visual_feature
        )

        region = self.region_refine(
            region_feature
        )

        routing_weights = (
            torch.softmax(
                self.routing(
                    region
                ),
                dim=1,
            )
        )

        branch_outputs = [
            branch(
                visual
            )
            for branch
            in self.branches
        ]

        mixed_feature = (
            torch.zeros_like(
                visual
            )
        )

        for (
            branch_index,
            branch_feature,
        ) in enumerate(
            branch_outputs
        ):
            mixed_feature = (
                mixed_feature
                + branch_feature
                * routing_weights[
                    :,
                    branch_index:branch_index + 1,
                ]
            )

        mixed_feature = (
            self.mixed_projection(
                mixed_feature
            )
        )

        channel_parameters = (
            self.channel_condition(
                region
            )
        )

        (
            gamma,
            beta,
        ) = torch.chunk(
            channel_parameters,
            chunks=2,
            dim=1,
        )

        gamma = (
            0.5
            * torch.tanh(
                gamma
            )
        )

        beta = (
            0.5
            * torch.tanh(
                beta
            )
        )

        conditioned_feature = (
            mixed_feature
            * (
                1.0
                + gamma
            )
            + beta
        )

        region_weight = (
            torch.sigmoid(
                self.region_gate(
                    region
                )
            )
        )

        conditioned_feature = (
            conditioned_feature
            * (
                1.0
                + region_weight
            )
        )

        fused = torch.cat(
            [
                visual_feature,
                region,
                conditioned_feature,
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


class LocalDetailReconstruction(nn.Module):
    """
    Stage1 high-resolution local reconstruction.

    RGB-M60 provides the residual appearance inside the finest
    Hier60 region partition.
    """

    def __init__(
        self,
        channels: int,
        inner_channels: int = 48,
    ) -> None:
        super().__init__()

        self.visual_projection = (
            ConvNormAct(
                channels,
                inner_channels,
                kernel_size=1,
                padding=0,
            )
        )

        self.detail_projection = (
            ConvNormAct(
                channels,
                inner_channels,
                kernel_size=1,
                padding=0,
            )
        )

        self.visual_local = nn.Sequential(
            nn.Conv2d(
                inner_channels,
                inner_channels,
                kernel_size=3,
                padding=1,
                groups=inner_channels,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                inner_channels,
            ),
            nn.GELU(),
        )

        self.detail_local = nn.Sequential(
            nn.Conv2d(
                inner_channels,
                inner_channels,
                kernel_size=3,
                padding=1,
                groups=inner_channels,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                inner_channels,
            ),
            nn.GELU(),
        )

        self.visual_gate = nn.Conv2d(
            inner_channels,
            inner_channels,
            kernel_size=1,
        )

        self.detail_gate = nn.Conv2d(
            inner_channels,
            inner_channels,
            kernel_size=1,
        )

        self.interaction_fusion = (
            nn.Sequential(
                ConvNormAct(
                    inner_channels * 3,
                    inner_channels,
                    kernel_size=1,
                    padding=0,
                ),
                ResidualConvBlock(
                    inner_channels
                ),
            )
        )

        self.output_projection = (
            nn.Sequential(
                ConvNormAct(
                    inner_channels,
                    channels,
                    kernel_size=1,
                    padding=0,
                ),
                ResidualConvBlock(
                    channels
                ),
            )
        )

    def forward(
        self,
        visual_feature: torch.Tensor,
        detail_feature: torch.Tensor,
    ) -> torch.Tensor:
        visual_reduced = (
            self.visual_projection(
                visual_feature
            )
        )

        detail_reduced = (
            self.detail_projection(
                detail_feature
            )
        )

        visual_local = (
            self.visual_local(
                visual_reduced
            )
        )

        detail_local = (
            self.detail_local(
                detail_reduced
            )
        )

        visual_weight = (
            2.0
            * torch.sigmoid(
                self.visual_gate(
                    detail_local
                )
            )
        )

        detail_weight = (
            torch.sigmoid(
                self.detail_gate(
                    visual_local
                )
            )
        )

        interaction = (
            visual_local
            * visual_weight
            + detail_local
            * detail_weight
        )

        interaction = (
            self.interaction_fusion(
                torch.cat(
                    [
                        visual_reduced,
                        detail_reduced,
                        interaction,
                    ],
                    dim=1,
                )
            )
        )

        reconstruction = (
            self.output_projection(
                interaction
            )
        )

        return (
            visual_feature
            + reconstruction
        )


class NAMLabHybrid(nn.Module):
    """
    Reusable NAMLab Hybrid feature component.

    Inputs:
        - VMamba/MambaVision four-stage visual feature pyramid
        - normalized RGB image
        - Hier60 region-mean RGB map

    Output:
        - four enhanced visual features with exactly the same
          shapes/channels as the input pyramid

    The implementation preserves the validated Region Hybrid design:
        Stage1:
            RGB - M60 detail
            -> original detail encoder
            -> local detail reconstruction

        Stage2:
            M60
            -> hybrid region encoder
            -> region-conditioned local reconstruction

        Stage3/4:
            M60
            -> hybrid region encoder
            -> bidirectional region-visual interaction
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
    ) -> None:
        super().__init__()

        (
            stage1_channels,
            stage2_channels,
            stage3_channels,
            stage4_channels,
        ) = stage_channels

        self.region_input = (
            Hier60RegionInput()
        )

        self.region_encoder = (
            HybridHier60RegionEncoder(
                stage_channels=stage_channels,
                initial_context_scale=initial_context_scale,
            )
        )

        self.stage4_region_interaction = (
            BidirectionalRegionVisualInteraction(
                channels=stage4_channels,
                inner_channels=128,
                num_heads=4,
            )
        )

        self.stage3_region_interaction = (
            BidirectionalRegionVisualInteraction(
                channels=stage3_channels,
                inner_channels=96,
                num_heads=4,
            )
        )

        self.stage2_region_reconstruction = (
            RegionConditionedLocalReconstruction(
                channels=stage2_channels,
            )
        )

        self.stage1_detail_reconstruction = (
            LocalDetailReconstruction(
                channels=stage1_channels,
                inner_channels=48,
            )
        )

    def forward(
        self,
        features: FeaturePyramid,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> FeaturePyramid:
        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = features

        (
            detail_region,
            mean_60_normalized,
        ) = self.region_input(
            image=image,
            mean_60=mean_60,
        )

        region_features = (
            self.region_encoder(
                detail_region=detail_region,
                mean_60=mean_60_normalized,
                feature_sizes=(
                    stage1.shape[-2:],
                    stage2.shape[-2:],
                    stage3.shape[-2:],
                    stage4.shape[-2:],
                ),
            )
        )

        (
            region1,
            region2,
            region3,
            region4,
        ) = region_features

        stage4 = (
            self.stage4_region_interaction(
                visual_feature=stage4,
                region_feature=region4,
            )
        )

        stage3 = (
            self.stage3_region_interaction(
                visual_feature=stage3,
                region_feature=region3,
            )
        )

        stage2 = (
            self.stage2_region_reconstruction(
                visual_feature=stage2,
                region_feature=region2,
            )
        )

        stage1 = (
            self.stage1_detail_reconstruction(
                visual_feature=stage1,
                detail_feature=region1,
            )
        )

        return (
            stage1,
            stage2,
            stage3,
            stage4,
        )
