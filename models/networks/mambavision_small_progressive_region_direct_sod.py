# models/networks/mambavision_small_progressive_region_direct_sod.py
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.mambavision import (
    MambaVisionBackbone,
    mamba_vision_small,
)
from models.components.sod_blocks import (
    BoundaryRefinementBlock,
    ConvNormAct,
    PredictionHead,
    PyramidContextBlock,
    ResidualConvBlock,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "mambavision"
    / "mambavision_small_1k.pth.tar"
)


class DirectRegionHierarchy(nn.Module):
    """
    Direct hierarchical region representation.

        Stage1 <- RGB - M60
        Stage2 <- M60
        Stage3 <- M40
        Stage4 <- M20

    RGB is already ImageNet-normalized by the dataset.
    Region-mean maps are loaded in [0, 1], so they are normalized
    into the same space before use.
    """

    def __init__(self) -> None:
        super().__init__()

        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

    def normalize_mean_map(
        self,
        mean_map: torch.Tensor,
    ) -> torch.Tensor:
        return (
            mean_map
            - self.image_mean
        ) / self.image_std

    def forward(
        self,
        image: torch.Tensor,
        mean_20: torch.Tensor,
        mean_40: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        mean20 = self.normalize_mean_map(
            mean_20
        )

        mean40 = self.normalize_mean_map(
            mean_40
        )

        mean60 = self.normalize_mean_map(
            mean_60
        )

        detail_region = (
            image
            - mean60
        )

        fine_region = mean60
        middle_region = mean40
        coarse_region = mean20

        return (
            detail_region,
            fine_region,
            middle_region,
            coarse_region,
        )


class RegionScaleEncoder(nn.Module):
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


class RegionPyramidEncoder(nn.Module):
    """
    Encode the direct region hierarchy into native backbone channels.

        R1 = RGB - M60 -> Stage1
        R2 = M60       -> Stage2
        R3 = M40       -> Stage3
        R4 = M20       -> Stage4
    """

    def __init__(
        self,
        stage1_channels: int,
        stage2_channels: int,
        stage3_channels: int,
        stage4_channels: int,
    ) -> None:
        super().__init__()

        self.stage1_encoder = RegionScaleEncoder(
            out_channels=stage1_channels,
        )

        self.stage2_encoder = RegionScaleEncoder(
            out_channels=stage2_channels,
        )

        self.stage3_encoder = RegionScaleEncoder(
            out_channels=stage3_channels,
        )

        self.stage4_encoder = RegionScaleEncoder(
            out_channels=stage4_channels,
        )

    def forward(
        self,
        detail_region: torch.Tensor,
        fine_region: torch.Tensor,
        middle_region: torch.Tensor,
        coarse_region: torch.Tensor,
        stage1_size: tuple[int, int],
        stage2_size: tuple[int, int],
        stage3_size: tuple[int, int],
        stage4_size: tuple[int, int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        region1 = self.stage1_encoder(
            detail_region,
            target_size=stage1_size,
        )

        region2 = self.stage2_encoder(
            fine_region,
            target_size=stage2_size,
        )

        region3 = self.stage3_encoder(
            middle_region,
            target_size=stage3_size,
        )

        region4 = self.stage4_encoder(
            coarse_region,
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

        self.inner_channels = inner_channels
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
        height, width = (
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

        visual_region_attention = torch.softmax(
            visual_region_attention,
            dim=-1,
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

        region_visual_attention = torch.softmax(
            region_visual_attention,
            dim=-1,
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

        self.visual_refine = ResidualConvBlock(
            channels
        )

        self.region_refine = ResidualConvBlock(
            channels
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

        self.mixed_projection = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
            padding=0,
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

        routing_weights = torch.softmax(
            self.routing(
                region
            ),
            dim=1,
        )

        branch_outputs = [
            branch(
                visual
            )
            for branch
            in self.branches
        ]

        mixed_feature = torch.zeros_like(
            visual
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

        gamma, beta = torch.chunk(
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

        region_weight = torch.sigmoid(
            self.region_gate(
                region
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
    region partition.
    """

    def __init__(
        self,
        channels: int,
        inner_channels: int = 48,
    ) -> None:
        super().__init__()

        self.visual_projection = ConvNormAct(
            channels,
            inner_channels,
            kernel_size=1,
            padding=0,
        )

        self.detail_projection = ConvNormAct(
            channels,
            inner_channels,
            kernel_size=1,
            padding=0,
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

        self.interaction_fusion = nn.Sequential(
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

        self.output_projection = nn.Sequential(
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

        visual_local = self.visual_local(
            visual_reduced
        )

        detail_local = self.detail_local(
            detail_reduced
        )

        visual_weight = (
            2.0
            * torch.sigmoid(
                self.visual_gate(
                    detail_local
                )
            )
        )

        detail_weight = torch.sigmoid(
            self.detail_gate(
                visual_local
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


class GlobalSemanticBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        feature: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        context = F.adaptive_avg_pool2d(
            feature,
            output_size=1,
        )

        context = self.projection(
            context
        )

        context = F.interpolate(
            context,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return context


class ProgressiveSelectiveFusion(nn.Module):
    """
    Three-stream selective fusion:

        low feature
        top-down feature
        persistent Stage4 global feature
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.low_gate = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1,
        )

        self.high_gate = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1,
        )

        self.global_gate = nn.Conv2d(
            channels * 2,
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

        self._init_gates()

    def _init_gates(
        self,
    ) -> None:
        for gate in (
            self.low_gate,
            self.high_gate,
            self.global_gate,
        ):
            nn.init.zeros_(
                gate.weight
            )

            if gate.bias is not None:
                nn.init.zeros_(
                    gate.bias
                )

    def forward(
        self,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
        global_feature: torch.Tensor,
    ) -> torch.Tensor:
        target_size = (
            low_feature.shape[-2:]
        )

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

        low_weight = (
            2.0
            * torch.sigmoid(
                self.low_gate(
                    torch.cat(
                        [
                            high_feature,
                            global_feature,
                        ],
                        dim=1,
                    )
                )
            )
        )

        high_weight = (
            2.0
            * torch.sigmoid(
                self.high_gate(
                    torch.cat(
                        [
                            low_feature,
                            global_feature,
                        ],
                        dim=1,
                    )
                )
            )
        )

        global_weight = (
            2.0
            * torch.sigmoid(
                self.global_gate(
                    torch.cat(
                        [
                            low_feature,
                            high_feature,
                        ],
                        dim=1,
                    )
                )
            )
        )

        low_selected = (
            low_feature
            * low_weight
        )

        high_selected = (
            high_feature
            * high_weight
        )

        global_selected = (
            global_feature
            * global_weight
        )

        fused = torch.cat(
            [
                low_selected,
                high_selected,
                global_selected,
            ],
            dim=1,
        )

        return self.fusion(
            fused
        )


class MambaVisionSmallProgressiveRegionDirectSOD(
    nn.Module
):
    """
    MambaVision-S progressive decoder with direct region hierarchy.

    Region assignment:

        Stage1 <- RGB - M60
        Stage2 <- M60
        Stage3 <- M40
        Stage4 <- M20

    This is a controlled ablation of the difference-based hierarchy:

        Stage2 <- M60 - M40
        Stage3 <- M40 - M20

    All reconstruction modules and decoder structures remain
    unchanged.
    """

    input_keys = (
        "image",
        "mean_20",
        "mean_40",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
    ) -> None:
        super().__init__()

        self.backbone: MambaVisionBackbone = (
            mamba_vision_small(
                pretrained_path=pretrained_path,
            )
        )

        stage1_channels = (
            self.backbone.out_channels[0]
        )

        stage2_channels = (
            self.backbone.out_channels[1]
        )

        stage3_channels = (
            self.backbone.out_channels[2]
        )

        stage4_channels = (
            self.backbone.out_channels[3]
        )

        # -------------------------------------------------
        # Direct region hierarchy
        # -------------------------------------------------

        self.region_hierarchy = (
            DirectRegionHierarchy()
        )

        self.region_encoder = (
            RegionPyramidEncoder(
                stage1_channels=stage1_channels,
                stage2_channels=stage2_channels,
                stage3_channels=stage3_channels,
                stage4_channels=stage4_channels,
            )
        )

        # -------------------------------------------------
        # Region-conditioned backbone reconstruction
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Deep decoder entry
        # -------------------------------------------------

        self.deep_projection = ConvNormAct(
            stage4_channels,
            stage3_channels,
            kernel_size=1,
            padding=0,
        )

        self.context4 = PyramidContextBlock(
            stage3_channels
        )

        self.pred4 = PredictionHead(
            stage3_channels
        )

        # -------------------------------------------------
        # Persistent Stage4 global semantic stream
        # -------------------------------------------------

        self.global3 = GlobalSemanticBranch(
            in_channels=stage4_channels,
            out_channels=stage3_channels,
        )

        self.global2 = GlobalSemanticBranch(
            in_channels=stage4_channels,
            out_channels=stage2_channels,
        )

        self.global1 = GlobalSemanticBranch(
            in_channels=stage4_channels,
            out_channels=128,
        )

        # -------------------------------------------------
        # Stage3 decoder
        # -------------------------------------------------

        self.fusion3 = ProgressiveSelectiveFusion(
            channels=stage3_channels,
        )

        self.pred3 = PredictionHead(
            stage3_channels
        )

        self.reduce3 = ConvNormAct(
            stage3_channels,
            stage2_channels,
            kernel_size=1,
            padding=0,
        )

        # -------------------------------------------------
        # Stage2 decoder
        # -------------------------------------------------

        self.fusion2 = ProgressiveSelectiveFusion(
            channels=stage2_channels,
        )

        self.pred2 = PredictionHead(
            stage2_channels
        )

        self.reduce2 = ConvNormAct(
            stage2_channels,
            128,
            kernel_size=1,
            padding=0,
        )

        # -------------------------------------------------
        # Stage1 decoder
        # -------------------------------------------------

        self.stage1_adapter = ConvNormAct(
            stage1_channels,
            128,
            kernel_size=1,
            padding=0,
        )

        self.fusion1 = ProgressiveSelectiveFusion(
            channels=128,
        )

        self.stage2_boundary_adapter = (
            ConvNormAct(
                stage2_channels,
                128,
                kernel_size=1,
                padding=0,
            )
        )

        self.boundary_refinement = (
            BoundaryRefinementBlock(
                128
            )
        )

        self.pred1 = PredictionHead(
            128
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_20: torch.Tensor,
        mean_40: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor | list[torch.Tensor],
    ]:
        input_size = (
            image.shape[-2:]
        )

        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = self.backbone(
            image
        )

        # -------------------------------------------------
        # Direct region hierarchy
        # -------------------------------------------------

        (
            detail_region,
            fine_region,
            middle_region,
            coarse_region,
        ) = self.region_hierarchy(
            image=image,
            mean_20=mean_20,
            mean_40=mean_40,
            mean_60=mean_60,
        )

        # -------------------------------------------------
        # Region pyramid
        # -------------------------------------------------

        (
            region1,
            region2,
            region3,
            region4,
        ) = self.region_encoder(
            detail_region=detail_region,
            fine_region=fine_region,
            middle_region=middle_region,
            coarse_region=coarse_region,
            stage1_size=(
                stage1.shape[-2:]
            ),
            stage2_size=(
                stage2.shape[-2:]
            ),
            stage3_size=(
                stage3.shape[-2:]
            ),
            stage4_size=(
                stage4.shape[-2:]
            ),
        )

        # -------------------------------------------------
        # Backbone reconstruction
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Stage4
        # -------------------------------------------------

        decoded4 = self.deep_projection(
            stage4
        )

        decoded4 = self.context4(
            decoded4
        )

        prediction4 = self.pred4(
            decoded4
        )

        # -------------------------------------------------
        # Stage3
        # -------------------------------------------------

        global3 = self.global3(
            stage4,
            target_size=(
                stage3.shape[-2:]
            ),
        )

        decoded3 = self.fusion3(
            low_feature=stage3,
            high_feature=decoded4,
            global_feature=global3,
        )

        prediction3 = self.pred3(
            decoded3
        )

        decoded3_reduced = self.reduce3(
            decoded3
        )

        # -------------------------------------------------
        # Stage2
        # -------------------------------------------------

        global2 = self.global2(
            stage4,
            target_size=(
                stage2.shape[-2:]
            ),
        )

        decoded2 = self.fusion2(
            low_feature=stage2,
            high_feature=decoded3_reduced,
            global_feature=global2,
        )

        prediction2 = self.pred2(
            decoded2
        )

        decoded2_reduced = self.reduce2(
            decoded2
        )

        # -------------------------------------------------
        # Stage1
        # -------------------------------------------------

        stage1_feature = (
            self.stage1_adapter(
                stage1
            )
        )

        global1 = self.global1(
            stage4,
            target_size=(
                stage1.shape[-2:]
            ),
        )

        decoded1 = self.fusion1(
            low_feature=stage1_feature,
            high_feature=decoded2_reduced,
            global_feature=global1,
        )

        # -------------------------------------------------
        # Boundary refinement
        # -------------------------------------------------

        stage2_boundary = (
            self.stage2_boundary_adapter(
                stage2
            )
        )

        decoded1 = (
            self.boundary_refinement(
                shallow_feature=stage1_feature,
                semantic_feature=stage2_boundary,
                saliency_feature=decoded1,
            )
        )

        prediction1 = self.pred1(
            decoded1
        )

        prediction1 = F.interpolate(
            prediction1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": prediction1,
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> MambaVisionSmallProgressiveRegionDirectSOD:
    return MambaVisionSmallProgressiveRegionDirectSOD(
        pretrained_path=PRETRAINED_PATH,
    )