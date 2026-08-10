# models/networks/mambavision_small_region_sod.py
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
    SaliencyGuidedFusion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "mambavision"
    / "mambavision_small_1k.pth.tar"
)


class HierarchicalRegionDecomposition(nn.Module):
    """
    Hierarchical decomposition of RGB and region-mean maps.

    Inputs:
        image:
            ImageNet-normalized RGB image.

        mean_20 / mean_40 / mean_60:
            Region-mean RGB maps in [0, 1].

    Outputs:
        R1 = RGB - M60
        R2 = M60 - M40
        R3 = M40 - M20
        R4 = M20
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

        fine_region = (
            mean60
            - mean40
        )

        middle_region = (
            mean40
            - mean20
        )

        coarse_region = mean20

        return (
            detail_region,
            fine_region,
            middle_region,
            coarse_region,
        )


class RegionScaleEncoder(nn.Module):
    """
    Encode one region-hierarchy component into the channel
    dimension of its corresponding MambaVision stage.
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


class RegionPyramidEncoder(nn.Module):
    """
    Map the region hierarchy to the four MambaVision stages.

        Stage1 <- RGB - M60
        Stage2 <- M60 - M40
        Stage3 <- M40 - M20
        Stage4 <- M20
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

        Visual queries attend to region keys/values.
        Region queries attend to visual keys/values.

    Both interaction results are fused back into the visual feature.
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
                "inner_channels must be divisible "
                "by num_heads."
            )

        self.inner_channels = (
            inner_channels
        )

        self.num_heads = (
            num_heads
        )

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
        batch_size = tokens.shape[0]

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
    """
    One depthwise dilated branch for Stage2 reconstruction.
    """

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


class RegionConditionedLocalReconstruction(
    nn.Module
):
    """
    Region-conditioned Stage2 reconstruction.

    The region feature predicts:
        1. Spatial routing weights for dilation 1/2/3 branches.
        2. Channel-wise affine modulation.
        3. A spatial-channel region gate.
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
    High-resolution Stage1 reconstruction.

    Full cross-attention is avoided at stride 4. RGB-M60 detail
    information interacts with the visual feature through local
    depthwise operations and reciprocal gates.
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


class MambaVisionSmallRegionSOD(nn.Module):
    """
    Original MambaVision-S SOD baseline with hierarchical
    region reconstruction inserted before the baseline decoder.

    Backbone:
        Stage1:  96 channels
        Stage2: 192 channels
        Stage3: 384 channels
        Stage4: 768 channels

    Region hierarchy:
        R1 = RGB - M60
        R2 = M60 - M40
        R3 = M40 - M20
        R4 = M20

    Reconstruction:
        Stage1:
            local detail reconstruction

        Stage2:
            region-conditioned multi-dilation reconstruction

        Stage3:
            bidirectional region-visual cross-attention

        Stage4:
            bidirectional region-visual cross-attention

    Decoder:
        Exactly follows the original baseline after the four
        reconstructed backbone features are produced.
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
        decoder_channels: int = 128,
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
        # Region hierarchy
        # -------------------------------------------------

        self.region_decomposition = (
            HierarchicalRegionDecomposition()
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
        # Original baseline decoder starts here
        # -------------------------------------------------

        self.projections = nn.ModuleList(
            [
                ConvNormAct(
                    in_channels,
                    decoder_channels,
                    kernel_size=1,
                    padding=0,
                )
                for in_channels
                in self.backbone.out_channels
            ]
        )

        self.context4 = PyramidContextBlock(
            decoder_channels
        )

        self.pred4 = PredictionHead(
            decoder_channels
        )

        self.fusion3 = SaliencyGuidedFusion(
            decoder_channels
        )

        self.pred3 = PredictionHead(
            decoder_channels
        )

        self.fusion2 = SaliencyGuidedFusion(
            decoder_channels
        )

        self.pred2 = PredictionHead(
            decoder_channels
        )

        self.fusion1 = SaliencyGuidedFusion(
            decoder_channels
        )

        self.boundary_refinement = (
            BoundaryRefinementBlock(
                decoder_channels
            )
        )

        self.pred1 = PredictionHead(
            decoder_channels
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
        # Hierarchical region decomposition
        # -------------------------------------------------

        (
            detail_region,
            fine_region,
            middle_region,
            coarse_region,
        ) = self.region_decomposition(
            image=image,
            mean_20=mean_20,
            mean_40=mean_40,
            mean_60=mean_60,
        )

        # -------------------------------------------------
        # Region pyramid encoding
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
        # Reconstruct native backbone features
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
        # Original baseline decoder
        # -------------------------------------------------

        (
            feature1,
            feature2,
            feature3,
            feature4,
        ) = [
            projection(
                feature
            )
            for (
                projection,
                feature,
            ) in zip(
                self.projections,
                (
                    stage1,
                    stage2,
                    stage3,
                    stage4,
                ),
            )
        ]

        decoded4 = self.context4(
            feature4
        )

        prediction4 = self.pred4(
            decoded4
        )

        decoded3 = self.fusion3(
            low_feature=feature3,
            high_feature=decoded4,
            guide_logits=prediction4,
        )

        prediction3 = self.pred3(
            decoded3
        )

        decoded2 = self.fusion2(
            low_feature=feature2,
            high_feature=decoded3,
            guide_logits=prediction3,
        )

        prediction2 = self.pred2(
            decoded2
        )

        decoded1 = self.fusion1(
            low_feature=feature1,
            high_feature=decoded2,
            guide_logits=prediction2,
        )

        decoded1 = (
            self.boundary_refinement(
                shallow_feature=feature1,
                semantic_feature=feature2,
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
) -> MambaVisionSmallRegionSOD:
    return MambaVisionSmallRegionSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )