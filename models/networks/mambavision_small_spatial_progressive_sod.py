# models/networks/mambavision_small_spatial_progressive_sod.py
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


class SpatialSemanticBranch(nn.Module):
    """
    Preserve the spatial layout of the highest-level Stage4 feature
    while projecting it to the channel dimension required by each
    decoder stage.

    Unlike the previous GlobalSemanticBranch, this branch does not
    perform global average pooling.

    Flow:
        Stage4
        -> channel projection
        -> spatial upsampling
        -> spatial refinement
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        # Keep the projection style of the previous progressive
        # decoder so that the main semantic-stream change is the
        # removal of GAP and preservation of spatial information.
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
        )

        self.refinement = nn.Sequential(
            ConvNormAct(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            ResidualConvBlock(
                out_channels
            ),
        )

    def forward(
        self,
        feature: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        context = self.projection(
            feature
        )

        context = F.interpolate(
            context,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        context = self.refinement(
            context
        )

        return context


class ProgressiveSelectiveFusion(nn.Module):
    """
    Three-stream selective fusion.

    Inputs:
        low_feature:
            current encoder-stage feature

        high_feature:
            top-down decoder feature

        global_feature:
            persistent Stage4 semantic feature

    Each stream is modulated by information from the other two
    streams before feature fusion.

    The gates are initialized to produce approximately identity
    scaling:
        2 * sigmoid(0) = 1
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

        low_weight = 2.0 * torch.sigmoid(
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

        high_weight = 2.0 * torch.sigmoid(
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

        global_weight = 2.0 * torch.sigmoid(
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


class MambaVisionSmallSpatialProgressiveSOD(nn.Module):
    """
    MambaVision-S with a progressive hierarchical decoder and
    spatially preserved persistent Stage4 semantic streams.

    Encoder:
        Stage1:  96 channels, stride 4
        Stage2: 192 channels, stride 8
        Stage3: 384 channels, stride 16
        Stage4: 768 channels, stride 32

    Main decoder:
        Stage4 768
            -> 384
            -> context
            -> fuse Stage3 384
            -> 384
            -> 192
            -> fuse Stage2 192
            -> 192
            -> 128
            -> fuse Stage1 128
            -> boundary refinement
            -> prediction

    Persistent Stage4 semantic streams:
        Stage4 768 x 11 x 11
            -> 384 -> upsample -> refine -> G3
            -> 192 -> upsample -> refine -> G2
            -> 128 -> upsample -> refine -> G1

    No AdaptiveAvgPool2d(1) is used in these persistent streams.
    """

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
        # Persistent Stage4 spatial semantic streams
        #
        # Stage4 spatial layout is preserved here.
        # No global average pooling is performed.
        # -------------------------------------------------

        self.global3 = SpatialSemanticBranch(
            in_channels=stage4_channels,
            out_channels=stage3_channels,
        )

        self.global2 = SpatialSemanticBranch(
            in_channels=stage4_channels,
            out_channels=stage2_channels,
        )

        self.global1 = SpatialSemanticBranch(
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

        # Preserve the previous progressive decoder
        # boundary refinement exactly.
        self.stage2_boundary_adapter = ConvNormAct(
            stage2_channels,
            128,
            kernel_size=1,
            padding=0,
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
    ) -> dict[
        str,
        torch.Tensor | list[torch.Tensor],
    ]:
        input_size = image.shape[-2:]

        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = self.backbone(
            image
        )

        # -------------------------------------------------
        # Stage4 -> D4
        # 768 -> 384
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
        #
        # Encoder:
        #   Stage3: 384 x 22 x 22
        #
        # Top-down:
        #   D4: 384 x 11 x 11
        #
        # Persistent Stage4 spatial semantics:
        #   Stage4 -> G3: 384 x 22 x 22
        # -------------------------------------------------

        global3 = self.global3(
            stage4,
            target_size=stage3.shape[-2:],
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
        #
        # Encoder:
        #   Stage2: 192 x 44 x 44
        #
        # Top-down:
        #   D3: 384 -> 192
        #
        # Persistent Stage4 spatial semantics:
        #   Stage4 -> G2: 192 x 44 x 44
        # -------------------------------------------------

        global2 = self.global2(
            stage4,
            target_size=stage2.shape[-2:],
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
        #
        # Encoder:
        #   Stage1: 96 -> 128
        #
        # Top-down:
        #   D2: 192 -> 128
        #
        # Persistent Stage4 spatial semantics:
        #   Stage4 -> G1: 128 x 88 x 88
        # -------------------------------------------------

        stage1_feature = self.stage1_adapter(
            stage1
        )

        global1 = self.global1(
            stage4,
            target_size=stage1.shape[-2:],
        )

        decoded1 = self.fusion1(
            low_feature=stage1_feature,
            high_feature=decoded2_reduced,
            global_feature=global1,
        )

        # -------------------------------------------------
        # Preserve the previous progressive decoder
        # BoundaryRefinementBlock.
        # -------------------------------------------------

        stage2_boundary = (
            self.stage2_boundary_adapter(
                stage2
            )
        )

        decoded1 = self.boundary_refinement(
            shallow_feature=stage1_feature,
            semantic_feature=stage2_boundary,
            saliency_feature=decoded1,
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


def build_model() -> MambaVisionSmallSpatialProgressiveSOD:
    return MambaVisionSmallSpatialProgressiveSOD(
        pretrained_path=PRETRAINED_PATH,
    )