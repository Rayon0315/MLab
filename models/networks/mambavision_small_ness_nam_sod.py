# models/networks/mambavision_small_ness_nam_sod.py
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
    SaliencyGuidedFusion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "mambavision"
    / "mambavision_small_1k.pth.tar"
)


class CALayer(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        hidden_channels = max(
            channels // reduction,
            1,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.PReLU(hidden_channels),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        attention = self.conv_du(
            self.avg_pool(x)
        )

        return x * attention


class RCAB(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            CALayer(
                channels=channels,
                reduction=reduction,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x + self.body(x)


class CBAMLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ) -> None:
        super().__init__()

        hidden_channels = max(
            channels // reduction,
            1,
        )

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
        )

        self.spatial_conv = nn.Conv2d(
            2,
            1,
            kernel_size=spatial_kernel,
            padding=spatial_kernel // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        channel_attention = self.sigmoid(
            self.channel_mlp(
                self.max_pool(x)
            )
            + self.channel_mlp(
                self.avg_pool(x)
            )
        )

        x = channel_attention * x

        max_feature = torch.max(
            x,
            dim=1,
            keepdim=True,
        ).values

        mean_feature = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        spatial_attention = self.sigmoid(
            self.spatial_conv(
                torch.cat(
                    [
                        max_feature,
                        mean_feature,
                    ],
                    dim=1,
                )
            )
        )

        return spatial_attention * x


class CBAMConvBlock(nn.Module):
    """
    NESS-Net deep feature refinement.

    The original NESS-Net applies this block to a clone of
    the deepest backbone feature and uses it as the fifth
    feature level for EdgeConstructModule.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        hidden_channels = channels // 2

        self.depthwise5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=True,
        )

        self.depthwise3_1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        self.depthwise3_2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        self.reduce = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            bias=True,
        )

        self.prelu = nn.PReLU(
            hidden_channels
        )

        self.expand = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.cbam = CBAMLayer(
            channels
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        identity = x

        x = self.depthwise5(x) + x
        x = self.depthwise3_1(x) + x
        x = self.depthwise3_2(x) + x

        x = self.reduce(x)
        x = self.prelu(x)
        x = self.expand(x) + identity

        return self.cbam(x)


class EdgeConstructModule(nn.Module):
    """
    NESS-Net EdgeConstructModule adapted to MambaVision-S.

    MambaVision-S features:
        stage1: 96 channels, stride 4
        stage2: 192 channels, stride 8
        stage3: 384 channels, stride 16
        stage4: 768 channels, stride 32
        stage5: CBAM-refined stage4, stride 32

    The output has 96 channels at stride 2.
    """

    def __init__(
        self,
        in_channels: tuple[
            int,
            int,
            int,
            int,
            int,
        ],
        mid_channels: int = 96,
    ) -> None:
        super().__init__()

        self.depthwise_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=5,
                    padding=2,
                    groups=channels,
                    bias=True,
                )
                for channels in in_channels
            ]
        )

        self.projections = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    mid_channels,
                    kernel_size=1,
                    bias=True,
                )
                for channels in in_channels
            ]
        )

        self.relu = nn.PReLU(
            mid_channels
        )

        self.conv_after_up = nn.ModuleList(
            [
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                )
                for _ in range(5)
            ]
        )

        self.se_block = CALayer(
            mid_channels
        )

        self.rcab = RCAB(
            mid_channels
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

    def _extract_edge_feature(
        self,
        feature: torch.Tensor,
        depthwise_conv: nn.Module,
        projection: nn.Module,
    ) -> torch.Tensor:
        feature = (
            depthwise_conv(feature)
            + feature
        )

        return self.relu(
            projection(feature)
        )

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
    ) -> torch.Tensor:
        inputs = (
            x1,
            x2,
            x3,
            x4,
            x5,
        )

        edge_features = [
            self._extract_edge_feature(
                feature=feature,
                depthwise_conv=depthwise_conv,
                projection=projection,
            )
            for feature, depthwise_conv, projection in zip(
                inputs,
                self.depthwise_convs,
                self.projections,
            )
        ]

        edge1 = edge_features[0]
        edge2 = edge_features[1]
        edge3 = edge_features[2]
        edge4 = edge_features[3]
        edge5 = edge_features[4]

        edge5 = (
            self.conv_after_up[4](
                edge5
            )
            + edge5
        )

        edge = edge5 + edge4

        edge = F.interpolate(
            edge,
            size=edge3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        edge = (
            self.conv_after_up[3](
                edge
            )
            + edge
            + edge3
        )

        edge = F.interpolate(
            edge,
            size=edge2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        edge = (
            self.conv_after_up[2](
                edge
            )
            + edge
            + edge2
        )

        edge = F.interpolate(
            edge,
            size=edge1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        edge = (
            self.conv_after_up[1](
                edge
            )
            + edge
            + edge1
        )

        edge = F.interpolate(
            edge,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        edge = (
            self.conv_after_up[0](
                edge
            )
            + edge
        )

        edge = (
            self.se_block(edge)
            + edge
        )

        edge = self.rcab(edge)

        return (
            self.classifier(edge)
            + edge
        )


class SeparableConv2d(nn.Module):
    """
    Direct adaptation of NESS-Net fuse_canny_edge.

    Input:
        96 learned ECM channels + 1 NAMLab channel.

    Output:
        1 edge logit channel.
    """

    def __init__(
        self,
        in_channels: int,
    ) -> None:
        super().__init__()

        self.depthwise1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels,
            bias=True,
        )

        self.depthwise2 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=True,
        )

        self.relu = nn.PReLU(
            in_channels
        )

        self.depthwise3 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=True,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        identity = x

        x = self.depthwise1(x) + x
        x = self.depthwise2(x) + x
        x = self.relu(x)
        x = self.depthwise3(x) + x
        x = x + identity

        return self.pointwise(x)


class SelfGatedPredUnit(nn.Module):
    """
    Direct adaptation of NESS-Net SelfGatedPredUnit.
    """

    def __init__(
        self,
        channels: int = 64,
    ) -> None:
        super().__init__()

        self.gate_depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            padding=2,
            groups=channels,
            bias=True,
        )

        self.gate_projection = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
            bias=True,
        )

        self.refine = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.prediction = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(
            self.gate_projection(
                self.gate_depthwise(x)
            )
        )

        x = gate * x + x
        x = self.refine(x) + x

        return self.prediction(x)


class NESSSaliencyEdgeFusion(nn.Module):
    """
    RGB-only adaptation of NESS-Net final saliency-edge fusion.

    NESS-Net normally combines RGB-edge and depth-edge features
    through CMAM. This version removes the depth path and keeps
    the saliency-edge concatenation, RCAB and self-gated head.
    """

    def __init__(
        self,
        saliency_channels: int,
        edge_channels: int = 32,
    ) -> None:
        super().__init__()

        self.saliency_projection = nn.Conv2d(
            saliency_channels,
            edge_channels,
            kernel_size=1,
            bias=True,
        )

        self.saliency_refine = nn.Conv2d(
            edge_channels,
            edge_channels,
            kernel_size=3,
            padding=1,
            groups=edge_channels,
            bias=True,
        )

        self.edge_projection = nn.Sequential(
            nn.Conv2d(
                1,
                1,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.Conv2d(
                1,
                edge_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        fused_channels = edge_channels * 2

        self.fused_refine = nn.Conv2d(
            fused_channels,
            fused_channels,
            kernel_size=3,
            padding=1,
            groups=fused_channels,
            bias=True,
        )

        self.rcab = RCAB(
            fused_channels
        )

        self.relu = nn.PReLU(
            fused_channels
        )

        self.prediction = SelfGatedPredUnit(
            fused_channels
        )

    def forward(
        self,
        saliency_feature: torch.Tensor,
        edge_logits: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        saliency_feature = (
            self.saliency_projection(
                saliency_feature
            )
        )

        saliency_feature = F.interpolate(
            saliency_feature,
            size=edge_logits.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        saliency_feature = (
            self.saliency_refine(
                saliency_feature
            )
            + saliency_feature
        )

        edge_feature = self.edge_projection(
            edge_logits
        )

        fused_feature = torch.cat(
            [
                saliency_feature,
                edge_feature,
            ],
            dim=1,
        )

        fused_feature = F.interpolate(
            fused_feature,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        fused_feature = (
            self.fused_refine(
                fused_feature
            )
            + fused_feature
        )

        fused_feature = self.rcab(
            fused_feature
        )

        return self.prediction(
            self.relu(
                fused_feature
            )
        )


class MambaVisionSmallNESSNAMSOD(nn.Module):
    input_keys = (
        "image",
        "nam_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decoder_channels: int = 128,
        edge_mid_channels: int = 96,
        edge_fusion_channels: int = 32,
    ) -> None:
        super().__init__()

        self.backbone: MambaVisionBackbone = (
            mamba_vision_small(
                pretrained_path=pretrained_path,
            )
        )

        backbone_channels = tuple(
            self.backbone.out_channels
        )

        self.projections = nn.ModuleList(
            [
                ConvNormAct(
                    in_channels,
                    decoder_channels,
                    kernel_size=1,
                    padding=0,
                )
                for in_channels in backbone_channels
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

        self.pre_edge_pred = PredictionHead(
            decoder_channels
        )

        self.deep_edge_refinement = (
            CBAMConvBlock(
                backbone_channels[3]
            )
        )

        self.edge_construct = (
            EdgeConstructModule(
                in_channels=(
                    backbone_channels[0],
                    backbone_channels[1],
                    backbone_channels[2],
                    backbone_channels[3],
                    backbone_channels[3],
                ),
                mid_channels=edge_mid_channels,
            )
        )

        self.nam_pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.edge_predictor = SeparableConv2d(
            edge_mid_channels + 1
        )

        self.saliency_edge_fusion = (
            NESSSaliencyEdgeFusion(
                saliency_channels=decoder_channels,
                edge_channels=edge_fusion_channels,
            )
        )

    def forward(
        self,
        image: torch.Tensor,
        nam_60: torch.Tensor,
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
        ) = self.backbone(image)

        deep_edge_feature = (
            self.deep_edge_refinement(
                stage4
            )
        )

        learned_edge_feature = (
            self.edge_construct(
                stage1,
                stage2,
                stage3,
                stage4,
                deep_edge_feature,
            )
        )

        nam_feature = self.nam_pool(
            nam_60
        )

        if (
            nam_feature.shape[-2:]
            != learned_edge_feature.shape[-2:]
        ):
            nam_feature = F.interpolate(
                nam_feature,
                size=learned_edge_feature.shape[-2:],
                mode="nearest",
            )

        edge_logits = self.edge_predictor(
            torch.cat(
                [
                    learned_edge_feature,
                    nam_feature,
                ],
                dim=1,
            )
        )

        (
            feature1,
            feature2,
            feature3,
            feature4,
        ) = [
            projection(feature)
            for projection, feature in zip(
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

        decoded1 = self.boundary_refinement(
            shallow_feature=feature1,
            semantic_feature=feature2,
            saliency_feature=decoded1,
        )

        pre_edge_prediction = (
            self.pre_edge_pred(
                decoded1
            )
        )

        final_prediction = (
            self.saliency_edge_fusion(
                saliency_feature=decoded1,
                edge_logits=edge_logits,
                output_size=input_size,
            )
        )

        return {
            "pred": final_prediction,
            "aux": [
                pre_edge_prediction,
                prediction2,
                prediction3,
                prediction4,
            ],
            "edge": edge_logits,
        }


def build_model() -> MambaVisionSmallNESSNAMSOD:
    return MambaVisionSmallNESSNAMSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
        edge_mid_channels=96,
        edge_fusion_channels=32,
    )