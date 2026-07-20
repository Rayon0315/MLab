# models/networks/mambavision_small_nam_sod.py

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


class NAMHierarchyEncoder(nn.Module):
    """
    分别编码 NAMLab 的三个 hierarchy，
    再融合为统一的边缘特征。
    """

    def __init__(
        self,
        out_channels: int,
    ) -> None:
        super().__init__()

        branch_channels = out_channels // 4

        self.hier_20_branch = nn.Sequential(
            ConvNormAct(
                1,
                branch_channels,
            ),
            ResidualConvBlock(
                branch_channels,
            ),
        )

        self.hier_40_branch = nn.Sequential(
            ConvNormAct(
                1,
                branch_channels,
            ),
            ResidualConvBlock(
                branch_channels,
            ),
        )

        self.hier_60_branch = nn.Sequential(
            ConvNormAct(
                1,
                branch_channels,
            ),
            ResidualConvBlock(
                branch_channels,
            ),
        )

        self.fusion = nn.Sequential(
            ConvNormAct(
                branch_channels * 3,
                out_channels,
            ),
            ResidualConvBlock(
                out_channels,
            ),
        )

    def forward(
        self,
        nam_20: torch.Tensor,
        nam_40: torch.Tensor,
        nam_60: torch.Tensor,
    ) -> torch.Tensor:
        feature_20 = self.hier_20_branch(
            nam_20
        )

        feature_40 = self.hier_40_branch(
            nam_40
        )

        feature_60 = self.hier_60_branch(
            nam_60
        )

        return self.fusion(
            torch.cat(
                [
                    feature_20,
                    feature_40,
                    feature_60,
                ],
                dim=1,
            )
        )


class NAMGuidedFusion(nn.Module):
    """
    使用 NAM 边缘特征增强 stride 4 RGB 浅层特征。

    NAM 提供边缘提示；
    RGB 特征仍然是主分支；
    输出尺寸和通道数均与 RGB 特征相同。
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.rgb_refine = ResidualConvBlock(
            channels
        )

        self.nam_refine = ResidualConvBlock(
            channels
        )

        self.nam_gate = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

        self.output_fusion = nn.Sequential(
            ConvNormAct(
                channels * 2,
                channels,
            ),
            ResidualConvBlock(
                channels,
            ),
        )

    def forward(
        self,
        rgb_feature: torch.Tensor,
        nam_feature: torch.Tensor,
    ) -> torch.Tensor:
        nam_feature = F.interpolate(
            nam_feature,
            size=rgb_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        rgb_feature = self.rgb_refine(
            rgb_feature
        )

        nam_feature = self.nam_refine(
            nam_feature
        )

        nam_attention = torch.sigmoid(
            self.nam_gate(nam_feature)
        )

        guided_rgb = rgb_feature * (
            1.0 + nam_attention
        )

        fused_feature = self.output_fusion(
            torch.cat(
                [
                    guided_rgb,
                    nam_feature,
                ],
                dim=1,
            )
        )

        return rgb_feature + fused_feature


class MambaVisionSmallNAMSOD(nn.Module):
    """
    MambaVision-Small + NAMLab hierarchy guidance。

    RGB backbone:
        stage1: stride 4, channels 96
        stage2: stride 8, channels 192
        stage3: stride 16, channels 384
        stage4: stride 32, channels 768

    NAM:
        hier_20
        hier_40
        hier_60

    NAM 三层首先独立编码，然后融合并注入 stage1。
    后续 decoder 与 MambaVisionSmallSOD baseline 保持一致。
    """

    input_keys = (
        "image",
        "nam_20",
        "nam_40",
        "nam_60",
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

        self.nam_encoder = NAMHierarchyEncoder(
            out_channels=decoder_channels,
        )

        self.nam_fusion = NAMGuidedFusion(
            channels=decoder_channels,
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
        nam_20: torch.Tensor,
        nam_40: torch.Tensor,
        nam_60: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor | list[torch.Tensor],
    ]:
        input_size = image.shape[-2:]

        stage1, stage2, stage3, stage4 = (
            self.backbone(image)
        )

        feature1, feature2, feature3, feature4 = [
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

        nam_feature = self.nam_encoder(
            nam_20=nam_20,
            nam_40=nam_40,
            nam_60=nam_60,
        )

        feature1 = self.nam_fusion(
            rgb_feature=feature1,
            nam_feature=nam_feature,
        )

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


def build_model() -> MambaVisionSmallNAMSOD:
    return MambaVisionSmallNAMSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )