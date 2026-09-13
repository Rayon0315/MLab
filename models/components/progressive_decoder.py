from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import ConvNormAct, PredictionHead


FeaturePyramid = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class HierarchicalTopDownBlock(nn.Module):
    """
    Vanilla FPN fusion at one hierarchy level:

        low + upsample(high) -> 3x3 ConvNormAct
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = ConvNormAct(
            channels,
            channels,
            kernel_size=3,
        )

    def forward(
        self,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
    ) -> torch.Tensor:
        high_feature = F.interpolate(
            high_feature,
            size=low_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(
            low_feature + high_feature
        )


class HierarchicalFPNDecoder(nn.Module):
    """
    FPN with a hierarchical channel schedule.

    Default schedule for VMamba-S:

        Stage4: 768 -> 384
        Stage3: 384 -> 384 -> 192
        Stage2: 192 -> 192 -> 128
        Stage1:  96 -> 128

    Only the channel schedule differs from the clean FPN idea.
    Fusion is still simple top-down addition.

    Intentionally excluded:
        - PyramidContext
        - persistent Stage4 global branch
        - selective/gated fusion
        - concat reconstruction
        - boundary refinement
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int, int],
        decoder_channels: tuple[int, int, int, int] = (
            128,
            192,
            384,
            384,
        ),
    ) -> None:
        super().__init__()

        in1, in2, in3, in4 = in_channels
        dec1, dec2, dec3, dec4 = decoder_channels

        self.in_channels = in_channels
        self.decoder_channels = decoder_channels

        self.lateral1 = ConvNormAct(
            in1,
            dec1,
            kernel_size=1,
            padding=0,
        )
        self.lateral2 = ConvNormAct(
            in2,
            dec2,
            kernel_size=1,
            padding=0,
        )
        self.lateral3 = ConvNormAct(
            in3,
            dec3,
            kernel_size=1,
            padding=0,
        )
        self.lateral4 = ConvNormAct(
            in4,
            dec4,
            kernel_size=1,
            padding=0,
        )

        self.refine4 = ConvNormAct(
            dec4,
            dec4,
            kernel_size=3,
        )
        self.pred4 = PredictionHead(
            dec4
        )

        if dec4 == dec3:
            self.reduce4 = nn.Identity()
        else:
            self.reduce4 = ConvNormAct(
                dec4,
                dec3,
                kernel_size=1,
                padding=0,
            )

        self.fusion3 = HierarchicalTopDownBlock(
            dec3
        )
        self.pred3 = PredictionHead(
            dec3
        )
        self.reduce3 = ConvNormAct(
            dec3,
            dec2,
            kernel_size=1,
            padding=0,
        )

        self.fusion2 = HierarchicalTopDownBlock(
            dec2
        )
        self.pred2 = PredictionHead(
            dec2
        )
        self.reduce2 = ConvNormAct(
            dec2,
            dec1,
            kernel_size=1,
            padding=0,
        )

        self.fusion1 = HierarchicalTopDownBlock(
            dec1
        )
        self.pred1 = PredictionHead(
            dec1
        )

    def forward(
        self,
        features: FeaturePyramid,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        stage1, stage2, stage3, stage4 = features

        feature1 = self.lateral1(stage1)
        feature2 = self.lateral2(stage2)
        feature3 = self.lateral3(stage3)
        feature4 = self.lateral4(stage4)

        decoded4 = self.refine4(
            feature4
        )
        prediction4 = self.pred4(
            decoded4
        )

        decoded4_for_stage3 = self.reduce4(
            decoded4
        )

        decoded3 = self.fusion3(
            low_feature=feature3,
            high_feature=decoded4_for_stage3,
        )
        prediction3 = self.pred3(
            decoded3
        )

        decoded3_for_stage2 = self.reduce3(
            decoded3
        )

        decoded2 = self.fusion2(
            low_feature=feature2,
            high_feature=decoded3_for_stage2,
        )
        prediction2 = self.pred2(
            decoded2
        )

        decoded2_for_stage1 = self.reduce2(
            decoded2
        )

        decoded1 = self.fusion1(
            low_feature=feature1,
            high_feature=decoded2_for_stage1,
        )
        prediction1 = self.pred1(
            decoded1
        )

        prediction1 = F.interpolate(
            prediction1,
            size=output_size,
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
