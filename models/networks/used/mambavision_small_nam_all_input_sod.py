# models/networks/mambavision_small_nam_all_input_sod.py

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


class MambaVisionSmallNAMAllInputSOD(nn.Module):
    """
    MambaVision-Small RGB + multi-level NAMLab input fusion.

    Input:
        image:  [B, 3, H, W]
        nam_20: [B, 1, H, W]
        nam_40: [B, 1, H, W]
        nam_60: [B, 1, H, W]

    The four inputs are concatenated into:

        [B, 6, H, W]

    The RGB channels inherit the ImageNet pretrained weights.
    The three NAM channels are initialized with zero weights.

    The decoder is identical to the RGB baseline.
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

        self._expand_input_convolution()

        self.projections = nn.ModuleList(
            [
                ConvNormAct(
                    in_channels,
                    decoder_channels,
                    kernel_size=1,
                    padding=0,
                )
                for in_channels in self.backbone.out_channels
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

    def _expand_input_convolution(
        self,
    ) -> None:
        rgb_conv: nn.Conv2d = (
            self.backbone
            .patch_embed
            .conv_down[0]
        )

        input_conv = nn.Conv2d(
            in_channels=6,
            out_channels=rgb_conv.out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            input_conv.weight[:, :3].copy_(
                rgb_conv.weight
            )

            input_conv.weight[:, 3:].zero_()

        self.backbone.patch_embed.conv_down[0] = (
            input_conv
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

        model_input = torch.cat(
            [
                image,
                nam_20,
                nam_40,
                nam_60,
            ],
            dim=1,
        )

        stage1, stage2, stage3, stage4 = (
            self.backbone(model_input)
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


def build_model() -> MambaVisionSmallNAMAllInputSOD:
    return MambaVisionSmallNAMAllInputSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )