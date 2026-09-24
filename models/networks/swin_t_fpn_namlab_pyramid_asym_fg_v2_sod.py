from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.swin import (
    SwinTinyBackbone,
    swin_tiny,
)
from models.components.asymmetric_foreground_support_v2 import (
    AsymmetricForegroundSupportV2,
)
from models.components.namlab_hybrid import (
    NAMLabHybrid,
)
from models.components.sod_blocks import (
    ConvNormAct,
    PredictionHead,
    PyramidContextBlock,
)
from models.networks.vmamba_small_fpn_sod import (
    TopDownFPNBlock,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "swin"
    / "swin_t-704ceda3.pth"
)


class SwinTinyFPNNAMLabPyramidAsymFGV2SOD(
    nn.Module
):
    """
    Swin-T replacement of:
        vmamba_small_fpn_namlab_pyramid_asym_fg_v2_sod
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()

        self.backbone: SwinTinyBackbone = (
            swin_tiny(
                pretrained_path=pretrained_path,
            )
        )

        self.namlab_hybrid = (
            NAMLabHybrid(
                stage_channels=tuple(
                    self.backbone.out_channels
                ),
                initial_context_scale=0.1,
            )
        )

        self.lateral1 = ConvNormAct(
            self.backbone.out_channels[0],
            decoder_channels,
            kernel_size=1,
            padding=0,
        )
        self.lateral2 = ConvNormAct(
            self.backbone.out_channels[1],
            decoder_channels,
            kernel_size=1,
            padding=0,
        )
        self.lateral3 = ConvNormAct(
            self.backbone.out_channels[2],
            decoder_channels,
            kernel_size=1,
            padding=0,
        )
        self.lateral4 = ConvNormAct(
            self.backbone.out_channels[3],
            decoder_channels,
            kernel_size=1,
            padding=0,
        )

        self.context4 = PyramidContextBlock(
            decoder_channels
        )

        self.refine4 = ConvNormAct(
            decoder_channels,
            decoder_channels,
            kernel_size=3,
        )

        self.fusion3 = TopDownFPNBlock(
            decoder_channels
        )
        self.fusion2 = TopDownFPNBlock(
            decoder_channels
        )
        self.fusion1 = TopDownFPNBlock(
            decoder_channels
        )

        self.pred4 = PredictionHead(
            decoder_channels
        )
        self.pred3 = PredictionHead(
            decoder_channels
        )
        self.pred2 = PredictionHead(
            decoder_channels
        )
        self.pred1 = PredictionHead(
            decoder_channels
        )

        self.foreground_support = (
            AsymmetricForegroundSupportV2(
                channels=decoder_channels,
                metric_channels=64,
                initial_support_scale=4.0,
                initial_residual_scale=0.0,
                eps=1e-6,
                detach_guide=True,
            )
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
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

        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = self.namlab_hybrid(
            features=(
                stage1,
                stage2,
                stage3,
                stage4,
            ),
            image=image,
            mean_60=mean_60,
        )

        feature1 = self.lateral1(stage1)
        feature2 = self.lateral2(stage2)
        feature3 = self.lateral3(stage3)
        feature4 = self.lateral4(stage4)

        feature4 = self.context4(feature4)

        decoded4 = self.refine4(feature4)
        prediction4 = self.pred4(decoded4)

        decoded3 = self.fusion3(
            low_feature=feature3,
            high_feature=decoded4,
        )
        prediction3 = self.pred3(decoded3)

        decoded2 = self.fusion2(
            low_feature=feature2,
            high_feature=decoded3,
        )
        prediction2 = self.pred2(decoded2)

        decoded1 = self.fusion1(
            low_feature=feature1,
            high_feature=decoded2,
        )

        fg = self.foreground_support(
            high_feature=decoded3,
            low_feature=decoded1,
            guide_logits=prediction3,
        )

        prediction1 = self.pred1(
            fg["refined_feature"]
        )

        def upsample(
            tensor: torch.Tensor,
        ) -> torch.Tensor:
            return F.interpolate(
                tensor,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return {
            "pred": upsample(prediction1),
            "fg_similarity": upsample(
                fg["fg_similarity"]
            ),
            "fg_support_logits": upsample(
                fg["fg_support_logits"]
            ),
            "fg_support": upsample(
                fg["fg_support"]
            ),
            "fg_guide_probability": upsample(
                fg["guide_foreground_probability"]
            ),
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> SwinTinyFPNNAMLabPyramidAsymFGV2SOD:
    return (
        SwinTinyFPNNAMLabPyramidAsymFGV2SOD(
            pretrained_path=PRETRAINED_PATH,
            decoder_channels=128,
        )
    )
