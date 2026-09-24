from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.smt import (
    SMTTinyBackbone,
    smt_tiny,
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
    / "smt"
    / "smt_tiny.pth"
)


class SMTTinyFPNNAMLabPyramidAsymFGV2SOD(
    nn.Module
):
    """
    SMT-T replacement of:
        vmamba_small_fpn_namlab_pyramid_asym_fg_v2_sod

    Controlled change:
        VMamba-S backbone
            ->
        SMT-T backbone

    Preserved:
        - NAMLabHybrid
        - 128-channel FPN laterals
        - Stage4 PyramidContextBlock
        - vanilla top-down FPN
        - AsymmetricForegroundSupportV2
        - prediction heads / aux outputs
        - image + mean_60 input interface
        - asymmetric foreground-support supervision interface

    SMT-T feature pyramid at 352x352:
        stage1:  64 x 88 x 88
        stage2: 128 x 44 x 44
        stage3: 256 x 22 x 22
        stage4: 512 x 11 x 11
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

        self.backbone: SMTTinyBackbone = (
            smt_tiny(
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
        ) = self.backbone(
            image
        )

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

        feature1 = self.lateral1(
            stage1
        )
        feature2 = self.lateral2(
            stage2
        )
        feature3 = self.lateral3(
            stage3
        )
        feature4 = self.lateral4(
            stage4
        )

        feature4 = self.context4(
            feature4
        )

        decoded4 = self.refine4(
            feature4
        )
        prediction4 = self.pred4(
            decoded4
        )

        decoded3 = self.fusion3(
            low_feature=feature3,
            high_feature=decoded4,
        )
        prediction3 = self.pred3(
            decoded3
        )

        decoded2 = self.fusion2(
            low_feature=feature2,
            high_feature=decoded3,
        )
        prediction2 = self.pred2(
            decoded2
        )

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
            "pred": upsample(
                prediction1
            ),
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
                fg[
                    "guide_foreground_probability"
                ]
            ),
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> SMTTinyFPNNAMLabPyramidAsymFGV2SOD:
    return (
        SMTTinyFPNNAMLabPyramidAsymFGV2SOD(
            pretrained_path=PRETRAINED_PATH,
            decoder_channels=128,
        )
    )


if __name__ == "__main__":
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_model().to(
        device
    )
    model.eval()

    image = torch.randn(
        1,
        3,
        352,
        352,
        device=device,
    )

    mean_60 = torch.rand(
        1,
        3,
        352,
        352,
        device=device,
    )

    with torch.no_grad():
        outputs = model(
            image=image,
            mean_60=mean_60,
        )

    print(
        "backbone channels:",
        model.backbone.out_channels,
    )
    print(
        "backbone strides:",
        model.backbone.out_strides,
    )
    print(
        "pred:",
        outputs["pred"].shape,
    )
    print(
        "aux:",
        [
            tensor.shape
            for tensor in outputs["aux"]
        ],
    )
    print(
        "fg_support_logits:",
        outputs[
            "fg_support_logits"
        ].shape,
    )
