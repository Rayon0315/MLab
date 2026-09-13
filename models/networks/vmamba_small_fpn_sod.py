from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.vmamba import (
    VMambaSmallBackbone,
    vmamba_small,
)
from models.components.sod_blocks import (
    ConvNormAct,
    PredictionHead,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "vmamba"
    / "vssm_small_0229_ckpt_epoch_222.pth"
)


class TopDownFPNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
    ) -> None:
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


class VMambaSmallFPNSOD(nn.Module):
    """
    Clean VMamba-S baseline.

    Backbone:
        VMamba-S [s2l15]

    Decoder:
        1x1 lateral projection
        + vanilla top-down addition
        + one 3x3 ConvNormAct per level

    Excluded on purpose:
        NAM / Region Mean
        Region Hybrid
        Dictionary Routing
        persistent Stage4 global branch
        selective/gated fusion
        boundary refinement
        disagreement refinement

    Aux predictions are retained only for deep supervision.
    """

    input_keys = (
        "image",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()

        self.backbone: VMambaSmallBackbone = (
            vmamba_small(
                pretrained_path=pretrained_path,
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
) -> VMambaSmallFPNSOD:
    return VMambaSmallFPNSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VMamba smoke test requires CUDA."
        )

    device = torch.device(
        "cuda"
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

    with torch.no_grad():
        outputs = model(
            image=image,
        )

    print(
        "pred:",
        outputs["pred"].shape,
    )

    print(
        "aux:",
        [
            tensor.shape
            for tensor
            in outputs["aux"]
        ],
    )
