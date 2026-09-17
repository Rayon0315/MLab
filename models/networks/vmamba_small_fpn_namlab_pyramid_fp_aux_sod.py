from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import PyramidContextBlock
from models.networks.vmamba_small_fpn_namlab_hybrid_sod import (
    VMambaSmallFPNNAMLabHybridSOD,
)
from models.networks.vmamba_small_fpn_sod import PRETRAINED_PATH


class VMambaSmallFPNNAMLabPyramidFPAuxSOD(
    VMambaSmallFPNNAMLabHybridSOD
):
    """
    VMamba-S + NAMLab Hybrid + Pure FPN + Pyramid Context
    + minimal false-positive auxiliary head.

    Main saliency path is unchanged from the Pyramid baseline.
    The auxiliary branch reads decoded Stage1 features and predicts
    false-positive risk only for training-time supervision.
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
        super().__init__(
            pretrained_path=pretrained_path,
            decoder_channels=decoder_channels,
        )

        self.context4 = PyramidContextBlock(
            decoder_channels
        )

        self.fp_risk_head = nn.Conv2d(
            decoder_channels,
            1,
            kernel_size=1,
        )

        # Match the original CFPS risk-head initialization so that
        # the ablation changes the evidence source, not head bias.
        nn.init.zeros_(self.fp_risk_head.weight)
        nn.init.constant_(self.fp_risk_head.bias, -2.0)

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
            raw_stage1,
            raw_stage2,
            raw_stage3,
            raw_stage4,
        ) = self.backbone(image)

        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = self.namlab_hybrid(
            features=(
                raw_stage1,
                raw_stage2,
                raw_stage3,
                raw_stage4,
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

        coarse_logits = self.pred1(decoded1)
        fp_risk_logits = self.fp_risk_head(decoded1)

        prediction = F.interpolate(
            coarse_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        fp_risk_prediction = F.interpolate(
            fp_risk_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": prediction,
            "coarse_pred": prediction,
            "fp_risk": fp_risk_prediction,
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> VMambaSmallFPNNAMLabPyramidFPAuxSOD:
    return VMambaSmallFPNNAMLabPyramidFPAuxSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VMamba smoke test requires CUDA."
        )

    device = torch.device("cuda")
    model = build_model().to(device)
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

    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(key, tuple(value.shape))
        else:
            print(
                key,
                [tuple(tensor.shape) for tensor in value],
            )
