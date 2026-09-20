from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.components.asymmetric_foreground_support import (
    AsymmetricForegroundSupport,
)
from models.components.sod_blocks import (
    PyramidContextBlock,
)
from models.networks.vmamba_small_fpn_namlab_hybrid_sod import (
    VMambaSmallFPNNAMLabHybridSOD,
)
from models.networks.vmamba_small_fpn_sod import (
    PRETRAINED_PATH,
)


class VMambaSmallFPNNAMLabPyramidAsymFGSOD(
    VMambaSmallFPNNAMLabHybridSOD
):
    """
    FG-only asymmetric evidence ablation.

    Pyramid FPN
        -> positive foreground prototype support
        -> zero-init residual foreground enhancement
        -> saliency prediction

    No background prototype, no FG/BG softmax, no FP verifier.
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

        self.foreground_support = (
            AsymmetricForegroundSupport(
                channels=decoder_channels,
                metric_channels=64,
                initial_support_scale=4.0,
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
            "fg_support": upsample(
                fg["fg_support"]
            ),
            "fg_similarity": upsample(
                fg["fg_similarity"]
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
) -> VMambaSmallFPNNAMLabPyramidAsymFGSOD:
    return VMambaSmallFPNNAMLabPyramidAsymFGSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )
