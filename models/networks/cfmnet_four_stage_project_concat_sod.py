# models/networks/cfmnet_four_stage_project_concat_sod.py

from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
    build_cfmnet,
)
from models.components.cfm_stage_fusion import (
    FourStageProjectConcatHead,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "cfmnet"
    / "cfmnet_imagenet1k.pth"
)


class CFMNetFourStageProjectConcatSOD(
    nn.Module
):
    """
    Experiment A:
        use the normal four fused CFMNet stage features directly.

    S1 -> 1x1 project --\
    S2 -> 1x1 project ----\
    S3 -> 1x1 project ------> resize to S1 -> concat -> 3x3 -> pred
    S4 -> 1x1 project ----/

    This intentionally removes UNetFormer top-down decoding so the
    experiment asks whether CFMNet's four-level hierarchy itself is
    already sufficient for SOD with a minimal task head.
    """

    input_keys = (
        "image",
    )

    def __init__(
        self,
        pretrained_path: str
        | Path
        | None,
        decode_channels: int = 64,
    ) -> None:
        super().__init__()

        self.backbone = (
            build_cfmnet(
                pretrained_path=(
                    pretrained_path
                )
            )
        )

        self.decoder = (
            FourStageProjectConcatHead(
                encoder_channels=(
                    CFMNET_OUT_CHANNELS
                ),
                decode_channels=(
                    decode_channels
                ),
                dropout=0.1,
            )
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        output_size = (
            image.shape[-2:]
        )

        features = (
            self.backbone(
                image
            )
        )

        prediction = (
            self.decoder(
                features=features,
                output_size=output_size,
            )
        )

        return {
            "pred": prediction,
        }


def build_model(
) -> CFMNetFourStageProjectConcatSOD:
    pretrained_path = (
        PRETRAINED_PATH
        if PRETRAINED_PATH.exists()
        else None
    )

    if pretrained_path is None:
        warnings.warn(
            (
                "CFMNet ImageNet-1K checkpoint "
                "was not found at "
                f"{PRETRAINED_PATH}. "
                "The backbone will train "
                "from scratch."
            ),
            RuntimeWarning,
        )

    return (
        CFMNetFourStageProjectConcatSOD(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=64,
        )
    )
