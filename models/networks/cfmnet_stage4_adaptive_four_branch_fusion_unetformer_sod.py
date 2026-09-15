# models/networks/cfmnet_stage4_adaptive_four_branch_fusion_unetformer_sod.py

from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
    build_cfmnet,
)
from models.components.cfm_branch_interaction import (
    AdaptiveFourBranchFusion,
    Stage4BranchTap,
)
from models.networks.cfmnet_unetformer_sod import (
    CFMNetUNetFormerDecoder,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "cfmnet"
    / "cfmnet_imagenet1k.pth"
)


class CFMNetStage4AdaptiveFourBranchFusionUNetFormerSOD(
    nn.Module
):
    """
    Stage-4 four-branch adaptive fusion experiment.

    Stage 1/2/3:
        keep the standard fused CFMNet stage outputs.

    Stage 4:
        capture the final CFMBlock's four branch outputs:
            TCRM / SSRM / LDFM / EGPCM

        predict four spatially-varying softmax weights from their
        concatenation, rescale the four branches competitively, then
        concatenate them back to 768 channels and feed the unchanged
        UNetFormer decoder.

    The gate's last layer is zero-initialized. At initialization the
    softmax is uniform (1/4), then multiplied by 4, so this network
    starts exactly from the Stage4 raw-concat behavior.
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
        window_size: int = 8,
    ) -> None:
        super().__init__()

        self.backbone = build_cfmnet(
            pretrained_path=pretrained_path
        )

        self.stage4_branch_tap = (
            Stage4BranchTap(
                self.backbone
            )
        )

        branch_channels = (
            CFMNET_OUT_CHANNELS[-1]
            // 4
        )

        self.branch_fusion = (
            AdaptiveFourBranchFusion(
                branch_channels=(
                    branch_channels
                ),
                hidden_channels=(
                    branch_channels
                ),
            )
        )

        self.decoder = (
            CFMNetUNetFormerDecoder(
                encoder_channels=(
                    CFMNET_OUT_CHANNELS
                ),
                decode_channels=(
                    decode_channels
                ),
                dropout=0.1,
                window_size=(
                    window_size
                ),
                num_classes=1,
            )
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor
        | list[
            torch.Tensor
        ],
    ]:
        output_size = (
            image.shape[-2:]
        )

        self.stage4_branch_tap.clear()

        features = list(
            self.backbone(
                image
            )
        )

        stage4_branches = (
            self.stage4_branch_tap.pop()
        )

        features[-1] = (
            self.branch_fusion(
                stage4_branches
            )
        )

        (
            prediction,
            auxiliary,
        ) = self.decoder(
            features=features,
            output_size=output_size,
        )

        outputs: dict[
            str,
            torch.Tensor
            | list[
                torch.Tensor
            ],
        ] = {
            "pred": prediction,
        }

        if auxiliary is not None:
            outputs["aux"] = [
                auxiliary
            ]

        return outputs


def build_model(
) -> CFMNetStage4AdaptiveFourBranchFusionUNetFormerSOD:
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
        CFMNetStage4AdaptiveFourBranchFusionUNetFormerSOD(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=64,
            window_size=8,
        )
    )
