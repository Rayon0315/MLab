# models/networks/cfmnet_stage4_complete_graph_attention_unetformer_sod.py

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
    CompleteGraphBranchAttention,
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


class CFMNetStage4CompleteGraphAttentionUNetFormerSOD(
    nn.Module
):
    """
    Stage-4 complete-graph branch interaction experiment.

    Stage 1/2/3:
        keep the standard fused CFMNet stage outputs.

    Stage 4:
        TCRM / SSRM / LDFM / EGPCM are four aligned branch tokens.
        At each spatial position a 4-token attention graph models all
        C(4, 2) branch pairs. Self edges are masked, so each branch only
        receives messages from the other three branches.

    No spatial token is moved or matched to another position.

    A zero-initialized residual scale is used for every branch, so the
    network starts exactly from the Stage4 raw-concat behavior and then
    learns pairwise branch interactions during training.
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

        self.branch_attention = (
            CompleteGraphBranchAttention(
                branch_channels=(
                    branch_channels
                ),
                num_heads=4,
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
            self.branch_attention(
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
) -> CFMNetStage4CompleteGraphAttentionUNetFormerSOD:
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
        CFMNetStage4CompleteGraphAttentionUNetFormerSOD(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=64,
            window_size=8,
        )
    )
