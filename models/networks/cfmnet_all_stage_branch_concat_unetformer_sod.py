# models/networks/cfmnet_all_stage_branch_concat_unetformer_sod.py

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
    BRANCH_ORDER,
    concat_stage_branches,
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


class AllStageBranchTap:
    """
    Capture TCRM / SSRM / LDFM / EGPCM from the last CFMBlock
    of all four paper stages.

    CFMNet layout:
        stages[0] -> Stage 1
        stages[2] -> Stage 2
        stages[4] -> Stage 3
        stages[6] -> Stage 4
    """

    STAGES = {
        "stage1": 0,
        "stage2": 2,
        "stage3": 4,
        "stage4": 6,
    }

    BACKBONE_NAMES = {
        "tcrm": "TCRM",
        "ssrm": "SSRM",
        "ldfm": "LDFM",
        "egpcm": "EGPCM",
    }

    def __init__(
        self,
        backbone: nn.Module,
    ) -> None:
        self._outputs: dict[
            str,
            dict[
                str,
                torch.Tensor,
            ],
        ] = {
            stage_name: {}
            for stage_name
            in self.STAGES
        }

        self._handles = []

        for (
            stage_name,
            stage_index,
        ) in self.STAGES.items():
            block = (
                backbone
                .stages[
                    stage_index
                ]
                .blocks[-1]
            )

            for branch_name in (
                BRANCH_ORDER
            ):
                self._register(
                    stage_name=(
                        stage_name
                    ),
                    branch_name=(
                        branch_name
                    ),
                    module=getattr(
                        block,
                        self.BACKBONE_NAMES[
                            branch_name
                        ],
                    ),
                )

    def _register(
        self,
        stage_name: str,
        branch_name: str,
        module: nn.Module,
    ) -> None:
        def hook(
            _module: nn.Module,
            _inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            self._outputs[
                stage_name
            ][
                branch_name
            ] = output

        self._handles.append(
            module.register_forward_hook(
                hook
            )
        )

    def clear(
        self,
    ) -> None:
        for stage_name in (
            self._outputs
        ):
            self._outputs[
                stage_name
            ].clear()

    def pop(
        self,
    ) -> list[
        dict[
            str,
            torch.Tensor,
        ]
    ]:
        stages = []

        for stage_name in (
            self.STAGES
        ):
            expected = set(
                BRANCH_ORDER
            )

            missing = (
                expected
                - self._outputs[
                    stage_name
                ].keys()
            )

            if missing:
                raise RuntimeError(
                    "Failed to capture "
                    f"{stage_name}: "
                    + ", ".join(
                        sorted(
                            missing
                        )
                    )
                )

            stages.append(
                self._outputs[
                    stage_name
                ]
            )

        self._outputs = {
            stage_name: {}
            for stage_name
            in self.STAGES
        }

        return stages


class CFMNetAllStageBranchConcatUNetFormerSOD(
    nn.Module
):
    """
    Experiment B:
        at every CFMNet stage, bypass the last CFMBlock's
        post-branch MLP/norm/residual representation used by the task head.

    For each stage:
        TCRM --\
        SSRM ----\
        LDFM ------> raw branch concat -> canonical stage width
        EGPCM ----/

    The resulting four stage tensors keep the normal CFMNet channel widths:
        96 / 192 / 384 / 768

    They are then fed unchanged into the existing UNetFormer decoder.
    No branch-role assignment, attention, adaptive weighting, or extra
    stage-wise projection is added before the normal decoder.
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

        self.backbone = (
            build_cfmnet(
                pretrained_path=(
                    pretrained_path
                )
            )
        )

        self.branch_tap = (
            AllStageBranchTap(
                self.backbone
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

        self.branch_tap.clear()

        # The canonical backbone forward is still executed so all
        # hooks observe the actual branch outputs with normal gradients.
        _ = self.backbone(
            image
        )

        branch_features = (
            self.branch_tap.pop()
        )

        stage_features = [
            concat_stage_branches(
                branches
            )
            for branches
            in branch_features
        ]

        for (
            feature,
            expected_channels,
        ) in zip(
            stage_features,
            CFMNET_OUT_CHANNELS,
        ):
            if (
                feature.shape[1]
                != expected_channels
            ):
                raise RuntimeError(
                    "Unexpected branch-concat "
                    "stage width: "
                    f"{feature.shape[1]} "
                    f"(expected "
                    f"{expected_channels})."
                )

        (
            prediction,
            auxiliary,
        ) = self.decoder(
            features=stage_features,
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
            outputs[
                "aux"
            ] = [
                auxiliary
            ]

        return outputs


def build_model(
) -> CFMNetAllStageBranchConcatUNetFormerSOD:
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
        CFMNetAllStageBranchConcatUNetFormerSOD(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=64,
            window_size=8,
        )
    )
