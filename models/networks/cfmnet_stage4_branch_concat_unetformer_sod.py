# models/networks/cfmnet_stage4_branch_concat_unetformer_sod.py

from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
    build_cfmnet,
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


class Stage4BranchTap:
    """
    Capture TCRM / SSRM / LDFM / EGPCM outputs from the last
    CFMBlock of paper Stage 4.

    CFMNet stores the four paper stages at:
        stages[0], stages[2], stages[4], stages[6]

    Only Stage 4 is tapped in this experiment. The standard fused
    Stage 1/2/3 backbone outputs are kept unchanged.
    """

    BRANCHES = {
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
            torch.Tensor,
        ] = {}

        self._handles = []

        stage4_last_block = (
            backbone
            .stages[6]
            .blocks[-1]
        )

        for (
            branch_name,
            module_name,
        ) in self.BRANCHES.items():
            self._register(
                branch_name=branch_name,
                module=getattr(
                    stage4_last_block,
                    module_name,
                ),
            )

    def _register(
        self,
        branch_name: str,
        module: nn.Module,
    ) -> None:
        def hook(
            _module: nn.Module,
            _inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            self._outputs[
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
        self._outputs.clear()

    def pop(
        self,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        expected = set(
            self.BRANCHES
        )

        missing = (
            expected
            - self._outputs.keys()
        )

        if missing:
            raise RuntimeError(
                "Failed to capture Stage4 CFM branches: "
                + ", ".join(
                    sorted(missing)
                )
            )

        outputs = self._outputs
        self._outputs = {}

        return outputs


class CFMNetStage4BranchConcatUNetFormerSOD(
    nn.Module
):
    """
    CFMNet + UNetFormer SOD experiment.

    Stage 1/2/3:
        use the original fused CFMNet stage outputs.

    Stage 4:
        take the four raw outputs of the last CFMBlock:
            TCRM | SSRM | LDFM | EGPCM
        and concatenate them directly.

    Because CFMBlock splits channels equally, the concatenated Stage4
    tensor still has the original Stage4 channel count (768 for the
    current CFMNet). It is therefore sent directly into the existing
    UNetFormer decoder without an additional fusion/projection module.

    Important:
        The decoder's own original pre_conv (768 -> decode_channels)
        is intentionally retained. The removed operation is CFMBlock's
        post-concat MLP/norm/residual fusion for the Stage4 feature used
        by the decoder.
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

        self.stage4_branch_tap = (
            Stage4BranchTap(
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

    @staticmethod
    def _concat_stage4_branches(
        branches: dict[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        return torch.cat(
            [
                branches["tcrm"],
                branches["ssrm"],
                branches["ldfm"],
                branches["egpcm"],
            ],
            dim=1,
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

        stage4_concat = (
            self._concat_stage4_branches(
                stage4_branches
            )
        )

        expected_channels = (
            CFMNET_OUT_CHANNELS[-1]
        )

        if (
            stage4_concat.shape[1]
            != expected_channels
        ):
            raise RuntimeError(
                "Unexpected Stage4 branch-concat channels: "
                f"{stage4_concat.shape[1]} "
                f"(expected {expected_channels})."
            )

        # Only replace the Stage4 representation consumed by the decoder.
        # Stage1/2/3 remain the standard fused CFMNet outputs.
        features[-1] = (
            stage4_concat
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
) -> CFMNetStage4BranchConcatUNetFormerSOD:
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
        CFMNetStage4BranchConcatUNetFormerSOD(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=64,
            window_size=8,
        )
    )
