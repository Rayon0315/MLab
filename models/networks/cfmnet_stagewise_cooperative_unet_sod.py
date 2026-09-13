# models/networks/cfmnet_stagewise_cooperative_unet_sod.py

from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
    build_cfmnet,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "cfmnet"
    / "cfmnet_imagenet1k.pth"
)


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
    ) -> None:
        padding = kernel_size // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )


class DepthwiseSeparableRefine(
    nn.Module
):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class CFMBranchTap:
    """
    Capture TCRM / SSRM / LDFM / EGPCM from the last CFMBlock
    of each paper stage.

    CFMNet stages are stored as:
        stages[0] -> Stage 1
        stages[2] -> Stage 2
        stages[4] -> Stage 3
        stages[6] -> Stage 4

    No branch is recomputed and the pretrained backbone code
    does not need to be modified.
    """

    MODULE_NAMES = (
        "tcrm",
        "ssrm",
        "ldfm",
        "egpcm",
    )

    BACKBONE_NAMES = {
        "tcrm": "TCRM",
        "ssrm": "SSRM",
        "ldfm": "LDFM",
        "egpcm": "EGPCM",
    }

    STAGE_INDICES = (
        0,
        2,
        4,
        6,
    )

    def __init__(
        self,
        backbone: nn.Module,
    ) -> None:
        self._outputs: dict[
            str,
            dict[
                int,
                torch.Tensor,
            ],
        ] = {
            name: {}
            for name
            in self.MODULE_NAMES
        }

        self._handles = []

        for (
            stage_number,
            stage_index,
        ) in enumerate(
            self.STAGE_INDICES
        ):
            stage = (
                backbone
                .stages[
                    stage_index
                ]
            )

            block = (
                stage
                .blocks[-1]
            )

            for name in (
                self.MODULE_NAMES
            ):
                module = getattr(
                    block,
                    self.BACKBONE_NAMES[
                        name
                    ],
                )

                self._register(
                    name=name,
                    stage_number=(
                        stage_number
                    ),
                    module=module,
                )

    def _register(
        self,
        name: str,
        stage_number: int,
        module: nn.Module,
    ) -> None:
        def hook(
            _module: nn.Module,
            _inputs: tuple,
            output: torch.Tensor,
        ) -> None:
            self._outputs[
                name
            ][
                stage_number
            ] = output

        self._handles.append(
            module.register_forward_hook(
                hook
            )
        )

    def clear(
        self,
    ) -> None:
        for name in (
            self.MODULE_NAMES
        ):
            self._outputs[
                name
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

        for stage_number in range(
            4
        ):
            stage_output = {}

            for name in (
                self.MODULE_NAMES
            ):
                if (
                    stage_number
                    not in self._outputs[
                        name
                    ]
                ):
                    raise RuntimeError(
                        "Failed to capture "
                        f"{name.upper()} "
                        f"for Stage "
                        f"{stage_number + 1}."
                    )

                stage_output[
                    name
                ] = (
                    self._outputs[
                        name
                    ][
                        stage_number
                    ]
                )

            stages.append(
                stage_output
            )

        self.clear()

        return stages


class StageLocalCooperation(
    nn.Module
):
    """
    Local CFM cooperation inside one semantic stage.

    TCRM + EGPCM
        -> semantic/context

    SSRM + LDFM
        -> structure/detail

    semantic/context + structure/detail
        -> cooperative feature

    The normal fused CFMNet stage feature remains the main path:

        stage_feature
            = Proj(fused_stage)
            + alpha * cooperative_feature

    alpha starts from zero.
    """

    def __init__(
        self,
        stage_channels: int,
        decode_channels: int = 64,
    ) -> None:
        super().__init__()

        if (
            stage_channels
            % 4
            != 0
        ):
            raise ValueError(
                "CFMNet stage channels "
                "must be divisible by four."
            )

        branch_channels = (
            stage_channels
            // 4
        )

        pair_channels = (
            branch_channels
            * 2
        )

        self.semantic_fusion = (
            ConvBNReLU(
                pair_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.detail_fusion = (
            ConvBNReLU(
                pair_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.cooperative_reduce = (
            ConvBNReLU(
                decode_channels
                * 2,
                decode_channels,
                kernel_size=1,
            )
        )

        self.cooperative_refine = (
            DepthwiseSeparableRefine(
                decode_channels
            )
        )

        self.fused_proj = (
            ConvBNReLU(
                stage_channels,
                decode_channels,
                kernel_size=1,
            )
        )

        self.alpha = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        fused_stage: torch.Tensor,
        branches: dict[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        semantic = torch.cat(
            [
                branches[
                    "tcrm"
                ],
                branches[
                    "egpcm"
                ],
            ],
            dim=1,
        )

        detail = torch.cat(
            [
                branches[
                    "ssrm"
                ],
                branches[
                    "ldfm"
                ],
            ],
            dim=1,
        )

        semantic = (
            self.semantic_fusion(
                semantic
            )
        )

        detail = (
            self.detail_fusion(
                detail
            )
        )

        cooperative = (
            self.cooperative_reduce(
                torch.cat(
                    [
                        semantic,
                        detail,
                    ],
                    dim=1,
                )
            )
        )

        cooperative = (
            self.cooperative_refine(
                cooperative
            )
        )

        base = (
            self.fused_proj(
                fused_stage
            )
        )

        return (
            base
            + self.alpha
            * cooperative
        )


class UNetTopDownBlock(
    nn.Module
):
    """
    Plain U-Net-like cross-stage fusion.

    No attention, no gate, no transformer:
        shallower_stage + upsample(deeper_decoder)
            -> lightweight refinement
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.refine = (
            DepthwiseSeparableRefine(
                channels
            )
        )

    def forward(
        self,
        shallow: torch.Tensor,
        deep: torch.Tensor,
    ) -> torch.Tensor:
        deep = F.interpolate(
            deep,
            size=shallow.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return self.refine(
            shallow
            + deep
        )


class StagewiseCooperativeUNetDecoder(
    nn.Module
):
    """
    Stage-wise local CFM cooperation + plain U-Net top-down decoding.

    Step 1:
        At every stage, explicitly recover CFMNet's four-module
        cooperation locally.

    Step 2:
        Keep the normal fused CFMNet feature as the stable main path.

    Step 3:
        Decode the four resulting stage features with a simple
        U-Net top-down hierarchy.

    This preserves:
        - module identity inside a stage
        - hierarchy identity across stages
    """

    def __init__(
        self,
        encoder_channels: tuple[
            int,
            int,
            int,
            int,
        ] = CFMNET_OUT_CHANNELS,
        decode_channels: int = 64,
    ) -> None:
        super().__init__()

        self.stage1_coop = (
            StageLocalCooperation(
                encoder_channels[0],
                decode_channels,
            )
        )

        self.stage2_coop = (
            StageLocalCooperation(
                encoder_channels[1],
                decode_channels,
            )
        )

        self.stage3_coop = (
            StageLocalCooperation(
                encoder_channels[2],
                decode_channels,
            )
        )

        self.stage4_coop = (
            StageLocalCooperation(
                encoder_channels[3],
                decode_channels,
            )
        )

        self.deep_refine = (
            DepthwiseSeparableRefine(
                decode_channels
            )
        )

        self.up3 = UNetTopDownBlock(
            decode_channels
        )

        self.up2 = UNetTopDownBlock(
            decode_channels
        )

        self.up1 = UNetTopDownBlock(
            decode_channels
        )

        self.pred_head = (
            nn.Sequential(
                DepthwiseSeparableRefine(
                    decode_channels
                ),
                nn.Conv2d(
                    decode_channels,
                    1,
                    kernel_size=1,
                ),
            )
        )

        self.aux_head = (
            nn.Conv2d(
                decode_channels,
                1,
                kernel_size=1,
            )
        )

    def forward(
        self,
        fused_features: list[
            torch.Tensor
        ],
        branch_features: list[
            dict[
                str,
                torch.Tensor,
            ]
        ],
        output_size: tuple[
            int,
            int,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        (
            fused1,
            fused2,
            fused3,
            fused4,
        ) = fused_features

        (
            branches1,
            branches2,
            branches3,
            branches4,
        ) = branch_features

        stage1 = (
            self.stage1_coop(
                fused1,
                branches1,
            )
        )

        stage2 = (
            self.stage2_coop(
                fused2,
                branches2,
            )
        )

        stage3 = (
            self.stage3_coop(
                fused3,
                branches3,
            )
        )

        stage4 = (
            self.stage4_coop(
                fused4,
                branches4,
            )
        )

        d4 = self.deep_refine(
            stage4
        )

        d3 = self.up3(
            stage3,
            d4,
        )

        d2 = self.up2(
            stage2,
            d3,
        )

        d1 = self.up1(
            stage1,
            d2,
        )

        prediction = (
            self.pred_head(
                d1
            )
        )

        prediction = (
            F.interpolate(
                prediction,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        )

        auxiliary = None

        if self.training:
            auxiliary = (
                self.aux_head(
                    d2
                )
            )

            auxiliary = (
                F.interpolate(
                    auxiliary,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        return (
            prediction,
            auxiliary,
        )

    def cooperation_scales(
        self,
    ) -> list[
        torch.Tensor
    ]:
        return [
            self.stage1_coop.alpha,
            self.stage2_coop.alpha,
            self.stage3_coop.alpha,
            self.stage4_coop.alpha,
        ]


class CFMNetStagewiseCooperativeUNetSOD(
    nn.Module
):
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

        self.branch_tap = (
            CFMBranchTap(
                self.backbone
            )
        )

        self.decoder = (
            StagewiseCooperativeUNetDecoder(
                encoder_channels=(
                    CFMNET_OUT_CHANNELS
                ),
                decode_channels=(
                    decode_channels
                ),
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

        fused_features = (
            self.backbone(
                image
            )
        )

        branch_features = (
            self.branch_tap.pop()
        )

        (
            prediction,
            auxiliary,
        ) = self.decoder(
            fused_features=(
                fused_features
            ),
            branch_features=(
                branch_features
            ),
            output_size=(
                output_size
            ),
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
) -> CFMNetStagewiseCooperativeUNetSOD:
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
        CFMNetStagewiseCooperativeUNetSOD(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=64,
        )
    )


if __name__ == "__main__":
    model = build_model()

    image = torch.randn(
        2,
        3,
        352,
        352,
    )

    model.train()

    outputs = model(
        image=image
    )

    print(
        "train pred:",
        tuple(
            outputs[
                "pred"
            ].shape
        ),
    )

    print(
        "train aux:",
        [
            tuple(
                tensor.shape
            )
            for tensor
            in outputs.get(
                "aux",
                []
            )
        ],
    )

    print(
        "stage alpha:",
        [
            float(
                alpha
                .detach()
                .cpu()
            )
            for alpha
            in model
            .decoder
            .cooperation_scales()
        ],
    )

    model.eval()

    with torch.no_grad():
        outputs = model(
            image=image
        )

    print(
        "eval pred:",
        tuple(
            outputs[
                "pred"
            ].shape
        ),
    )
