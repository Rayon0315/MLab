# models/networks/cfmnet_functional_cooperative_unet_sod.py

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


def _conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
) -> nn.Sequential:
    padding = kernel_size // 2

    return nn.Sequential(
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


def _conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
) -> nn.Sequential:
    padding = kernel_size // 2

    return nn.Sequential(
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


class FunctionalUpBlock(
    nn.Module
):
    """
    One U-Net-like top-down step inside one functional stream.

    deep:
        already projected to stream_channels

    skip:
        same CFM branch from the next shallower stage
    """

    def __init__(
        self,
        skip_channels: int,
        stream_channels: int,
    ) -> None:
        super().__init__()

        self.skip_proj = (
            _conv_bn_relu(
                skip_channels,
                stream_channels,
                kernel_size=1,
            )
        )

        self.refine = (
            DepthwiseSeparableRefine(
                stream_channels
            )
        )

    def forward(
        self,
        deep: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        deep = F.interpolate(
            deep,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        skip = self.skip_proj(
            skip
        )

        return self.refine(
            deep + skip
        )


class FunctionalPyramid(
    nn.Module
):
    """
    Module-centric U-Net stream:

        B4 -> up + B3
           -> up + B2
           -> up + B1

    B is one fixed CFMNet branch identity:
        TCRM / SSRM / LDFM / EGPCM
    """

    def __init__(
        self,
        branch_channels: tuple[
            int,
            int,
            int,
            int,
        ],
        stream_channels: int = 32,
    ) -> None:
        super().__init__()

        c1, c2, c3, c4 = (
            branch_channels
        )

        self.deep_proj = (
            _conv_bn_relu(
                c4,
                stream_channels,
                kernel_size=1,
            )
        )

        self.up3 = FunctionalUpBlock(
            skip_channels=c3,
            stream_channels=(
                stream_channels
            ),
        )

        self.up2 = FunctionalUpBlock(
            skip_channels=c2,
            stream_channels=(
                stream_channels
            ),
        )

        self.up1 = FunctionalUpBlock(
            skip_channels=c1,
            stream_channels=(
                stream_channels
            ),
        )

    def forward(
        self,
        features: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        f1, f2, f3, f4 = (
            features
        )

        x = self.deep_proj(
            f4
        )

        x = self.up3(
            x,
            f3,
        )

        x = self.up2(
            x,
            f2,
        )

        x = self.up1(
            x,
            f1,
        )

        return x


class CFMBranchTap:
    """
    Read the last CFMBlock's four branch outputs at all four stages.

    This does not alter CFMNet or duplicate branch computation.

    CFMNet stage layout:
        stages[0] = paper Stage 1
        stages[2] = paper Stage 2
        stages[4] = paper Stage 3
        stages[6] = paper Stage 4

    Each CFMBlock splits its stage channels equally among:
        TCRM / SSRM / LDFM / EGPCM.
    """

    MODULE_NAMES = (
        "tcrm",
        "ssrm",
        "ldfm",
        "egpcm",
    )

    BACKBONE_MODULE_NAMES = {
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
                    self.BACKBONE_MODULE_NAMES[
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

        handle = (
            module
            .register_forward_hook(
                hook
            )
        )

        self._handles.append(
            handle
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
    ) -> dict[
        str,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        result = {}

        for name in (
            self.MODULE_NAMES
        ):
            stage_outputs = (
                self._outputs[
                    name
                ]
            )

            if (
                len(
                    stage_outputs
                )
                != 4
            ):
                raise RuntimeError(
                    "Failed to capture all "
                    f"four {name.upper()} "
                    "stage outputs."
                )

            result[
                name
            ] = tuple(
                stage_outputs[
                    stage_number
                ]
                for stage_number
                in range(4)
            )

        self.clear()

        return result


class BidirectionalCooperation(
    nn.Module
):
    """
    Two-way interaction between:

        semantic/context:
            TCRM + EGPCM

        structure/detail:
            SSRM + LDFM

    Both residual strengths start from zero so neither side
    initially overrides the other.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.detail_to_semantic = (
            _conv_bn(
                channels,
                channels,
                kernel_size=1,
            )
        )

        self.semantic_to_detail = (
            _conv_bn(
                channels,
                channels,
                kernel_size=1,
            )
        )

        self.alpha = nn.Parameter(
            torch.zeros(1)
        )

        self.beta = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        semantic: torch.Tensor,
        detail: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        semantic_delta = (
            self.detail_to_semantic(
                detail
            )
        )

        detail_delta = (
            self.semantic_to_detail(
                semantic
            )
        )

        semantic_out = (
            semantic
            + torch.tanh(
                self.alpha
            )
            * semantic_delta
        )

        detail_out = (
            detail
            + torch.tanh(
                self.beta
            )
            * detail_delta
        )

        return (
            semantic_out,
            detail_out,
        )


class FunctionalCooperativeDecoder(
    nn.Module
):
    """
    CFMNet-specific SOD decoder.

    1. Preserve branch identity across scales:
        T4->T3->T2->T1
        S4->S3->S2->S1
        L4->L3->L2->L1
        G4->G3->G2->G1

    2. Build two task-level functional groups:
        TCRM + EGPCM -> semantic/context
        SSRM + LDFM -> structure/detail

    3. Perform bidirectional residual cooperation.

    4. Keep CFMNet's normal fused Stage-1 output as a
       residual representation path.

    5. Directly supervise the functional cooperative feature
       with an auxiliary SOD prediction.
    """

    def __init__(
        self,
        encoder_channels: tuple[
            int,
            int,
            int,
            int,
        ] = CFMNET_OUT_CHANNELS,
        stream_channels: int = 32,
        decode_channels: int = 64,
    ) -> None:
        super().__init__()

        branch_channels = tuple(
            channels // 4
            for channels
            in encoder_channels
        )

        self.tcrm_pyramid = (
            FunctionalPyramid(
                branch_channels=(
                    branch_channels
                ),
                stream_channels=(
                    stream_channels
                ),
            )
        )

        self.ssrm_pyramid = (
            FunctionalPyramid(
                branch_channels=(
                    branch_channels
                ),
                stream_channels=(
                    stream_channels
                ),
            )
        )

        self.ldfm_pyramid = (
            FunctionalPyramid(
                branch_channels=(
                    branch_channels
                ),
                stream_channels=(
                    stream_channels
                ),
            )
        )

        self.egpcm_pyramid = (
            FunctionalPyramid(
                branch_channels=(
                    branch_channels
                ),
                stream_channels=(
                    stream_channels
                ),
            )
        )

        pair_channels = (
            stream_channels
            * 2
        )

        self.semantic_fusion = (
            _conv_bn_relu(
                pair_channels,
                decode_channels,
                kernel_size=3,
            )
        )

        self.detail_fusion = (
            _conv_bn_relu(
                pair_channels,
                decode_channels,
                kernel_size=3,
            )
        )

        self.cooperation = (
            BidirectionalCooperation(
                decode_channels
            )
        )

        self.cooperative_fusion = (
            nn.Sequential(
                nn.Conv2d(
                    decode_channels * 2,
                    decode_channels * 2,
                    kernel_size=3,
                    padding=1,
                    groups=(
                        decode_channels
                        * 2
                    ),
                    bias=False,
                ),
                nn.BatchNorm2d(
                    decode_channels
                    * 2
                ),
                nn.ReLU(
                    inplace=True
                ),
                nn.Conv2d(
                    decode_channels * 2,
                    decode_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    decode_channels
                ),
                nn.ReLU(
                    inplace=True
                ),
            )
        )

        self.fused_stage1_proj = (
            _conv_bn_relu(
                encoder_channels[0],
                decode_channels,
                kernel_size=1,
            )
        )

        self.final_fusion = (
            nn.Sequential(
                nn.Conv2d(
                    decode_channels * 2,
                    decode_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    decode_channels
                ),
                nn.ReLU(
                    inplace=True
                ),
                DepthwiseSeparableRefine(
                    decode_channels
                ),
            )
        )

        self.pred_head = (
            nn.Conv2d(
                decode_channels,
                1,
                kernel_size=1,
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
        branch_features: dict[
            str,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ],
        output_size: tuple[
            int,
            int,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        stage1 = (
            fused_features[0]
        )

        tcrm = (
            self.tcrm_pyramid(
                branch_features[
                    "tcrm"
                ]
            )
        )

        ssrm = (
            self.ssrm_pyramid(
                branch_features[
                    "ssrm"
                ]
            )
        )

        ldfm = (
            self.ldfm_pyramid(
                branch_features[
                    "ldfm"
                ]
            )
        )

        egpcm = (
            self.egpcm_pyramid(
                branch_features[
                    "egpcm"
                ]
            )
        )

        semantic = (
            self.semantic_fusion(
                torch.cat(
                    [
                        tcrm,
                        egpcm,
                    ],
                    dim=1,
                )
            )
        )

        detail = (
            self.detail_fusion(
                torch.cat(
                    [
                        ssrm,
                        ldfm,
                    ],
                    dim=1,
                )
            )
        )

        (
            semantic,
            detail,
        ) = self.cooperation(
            semantic,
            detail,
        )

        cooperative = (
            self.cooperative_fusion(
                torch.cat(
                    [
                        semantic,
                        detail,
                    ],
                    dim=1,
                )
            )
        )

        fused_stage1 = (
            self.fused_stage1_proj(
                stage1
            )
        )

        final_feature = (
            self.final_fusion(
                torch.cat(
                    [
                        cooperative,
                        fused_stage1,
                    ],
                    dim=1,
                )
            )
        )

        prediction = (
            self.pred_head(
                final_feature
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
                    cooperative
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


class CFMNetFunctionalCooperativeUNetSOD(
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
        stream_channels: int = 32,
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
            FunctionalCooperativeDecoder(
                encoder_channels=(
                    CFMNET_OUT_CHANNELS
                ),
                stream_channels=(
                    stream_channels
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
) -> CFMNetFunctionalCooperativeUNetSOD:
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
        CFMNetFunctionalCooperativeUNetSOD(
            pretrained_path=(
                pretrained_path
            ),
            stream_channels=32,
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
        "cooperation alpha:",
        float(
            model
            .decoder
            .cooperation
            .alpha
            .detach()
            .cpu()
        ),
    )

    print(
        "cooperation beta:",
        float(
            model
            .decoder
            .cooperation
            .beta
            .detach()
            .cpu()
        ),
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
