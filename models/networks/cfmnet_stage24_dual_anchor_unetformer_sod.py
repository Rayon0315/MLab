# models/networks/cfmnet_stage24_dual_anchor_unetformer_sod.py

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
from models.networks.cfmnet_unetformer_sod import (
    CFMNetUNetFormerDecoder,
)
from models.networks.cfmnet_stage2_cross_matching_unetformer_sod import (
    Stage2LocalCrossMatching,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "cfmnet"
    / "cfmnet_imagenet1k.pth"
)


BRANCH_ORDER = (
    "tcrm",
    "ssrm",
    "ldfm",
    "egpcm",
)


def concat_stage4_branches(
    branches: dict[
        str,
        torch.Tensor,
    ],
) -> torch.Tensor:
    """
    Preserve all four raw outputs of the last Stage-4 CFMBlock.

    CFMNet splits Stage4 channels equally across:
        TCRM / SSRM / LDFM / EGPCM

    Concatenation restores the canonical Stage4 width:
        C/4 + C/4 + C/4 + C/4 = C
    """

    return torch.cat(
        [
            branches[name]
            for name in BRANCH_ORDER
        ],
        dim=1,
    )


class Stage24BranchTap:
    """
    Capture only the two stages that already showed positive signals.

    Stage 2:
        branch features are used by the validated local cross matching.

    Stage 4:
        raw branch outputs replace only the decoder's Stage4 input.

    Stage 1 and Stage 3 remain untouched.
    """

    STAGES = {
        "stage2": 2,
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
    ) -> dict[
        str,
        dict[
            str,
            torch.Tensor,
        ],
    ]:
        expected = set(
            BRANCH_ORDER
        )

        for stage_name in (
            self.STAGES
        ):
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

        outputs = self._outputs

        self._outputs = {
            stage_name: {}
            for stage_name
            in self.STAGES
        }

        return outputs


class CFMNetStage24DualAnchorDecoder(
    CFMNetUNetFormerDecoder
):
    """
    Dual-anchor decoder.

    Stage 4 anchor:
        use raw [TCRM | SSRM | LDFM | EGPCM] as the deep decoder input.

    Stage 3:
        unchanged normal fused CFMNet feature and normal UNetFormer fusion.

    Stage 2 anchor:
        keep the normal fused Stage2 feature as the main path, then apply
        the already validated local cross matching:
            Q <- Stage3 decoder feature
            K <- Stage2 TCRM + EGPCM
            V <- Stage2 SSRM + LDFM

    Stage 1:
        unchanged normal UNetFormer FeatureRefinementHead.

    No new fusion or role assignment is introduced beyond combining the
    two independently positive interventions.
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
        dropout: float = 0.1,
        window_size: int = 8,
        num_classes: int = 1,
        matching_window: int = 7,
    ) -> None:
        super().__init__(
            encoder_channels=(
                encoder_channels
            ),
            decode_channels=(
                decode_channels
            ),
            dropout=dropout,
            window_size=(
                window_size
            ),
            num_classes=(
                num_classes
            ),
        )

        stage2_channels = (
            encoder_channels[1]
        )

        if (
            stage2_channels
            % 4
            != 0
        ):
            raise ValueError(
                "Stage2 channels must be "
                "divisible by four."
            )

        self.cross_matching = (
            Stage2LocalCrossMatching(
                branch_pair_channels=(
                    stage2_channels
                    // 2
                ),
                decode_channels=(
                    decode_channels
                ),
                window_size=(
                    matching_window
                ),
            )
        )

        self.last_matching_peak = None

    def forward(
        self,
        features: list[
            torch.Tensor
        ],
        branch_features: dict[
            str,
            dict[
                str,
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
        (
            stage1,
            stage2,
            stage3,
            _stage4_fused,
        ) = features

        stage4_raw = (
            concat_stage4_branches(
                branch_features[
                    "stage4"
                ]
            )
        )

        expected_stage4_channels = (
            CFMNET_OUT_CHANNELS[-1]
        )

        if (
            stage4_raw.shape[1]
            != expected_stage4_channels
        ):
            raise RuntimeError(
                "Unexpected Stage4 raw-concat "
                "channel count: "
                f"{stage4_raw.shape[1]} "
                f"(expected "
                f"{expected_stage4_channels})."
            )

        # Stage 4 anchor:
        # preserve the four raw CFM branches for the deep decoder input.
        x4 = self.block4(
            self.pre_conv(
                stage4_raw
            )
        )

        # Stage 3:
        # unchanged fused feature and unchanged UNetFormer top-down path.
        x3 = self.block3(
            self.fuse3(
                x4,
                stage3,
            )
        )

        # Stage 2 main path stays the normal fused Stage2 feature.
        x2_base = self.fuse2(
            x3,
            stage2,
        )

        # Stage 2 anchor:
        # local semantic-to-structure matching.
        (
            x2_matched,
            matching_peak,
        ) = self.cross_matching(
            stage2_base=(
                x2_base
            ),
            decoder_stage3=(
                x3
            ),
            branches=(
                branch_features[
                    "stage2"
                ]
            ),
        )

        self.last_matching_peak = (
            matching_peak.detach()
        )

        x2 = self.block2(
            x2_matched
        )

        # Stage 1 remains the standard UNetFormer refinement path.
        x1 = self.refine1(
            x2,
            stage1,
        )

        prediction = (
            self.segmentation_head(
                x1
            )
        )

        prediction = F.interpolate(
            prediction,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        auxiliary = None

        if self.training:
            aux_size = (
                x2.shape[-2:]
            )

            aux4 = F.interpolate(
                x4,
                size=aux_size,
                mode="bilinear",
                align_corners=False,
            )

            aux3 = F.interpolate(
                x3,
                size=aux_size,
                mode="bilinear",
                align_corners=False,
            )

            auxiliary = (
                self.aux_head(
                    aux4
                    + aux3
                    + x2,
                    output_size=(
                        output_size
                    ),
                )
            )

        return (
            prediction,
            auxiliary,
        )

    def matching_scale(
        self,
    ) -> torch.Tensor:
        return (
            self.cross_matching
            .scale
        )


class CFMNetStage24DualAnchorUNetFormerSOD(
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
        window_size: int = 8,
        matching_window: int = 7,
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
            Stage24BranchTap(
                self.backbone
            )
        )

        self.decoder = (
            CFMNetStage24DualAnchorDecoder(
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
                matching_window=(
                    matching_window
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

        features = self.backbone(
            image
        )

        branch_features = (
            self.branch_tap.pop()
        )

        (
            prediction,
            auxiliary,
        ) = self.decoder(
            features=features,
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
) -> CFMNetStage24DualAnchorUNetFormerSOD:
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
        CFMNetStage24DualAnchorUNetFormerSOD(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=64,
            window_size=8,
            matching_window=7,
        )
    )
