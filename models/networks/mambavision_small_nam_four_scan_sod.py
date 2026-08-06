# models/networks/mambavision_small_nam_four_scan_sod.py
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.mambavision_nam_four_scan import (
    NAMFourScanMambaVisionBackbone,
    ScanMode,
    mamba_vision_small_nam_four_scan,
)
from models.backbones.mambavision_nam_scan import (
    VALID_SCAN_MODES,
)
from models.components.sod_blocks import (
    BoundaryRefinementBlock,
    ConvNormAct,
    PredictionHead,
    PyramidContextBlock,
    SaliencyGuidedFusion,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "mambavision"
    / "mambavision_small_1k.pth.tar"
)


def read_boolean_environment(
    name: str,
    default: bool = False,
) -> bool:
    value = os.environ.get(
        name
    )

    if value is None:
        return default

    normalized = (
        value
        .strip()
        .lower()
    )

    if normalized in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True

    if normalized in (
        "0",
        "false",
        "no",
        "off",
    ):
        return False

    raise ValueError(
        "Invalid boolean environment "
        f"variable {name}={value}"
    )


class MambaVisionSmallNAMFourScanSOD(
    nn.Module
):
    input_keys = (
        "image",
        "nam_20",
        "nam_40",
        "nam_60",
    )

    def __init__(
        self,
        pretrained_path: (
            str | Path | None
        ),
        decoder_channels: int = 128,
        scan_mode: ScanMode = (
            "nam_hierarchical"
        ),
        debug_validate_permutations: (
            bool
        ) = False,
    ) -> None:
        super().__init__()

        self.scan_mode = scan_mode

        self.backbone: (
            NAMFourScanMambaVisionBackbone
        ) = (
            mamba_vision_small_nam_four_scan(
                pretrained_path=(
                    pretrained_path
                ),
                scan_mode=scan_mode,
                debug_validate_permutations=(
                    debug_validate_permutations
                ),
            )
        )

        self.projections = (
            nn.ModuleList(
                [
                    ConvNormAct(
                        in_channels,
                        decoder_channels,
                        kernel_size=1,
                        padding=0,
                    )
                    for in_channels
                    in self.backbone.out_channels
                ]
            )
        )

        self.context4 = (
            PyramidContextBlock(
                decoder_channels
            )
        )

        self.pred4 = PredictionHead(
            decoder_channels
        )

        self.fusion3 = (
            SaliencyGuidedFusion(
                decoder_channels
            )
        )

        self.pred3 = PredictionHead(
            decoder_channels
        )

        self.fusion2 = (
            SaliencyGuidedFusion(
                decoder_channels
            )
        )

        self.pred2 = PredictionHead(
            decoder_channels
        )

        self.fusion1 = (
            SaliencyGuidedFusion(
                decoder_channels
            )
        )

        self.boundary_refinement = (
            BoundaryRefinementBlock(
                decoder_channels
            )
        )

        self.pred1 = PredictionHead(
            decoder_channels
        )

    def forward(
        self,
        image: torch.Tensor,
        nam_20: torch.Tensor,
        nam_40: torch.Tensor,
        nam_60: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor
        | list[torch.Tensor],
    ]:
        input_size = (
            image.shape[-2:]
        )

        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = self.backbone(
            image=image,
            nam_20=nam_20,
            nam_40=nam_40,
            nam_60=nam_60,
        )

        (
            feature1,
            feature2,
            feature3,
            feature4,
        ) = [
            projection(
                feature
            )
            for (
                projection,
                feature,
            ) in zip(
                self.projections,
                (
                    stage1,
                    stage2,
                    stage3,
                    stage4,
                ),
            )
        ]

        decoded4 = self.context4(
            feature4
        )

        prediction4 = self.pred4(
            decoded4
        )

        decoded3 = self.fusion3(
            low_feature=feature3,
            high_feature=decoded4,
            guide_logits=prediction4,
        )

        prediction3 = self.pred3(
            decoded3
        )

        decoded2 = self.fusion2(
            low_feature=feature2,
            high_feature=decoded3,
            guide_logits=prediction3,
        )

        prediction2 = self.pred2(
            decoded2
        )

        decoded1 = self.fusion1(
            low_feature=feature1,
            high_feature=decoded2,
            guide_logits=prediction2,
        )

        decoded1 = (
            self.boundary_refinement(
                shallow_feature=feature1,
                semantic_feature=feature2,
                saliency_feature=decoded1,
            )
        )

        prediction1 = self.pred1(
            decoded1
        )

        prediction1 = F.interpolate(
            prediction1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": prediction1,
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> MambaVisionSmallNAMFourScanSOD:
    scan_mode = os.environ.get(
        "MLAB_NAM_SCAN_MODE",
        "nam_hierarchical",
    ).strip()

    if (
        scan_mode
        not in VALID_SCAN_MODES
    ):
        choices = ", ".join(
            VALID_SCAN_MODES
        )

        raise ValueError(
            "Invalid "
            f"MLAB_NAM_SCAN_MODE={scan_mode}. "
            f"Expected one of: {choices}"
        )

    debug_validate = (
        read_boolean_environment(
            "MLAB_NAM_SCAN_DEBUG",
            default=False,
        )
    )

    print(
        "MambaVision NAM four-path network | "
        f"mode={scan_mode} | "
        "validate_permutations="
        f"{debug_validate}"
    )

    return (
        MambaVisionSmallNAMFourScanSOD(
            pretrained_path=(
                PRETRAINED_PATH
            ),
            decoder_channels=128,
            scan_mode=scan_mode,
            debug_validate_permutations=(
                debug_validate
            ),
        )
    )