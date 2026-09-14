from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.components.hard_background_verifier import (
    HardBackgroundVerifier,
)
from models.components.relative_saliency_candidate_verifier import (
    RelativeSaliencyCandidateVerifier,
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


class VMambaSmallFPNNAMLabPyramidCFPSRSCVSOD(
    VMambaSmallFPNNAMLabHybridSOD
):
    """
    VMamba-S
        + NAMLab Hybrid
        + Pure FPN
        + Pyramid Context
        + CFPS
        + Relative Saliency Candidate Verification (RSCV)

    Responsibilities:
        Pyramid Context:
            improve high-level multi-scale target representation.

        CFPS:
            pixel/region-level hard-background suppression using
            pre-NAM raw Stage2/Stage3 counter-evidence.

        RSCV:
            candidate-level relative saliency verification for
            semantically complete distractor objects that survive CFPS.

    RSCV uses decoded Stage3 as its high-level semantic feature and
    discovers Top-K candidates from the CFPS prediction.
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

        self.background_verifier = (
            HardBackgroundVerifier(
                stage2_channels=(
                    self.backbone.out_channels[1]
                ),
                stage3_channels=(
                    self.backbone.out_channels[2]
                ),
                evidence_channels=64,
                hidden_channels=decoder_channels,
                confidence_power=2.0,
                max_suppression=4.0,
                initial_suppression_scale=0.25,
            )
        )

        self.candidate_verifier = (
            RelativeSaliencyCandidateVerifier(
                channels=decoder_channels,
                num_candidates=4,
                num_heads=4,
                ff_channels=256,
                nms_kernel=5,
                similarity_temperature=0.2,
                initial_accept_bias=2.0,
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
            raw_stage1,
            raw_stage2,
            raw_stage3,
            raw_stage4,
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
                raw_stage1,
                raw_stage2,
                raw_stage3,
                raw_stage4,
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

        coarse_logits = self.pred1(
            decoded1
        )

        # -------------------------------------------------
        # 1. CFPS: hard-background counter-evidence
        # -------------------------------------------------
        verification = self.background_verifier(
            raw_stage2=raw_stage2,
            raw_stage3=raw_stage3,
            coarse_logits=coarse_logits,
        )

        cfps_logits = verification[
            "refined_logits"
        ]

        # -------------------------------------------------
        # 2. RSCV: candidate-level relative saliency
        # -------------------------------------------------
        candidate_verification = (
            self.candidate_verifier(
                semantic_feature=decoded3,
                saliency_logits=cfps_logits,
            )
        )

        final_logits = (
            candidate_verification[
                "refined_logits"
            ]
        )

        prediction = F.interpolate(
            final_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        coarse_prediction = F.interpolate(
            coarse_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        cfps_prediction = F.interpolate(
            cfps_logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        fp_risk_logits = F.interpolate(
            verification[
                "fp_risk_logits"
            ],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        cfps_suppression = F.interpolate(
            verification[
                "suppression"
            ],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        candidate_suppression = F.interpolate(
            candidate_verification[
                "candidate_suppression"
            ],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": prediction,
            "coarse_pred": coarse_prediction,
            "cfps_pred": cfps_prediction,
            "fp_risk": fp_risk_logits,
            "suppression": cfps_suppression,
            "candidate_logits": (
                candidate_verification[
                    "candidate_logits"
                ]
            ),
            "candidate_regions": (
                candidate_verification[
                    "candidate_regions"
                ]
            ),
            "candidate_peak_scores": (
                candidate_verification[
                    "candidate_peak_scores"
                ]
            ),
            "candidate_suppression": (
                candidate_suppression
            ),
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> VMambaSmallFPNNAMLabPyramidCFPSRSCVSOD:
    return VMambaSmallFPNNAMLabPyramidCFPSRSCVSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )
