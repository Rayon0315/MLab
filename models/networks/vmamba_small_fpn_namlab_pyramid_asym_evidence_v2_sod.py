from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.components.asymmetric_foreground_support_v2 import (
    AsymmetricForegroundSupportV2,
)
from models.components.foreground_aware_background_verifier import (
    ForegroundAwareBackgroundVerifier,
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


class VMambaSmallFPNNAMLabPyramidAsymEvidenceV2SOD(
    VMambaSmallFPNNAMLabHybridSOD
):
    """
    Asymmetric FG/BG Evidence v2.

    FG:
        semantically supervised positive support;
        allowed to shape the main saliency representation.

    BG:
        detached counter-evidence verifier;
        only learns rejection / false-positive risk;
        cannot reshape the FG metric space through its auxiliary loss.

    The v1 BG verifier is deliberately unchanged so this experiment isolates
    the effect of semantic anchoring of FG support.
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

        self.foreground_support = (
            AsymmetricForegroundSupportV2(
                channels=decoder_channels,
                metric_channels=64,
                initial_support_scale=4.0,
                initial_residual_scale=0.0,
                eps=1e-6,
                detach_guide=True,
            )
        )

        self.background_verifier = (
            ForegroundAwareBackgroundVerifier(
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

        fg = self.foreground_support(
            high_feature=decoded3,
            low_feature=decoded1,
            guide_logits=prediction3,
        )

        coarse_logits = self.pred1(
            fg["refined_feature"]
        )

        bg = self.background_verifier(
            raw_stage2=raw_stage2,
            raw_stage3=raw_stage3,
            coarse_logits=coarse_logits,
            fg_support=fg["fg_support"],
        )

        final_logits = bg[
            "refined_logits"
        ]

        def upsample(
            tensor: torch.Tensor,
        ) -> torch.Tensor:
            return F.interpolate(
                tensor,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        return {
            "pred": upsample(
                final_logits
            ),
            "coarse_pred": upsample(
                coarse_logits
            ),
            "fg_similarity": upsample(
                fg["fg_similarity"]
            ),
            "fg_support_logits": upsample(
                fg[
                    "fg_support_logits"
                ]
            ),
            "fg_support": upsample(
                fg["fg_support"]
            ),
            "fg_guide_probability": upsample(
                fg[
                    "guide_foreground_probability"
                ]
            ),
            "fp_risk": upsample(
                bg["fp_risk_logits"]
            ),
            "suppression_strength": upsample(
                bg[
                    "suppression_strength"
                ]
            ),
            "suppression": upsample(
                bg["suppression"]
            ),
            "prediction_support_mismatch": upsample(
                bg[
                    "prediction_support_mismatch"
                ]
            ),
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> VMambaSmallFPNNAMLabPyramidAsymEvidenceV2SOD:
    return VMambaSmallFPNNAMLabPyramidAsymEvidenceV2SOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )
