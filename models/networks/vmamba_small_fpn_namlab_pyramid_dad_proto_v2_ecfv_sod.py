from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.components.dad_conditioned_background_verifier import (
    DADConditionedBackgroundVerifier,
)
from models.components.dad_prototype_v2 import (
    DifferenceAwarePrototypeGenerationV2,
)
from models.components.sod_blocks import (
    ConvNormAct,
    PyramidContextBlock,
)
from models.networks.vmamba_small_fpn_namlab_hybrid_sod import (
    VMambaSmallFPNNAMLabHybridSOD,
)
from models.networks.vmamba_small_fpn_sod import (
    PRETRAINED_PATH,
)


class VMambaSmallFPNNAMLabPyramidDADProtoV2ECFVSOD(
    VMambaSmallFPNNAMLabHybridSOD
):
    """
    VMamba-S + NAMLab + Pyramid FPN
    + DAD-v2
    + DAD-conditioned counter-evidence verifier (ECFV).

    Compared with the failed direct DAD-v2 + CFPS combination:

        1. DAD-v2 remains the main FG/BG evidence mechanism.

        2. CFPS no longer works as an independent classifier only from
           raw features + coarse confidence. It additionally reads:
               - DAD ambiguity
               - DAD-guide agreement

        3. All main-network inputs entering the verifier are detached:
               raw Stage2 / Stage3
               coarse confidence
               DAD margin
               DAD guide
           Therefore FP-risk supervision cannot reshape the shared
           backbone / decoder / DAD feature geometry.

        4. The original FP loss and suppression equation are unchanged.

    This experiment tests whether coordinating the two evidence sources,
    while isolating verifier gradients, preserves the Pure DAD-v2 gain.
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

        self.dad_prototype_v2 = (
            DifferenceAwarePrototypeGenerationV2(
                temperature=1.0,
                eps=1e-6,
                detach_guide=True,
            )
        )

        # Keep Pure DAD-v2 outer fusion unchanged.
        self.dad_prototype_fusion = ConvNormAct(
            decoder_channels * 2,
            decoder_channels,
            kernel_size=3,
        )

        self.background_verifier = (
            DADConditionedBackgroundVerifier(
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

        dad = self.dad_prototype_v2(
            high_feature=decoded3,
            low_feature=decoded1,
            guide_logits=prediction3,
        )

        prototype_feature = (
            self.dad_prototype_fusion(
                torch.cat(
                    [
                        dad[
                            "foreground_feature"
                        ],
                        dad[
                            "background_feature"
                        ],
                    ],
                    dim=1,
                )
            )
        )

        coarse_logits = self.pred1(
            prototype_feature
        )

        verification = (
            self.background_verifier(
                raw_stage2=raw_stage2,
                raw_stage3=raw_stage3,
                coarse_logits=coarse_logits,
                dad_semantic_margin=dad[
                    "semantic_margin"
                ],
                dad_guide_logits=prediction3,
            )
        )

        refined_logits = verification[
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
                refined_logits
            ),
            "coarse_pred": upsample(
                coarse_logits
            ),
            "fp_risk": upsample(
                verification[
                    "fp_risk_logits"
                ]
            ),
            "suppression": upsample(
                verification[
                    "suppression"
                ]
            ),
            "suppression_strength": upsample(
                verification[
                    "suppression_strength"
                ]
            ),
            "dad_fg_evidence": upsample(
                dad[
                    "foreground_evidence"
                ]
            ),
            "dad_bg_evidence": upsample(
                dad[
                    "background_evidence"
                ]
            ),
            "dad_semantic_margin": upsample(
                dad[
                    "semantic_margin"
                ]
            ),
            "dad_ambiguity": upsample(
                verification[
                    "dad_ambiguity"
                ]
            ),
            "dad_agreement": upsample(
                verification[
                    "dad_agreement"
                ]
            ),
            "dad_guide_probability": upsample(
                verification[
                    "guide_probability"
                ]
            ),
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> VMambaSmallFPNNAMLabPyramidDADProtoV2ECFVSOD:
    return VMambaSmallFPNNAMLabPyramidDADProtoV2ECFVSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VMamba smoke test requires CUDA."
        )

    device = torch.device(
        "cuda"
    )

    model = build_model().to(
        device
    )
    model.eval()

    image = torch.randn(
        1,
        3,
        352,
        352,
        device=device,
    )
    mean_60 = torch.rand(
        1,
        3,
        352,
        352,
        device=device,
    )

    with torch.no_grad():
        outputs = model(
            image=image,
            mean_60=mean_60,
        )

    for key, value in outputs.items():
        if isinstance(
            value,
            torch.Tensor,
        ):
            print(
                key,
                tuple(value.shape),
            )
        else:
            print(
                key,
                [
                    tuple(t.shape)
                    for t in value
                ],
            )
