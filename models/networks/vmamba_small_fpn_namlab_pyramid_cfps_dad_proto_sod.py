from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.components.dad_prototype import (
    DifferenceAwarePrototypeGeneration,
)
from models.components.hard_background_verifier import (
    HardBackgroundVerifier,
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


class VMambaSmallFPNNAMLabPyramidCFPSDADProtoSOD(
    VMambaSmallFPNNAMLabHybridSOD
):
    """
    VMamba-S + NAMLab Hybrid + Pyramid FPN
    + DAD FG/BG prototype evidence
    + CFPS hard-background branch.

    DAD evidence path:
        decoded3 + prediction3
            -> foreground/background prototypes
            -> cosine similarity against decoded1
            -> FG-aware / BG-aware low-level features
            -> concat + 3x3 fusion
            -> coarse saliency

    CFPS path is kept unchanged:
        raw Stage2 / Stage3 BEFORE NAMLab
            -> HardBackgroundVerifier
            -> FP-risk supervision
            -> one-way suppression

    Only the Difference-Aware Prototype Generation part of DAD is
    introduced here. The overlapped-window cross-level guidance from
    DAD is intentionally excluded for a clean evidence ablation.
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

        self.dad_prototype = (
            DifferenceAwarePrototypeGeneration()
        )

        # DAD produces two prototype-aware low-level branches:
        # foreground-aware Ff and background-aware Fb.
        # Fuse only these two branches so this experiment measures
        # the prototype evidence itself instead of adding a third
        # unchanged bypass branch.
        self.dad_prototype_fusion = ConvNormAct(
            decoder_channels * 2,
            decoder_channels,
            kernel_size=3,
        )

        self.background_verifier = HardBackgroundVerifier(
            stage2_channels=self.backbone.out_channels[1],
            stage3_channels=self.backbone.out_channels[2],
            evidence_channels=64,
            hidden_channels=decoder_channels,
            confidence_power=2.0,
            max_suppression=4.0,
            initial_suppression_scale=0.25,
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

        # DAD mapping:
        #   Fh <- decoded3      (high-level semantic feature)
        #   C0 <- prediction3  (already aux-supervised coarse guide)
        #   Fl <- decoded1      (low-level detail feature)
        prototype_evidence = self.dad_prototype(
            high_feature=decoded3,
            low_feature=decoded1,
            guide_logits=prediction3,
        )

        prototype_feature = self.dad_prototype_fusion(
            torch.cat(
                [
                    prototype_evidence[
                        "foreground_feature"
                    ],
                    prototype_evidence[
                        "background_feature"
                    ],
                ],
                dim=1,
            )
        )

        coarse_logits = self.pred1(
            prototype_feature
        )

        # Keep the original CFPS branch unchanged so the experiment
        # isolates the added FG/BG prototype evidence.
        verification = self.background_verifier(
            raw_stage2=raw_stage2,
            raw_stage3=raw_stage3,
            coarse_logits=coarse_logits,
        )

        refined_logits = verification[
            "refined_logits"
        ]

        prediction = F.interpolate(
            refined_logits,
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

        fp_risk_logits = F.interpolate(
            verification["fp_risk_logits"],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        suppression = F.interpolate(
            verification["suppression"],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        dad_fg_similarity = F.interpolate(
            prototype_evidence[
                "foreground_similarity"
            ],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        dad_bg_similarity = F.interpolate(
            prototype_evidence[
                "background_similarity"
            ],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        dad_semantic_margin = F.interpolate(
            prototype_evidence[
                "semantic_margin"
            ],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": prediction,
            "coarse_pred": coarse_prediction,
            "fp_risk": fp_risk_logits,
            "suppression": suppression,
            "dad_fg_similarity": dad_fg_similarity,
            "dad_bg_similarity": dad_bg_similarity,
            "dad_semantic_margin": dad_semantic_margin,
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> VMambaSmallFPNNAMLabPyramidCFPSDADProtoSOD:
    return VMambaSmallFPNNAMLabPyramidCFPSDADProtoSOD(
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
        if isinstance(value, torch.Tensor):
            print(
                key,
                tuple(value.shape),
            )
        else:
            print(
                key,
                [
                    tuple(tensor.shape)
                    for tensor in value
                ],
            )
