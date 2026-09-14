from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.components.hard_background_verifier import (
    HardBackgroundVerifier,
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


class VMambaSmallFPNNAMLabPyramidCFPSSOD(
    VMambaSmallFPNNAMLabHybridSOD
):
    """
    VMamba-S + NAMLab Hybrid + Pure FPN + Pyramid Context
    + Counter-Evidence False Positive Suppression (CFPS).

    Main saliency path:
        VMamba
            -> NAMLab
            -> Pure FPN
            -> Pyramid Context on Stage4 FPN feature
            -> coarse saliency

    Counter-evidence path:
        raw Stage2 / Stage3 BEFORE NAMLab
            -> HardBackgroundVerifier
            -> false-positive risk
            -> one-way negative logit correction

    Deliberately excluded:
        - hierarchical FPN channels
        - persistent Stage4 global branch
        - three-way fusion
        - mutual gates
        - boundary refinement

    The new branch is therefore aimed specifically at:
        high-confidence background false positives.
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

        # Keep the currently useful Pyramid Context component.
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

        # Coarse saliency is intentionally formed BEFORE
        # the false-positive verifier.
        coarse_logits = self.pred1(
            decoded1
        )

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

        return {
            "pred": prediction,
            "coarse_pred": coarse_prediction,
            "fp_risk": fp_risk_logits,
            "suppression": suppression,
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> VMambaSmallFPNNAMLabPyramidCFPSSOD:
    return VMambaSmallFPNNAMLabPyramidCFPSSOD(
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

    print(
        "suppression scale:",
        float(
            model.background_verifier.max_suppression
            * torch.sigmoid(
                model.background_verifier
                .suppression_scale_logit
            )
        ),
    )
