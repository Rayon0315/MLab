from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

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


class VMambaSmallFPNNAMLabPyramidDADProtoV2SOD(
    VMambaSmallFPNNAMLabHybridSOD
):
    """
    Pure DAD-v2 control:
        VMamba-S + NAMLab Hybrid + Pyramid FPN
        + revised FG/BG prototype evidence.

    No CFPS, no FP-risk loss, no suppression.

    The outer topology is intentionally kept the same as Pure DAD v1:
        decoded3 + prediction3
            -> DAD prototype module
        decoded1
            -> FG-aware / BG-aware branches
            -> concat + 3x3 fusion
            -> pred1

    Only the internal evidence expression is changed.
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

        # Keep identical outer fusion form to DAD v1 so this run tests
        # the revised evidence expression instead of a new fusion design.
        self.dad_prototype_fusion = ConvNormAct(
            decoder_channels * 2,
            decoder_channels,
            kernel_size=3,
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
            stage1,
            stage2,
            stage3,
            stage4,
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
                stage1,
                stage2,
                stage3,
                stage4,
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

        prototype_evidence = (
            self.dad_prototype_v2(
                high_feature=decoded3,
                low_feature=decoded1,
                guide_logits=prediction3,
            )
        )

        prototype_feature = (
            self.dad_prototype_fusion(
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
        )

        prediction1 = self.pred1(
            prototype_feature
        )

        prediction = F.interpolate(
            prediction1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        dad_fg_evidence = F.interpolate(
            prototype_evidence[
                "foreground_evidence"
            ],
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        dad_bg_evidence = F.interpolate(
            prototype_evidence[
                "background_evidence"
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

        return {
            "pred": prediction,
            "dad_fg_evidence": dad_fg_evidence,
            "dad_bg_evidence": dad_bg_evidence,
            "dad_semantic_margin": dad_semantic_margin,
            "dad_fg_similarity": dad_fg_similarity,
            "dad_bg_similarity": dad_bg_similarity,
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> VMambaSmallFPNNAMLabPyramidDADProtoV2SOD:
    return VMambaSmallFPNNAMLabPyramidDADProtoV2SOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "VMamba smoke test requires CUDA."
        )

    device = torch.device("cuda")
    model = build_model().to(device)
    model.eval()

    image = torch.randn(
        1, 3, 352, 352,
        device=device,
    )
    mean_60 = torch.rand(
        1, 3, 352, 352,
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
