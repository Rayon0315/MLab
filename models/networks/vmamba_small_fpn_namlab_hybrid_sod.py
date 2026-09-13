from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.components.namlab_hybrid import (
    NAMLabHybrid,
)
from models.networks.vmamba_small_fpn_sod import (
    PRETRAINED_PATH,
    VMambaSmallFPNSOD,
)


class VMambaSmallFPNNAMLabHybridSOD(
    VMambaSmallFPNSOD
):
    """
    VMamba-S + NAMLab Hybrid + vanilla FPN.

    Controlled change from vmamba_small_fpn_sod.py:

        VMamba backbone
            ->
        NAMLabHybrid
            ->
        unchanged FPN decoder

    Unchanged:
        - VMamba-S backbone
        - 1x1 FPN lateral projections
        - top-down addition
        - 3x3 FPN refinement
        - prediction heads
        - auxiliary supervision interface
        - loss/training protocol

    Added:
        - Hier60 NAMLab Hybrid feature enhancement
          before the FPN lateral projections
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

        self.namlab_hybrid = (
            NAMLabHybrid(
                stage_channels=tuple(
                    self.backbone.out_channels
                ),
                initial_context_scale=0.1,
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
        input_size = (
            image.shape[-2:]
        )

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

        # From here downward the code is intentionally
        # identical to vmamba_small_fpn_sod.py.

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
) -> VMambaSmallFPNNAMLabHybridSOD:
    return VMambaSmallFPNNAMLabHybridSOD(
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

    print(
        "pred:",
        outputs["pred"].shape,
    )

    print(
        "aux:",
        [
            tensor.shape
            for tensor
            in outputs["aux"]
        ],
    )

    print(
        "backbone channels:",
        model.backbone.out_channels,
    )

    print(
        "NAMLab context scales:",
        float(
            model.namlab_hybrid
            .region_encoder
            .stage2_encoder
            .context_scale
        ),
        float(
            model.namlab_hybrid
            .region_encoder
            .stage3_encoder
            .context_scale
        ),
        float(
            model.namlab_hybrid
            .region_encoder
            .stage4_encoder
            .context_scale
        ),
    )
