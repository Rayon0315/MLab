# models/networks/cfmnet_stage2_cross_matching_namlab_hybrid_sod.py

from __future__ import annotations

from pathlib import Path

import torch

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
)
from models.components.namlab_hybrid import (
    NAMLabHybrid,
)
from models.networks.cfmnet_stage2_cross_matching_unetformer_sod import (
    PRETRAINED_PATH,
    CFMNetStage2CrossMatchingUNetFormerSOD,
)


class CFMNetStage2CrossMatchingNAMLabHybridSOD(
    CFMNetStage2CrossMatchingUNetFormerSOD
):
    """
    CFMNet + NAMLab Hybrid + Stage2 Cross-Matching UNetFormer.

    Controlled change from:
        cfmnet_stage2_cross_matching_unetformer_sod.py

    Pipeline:
        image
          |
        CFMNet backbone
          |
          |---- raw Stage2 T/S/L/G branches --------|
          |                                           |
          |                                  local Q/K/V matching
          |
        fused CFM features
          |
        NAMLabHybrid(image, mean_60)
          |
        region-enhanced fused features
          |
        unchanged Stage2 Cross-Matching UNetFormer
          |
        prediction

    Matching roles remain:
        Q <- decoder Stage3 feature
             (now decoded from NAMLab-enhanced fused features)

        K <- raw Stage2 TCRM + EGPCM

        V <- raw Stage2 SSRM + LDFM

    Thus NAMLab does not overwrite the CFM functional branch
    identity used by K/V.
    """

    input_keys = (
        "image",
        "mean_60",
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
        super().__init__(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=(
                decode_channels
            ),
            window_size=(
                window_size
            ),
            matching_window=(
                matching_window
            ),
        )

        self.namlab_hybrid = (
            NAMLabHybrid(
                stage_channels=(
                    CFMNET_OUT_CHANNELS
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
        torch.Tensor
        | list[
            torch.Tensor
        ],
    ]:
        output_size = (
            image.shape[-2:]
        )

        # Stage2 branch identity is captured before NAMLab.
        self.stage2_tap.clear()

        features = (
            self.backbone(
                image
            )
        )

        stage2_branches = (
            self.stage2_tap.pop()
        )

        # Region-aware reconstruction modifies only fused features.
        enhanced_features = (
            self.namlab_hybrid(
                features=tuple(
                    features
                ),
                image=image,
                mean_60=mean_60,
            )
        )

        (
            prediction,
            auxiliary,
        ) = self.decoder(
            features=list(
                enhanced_features
            ),
            stage2_branches=(
                stage2_branches
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
) -> CFMNetStage2CrossMatchingNAMLabHybridSOD:
    return (
        CFMNetStage2CrossMatchingNAMLabHybridSOD(
            pretrained_path=(
                PRETRAINED_PATH
            ),
            decode_channels=64,
            window_size=8,
            matching_window=7,
        )
    )


if __name__ == "__main__":
    model = build_model()

    image = torch.randn(
        1,
        3,
        352,
        352,
    )

    mean_60 = torch.rand(
        1,
        3,
        352,
        352,
    )

    model.train()

    outputs = model(
        image=image,
        mean_60=mean_60,
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
        "matching scale:",
        float(
            model
            .decoder
            .matching_scale()
            .detach()
        ),
    )

    print(
        "NAMLab context scales:",
        float(
            model
            .namlab_hybrid
            .region_encoder
            .stage2_encoder
            .context_scale
            .detach()
        ),
        float(
            model
            .namlab_hybrid
            .region_encoder
            .stage3_encoder
            .context_scale
            .detach()
        ),
        float(
            model
            .namlab_hybrid
            .region_encoder
            .stage4_encoder
            .context_scale
            .detach()
        ),
    )

    model.eval()

    with torch.no_grad():
        outputs = model(
            image=image,
            mean_60=mean_60,
        )

    print(
        "eval pred:",
        tuple(
            outputs[
                "pred"
            ].shape
        ),
    )
