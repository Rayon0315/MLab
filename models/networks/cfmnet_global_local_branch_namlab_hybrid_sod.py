# models/networks/cfmnet_global_local_branch_namlab_hybrid_sod.py

from __future__ import annotations

from pathlib import Path

import torch

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
)
from models.components.namlab_hybrid import (
    NAMLabHybrid,
)
from models.networks.cfmnet_global_local_branch_unetformer_sod import (
    PRETRAINED_PATH,
    CFMNetGlobalLocalBranchUNetFormerSOD,
)


class CFMNetGlobalLocalBranchNAMLabHybridSOD(
    CFMNetGlobalLocalBranchUNetFormerSOD
):
    """
    CFMNet + NAMLab Hybrid + Global/Local Branch UNetFormer.

    Controlled change from:
        cfmnet_global_local_branch_unetformer_sod.py

    Pipeline:
        image
          |
        CFMNet backbone
          |
          |---- raw CFM branches ----------------------|
          |                                             |
          |                                      role-aware
          |                                global/local guidance
          |
        fused CFM features
          |
        NAMLabHybrid(image, mean_60)
          |
        region-enhanced fused features
          |
        unchanged Global/Local Branch UNetFormer
          |
        prediction

    Important:
        - CFM branch taps are captured from the original CFMNet
          backbone output before NAMLab modification.
        - NAMLab only enhances the normal fused feature pyramid.
        - CFM branch role guidance and NAMLab region guidance
          therefore remain two independent information sources.
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

        # Capture CFM functional branches from the original
        # backbone representation.
        self.branch_tap.clear()

        features = (
            self.backbone(
                image
            )
        )

        branch_features = (
            self.branch_tap.pop()
        )

        # NAMLab modifies only the normal fused feature pyramid.
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
) -> CFMNetGlobalLocalBranchNAMLabHybridSOD:
    return (
        CFMNetGlobalLocalBranchNAMLabHybridSOD(
            pretrained_path=(
                PRETRAINED_PATH
            ),
            decode_channels=64,
            window_size=8,
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
