# models/networks/mambavision_small_progressive_region_direct_hier60_sod.py
from __future__ import annotations

from pathlib import Path

import torch

from models.networks.mambavision_small_progressive_region_direct_sod import (
    MambaVisionSmallProgressiveRegionDirectSOD,
    PRETRAINED_PATH,
)


class MambaVisionSmallProgressiveRegionDirectHier60SOD(
    MambaVisionSmallProgressiveRegionDirectSOD
):
    """
    Simplified Direct Region Mean model using only Hier-60.

    Region assignment:

        Stage1 <- RGB - M60
        Stage2 <- M60
        Stage3 <- M60
        Stage4 <- M60

    All region encoders, region-visual interaction modules,
    progressive decoder modules, boundary refinement, prediction
    heads, and initialization remain identical to the validated
    Direct Region Mean parent network.

    Only the region inputs are simplified. The data pipeline
    therefore needs image + mean_60 only.
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor | list[torch.Tensor],
    ]:
        # Reuse the validated parent implementation.
        #
        # Passing M60 into all three hierarchy slots makes the
        # inherited DirectRegionHierarchy produce:
        #
        #   detail_region = RGB - M60
        #   fine_region   = M60
        #   middle_region = M60
        #   coarse_region = M60
        #
        # Therefore the rest of the network remains unchanged.
        return super().forward(
            image=image,
            mean_20=mean_60,
            mean_40=mean_60,
            mean_60=mean_60,
        )


def build_model(
) -> MambaVisionSmallProgressiveRegionDirectHier60SOD:
    return MambaVisionSmallProgressiveRegionDirectHier60SOD(
        pretrained_path=PRETRAINED_PATH,
    )


if __name__ == "__main__":
    model = build_model()
    model.eval()

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

    with torch.no_grad():
        outputs = model(
            image=image,
            mean_60=mean_60,
        )

    print(
        "pred:",
        outputs["pred"].shape,
    )

    if "aux" in outputs:
        print(
            "aux:",
            [
                tensor.shape
                for tensor in outputs["aux"]
            ],
        )