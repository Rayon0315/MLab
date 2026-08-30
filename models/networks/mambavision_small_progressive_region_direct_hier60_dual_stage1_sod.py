from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.networks.mambavision_small_progressive_region_direct_hier60_sod import (
    MambaVisionSmallProgressiveRegionDirectHier60SOD,
)
from models.networks.mambavision_small_progressive_region_direct_sod import (
    PRETRAINED_PATH,
    RegionScaleEncoder,
)


class Stage1DualRegionPyramidEncoder(nn.Module):
    """
    Hier-60 region encoder with complementary Stage1 inputs.

    Stage1:
        structure branch <- M60
        detail branch    <- RGB - M60
        concat            -> native Stage1 channels

    Stage2/3/4 remain identical to the pure Hier60 model:
        Stage2 <- M60
        Stage3 <- M60
        Stage4 <- M60
    """

    def __init__(
        self,
        stage1_channels: int,
        stage2_channels: int,
        stage3_channels: int,
        stage4_channels: int,
    ) -> None:
        super().__init__()

        # MambaVision-S Stage1 = 96 channels,
        # so the two complementary branches each use 48 channels.
        stage1_branch_channels = (
            stage1_channels // 2
        )

        self.stage1_structure_encoder = (
            RegionScaleEncoder(
                out_channels=stage1_branch_channels,
            )
        )

        self.stage1_detail_encoder = (
            RegionScaleEncoder(
                out_channels=stage1_branch_channels,
            )
        )

        # Stage2-4 stay exactly the same as pure Hier60.
        self.stage2_encoder = RegionScaleEncoder(
            out_channels=stage2_channels,
        )

        self.stage3_encoder = RegionScaleEncoder(
            out_channels=stage3_channels,
        )

        self.stage4_encoder = RegionScaleEncoder(
            out_channels=stage4_channels,
        )

    def forward(
        self,
        detail_region: torch.Tensor,
        fine_region: torch.Tensor,
        middle_region: torch.Tensor,
        coarse_region: torch.Tensor,
        stage1_size: tuple[int, int],
        stage2_size: tuple[int, int],
        stage3_size: tuple[int, int],
        stage4_size: tuple[int, int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # In the pure Hier60 parent:
        #
        # detail_region = RGB - M60
        # fine_region   = M60
        # middle_region = M60
        # coarse_region = M60

        # -------------------------------------------------
        # Stage1:
        # preserve both region structure and local residual
        # -------------------------------------------------

        region1_structure = (
            self.stage1_structure_encoder(
                fine_region,
                target_size=stage1_size,
            )
        )

        region1_detail = (
            self.stage1_detail_encoder(
                detail_region,
                target_size=stage1_size,
            )
        )

        region1 = torch.cat(
            [
                region1_structure,
                region1_detail,
            ],
            dim=1,
        )

        # -------------------------------------------------
        # Stage2-4:
        # unchanged pure Hier60 path
        # -------------------------------------------------

        region2 = self.stage2_encoder(
            fine_region,
            target_size=stage2_size,
        )

        region3 = self.stage3_encoder(
            middle_region,
            target_size=stage3_size,
        )

        region4 = self.stage4_encoder(
            coarse_region,
            target_size=stage4_size,
        )

        return (
            region1,
            region2,
            region3,
            region4,
        )


class MambaVisionSmallProgressiveRegionDirectHier60DualStage1SOD(
    MambaVisionSmallProgressiveRegionDirectHier60SOD
):
    """
    Pure Hier60 model with complementary Stage1 region information.

    Original pure Hier60:

        Stage1 <- RGB - M60
        Stage2 <- M60
        Stage3 <- M60
        Stage4 <- M60

    This variant:

        Stage1 <- concat(
            Encode(M60),
            Encode(RGB - M60)
        )

        Stage2 <- M60
        Stage3 <- M60
        Stage4 <- M60

    Only the region encoder is replaced.
    Backbone reconstruction, progressive decoder,
    boundary refinement and prediction heads remain unchanged.
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
        )

        (
            stage1_channels,
            stage2_channels,
            stage3_channels,
            stage4_channels,
        ) = self.backbone.out_channels

        # Replace only the region encoder.
        self.region_encoder = (
            Stage1DualRegionPyramidEncoder(
                stage1_channels=stage1_channels,
                stage2_channels=stage2_channels,
                stage3_channels=stage3_channels,
                stage4_channels=stage4_channels,
            )
        )


def build_model(
) -> MambaVisionSmallProgressiveRegionDirectHier60DualStage1SOD:
    return (
        MambaVisionSmallProgressiveRegionDirectHier60DualStage1SOD(
            pretrained_path=PRETRAINED_PATH,
        )
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

    print(
        "aux:",
        [
            tensor.shape
            for tensor in outputs["aux"]
        ],
    )