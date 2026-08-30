from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from models.networks.mambavision_small_progressive_region_direct_hier60_sod import (
    MambaVisionSmallProgressiveRegionDirectHier60SOD,
)
from models.networks.mambavision_small_progressive_region_direct_sod import (
    PRETRAINED_PATH,
    LocalDetailReconstruction,
    RegionScaleEncoder,
)


class GuidedStage1RegionPyramidEncoder(nn.Module):
    """
    Pure Hier60 region pyramid with an additional M60 guide
    for Stage1 reconstruction.

    Stage1 content:
        detail feature    <- RGB - M60, 96ch

    Stage1 guide:
        structure feature <- M60,       96ch

    They are packed together only for passing through the
    unchanged parent forward path.

    Stage2/3/4 remain identical to Pure Hier60:
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

        # Keep the original Stage1 detail capacity unchanged.
        self.stage1_detail_encoder = RegionScaleEncoder(
            out_channels=stage1_channels,
        )

        # Additional M60 structural guide.
        self.stage1_structure_encoder = RegionScaleEncoder(
            out_channels=stage1_channels,
        )

        # Pure Hier60 Stage2-4.
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
        # Pure Hier60 parent provides:
        #
        # detail_region = RGB - M60
        # fine_region   = M60
        # middle_region = M60
        # coarse_region = M60

        # -------------------------------------------------
        # Stage1
        # -------------------------------------------------

        detail_feature = self.stage1_detail_encoder(
            detail_region,
            target_size=stage1_size,
        )

        structure_feature = self.stage1_structure_encoder(
            fine_region,
            target_size=stage1_size,
        )

        # 96ch detail + 96ch M60 guide.
        #
        # They are NOT fused here.
        # The reconstruction module will split them again.
        region1 = torch.cat(
            [
                detail_feature,
                structure_feature,
            ],
            dim=1,
        )

        # -------------------------------------------------
        # Stage2-4: unchanged Pure Hier60
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


class RegionGuidedLocalDetailReconstruction(nn.Module):
    """
    Preserve the original LocalDetailReconstruction and let
    M60 modulate only the residual correction magnitude.

    Original:
        restored = visual + delta(detail)

    Guided:
        restored = visual + delta(detail) * guide(M60)

    M60 therefore acts as structural guidance rather than
    another feature directly mixed into Stage1.
    """

    def __init__(
        self,
        channels: int,
        inner_channels: int = 48,
        guide_strength: float = 0.5,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.guide_strength = guide_strength

        # Exactly preserve the original Stage1 detail module.
        self.detail_reconstruction = (
            LocalDetailReconstruction(
                channels=channels,
                inner_channels=inner_channels,
            )
        )

        # Spatial + channel-wise M60 guidance.
        self.structure_gate = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

        # Start from the original reconstruction behaviour:
        #
        # tanh(0) = 0
        # guide = 1
        nn.init.zeros_(
            self.structure_gate.weight
        )

        if self.structure_gate.bias is not None:
            nn.init.zeros_(
                self.structure_gate.bias
            )

    def forward(
        self,
        visual_feature: torch.Tensor,
        detail_feature: torch.Tensor,
    ) -> torch.Tensor:
        # region1 was packed as:
        #
        # [detail 96ch | M60 structure 96ch]

        residual_feature = detail_feature[
            :,
            :self.channels,
        ]

        structure_feature = detail_feature[
            :,
            self.channels:,
        ]

        # -------------------------------------------------
        # Original Pure Hier60 Stage1 reconstruction
        # -------------------------------------------------

        reconstructed = self.detail_reconstruction(
            visual_feature=visual_feature,
            detail_feature=residual_feature,
        )

        # Extract only the correction proposed by I-M60.
        correction = (
            reconstructed
            - visual_feature
        )

        # -------------------------------------------------
        # M60-guided correction
        # -------------------------------------------------

        guide = (
            1.0
            + self.guide_strength
            * torch.tanh(
                self.structure_gate(
                    structure_feature
                )
            )
        )

        # M60 does not directly enter the restored feature.
        # It only determines where/how strongly the
        # detail correction should be applied.
        return (
            visual_feature
            + correction * guide
        )


class MambaVisionSmallProgressiveRegionDirectHier60Stage1GuidedSOD(
    MambaVisionSmallProgressiveRegionDirectHier60SOD
):
    """
    Pure Hier60 + M60-guided Stage1 detail reconstruction.

    Pure Hier60:
        Stage1 <- RGB - M60
        Stage2 <- M60
        Stage3 <- M60
        Stage4 <- M60

    This version:
        Stage1 detail <- RGB - M60
        Stage1 guide  <- M60

        restored Stage1
            = visual
            + detail_correction * guide(M60)

        Stage2 <- M60
        Stage3 <- M60
        Stage4 <- M60

    All later modules remain unchanged.
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

        # Replace region encoder so Stage1 receives:
        # 96ch residual + 96ch M60 guide.
        self.region_encoder = (
            GuidedStage1RegionPyramidEncoder(
                stage1_channels=stage1_channels,
                stage2_channels=stage2_channels,
                stage3_channels=stage3_channels,
                stage4_channels=stage4_channels,
            )
        )

        # Replace only Stage1 reconstruction.
        self.stage1_detail_reconstruction = (
            RegionGuidedLocalDetailReconstruction(
                channels=stage1_channels,
                inner_channels=48,
                guide_strength=0.5,
            )
        )


def build_model(
) -> MambaVisionSmallProgressiveRegionDirectHier60Stage1GuidedSOD:
    return (
        MambaVisionSmallProgressiveRegionDirectHier60Stage1GuidedSOD(
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