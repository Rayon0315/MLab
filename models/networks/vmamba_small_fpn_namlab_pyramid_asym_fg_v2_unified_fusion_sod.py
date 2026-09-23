from __future__ import annotations

from pathlib import Path

from models.components.namlab_stage234_unified import (
    RegionVisualUnifiedFusion,
)
from models.networks.vmamba_small_fpn_namlab_pyramid_asym_fg_v2_sod import (
    VMambaSmallFPNNAMLabPyramidAsymFGV2SOD,
)
from models.networks.vmamba_small_fpn_sod import (
    PRETRAINED_PATH,
)


class VMambaSmallFPNNAMLabPyramidAsymFGV2UnifiedFusionSOD(
    VMambaSmallFPNNAMLabPyramidAsymFGV2SOD
):
    """
    FG-only v2 + unified Stage2/3/4 region fusion.

    Stage1 and the Stage2/3/4 M60 encoders stay unchanged.
    Only the region-visual interaction at Stage2/3/4 is replaced.
    """

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            decoder_channels=decoder_channels,
        )

        (
            _,
            stage2_channels,
            stage3_channels,
            stage4_channels,
        ) = tuple(
            self.backbone.out_channels
        )

        self.namlab_hybrid.stage2_region_reconstruction = (
            RegionVisualUnifiedFusion(
                channels=stage2_channels,
            )
        )

        self.namlab_hybrid.stage3_region_interaction = (
            RegionVisualUnifiedFusion(
                channels=stage3_channels,
            )
        )

        self.namlab_hybrid.stage4_region_interaction = (
            RegionVisualUnifiedFusion(
                channels=stage4_channels,
            )
        )


def build_model(
) -> VMambaSmallFPNNAMLabPyramidAsymFGV2UnifiedFusionSOD:
    return VMambaSmallFPNNAMLabPyramidAsymFGV2UnifiedFusionSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )
