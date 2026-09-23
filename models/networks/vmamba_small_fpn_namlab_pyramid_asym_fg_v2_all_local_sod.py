from __future__ import annotations

from pathlib import Path

from models.components.namlab_hybrid import (
    RegionConditionedLocalReconstruction,
)
from models.networks.vmamba_small_fpn_namlab_pyramid_asym_fg_v2_sod import (
    VMambaSmallFPNNAMLabPyramidAsymFGV2SOD,
)
from models.networks.vmamba_small_fpn_sod import (
    PRETRAINED_PATH,
)


class VMambaSmallFPNNAMLabPyramidAsymFGV2AllLocalSOD(
    VMambaSmallFPNNAMLabPyramidAsymFGV2SOD
):
    """
    FG-only v2 + RegionConditionedLocalReconstruction at Stage2/3/4.

    Stage2 keeps the original validated local reconstruction instance.
    Only Stage3/4 are replaced.
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
            _,
            stage3_channels,
            stage4_channels,
        ) = tuple(
            self.backbone.out_channels
        )

        self.namlab_hybrid.stage3_region_interaction = (
            RegionConditionedLocalReconstruction(
                channels=stage3_channels,
            )
        )

        self.namlab_hybrid.stage4_region_interaction = (
            RegionConditionedLocalReconstruction(
                channels=stage4_channels,
            )
        )


def build_model(
) -> VMambaSmallFPNNAMLabPyramidAsymFGV2AllLocalSOD:
    return VMambaSmallFPNNAMLabPyramidAsymFGV2AllLocalSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )
