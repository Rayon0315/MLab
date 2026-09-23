from __future__ import annotations

from pathlib import Path

from models.components.namlab_selective_unified import (
    SelectiveUnifiedNAMLab,
)
from models.networks.vmamba_small_fpn_namlab_pyramid_asym_fg_v2_sod import (
    VMambaSmallFPNNAMLabPyramidAsymFGV2SOD,
)


class VMambaSmallFPNNAMLabPyramidAsymFGV2SelectiveUnifiedSOD(
    VMambaSmallFPNNAMLabPyramidAsymFGV2SOD
):
    """
    FG-only v2 with selective Stage2/3/4 M60 guidance.

    Fixed:
        - VMamba-S backbone
        - Stage1 RGB-M60 detail path
        - Stage2/3/4 HybridRegionScaleEncoder definitions
        - FPN / PyramidContext / FG Support v2 / loss interface

    Variable:
        which Stage2/3/4 receives M60 guidance.
    """

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decoder_channels: int = 128,
        enabled_stages: tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            decoder_channels=decoder_channels,
        )

        self.namlab_hybrid = SelectiveUnifiedNAMLab(
            original_hybrid=self.namlab_hybrid,
            stage_channels=tuple(self.backbone.out_channels),
            enabled_stages=enabled_stages,
        )
