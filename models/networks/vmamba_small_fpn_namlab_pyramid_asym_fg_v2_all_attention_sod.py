from __future__ import annotations

from pathlib import Path

from models.components.namlab_hybrid import (
    BidirectionalRegionVisualInteraction,
)
from models.networks.vmamba_small_fpn_namlab_pyramid_asym_fg_v2_sod import (
    VMambaSmallFPNNAMLabPyramidAsymFGV2SOD,
)
from models.networks.vmamba_small_fpn_sod import (
    PRETRAINED_PATH,
)


class VMambaSmallFPNNAMLabPyramidAsymFGV2AllAttentionSOD(
    VMambaSmallFPNNAMLabPyramidAsymFGV2SOD
):
    """
    FG-only v2 + BidirectionalRegionVisualInteraction at Stage2/3/4.

    Stage3/4 keep their current attention instances.
    Only Stage2 is replaced with the same topology.
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
            _,
            _,
        ) = tuple(
            self.backbone.out_channels
        )

        self.namlab_hybrid.stage2_region_reconstruction = (
            BidirectionalRegionVisualInteraction(
                channels=stage2_channels,
                inner_channels=64,
                num_heads=4,
            )
        )


def build_model(
) -> VMambaSmallFPNNAMLabPyramidAsymFGV2AllAttentionSOD:
    return VMambaSmallFPNNAMLabPyramidAsymFGV2AllAttentionSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )
