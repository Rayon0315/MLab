from __future__ import annotations

from pathlib import Path

from models.components.three_way_fpn import (
    ThreeWayAttentionFPNFusion,
)
from models.networks.vmamba_small_fpn_namlab_threeway_sod import (
    VMambaSmallFPNNAMLabThreeWaySOD,
)
from models.networks.vmamba_small_fpn_sod import (
    PRETRAINED_PATH,
)


class VMambaSmallFPNNAMLabThreeWayAttentionSOD(
    VMambaSmallFPNNAMLabThreeWaySOD
):
    """
    Attention version of the three-source FPN.

    The feature sources are exactly the same as the simple three-way model:

        1. NAM-enhanced lateral feature
        2. top-down decoder feature
        3. persistent Stage4 global semantic descriptor

    Only the fusion rule changes:

        simple:
            concat -> projection -> residual refinement

        attention:
            spatial 3-way softmax
            -> weighted sum
            -> residual refinement

    This file should only be compared after the simple three-way
    model has established that the third global source itself is useful.
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

        self.fusion3 = ThreeWayAttentionFPNFusion(
            decoder_channels
        )
        self.fusion2 = ThreeWayAttentionFPNFusion(
            decoder_channels
        )
        self.fusion1 = ThreeWayAttentionFPNFusion(
            decoder_channels
        )


def build_model(
) -> VMambaSmallFPNNAMLabThreeWayAttentionSOD:
    return VMambaSmallFPNNAMLabThreeWayAttentionSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )
