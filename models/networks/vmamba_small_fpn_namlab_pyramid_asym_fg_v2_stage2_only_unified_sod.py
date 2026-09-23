from __future__ import annotations

from models.networks.vmamba_small_fpn_namlab_pyramid_asym_fg_v2_selective_unified_sod import (
    VMambaSmallFPNNAMLabPyramidAsymFGV2SelectiveUnifiedSOD,
)
from models.networks.vmamba_small_fpn_sod import PRETRAINED_PATH


def build_model(
) -> VMambaSmallFPNNAMLabPyramidAsymFGV2SelectiveUnifiedSOD:
    return VMambaSmallFPNNAMLabPyramidAsymFGV2SelectiveUnifiedSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
        enabled_stages=(2,),
    )
