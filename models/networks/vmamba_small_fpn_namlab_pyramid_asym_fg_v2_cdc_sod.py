from __future__ import annotations

from models.networks.vmamba_small_fpn_namlab_pyramid_asym_fg_v2_multi_evidence_sod import (
    VMambaSmallFPNNAMLabPyramidAsymFGV2MultiEvidenceSOD,
)
from models.networks.vmamba_small_fpn_sod import PRETRAINED_PATH


def build_model(
) -> VMambaSmallFPNNAMLabPyramidAsymFGV2MultiEvidenceSOD:
    return VMambaSmallFPNNAMLabPyramidAsymFGV2MultiEvidenceSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
        use_cdc=True,
        use_oad=False,
        initial_cdc_scale=0.05,
    )
