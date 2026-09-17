# models/networks/cfmnet_f23_iram_region_consistency_gate_unetformer_sod.py

from __future__ import annotations

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
)
from models.components.cfm_iram_selective_necks import (
    IRAMRegionConsistencyGateNeck,
)
from models.networks.cfmnet_iram_x_base import (
    CFMNetIRAMXUNetFormerSOD,
    resolve_pretrained_path,
)


class CFMNetF23IRAMRegionConsistencyGateUNetFormerSOD(
    CFMNetIRAMXUNetFormerSOD
):
    def __init__(
        self,
        pretrained_path,
    ) -> None:
        super().__init__(
            pretrained_path=(
                pretrained_path
            ),
            neck=IRAMRegionConsistencyGateNeck(
                stage2_channels=(
                    CFMNET_OUT_CHANNELS[1]
                ),
                stage3_channels=(
                    CFMNET_OUT_CHANNELS[2]
                ),
                stage4_channels=(
                    CFMNET_OUT_CHANNELS[3]
                ),
                region_kernel=3,
            ),
            decode_channels=64,
            window_size=8,
        )


def build_model(
) -> CFMNetF23IRAMRegionConsistencyGateUNetFormerSOD:
    return CFMNetF23IRAMRegionConsistencyGateUNetFormerSOD(
        pretrained_path=(
            resolve_pretrained_path()
        )
    )
