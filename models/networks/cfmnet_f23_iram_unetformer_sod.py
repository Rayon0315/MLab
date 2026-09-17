# models/networks/cfmnet_f23_iram_unetformer_sod.py

from __future__ import annotations

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
)
from models.components.cfm_external_necks import (
    IRAMNeck,
)
from models.networks.cfmnet_external_neck_base import (
    CFMNetExternalNeckUNetFormerSOD,
    resolve_pretrained_path,
)


class CFMNetF23IRAMUNetFormerSOD(
    CFMNetExternalNeckUNetFormerSOD
):
    def __init__(
        self,
        pretrained_path,
    ) -> None:
        super().__init__(
            pretrained_path=(
                pretrained_path
            ),
            neck=IRAMNeck(
                stage2_channels=(
                    CFMNET_OUT_CHANNELS[1]
                ),
                stage3_channels=(
                    CFMNET_OUT_CHANNELS[2]
                ),
                stage2_index=1,
                stage3_index=2,
            ),
            decode_channels=64,
            window_size=8,
        )


def build_model(
) -> CFMNetF23IRAMUNetFormerSOD:
    return CFMNetF23IRAMUNetFormerSOD(
        pretrained_path=(
            resolve_pretrained_path()
        )
    )
