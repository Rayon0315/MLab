# models/networks/cfmnet_f2_sdr_unetformer_sod.py

from __future__ import annotations

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
)
from models.components.cfm_external_necks import (
    SDRNeck,
)
from models.networks.cfmnet_external_neck_base import (
    CFMNetExternalNeckUNetFormerSOD,
    resolve_pretrained_path,
)


class CFMNetF2SDRUNetFormerSOD(
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
            neck=SDRNeck(
                channels=(
                    CFMNET_OUT_CHANNELS[1]
                ),
                stage_index=1,
            ),
            decode_channels=64,
            window_size=8,
        )


def build_model(
) -> CFMNetF2SDRUNetFormerSOD:
    return CFMNetF2SDRUNetFormerSOD(
        pretrained_path=(
            resolve_pretrained_path()
        )
    )
