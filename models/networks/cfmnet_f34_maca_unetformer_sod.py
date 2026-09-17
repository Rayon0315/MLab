# models/networks/cfmnet_f34_maca_unetformer_sod.py

from __future__ import annotations

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
)
from models.components.cfm_external_necks import (
    MaCANeck,
)
from models.networks.cfmnet_external_neck_base import (
    CFMNetExternalNeckUNetFormerSOD,
    resolve_pretrained_path,
)


class CFMNetF34MaCAUNetFormerSOD(
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
            neck=MaCANeck(
                stage3_channels=(
                    CFMNET_OUT_CHANNELS[2]
                ),
                stage4_channels=(
                    CFMNET_OUT_CHANNELS[3]
                ),
                stage3_index=2,
                stage4_index=3,
            ),
            decode_channels=64,
            window_size=8,
        )


def build_model(
) -> CFMNetF34MaCAUNetFormerSOD:
    return CFMNetF34MaCAUNetFormerSOD(
        pretrained_path=(
            resolve_pretrained_path()
        )
    )
