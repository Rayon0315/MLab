"""CFMNet + F2/F3 IRAM + SACA-v2 + M60 Region Relation + UNetFormer."""

from models.components.cfm_iram_saca_m60_region_relation import (
    SACAv2M60RegionRelationStream,
)
from models.networks.cfmnet_external_neck_base import (
    resolve_pretrained_path,
)
from models.networks.cfmnet_iram_m60_strong_stream_base import (
    CFMNetIRAMM60StrongStreamSOD,
)


class CFMNetF23IRAMSACAv2M60RegionRelationUNetFormerSOD(
    CFMNetIRAMM60StrongStreamSOD
):
    def __init__(
        self,
        pretrained_path,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAv2M60RegionRelationStream(
                channels=128,
                window_size=5,
            ),
        )


def build_model():
    return CFMNetF23IRAMSACAv2M60RegionRelationUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
