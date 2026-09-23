"""CFMNet + F2/F3 IRAM + SACA-v2 T->R-only + unchanged UNetFormer."""

from models.components.cfm_iram_saca_v2_cooperation_variants import (
    SACAv2StructureToRegionOnlyStream,
)
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_strong_stream_base import (
    CFMNetIRAMStrongStreamSOD,
)


class CFMNetF23IRAMSACAv2T2RUNetFormerSOD(
    CFMNetIRAMStrongStreamSOD
):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAv2StructureToRegionOnlyStream(),
        )


def build_model():
    return CFMNetF23IRAMSACAv2T2RUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
