"""CFMNet + F2/F3 IRAM + SACA-v2 R->T-only + unchanged UNetFormer."""

from models.components.cfm_iram_saca_v2_cooperation_variants import (
    SACAv2RegionToStructureOnlyStream,
)
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_strong_stream_base import (
    CFMNetIRAMStrongStreamSOD,
)


class CFMNetF23IRAMSACAv2R2TUNetFormerSOD(
    CFMNetIRAMStrongStreamSOD
):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAv2RegionToStructureOnlyStream(),
        )


def build_model():
    return CFMNetF23IRAMSACAv2R2TUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
