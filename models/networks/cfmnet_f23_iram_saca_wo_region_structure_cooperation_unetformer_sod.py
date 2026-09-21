"""SACA ablation: remove explicit bidirectional Region <-> Structure cooperation."""

from models.components.cfm_iram_saca import SACAStream
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_strong_stream_base import CFMNetIRAMStrongStreamSOD


class CFMNetF23IRAMSACAWoRegionStructureCooperationUNetFormerSOD(
    CFMNetIRAMStrongStreamSOD
):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAStream(ablation="no_region_structure_cooperation"),
        )


def build_model():
    return CFMNetF23IRAMSACAWoRegionStructureCooperationUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
