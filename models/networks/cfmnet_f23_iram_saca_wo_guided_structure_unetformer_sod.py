"""SACA ablation: raw structure remains, semantic/region structure guidance is removed."""

from models.components.cfm_iram_saca import SACAStream
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_strong_stream_base import CFMNetIRAMStrongStreamSOD


class CFMNetF23IRAMSACAWoGuidedStructureUNetFormerSOD(
    CFMNetIRAMStrongStreamSOD
):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAStream(ablation="no_guided_structure"),
        )


def build_model():
    return CFMNetF23IRAMSACAWoGuidedStructureUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
