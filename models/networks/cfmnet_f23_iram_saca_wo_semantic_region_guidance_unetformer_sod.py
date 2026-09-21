"""SACA ablation: remove semantic -> region guidance only."""

from models.components.cfm_iram_saca import SACAStream
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_strong_stream_base import CFMNetIRAMStrongStreamSOD


class CFMNetF23IRAMSACAWoSemanticRegionGuidanceUNetFormerSOD(
    CFMNetIRAMStrongStreamSOD
):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAStream(ablation="no_semantic_region_guidance"),
        )


def build_model():
    return CFMNetF23IRAMSACAWoSemanticRegionGuidanceUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
