"""SACA-v2 ablation: remove coarse-saliency semantic boundary evidence."""

from models.components.cfm_iram_saca_v2 import SACAv2Stream
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_strong_stream_base import CFMNetIRAMStrongStreamSOD


class CFMNetF23IRAMSACAv2WoSemanticBoundaryUNetFormerSOD(CFMNetIRAMStrongStreamSOD):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAv2Stream(ablation="no_semantic_boundary"),
        )


def build_model():
    return CFMNetF23IRAMSACAv2WoSemanticBoundaryUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
