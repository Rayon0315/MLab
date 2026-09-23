"""SACA-v2 x NAM scheme A: late evidence interaction."""
from models.components.cfm_iram_saca_nam_structural_evidence import (
    SACAv2NAMLateInteractionStream,
)
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_nam_late_interaction_base import (
    CFMNetIRAMNAMLateInteractionSOD,
)


class CFMNetF23IRAMSACAv2NAMLateInteractionUNetFormerSOD(
    CFMNetIRAMNAMLateInteractionSOD
):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAv2NAMLateInteractionStream(),
        )


def build_model():
    return CFMNetF23IRAMSACAv2NAMLateInteractionUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
