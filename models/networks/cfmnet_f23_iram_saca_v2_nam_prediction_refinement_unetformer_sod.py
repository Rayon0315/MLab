"""SACA-v2 x NAM scheme B: prediction structure refinement."""
from models.components.cfm_iram_saca_v2 import SACAv2Stream
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_nam_prediction_refinement_base import (
    CFMNetIRAMNAMPredictionRefinementSOD,
)


class CFMNetF23IRAMSACAv2NAMPredictionRefinementUNetFormerSOD(
    CFMNetIRAMNAMPredictionRefinementSOD
):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAv2Stream(ablation="full"),
        )


def build_model():
    return CFMNetF23IRAMSACAv2NAMPredictionRefinementUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
