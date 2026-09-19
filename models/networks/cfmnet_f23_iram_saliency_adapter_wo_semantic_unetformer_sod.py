"""IRAM + Saliency Adapter leave-one-out: remove semantic evidence only."""
from models.components.cfm_iram_saliency_adapter_ablation import (
    SaliencyAdapterLeaveOneOutStream,
)
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_strong_stream_base import CFMNetIRAMStrongStreamSOD


class CFMNetF23IRAMSaliencyAdapterWithoutSemanticUNetFormerSOD(
    CFMNetIRAMStrongStreamSOD
):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SaliencyAdapterLeaveOneOutStream(ablated="semantic"),
        )


def build_model():
    return CFMNetF23IRAMSaliencyAdapterWithoutSemanticUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
