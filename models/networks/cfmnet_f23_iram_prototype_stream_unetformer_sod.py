"""Independent prototype_stream ablation; RGB input and unchanged canonical decoder."""
from models.components.cfm_iram_prototype_stream import PrototypeStream
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_strong_stream_base import CFMNetIRAMStrongStreamSOD


class CFMNetF23IRAMPrototypeStreamUNetFormerSOD(CFMNetIRAMStrongStreamSOD):
    def __init__(self, pretrained_path):
        super().__init__(pretrained_path=pretrained_path, x_stream=PrototypeStream())


def build_model():
    return CFMNetF23IRAMPrototypeStreamUNetFormerSOD(pretrained_path=resolve_pretrained_path())
