"""CFMNet + IRAM + SACA-v1 + NRTR + UNetFormer."""
from models.components.cfm_iram_saca import SACAStream
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_saca_nrtr_base import CFMNetIRAMSACANRTRSOD

class CFMNetF23IRAMSACANRTRUNetFormerSOD(CFMNetIRAMSACANRTRSOD):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAStream(ablation='full'),
            token_dim=128,
            max_regions=96,
        )

def build_model():
    return CFMNetF23IRAMSACANRTRUNetFormerSOD(pretrained_path=resolve_pretrained_path())
