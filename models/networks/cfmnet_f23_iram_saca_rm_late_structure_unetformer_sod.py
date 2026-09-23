"""SACA-v1 + Region-Mean late structural evidence."""
from models.components.cfm_iram_saca import SACAStream
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_saca_region_mean_structural_base import (
    CFMNetIRAMSACARegionMeanStructuralSOD,
)

class CFMNetF23IRAMSACARMLateStructureUNetFormerSOD(
    CFMNetIRAMSACARegionMeanStructuralSOD
):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAStream(ablation="full"),
            mode="late",
            structural_channels=96,
        )

def build_model():
    return CFMNetF23IRAMSACARMLateStructureUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
