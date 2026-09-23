"""SACA-v2 + Region-Mean F1 structural reconstruction."""
from models.components.cfm_iram_saca_v2 import SACAv2Stream
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_saca_region_mean_structural_base import (
    CFMNetIRAMSACARegionMeanStructuralSOD,
)

class CFMNetF23IRAMSACAv2RMF1StructureUNetFormerSOD(
    CFMNetIRAMSACARegionMeanStructuralSOD
):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAv2Stream(ablation="full"),
            mode="f1",
            structural_channels=96,
        )

def build_model():
    return CFMNetF23IRAMSACAv2RMF1StructureUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
