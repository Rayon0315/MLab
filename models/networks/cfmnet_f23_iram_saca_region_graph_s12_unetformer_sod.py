"""CFMNet + Stage1/Stage2 RegionGraph + F2/F3 IRAM + SACA v1 + UNetFormer."""

from models.components.cfm_iram_saca import SACAStream
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_saca_region_graph_s12_base import (
    CFMNetIRAMSACARegionGraphS12SOD,
)


class CFMNetF23IRAMSACARegionGraphS12UNetFormerSOD(
    CFMNetIRAMSACARegionGraphS12SOD
):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAStream(ablation="full"),
            graph_depth=2,
            num_heads=4,
        )


def build_model():
    return CFMNetF23IRAMSACARegionGraphS12UNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
