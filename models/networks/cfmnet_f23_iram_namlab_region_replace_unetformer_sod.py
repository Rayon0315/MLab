from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_iram_namlab_internal_base import CFMNetIRAMNAMLabInternalSOD


class CFMNetF23IRAMNAMLabRegionReplaceUNetFormerSOD(CFMNetIRAMNAMLabInternalSOD):
    def __init__(self, pretrained_path):
        super().__init__(
            pretrained_path=pretrained_path,
            mode="replace",
        )


def build_model():
    return CFMNetF23IRAMNAMLabRegionReplaceUNetFormerSOD(
        pretrained_path=resolve_pretrained_path()
    )
