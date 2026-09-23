"""CFMNet + F2/F3 IRAM + SACA-v2 + NAM Structure Prior + UNetFormer."""

from models.components.cfm_iram_saca_nam_structure_prior import (
    SACAv2NAMStructurePriorStream,
)
from models.networks.cfmnet_external_neck_base import (
    resolve_pretrained_path,
)
from models.networks.cfmnet_iram_nam_structure_prior_base import (
    CFMNetIRAMNAMStructurePriorSOD,
)


class CFMNetF23IRAMSACAv2NAMStructurePriorUNetFormerSOD(
    CFMNetIRAMNAMStructurePriorSOD
):
    def __init__(
        self,
        pretrained_path,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            x_stream=SACAv2NAMStructurePriorStream(
                channels=128,
            ),
        )


def build_model():
    return (
        CFMNetF23IRAMSACAv2NAMStructurePriorUNetFormerSOD(
            pretrained_path=resolve_pretrained_path()
        )
    )
