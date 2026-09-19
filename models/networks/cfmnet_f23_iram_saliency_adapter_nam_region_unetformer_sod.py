"""IRAM + Saliency Adapter + NAM region color ablation."""
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_saliency_evidence_base import CFMNetSaliencyEvidenceSOD


class CFMNetF23IRAMSaliencyAdapterNAMRegionUNetFormerSOD(CFMNetSaliencyEvidenceSOD):
    input_keys = ("image", "mean_60")

    def __init__(self, pretrained_path):
        super().__init__(pretrained_path=pretrained_path, use_iram=True, use_nam=True)

    def forward(self, image, mean_60):
        return self._predict(image, mean_60)


def build_model():
    return CFMNetF23IRAMSaliencyAdapterNAMRegionUNetFormerSOD(pretrained_path=resolve_pretrained_path())
