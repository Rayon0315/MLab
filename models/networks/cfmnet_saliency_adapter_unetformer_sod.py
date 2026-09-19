"""Saliency Adapter ablation."""
from models.networks.cfmnet_external_neck_base import resolve_pretrained_path
from models.networks.cfmnet_saliency_evidence_base import CFMNetSaliencyEvidenceSOD


class CFMNetSaliencyAdapterUNetFormerSOD(CFMNetSaliencyEvidenceSOD):
    input_keys = ("image",)

    def __init__(self, pretrained_path):
        super().__init__(pretrained_path=pretrained_path, use_iram=False, use_nam=False)

    def forward(self, image):
        return self._predict(image)


def build_model():
    return CFMNetSaliencyAdapterUNetFormerSOD(pretrained_path=resolve_pretrained_path())
