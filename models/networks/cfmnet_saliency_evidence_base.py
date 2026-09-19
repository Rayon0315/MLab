"""Unchanged canonical backbone/decoder, current-best auxiliary protocol."""
from models.components.cfm_iram_saliency_adapter import SaliencyAdapterStream
from models.components.cfm_saliency_nam_necks import SaliencyEvidenceNeck
from models.networks.cfmnet_unetformer_sod import CFMNetUNetFormerSOD


class CFMNetSaliencyEvidenceSOD(CFMNetUNetFormerSOD):
    def __init__(self, pretrained_path, use_iram, use_nam):
        # Preserve the current best's construction order for shared adapter/backbone.
        x_stream = SaliencyAdapterStream()
        super().__init__(pretrained_path=pretrained_path, decode_channels=64, window_size=8)
        self.neck = SaliencyEvidenceNeck(use_iram=use_iram, use_nam=use_nam, x_stream=x_stream)

    def _predict(self, image, mean_60=None):
        features, coarse = self.neck(self.backbone(image), mean_60)
        prediction, auxiliary = self.decoder(features=features, output_size=image.shape[-2:])
        outputs = {"pred": prediction}
        if self.training:
            outputs["aux"] = [auxiliary, coarse]
        return outputs
