from __future__ import annotations

from pathlib import Path

import torch

from models.components.namlab_hybrid import NAMLabHybrid
from models.components.pafr_ecm import FeatureConcatFusion
from models.networks.smt_t_pafr_ecm_sod import SMTTinyPAFRECMSOD
from models.networks.smt_t_pafr_sod import PRETRAINED_PATH


class SMTTinyPAFRNAMFeatureConcatECMSOD(SMTTinyPAFRECMSOD):
    """SMT-T + PAFR + NAM feature adjustment + feature-only ECM.

    PAFR keeps the original SMT features. NAMLabHybrid creates adjusted
    features only for the ECM branch. Original and adjusted features are
    concatenated stage-wise and projected back to the native channel width.
    Raw mean_60 is never concatenated into ECM.
    """

    input_keys = ("image", "mean_60")

    def __init__(
        self,
        pretrained_path: str | Path | None,
        ecm_channels: int = 96,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            ecm_channels=ecm_channels,
        )

        self.namlab_hybrid = NAMLabHybrid(
            stage_channels=tuple(self.backbone.out_channels),
            initial_context_scale=0.1,
        )
        self.ecm_feature_fusions = torch.nn.ModuleList(
            [
                FeatureConcatFusion(channels)
                for channels in self.backbone.out_channels
            ]
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        input_size = image.shape[-2:]

        stage1, stage2, stage3, stage4, stage5 = (
            self._encode_for_ecm(image)
        )
        raw_features = (stage1, stage2, stage3, stage4)

        adjusted_features = self.namlab_hybrid(
            features=raw_features,
            image=image,
            mean_60=mean_60,
        )

        fused_features = tuple(
            fusion(original, adjusted)
            for fusion, original, adjusted in zip(
                self.ecm_feature_fusions,
                raw_features,
                adjusted_features,
            )
        )
        ecm_features = (*fused_features, stage5)

        (
            opm_feature,
            decoded5,
            decoded4,
            decoded3,
            decoded2,
            decoded1,
        ) = self._decode_for_ecm(
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
        )

        structure_feature = self.ecm(ecm_features)
        decoded1 = self.structure_fusion(
            torch.cat((decoded1, structure_feature), dim=1)
        )

        return self._build_outputs_for_ecm(
            input_size,
            opm_feature,
            decoded5,
            decoded4,
            decoded3,
            decoded2,
            decoded1,
        )


def build_model() -> SMTTinyPAFRNAMFeatureConcatECMSOD:
    return SMTTinyPAFRNAMFeatureConcatECMSOD(
        pretrained_path=PRETRAINED_PATH,
    )
