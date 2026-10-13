from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.namlab_hybrid import (
    HybridRegionScaleEncoder,
    RegionConditionedLocalReconstruction,
)
from models.networks.smt_t_pafr_shallow_ecm_sod import (
    SMTTinyPAFRShallowECMSOD,
)
from models.networks.smt_t_pafr_sod import PRETRAINED_PATH


class M60Normalizer(nn.Module):
    """Normalize the exported M60 region-mean map to ImageNet space."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, mean_60: torch.Tensor) -> torch.Tensor:
        return (mean_60 - self.image_mean) / self.image_std


class M60LocalReconstruction(nn.Module):
    """M60-guided local reconstruction for one shallow SMT stage."""

    def __init__(
        self,
        channels: int,
        initial_context_scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.region_encoder = HybridRegionScaleEncoder(
            out_channels=channels,
            initial_context_scale=initial_context_scale,
        )
        self.local_reconstruction = RegionConditionedLocalReconstruction(
            channels=channels,
        )

    def forward(
        self,
        visual_feature: torch.Tensor,
        mean_60_normalized: torch.Tensor,
    ) -> torch.Tensor:
        region_feature = self.region_encoder(
            mean_60_normalized,
            target_size=visual_feature.shape[-2:],
        )
        return self.local_reconstruction(
            visual_feature=visual_feature,
            region_feature=region_feature,
        )


class OriginalAdjustedFusion(nn.Module):
    """Fuse an original stage feature with its M60-adjusted counterpart."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        original: torch.Tensor,
        adjusted: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion(
            torch.cat((original, adjusted), dim=1)
        )


class SMTTinyPAFRNAMM60ShallowECMSOD(SMTTinyPAFRShallowECMSOD):
    """SMT-T + PAFR + M60 local reconstruction on S1/S2 + Shallow ECM.

    PAFR's semantic path is unchanged and still consumes the original SMT
    features. Only the two inputs of Shallow ECM are rebuilt as:

        S1* = Proj([S1, LocalRecon(S1, M60)])
        S2* = Proj([S2, LocalRecon(S2, M60)])

    Stage3, stage4 and PAFR stage5 are not NAMLab-adjusted and do not enter
    Shallow ECM.
    """

    input_keys = ("image", "mean_60")

    def __init__(
        self,
        pretrained_path: str | Path | None,
        ecm_channels: int = 96,
        initial_context_scale: float = 0.1,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            ecm_channels=ecm_channels,
        )

        stage1_channels = self.backbone.out_channels[0]
        stage2_channels = self.backbone.out_channels[1]

        self.m60_normalizer = M60Normalizer()

        self.stage1_m60_reconstruction = M60LocalReconstruction(
            channels=stage1_channels,
            initial_context_scale=initial_context_scale,
        )
        self.stage2_m60_reconstruction = M60LocalReconstruction(
            channels=stage2_channels,
            initial_context_scale=initial_context_scale,
        )

        self.stage1_ecm_fusion = OriginalAdjustedFusion(
            stage1_channels
        )
        self.stage2_ecm_fusion = OriginalAdjustedFusion(
            stage2_channels
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor | list[torch.Tensor],
    ]:
        input_size = image.shape[-2:]

        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = self.backbone(image)

        mean_60_normalized = self.m60_normalizer(mean_60)

        stage1_adjusted = self.stage1_m60_reconstruction(
            visual_feature=stage1,
            mean_60_normalized=mean_60_normalized,
        )
        stage2_adjusted = self.stage2_m60_reconstruction(
            visual_feature=stage2,
            mean_60_normalized=mean_60_normalized,
        )

        ecm_stage1 = self.stage1_ecm_fusion(
            stage1,
            stage1_adjusted,
        )
        ecm_stage2 = self.stage2_ecm_fusion(
            stage2,
            stage2_adjusted,
        )

        stage5 = self.pool4(stage4)
        stage5 = self.stage5_projection(stage5)
        stage5 = self.stage5(stage5)

        opm_feature = self.opm(stage5)

        decoded5, _ = self.crm5(
            opm_feature,
            stage5,
        )

        x = self._resize_local(
            decoded5,
            stage4.shape[-2:],
        )
        decoded4, _ = self.crm4(
            x,
            stage4,
        )

        x = self._resize_local(
            decoded4,
            stage3.shape[-2:],
        )
        decoded3, _ = self.crm3(
            x,
            stage3,
        )

        x = self._resize_local(
            decoded3,
            stage2.shape[-2:],
        )
        decoded2, _ = self.crm2(
            x,
            stage2,
        )

        x = self._resize_local(
            decoded2,
            stage1.shape[-2:],
        )
        decoded1, _ = self.crm1(
            x,
            stage1,
        )

        shallow_structure = self.ecm((ecm_stage1, ecm_stage2))

        decoded1 = self.structure_fusion(
            torch.cat(
                (
                    decoded1,
                    shallow_structure,
                ),
                dim=1,
            )
        )

        opm_logits = self._resize_local(
            self.out_opm(opm_feature),
            input_size,
        )
        side5_logits = self._resize_local(
            self.out5(decoded5),
            input_size,
        )
        side4_logits = self._resize_local(
            self.out4(decoded4),
            input_size,
        )
        side3_logits = self._resize_local(
            self.out3(decoded3),
            input_size,
        )
        side2_logits = self._resize_local(
            self.out2(decoded2),
            input_size,
        )
        side1_logits = self._resize_local(
            self.out1(decoded1),
            input_size,
        )

        final_logits = self.refunet(side1_logits)

        pafr_logits = [
            final_logits,
            side1_logits,
            side2_logits,
            side3_logits,
            side4_logits,
            side5_logits,
            opm_logits,
        ]

        return {
            "pred": final_logits,
            "aux": pafr_logits[1:],
            "pafr_logits": pafr_logits,
        }


def build_model() -> SMTTinyPAFRNAMM60ShallowECMSOD:
    return SMTTinyPAFRNAMM60ShallowECMSOD(
        pretrained_path=PRETRAINED_PATH,
    )
