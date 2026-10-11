from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.pafr_ecm import FeatureECM
from models.networks.smt_t_pafr_sod import (
    PRETRAINED_PATH,
    SMTTinyPAFRSOD,
)


class SMTTinyPAFRECMSOD(SMTTinyPAFRSOD):
    """SMT-T + PAFR with a feature-only ECM structural branch.

    This subclass intentionally does not depend on private helper methods from
    ``SMTTinyPAFRSOD``. The existing baseline exposes the PAFR modules but its
    forward path is monolithic, so the encode/decode/output steps needed by ECM
    are implemented locally here.
    """

    input_keys = ("image",)

    def __init__(
        self,
        pretrained_path: str | Path | None,
        ecm_channels: int = 96,
    ) -> None:
        super().__init__(pretrained_path=pretrained_path)

        self.ecm = FeatureECM(
            input_channels=(
                self.backbone.out_channels[0],
                self.backbone.out_channels[1],
                self.backbone.out_channels[2],
                self.backbone.out_channels[3],
                512,
            ),
            edge_channels=ecm_channels,
        )
        self.structure_fusion = nn.Sequential(
            nn.Conv2d(
                64 + ecm_channels,
                64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _resize_local(
        x: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def _encode_for_ecm(
        self,
        image: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        stage1, stage2, stage3, stage4 = self.backbone(image)

        stage5 = self.pool4(stage4)
        stage5 = self.stage5_projection(stage5)
        stage5 = self.stage5(stage5)

        return stage1, stage2, stage3, stage4, stage5

    def _decode_for_ecm(
        self,
        stage1: torch.Tensor,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
        stage5: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        opm_feature = self.opm(stage5)

        decoded5, _ = self.crm5(opm_feature, stage5)

        x = self._resize_local(decoded5, stage4.shape[-2:])
        decoded4, _ = self.crm4(x, stage4)

        x = self._resize_local(decoded4, stage3.shape[-2:])
        decoded3, _ = self.crm3(x, stage3)

        x = self._resize_local(decoded3, stage2.shape[-2:])
        decoded2, _ = self.crm2(x, stage2)

        x = self._resize_local(decoded2, stage1.shape[-2:])
        decoded1, _ = self.crm1(x, stage1)

        return (
            opm_feature,
            decoded5,
            decoded4,
            decoded3,
            decoded2,
            decoded1,
        )

    def _build_outputs_for_ecm(
        self,
        input_size: tuple[int, int],
        opm_feature: torch.Tensor,
        decoded5: torch.Tensor,
        decoded4: torch.Tensor,
        decoded3: torch.Tensor,
        decoded2: torch.Tensor,
        decoded1: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
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

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        input_size = image.shape[-2:]
        stage1, stage2, stage3, stage4, stage5 = (
            self._encode_for_ecm(image)
        )

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

        structure_feature = self.ecm(
            (stage1, stage2, stage3, stage4, stage5)
        )
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


def build_model() -> SMTTinyPAFRECMSOD:
    return SMTTinyPAFRECMSOD(
        pretrained_path=PRETRAINED_PATH,
    )
