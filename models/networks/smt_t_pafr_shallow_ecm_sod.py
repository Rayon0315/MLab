from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.smt_t_pafr_sod import (
    PRETRAINED_PATH,
    SMTTinyPAFRSOD,
)


class _GroupedActivation5x5(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        groups = math.gcd(
            in_channels,
            out_channels,
        )

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                padding=2,
                groups=groups,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class _UpsampleConv3x3(nn.Module):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return self.relu(
            self.conv(x)
        )


class ShallowFeatureECM(nn.Module):
    """Two-level ECM restricted to SMT-T stage1 and stage2.

    The module extracts a shallow structural feature only from the two
    highest-resolution backbone stages. Stage3, stage4 and PAFR stage5 are
    deliberately excluded so that the original high-level semantic recovery
    path remains untouched.
    """

    def __init__(
        self,
        input_channels: tuple[int, int],
        edge_channels: int = 96,
    ) -> None:
        super().__init__()

        if len(input_channels) != 2:
            raise ValueError(
                "ShallowFeatureECM requires exactly two feature levels."
            )

        self.stage1_activation = _GroupedActivation5x5(
            input_channels[0],
            edge_channels,
        )
        self.stage2_activation = _GroupedActivation5x5(
            input_channels[1],
            edge_channels,
        )
        self.stage2_to_stage1 = _UpsampleConv3x3(
            edge_channels
        )
        self.final_structure = nn.Sequential(
            nn.Conv2d(
                edge_channels,
                edge_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        features: tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        if len(features) != 2:
            raise ValueError(
                "ShallowFeatureECM requires exactly two feature tensors."
            )

        stage1, stage2 = features

        stage1_structure = self.stage1_activation(
            stage1
        )
        stage2_structure = self.stage2_activation(
            stage2
        )

        stage2_structure = self.stage2_to_stage1(
            stage2_structure,
            stage1_structure.shape[-2:],
        )

        return self.final_structure(
            stage1_structure
            + stage2_structure
        )


class SMTTinyPAFRShallowECMSOD(SMTTinyPAFRSOD):
    """SMT-T + PAFR with an ECM branch restricted to stage1/stage2.

    PAFR's semantic path stays unchanged:
        stage5 -> OPM -> CRM5 -> ... -> CRM1

    The only added path is:
        stage1 + stage2 -> ShallowFeatureECM -> decoded1 fusion
    """

    input_keys = ("image",)

    def __init__(
        self,
        pretrained_path: str | Path | None,
        ecm_channels: int = 96,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path
        )

        self.ecm = ShallowFeatureECM(
            input_channels=(
                self.backbone.out_channels[0],
                self.backbone.out_channels[1],
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

    def forward(
        self,
        image: torch.Tensor,
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

        stage5 = self.pool4(
            stage4
        )
        stage5 = self.stage5_projection(
            stage5
        )
        stage5 = self.stage5(
            stage5
        )

        opm_feature = self.opm(
            stage5
        )

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

        shallow_structure = self.ecm((stage1, stage2))

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

        final_logits = self.refunet(
            side1_logits
        )

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


def build_model() -> SMTTinyPAFRShallowECMSOD:
    return SMTTinyPAFRShallowECMSOD(
        pretrained_path=PRETRAINED_PATH,
    )
