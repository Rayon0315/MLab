from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.vmamba_small_pafr_sod import (
    CRM,
    PRETRAINED_PATH,
    VMambaSmallPAFRSOD,
)


class _GroupedActivation5x5(nn.Module):
    """NESS-style receptive-field activation adapted to arbitrary channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        groups = math.gcd(in_channels, out_channels)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpsampleConv3x3(nn.Module):
    """2x-style top-down sampling step with exact target-size alignment."""

    def __init__(self, channels: int) -> None:
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
        return self.relu(self.conv(x))


class _DepthwiseSeparablePriorFusion(nn.Module):
    """Depthwise-separable fusion used for the final NAM structural prior."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RegionMeanECM(nn.Module):
    """NESS-style Edge Construction Module using NAMLab region mean.

    The five semantic feature levels are first projected to a shared channel
    width with 5x5 grouped convolutions. A top-down U-shaped cascade builds a
    multi-level structural feature. Unlike NESS-Net, which concatenates a
    1-channel NAMLab binary edge prior, this variant concatenates the 3-channel
    NAMLab region-mean representation. The prior is used only here.
    """

    def __init__(
        self,
        input_channels: tuple[int, int, int, int, int],
        edge_channels: int = 96,
    ) -> None:
        super().__init__()
        if len(input_channels) != 5:
            raise ValueError("RegionMeanECM requires exactly five feature levels")

        self.edge_channels = int(edge_channels)
        self.stage_activations = nn.ModuleList(
            [
                _GroupedActivation5x5(channels, self.edge_channels)
                for channels in input_channels
            ]
        )
        self.top_down_steps = nn.ModuleList(
            [_UpsampleConv3x3(self.edge_channels) for _ in range(4)]
        )
        self.final_structure = nn.Sequential(
            nn.Conv2d(
                self.edge_channels,
                self.edge_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )
        self.prior_fusion = _DepthwiseSeparablePriorFusion(
            in_channels=self.edge_channels + 3,
            out_channels=self.edge_channels,
        )

    def forward(
        self,
        features: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        region_mean: torch.Tensor,
    ) -> torch.Tensor:
        if len(features) != 5:
            raise ValueError("RegionMeanECM requires exactly five feature tensors")
        if region_mean.ndim != 4 or region_mean.shape[1] != 3:
            raise ValueError("region_mean must be a 3-channel NCHW tensor")

        activated = [
            block(feature)
            for block, feature in zip(self.stage_activations, features)
        ]

        top_down = self.top_down_steps[0](
            activated[4],
            activated[3].shape[-2:],
        )
        top_down = activated[3] + top_down

        top_down = self.top_down_steps[1](
            top_down,
            activated[2].shape[-2:],
        )
        top_down = activated[2] + top_down

        top_down = self.top_down_steps[2](
            top_down,
            activated[1].shape[-2:],
        )
        top_down = activated[1] + top_down

        top_down = self.top_down_steps[3](
            top_down,
            activated[0].shape[-2:],
        )
        semantic_structure = self.final_structure(
            activated[0] + top_down
        )

        mean = F.interpolate(
            region_mean.float(),
            size=semantic_structure.shape[-2:],
            mode="nearest",
        ).to(
            device=semantic_structure.device,
            dtype=semantic_structure.dtype,
        )

        return self.prior_fusion(
            torch.cat((semantic_structure, mean), dim=1)
        )


class VMambaSmallPAFRNAMRMECMSOD(VMambaSmallPAFRSOD):
    """VMamba-PAFR with a NESS-style region-mean ECM structural branch.

    PAFR's semantic/progressive recovery path is left intact. NAMLab evidence
    appears in one functional module only: RegionMeanECM. Its output is fused
    with decoded1 immediately before the original out1/RefUnet prediction path.
    """

    input_keys = ("image", "mean_60")

    def __init__(
        self,
        pretrained_path: str | Path | None,
        ecm_channels: int = 96,
    ) -> None:
        super().__init__(pretrained_path=pretrained_path)

        self.ecm = RegionMeanECM(
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

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        input_size = image.shape[-2:]
        stage1, stage2, stage3, stage4 = self.backbone(image)

        stage5 = self.pool4(stage4)
        stage5 = self.stage5_projection(stage5)
        stage5 = self.stage5(stage5)

        opm_feature = self.opm(stage5)

        decoded5, _ = self.crm5(opm_feature, stage5)
        x = self._resize(decoded5, stage4.shape[-2:])
        decoded4, _ = self.crm4(x, stage4)

        x = self._resize(decoded4, stage3.shape[-2:])
        decoded3, _ = self.crm3(x, stage3)

        x = self._resize(decoded3, stage2.shape[-2:])
        decoded2, _ = self.crm2(x, stage2)

        x = self._resize(decoded2, stage1.shape[-2:])
        decoded1, _ = self.crm1(x, stage1)

        structure_feature = self.ecm(
            (stage1, stage2, stage3, stage4, stage5),
            mean_60,
        )
        fused_decoded1 = self.structure_fusion(
            torch.cat((decoded1, structure_feature), dim=1)
        )

        opm_logits = self._resize(self.out_opm(opm_feature), input_size)
        side5_logits = self._resize(self.out5(decoded5), input_size)
        side4_logits = self._resize(self.out4(decoded4), input_size)
        side3_logits = self._resize(self.out3(decoded3), input_size)
        side2_logits = self._resize(self.out2(decoded2), input_size)
        side1_logits = self._resize(self.out1(fused_decoded1), input_size)
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


def build_model() -> VMambaSmallPAFRNAMRMECMSOD:
    return VMambaSmallPAFRNAMRMECMSOD(
        pretrained_path=PRETRAINED_PATH,
    )
