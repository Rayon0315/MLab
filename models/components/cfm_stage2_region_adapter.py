# models/components/cfm_stage2_region_adapter.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.namlab_hybrid import Hier60RegionInput


class LDFMRegionDetailAdapter(nn.Module):
    """
    Step 1:
        RGB - M60 provides region-relative local appearance detail.
        It supplements Stage2 LDFM through a gated residual.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.detail_encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.support_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )

        self.detail_projection = nn.Conv2d(
            channels,
            channels,
            1,
            bias=False,
        )

        self.scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        ldfm: torch.Tensor,
        detail_region: torch.Tensor,
    ) -> torch.Tensor:
        detail_region = F.interpolate(
            detail_region,
            size=ldfm.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        detail = self.detail_encoder(detail_region)

        support = self.support_gate(
            torch.cat([ldfm, detail], dim=1)
        )

        correction = self.detail_projection(detail) * support

        return ldfm + self.scale * correction


class TCRMRegionStatisticsAdapter(nn.Module):
    """
    Step 2:
        TCRM models channel/subspace relationships.
        M60 therefore enters through image-level channel statistics.

        T' = T + alpha_T * gamma(M60) * T
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.region_encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.statistics_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Tanh(),
        )

        self.scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        tcrm: torch.Tensor,
        mean_60_normalized: torch.Tensor,
    ) -> torch.Tensor:
        region = F.interpolate(
            mean_60_normalized,
            size=tcrm.shape[-2:],
            mode="nearest",
        )

        region = self.region_encoder(region)

        mean = region.mean(
            dim=(2, 3),
            keepdim=True,
        )

        variance = (region - mean).pow(2).mean(
            dim=(2, 3),
            keepdim=True,
        )

        std = torch.sqrt(variance + 1e-6)

        gamma = self.statistics_mlp(
            torch.cat([mean, std], dim=1)
        )

        correction = tcrm * gamma

        return tcrm + self.scale * correction


class EGPCMRegionSpatialAdapter(nn.Module):
    """
    Step 3:
        EGPCM already performs dynamic / large-kernel spatial modeling.
        M60 is used only as a light spatial context conditioner.

        G' = G + alpha_G * spatial_gate(M60) * G
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.region_encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Tanh(),
        )

        self.scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        egpcm: torch.Tensor,
        mean_60_normalized: torch.Tensor,
    ) -> torch.Tensor:
        region = F.interpolate(
            mean_60_normalized,
            size=egpcm.shape[-2:],
            mode="nearest",
        )

        region = self.region_encoder(region)
        gate = self.spatial_gate(region)
        correction = egpcm * gate

        return egpcm + self.scale * correction


class CFMStage2RegionAdapter(nn.Module):
    """
    Cumulative Stage2 CFM-aware region adapter.

    Step 1:
        LDFM <- RGB - M60

    Step 2:
        Step 1 + TCRM <- M60 channel statistics

    Step 3:
        Step 2 + EGPCM <- M60 spatial context

    SSRM is intentionally untouched in these three experiments.
    """

    def __init__(
        self,
        branch_channels: int,
        enable_ldfm: bool = True,
        enable_tcrm: bool = False,
        enable_egpcm: bool = False,
    ) -> None:
        super().__init__()

        self.region_input = Hier60RegionInput()

        self.ldfm_adapter = (
            LDFMRegionDetailAdapter(branch_channels)
            if enable_ldfm
            else None
        )

        self.tcrm_adapter = (
            TCRMRegionStatisticsAdapter(branch_channels)
            if enable_tcrm
            else None
        )

        self.egpcm_adapter = (
            EGPCMRegionSpatialAdapter(branch_channels)
            if enable_egpcm
            else None
        )

    def forward(
        self,
        branches: dict[str, torch.Tensor],
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        detail_region, mean_60_normalized = self.region_input(
            image=image,
            mean_60=mean_60,
        )

        adapted = dict(branches)

        if self.ldfm_adapter is not None:
            adapted["ldfm"] = self.ldfm_adapter(
                ldfm=branches["ldfm"],
                detail_region=detail_region,
            )

        if self.tcrm_adapter is not None:
            adapted["tcrm"] = self.tcrm_adapter(
                tcrm=branches["tcrm"],
                mean_60_normalized=mean_60_normalized,
            )

        if self.egpcm_adapter is not None:
            adapted["egpcm"] = self.egpcm_adapter(
                egpcm=branches["egpcm"],
                mean_60_normalized=mean_60_normalized,
            )

        return adapted

    def adapter_scales(self) -> dict[str, torch.Tensor]:
        scales: dict[str, torch.Tensor] = {}

        if self.ldfm_adapter is not None:
            scales["ldfm_region_detail"] = self.ldfm_adapter.scale

        if self.tcrm_adapter is not None:
            scales["tcrm_region_statistics"] = self.tcrm_adapter.scale

        if self.egpcm_adapter is not None:
            scales["egpcm_region_spatial"] = self.egpcm_adapter.scale

        return scales
