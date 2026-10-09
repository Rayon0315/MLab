from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.vmamba import VMambaSmallBackbone, vmamba_small
from models.networks.vmamba_small_pafr_sod import (
    CRM,
    OPM,
    PAFRBasicBlock,
    PRETRAINED_PATH,
    RefUnet,
)


class RegionAffinityRecovery(nn.Module):
    """NAMLab region-mean guided local feature recovery.

    A single NAM evidence role is used throughout the decoder: local encoder
    evidence is aggregated strongly inside region-mean-consistent areas and
    weakly across region boundaries. The result is blended residually with
    the original feature through a bounded learnable gate.
    """

    _OFFSETS = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    def __init__(
        self,
        sigma: float = 0.1,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if not 0.0 < gamma_init < 1.0:
            raise ValueError("gamma_init must be in (0, 1)")

        self.sigma = float(sigma)
        gamma_logit = math.log(gamma_init / (1.0 - gamma_init))
        self.raw_gamma = nn.Parameter(
            torch.tensor(gamma_logit, dtype=torch.float32)
        )

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gamma)

    @staticmethod
    def _shift_view(
        padded: torch.Tensor,
        dy: int,
        dx: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        y0 = 1 + dy
        x0 = 1 + dx
        return padded[..., y0:y0 + height, x0:x0 + width]

    def forward(
        self,
        feature: torch.Tensor,
        region_mean: torch.Tensor,
    ) -> torch.Tensor:
        if feature.ndim != 4 or region_mean.ndim != 4:
            raise ValueError("feature and region_mean must be NCHW tensors")
        if feature.shape[0] != region_mean.shape[0]:
            raise ValueError("feature and region_mean batch sizes must match")

        height, width = feature.shape[-2:]
        mean = F.interpolate(
            region_mean.float(),
            size=(height, width),
            mode="nearest",
        )

        padded_feature = F.pad(feature, (1, 1, 1, 1), value=0.0)
        padded_mean = F.pad(mean, (1, 1, 1, 1), value=0.0)
        valid = torch.ones(
            (feature.shape[0], 1, height, width),
            dtype=torch.float32,
            device=feature.device,
        )
        padded_valid = F.pad(valid, (1, 1, 1, 1), value=0.0)

        numerator = feature
        denominator = torch.ones(
            (feature.shape[0], 1, height, width),
            dtype=feature.dtype,
            device=feature.device,
        )

        sigma_squared_twice = 2.0 * self.sigma * self.sigma

        for dy, dx in self._OFFSETS:
            neighbor_feature = self._shift_view(
                padded_feature,
                dy,
                dx,
                height,
                width,
            )
            neighbor_mean = self._shift_view(
                padded_mean,
                dy,
                dx,
                height,
                width,
            )
            neighbor_valid = self._shift_view(
                padded_valid,
                dy,
                dx,
                height,
                width,
            )

            distance_squared = (mean - neighbor_mean).square().sum(
                dim=1,
                keepdim=True,
            )
            affinity = torch.exp(
                -distance_squared / sigma_squared_twice
            )
            affinity = affinity * neighbor_valid
            affinity = affinity.to(dtype=feature.dtype)

            numerator = numerator + affinity * neighbor_feature
            denominator = denominator + affinity

        recovered = numerator / denominator.clamp_min(1e-6)
        gamma = self.gamma.to(dtype=feature.dtype, device=feature.device)
        return feature + gamma * (recovered - feature)


class RegionAwareCRM(nn.Module):
    """PAFR CRM whose encoder-side feature is recovered with NAM affinity."""

    def __init__(
        self,
        left_channels: int,
        encoder_channels: int,
        out_channels: int,
        region_sigma: float = 0.1,
        region_gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.region_recovery = RegionAffinityRecovery(
            sigma=region_sigma,
            gamma_init=region_gamma_init,
        )
        self.crm = CRM(
            left_channels=left_channels,
            encoder_channels=encoder_channels,
            out_channels=out_channels,
        )

    def forward(
        self,
        left: torch.Tensor,
        encoder_feature: torch.Tensor,
        region_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recovered_encoder = self.region_recovery(
            encoder_feature,
            region_mean,
        )
        return self.crm(left, recovered_encoder)


class VMambaSmallPAFRNAMRCRMSOD(nn.Module):
    """VMamba-PAFR with NAM region-constrained recovery at CRM3/CRM2/CRM1."""

    input_keys = ("image", "mean_60")

    def __init__(
        self,
        pretrained_path: str | Path | None,
        region_sigma: float = 0.1,
        region_gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone: VMambaSmallBackbone = vmamba_small(
            pretrained_path=pretrained_path,
        )

        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage5_projection = nn.Sequential(
            nn.Conv2d(self.backbone.out_channels[3], 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.stage5 = nn.Sequential(
            PAFRBasicBlock(512),
            PAFRBasicBlock(512),
            PAFRBasicBlock(512),
        )

        self.opm = OPM(512)
        self.crm5 = CRM(512, 512, 512)
        self.crm4 = CRM(512, self.backbone.out_channels[3], 256)
        self.crm3 = RegionAwareCRM(
            256,
            self.backbone.out_channels[2],
            128,
            region_sigma=region_sigma,
            region_gamma_init=region_gamma_init,
        )
        self.crm2 = RegionAwareCRM(
            128,
            self.backbone.out_channels[1],
            64,
            region_sigma=region_sigma,
            region_gamma_init=region_gamma_init,
        )
        self.crm1 = RegionAwareCRM(
            64,
            self.backbone.out_channels[0],
            64,
            region_sigma=region_sigma,
            region_gamma_init=region_gamma_init,
        )

        self.out_opm = nn.Conv2d(512, 1, 3, padding=1)
        self.out5 = nn.Conv2d(512, 1, 3, padding=1)
        self.out4 = nn.Conv2d(256, 1, 3, padding=1)
        self.out3 = nn.Conv2d(128, 1, 3, padding=1)
        self.out2 = nn.Conv2d(64, 1, 3, padding=1)
        self.out1 = nn.Conv2d(64, 1, 3, padding=1)

        self.refunet = RefUnet(1, 64)

    @staticmethod
    def _resize(
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
        decoded3, _ = self.crm3(x, stage3, mean_60)

        x = self._resize(decoded3, stage2.shape[-2:])
        decoded2, _ = self.crm2(x, stage2, mean_60)

        x = self._resize(decoded2, stage1.shape[-2:])
        decoded1, _ = self.crm1(x, stage1, mean_60)

        opm_logits = self._resize(self.out_opm(opm_feature), input_size)
        side5_logits = self._resize(self.out5(decoded5), input_size)
        side4_logits = self._resize(self.out4(decoded4), input_size)
        side3_logits = self._resize(self.out3(decoded3), input_size)
        side2_logits = self._resize(self.out2(decoded2), input_size)
        side1_logits = self._resize(self.out1(decoded1), input_size)
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


def build_model() -> VMambaSmallPAFRNAMRCRMSOD:
    return VMambaSmallPAFRNAMRCRMSOD(
        pretrained_path=PRETRAINED_PATH,
    )
