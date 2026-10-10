from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.vmamba import VMambaSmallBackbone, vmamba_small
from models.networks.vmamba_small_pafr_nam_rcrm_sod import RegionAwareCRM
from models.networks.vmamba_small_pafr_sod import (
    CRM,
    OPM,
    PAFRBasicBlock,
    PRETRAINED_PATH,
    RefUnet,
)


class VMambaSmallPAFRNAMRCRM12SOD(nn.Module):
    """VMamba-PAFR with NAM region-constrained recovery at CRM2 and CRM1.

    NAMLab region-mean evidence is used for one role only: local encoder-side
    recovery in the shallow progressive decoder. CRM3 remains the original
    PAFR CRM so the /16 semantic recovery stage is not region-averaged.
    """

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
        self.crm3 = CRM(256, self.backbone.out_channels[2], 128)
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
        decoded3, _ = self.crm3(x, stage3)

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


def build_model() -> VMambaSmallPAFRNAMRCRM12SOD:
    return VMambaSmallPAFRNAMRCRM12SOD(
        pretrained_path=PRETRAINED_PATH,
    )
