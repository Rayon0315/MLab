from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.vmamba import VMambaSmallBackbone, vmamba_small


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "vmamba"
    / "vssm_small_0229_ckpt_epoch_222.pth"
)


class PAFRBasicBlock(nn.Module):
    """ResNet BasicBlock used by the extra PAFR stage after VMamba stage 4."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class OPM(nn.Module):
    """Object Position Module from the official PAFR implementation."""

    def __init__(self, channels: int = 512) -> None:
        super().__init__()
        self.convx0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bnx0 = nn.BatchNorm2d(channels)
        self.convx1 = nn.Conv2d(channels, channels * 2, 3, padding=1)
        self.convx2 = nn.Conv2d(channels, channels, 1)
        self.bnx2 = nn.BatchNorm2d(channels)
        hidden = max(channels // 16, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_0 = F.relu(self.bnx0(self.convx0(x)), inplace=True)
        wb = self.convx1(x)
        weight, bias = wb.chunk(2, dim=1)
        transformed = F.relu(weight * x_0 + bias, inplace=True)

        projected = F.relu(self.bnx2(self.convx2(x)), inplace=True)
        channel_weight = self.conv_du(self.avg_pool(transformed))
        return projected * channel_weight


class CRM(nn.Module):
    """Context Refinement Module from PAFR, with channels adapted per VMamba stage."""

    def __init__(
        self,
        left_channels: int,
        encoder_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv_l1 = nn.Conv2d(left_channels, out_channels, 3, padding=1)
        self.bn_l1 = nn.BatchNorm2d(out_channels)
        self.conv_d1 = nn.Conv2d(encoder_channels, out_channels, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden = max(out_channels // 16, 1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(out_channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=True),
        )
        self.conv_d2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(
        self,
        left: torch.Tensor,
        encoder_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_feature = F.relu(self.bn_l1(self.conv_l1(left)), inplace=True)
        down_feature = F.relu(
            self.bn_d1(self.conv_d1(encoder_feature)),
            inplace=True,
        )
        down_feature = self.conv_d2(down_feature)

        channel_attention = torch.sigmoid(
            self.conv_du(self.avg_pool(left_feature))
        )
        channel_feature = channel_attention * left_feature

        aligned_down = down_feature
        if aligned_down.shape[-2:] != left_feature.shape[-2:]:
            aligned_down = F.interpolate(
                aligned_down,
                size=left_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        spatial_feature = F.relu(left_feature * aligned_down, inplace=True)
        fused_left = channel_feature + spatial_feature
        fused = torch.cat((fused_left, spatial_feature), dim=1)
        output = F.relu(self.bn3(self.conv3(fused)), inplace=True)
        return output, down_feature


class RefUnet(nn.Module):
    """PAFR refinement U-Net.

    The official code uses fixed x2 upsampling. Here interpolation targets the
    corresponding skip size, preserving the same topology while allowing the
    scaled SRC branch to use odd spatial sizes safely.
    """

    def __init__(self, in_channels: int = 1, stem_channels: int = 64) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, stem_channels, 3, padding=1)

        self.conv1 = nn.Conv2d(stem_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.conv_d4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d4 = nn.BatchNorm2d(64)
        self.conv_d3 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d3 = nn.BatchNorm2d(64)
        self.conv_d2 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d2 = nn.BatchNorm2d(64)
        self.conv_d1 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_d1 = nn.BatchNorm2d(64)
        self.conv_d0 = nn.Conv2d(64, 1, 3, padding=1)

    @staticmethod
    def _up_to(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x
        x = self.conv0(x)

        x1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.pool(x1)
        x2 = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.pool(x2)
        x3 = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.pool(x3)
        x4 = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.pool(x4)
        x5 = F.relu(self.bn5(self.conv5(x)), inplace=True)

        x = self._up_to(x5, x4)
        d4 = F.relu(self.bn_d4(self.conv_d4(torch.cat((x, x4), dim=1))), inplace=True)
        x = self._up_to(d4, x3)
        d3 = F.relu(self.bn_d3(self.conv_d3(torch.cat((x, x3), dim=1))), inplace=True)
        x = self._up_to(d3, x2)
        d2 = F.relu(self.bn_d2(self.conv_d2(torch.cat((x, x2), dim=1))), inplace=True)
        x = self._up_to(d2, x1)
        d1 = F.relu(self.bn_d1(self.conv_d1(torch.cat((x, x1), dim=1))), inplace=True)

        return residual_input + self.conv_d0(d1)


class VMambaSmallPAFRSOD(nn.Module):
    """PAFR decoder/training outputs with VMamba-S as the encoder.

    Mapping from official PAFR to VMamba:
      PAFR h1/h2/h3/h4 <- VMamba stage1/stage2/stage3/stage4
      PAFR h5          <- max-pool(stage4) + 768->512 projection + 3 BasicBlocks

    OPM, five-stage CRM recovery, seven side predictions and RefUnet are kept.
    Predictions are returned as logits so the standard MLab test.py can apply
    sigmoid exactly once.
    """

    input_keys = ("image",)

    def __init__(
        self,
        pretrained_path: str | Path | None,
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
        self.crm2 = CRM(128, self.backbone.out_channels[1], 64)
        self.crm1 = CRM(64, self.backbone.out_channels[0], 64)

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

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
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

        opm_logits = self._resize(self.out_opm(opm_feature), input_size)
        side5_logits = self._resize(self.out5(decoded5), input_size)
        side4_logits = self._resize(self.out4(decoded4), input_size)
        side3_logits = self._resize(self.out3(decoded3), input_size)
        side2_logits = self._resize(self.out2(decoded2), input_size)
        side1_logits = self._resize(self.out1(decoded1), input_size)

        # Official PAFR refines its full-resolution first decoder output.
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


def build_model() -> VMambaSmallPAFRSOD:
    return VMambaSmallPAFRSOD(
        pretrained_path=PRETRAINED_PATH,
    )
