# models/networks/mambavision_baseline.py
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.mambavision import (
    MambaVisionBackbone,
    mamba_vision_tiny,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAMBAVISION_PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "mambavision"
    / "mambavision_tiny_1k.pth.tar"
)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class FusionBlock(nn.Module):
    """
    将高层特征上采样，与当前尺度的 skip feature 拼接融合。
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.refine = nn.Sequential(
            ConvBNReLU(
                in_channels=channels * 2,
                out_channels=channels,
            ),
            ConvBNReLU(
                in_channels=channels,
                out_channels=channels,
            ),
        )

    def forward(
        self,
        skip: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        high = F.interpolate(
            high,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [skip, high],
            dim=1,
        )

        return self.refine(x)


class MambaVisionSOD(nn.Module):
    def __init__(
        self,
        backbone_pretrained_path: str | Path | None,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()

        self.backbone: MambaVisionBackbone = (
            mamba_vision_tiny(
                pretrained_path=backbone_pretrained_path,
            )
        )

        c1_channels, c2_channels, c3_channels, c4_channels = (
            self.backbone.out_channels
        )

        self.lateral1 = ConvBNReLU(
            in_channels=c1_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            padding=0,
        )

        self.lateral2 = ConvBNReLU(
            in_channels=c2_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            padding=0,
        )

        self.lateral3 = ConvBNReLU(
            in_channels=c3_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            padding=0,
        )

        self.lateral4 = ConvBNReLU(
            in_channels=c4_channels,
            out_channels=decoder_channels,
            kernel_size=1,
            padding=0,
        )

        self.refine4 = nn.Sequential(
            ConvBNReLU(
                decoder_channels,
                decoder_channels,
            ),
            ConvBNReLU(
                decoder_channels,
                decoder_channels,
            ),
        )

        self.fusion3 = FusionBlock(
            channels=decoder_channels,
        )

        self.fusion2 = FusionBlock(
            channels=decoder_channels,
        )

        self.fusion1 = FusionBlock(
            channels=decoder_channels,
        )

        self.prediction_head = nn.Sequential(
            ConvBNReLU(
                in_channels=decoder_channels,
                out_channels=decoder_channels // 2,
            ),
            nn.Conv2d(
                decoder_channels // 2,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]

        c1, c2, c3, c4 = self.backbone(image)

        p4 = self.refine4(
            self.lateral4(c4)
        )

        p3 = self.fusion3(
            skip=self.lateral3(c3),
            high=p4,
        )

        p2 = self.fusion2(
            skip=self.lateral2(c2),
            high=p3,
        )

        p1 = self.fusion1(
            skip=self.lateral1(c1),
            high=p2,
        )

        logits = self.prediction_head(p1)

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": logits,
        }


def build_model() -> MambaVisionSOD:
    if not MAMBAVISION_PRETRAINED_PATH.is_file():
        raise FileNotFoundError(
            "MambaVision pretrained weights not found: "
            f"{MAMBAVISION_PRETRAINED_PATH}"
        )

    return MambaVisionSOD(
        backbone_pretrained_path=MAMBAVISION_PRETRAINED_PATH,
        decoder_channels=128,
    )