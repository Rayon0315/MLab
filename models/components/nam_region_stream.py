"""Independent region-color feature stream using the existing mean_60 format."""
from __future__ import annotations

import torch
from torch import nn

from models.components.cfm_external_necks import ConvBNAct
from models.components.cfm_iram_strong_common import resize_like


def _downsample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class NAMRegionStream(nn.Module):
    """[0,1] RGB mean -> normalized color -> stride8/16 region representations.

    Uses the same ImageNet color normalization as historical Hier60RegionInput.
    Does not read RGB features except their spatial sizes and never gates them.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.stem = nn.Sequential(_downsample(3, 32), _downsample(32, 64))
        self.stage2 = _downsample(64, 128)
        self.stage3 = _downsample(128, 128)
        self.project2 = ConvBNAct(128, 128, kernel_size=1)
        self.project3 = ConvBNAct(128, 128, kernel_size=1)

    def forward(self, mean_60, feature2, feature3):
        normalized = (mean_60 - self.image_mean) / self.image_std
        region2 = self.stage2(self.stem(normalized))
        region3 = self.stage3(region2)
        return (resize_like(self.project2(region2), feature2),
                resize_like(self.project3(region3), feature3))
