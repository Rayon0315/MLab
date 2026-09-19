"""A: supervised FG/BG relations reconstructed as a full feature stream."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import ConvBNAct


class PrototypeStream(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.anchor = ConvBNAct(768, channels, kernel_size=1)
        self.coarse_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.target2 = ConvBNAct(192, channels, kernel_size=1)
        self.target3 = ConvBNAct(384, channels, kernel_size=1)
        self.relation2 = self._relation_encoder(channels)
        self.relation3 = self._relation_encoder(channels)

    @staticmethod
    def _relation_encoder(channels):
        return nn.Sequential(
            ConvBNAct(3 * channels + 1, channels, kernel_size=1),
            ConvBNAct(channels, channels, kernel_size=3),
        )

    @staticmethod
    def _prototype(anchor, weights):
        # Accumulation in FP32 also handles near-empty soft FG/BG under AMP.
        numerator = (anchor * weights).sum(dim=(-2, -1), keepdim=True)
        return numerator / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _relations(feature, foreground, background):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            unit = F.normalize(feature.float(), dim=1, eps=1e-6)
            fg = (unit * F.normalize(foreground, dim=1, eps=1e-6)).sum(1, keepdim=True)
            bg = (unit * F.normalize(background, dim=1, eps=1e-6)).sum(1, keepdim=True)
        fg, bg = fg.to(feature.dtype), bg.to(feature.dtype)
        return torch.cat([feature, feature * fg, feature * bg, fg - bg], dim=1)

    def forward(self, features):
        _, f2, f3, f4 = features
        anchor = self.anchor(f4)
        coarse = self.coarse_head(anchor)
        with torch.autocast(device_type=anchor.device.type, enabled=False):
            probability = coarse.float().sigmoid()
            foreground = self._prototype(anchor.float(), probability)
            background = self._prototype(anchor.float(), 1.0 - probability)
        proto2 = self.relation2(self._relations(self.target2(f2), foreground, background))
        proto3 = self.relation3(self._relations(self.target3(f3), foreground, background))
        return proto2, proto3, coarse
