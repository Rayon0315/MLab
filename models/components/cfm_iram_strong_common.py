"""Shared strong fusion; the canonical IRAM implementation is reused verbatim."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import ConvBNAct, FrequencyReconstructionAdapter


def resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)


class ThreeStreamFusion(nn.Module):
    """[base C, frequency C, X 128] -> C, without an outer residual/gate."""

    def __init__(self, channels: int, stream_channels: int = 128):
        super().__init__()
        self.reconstruction = nn.Sequential(
            ConvBNAct(2 * channels + stream_channels, channels, kernel_size=1),
            ConvBNAct(channels, channels, kernel_size=3),
        )

    def forward(self, base, frequency, task_feature):
        return self.reconstruction(torch.cat([base, frequency, task_feature], dim=1))


class IRAMStrongNeck(nn.Module):
    """Parallel IRAM and X; X never controls the IRAM branch.

    The inherited adapter computes F + scale * delta_IRAM. Only the initial
    value of scale in these NEW instances is changed to 1; its FFT, frequency
    weights, spatial attention and refinement remain the canonical code.
    """

    def __init__(self, x_stream: nn.Module):
        super().__init__()
        self.x_stream = x_stream
        self.freq2 = FrequencyReconstructionAdapter(192)
        self.freq3 = FrequencyReconstructionAdapter(384)
        with torch.no_grad():
            self.freq2.scale.fill_(1.0)
            self.freq3.scale.fill_(1.0)
        self.fuse2 = ThreeStreamFusion(192)
        self.fuse3 = ThreeStreamFusion(384)

    def forward(self, features):
        f1, f2, f3, f4 = features
        x2, x3, coarse = self.x_stream(features)
        out2 = self.fuse2(f2, self.freq2(f2), x2)
        out3 = self.fuse3(f3, self.freq3(f3), x3)
        return [f1, out2, out3, f4], coarse
