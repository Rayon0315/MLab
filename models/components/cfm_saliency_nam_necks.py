"""Evidence ablations retaining the exact successful SaliencyAdapterStream."""
from __future__ import annotations

import torch
from torch import nn

from models.components.cfm_external_necks import ConvBNAct, FrequencyReconstructionAdapter
from models.components.cfm_iram_saliency_adapter import SaliencyAdapterStream
from models.components.nam_region_stream import NAMRegionStream


class EvidenceFusion(nn.Module):
    """Same strong reconstruction as current best; only concat width changes."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.reconstruction = nn.Sequential(
            ConvBNAct(input_channels, output_channels, kernel_size=1),
            ConvBNAct(output_channels, output_channels, kernel_size=3),
        )

    def forward(self, streams):
        return self.reconstruction(torch.cat(streams, dim=1))


class SaliencyEvidenceNeck(nn.Module):
    def __init__(self, use_iram=False, use_nam=False, x_stream=None):
        super().__init__()
        self.x_stream = x_stream if x_stream is not None else SaliencyAdapterStream()
        self.freq2 = FrequencyReconstructionAdapter(192) if use_iram else None
        self.freq3 = FrequencyReconstructionAdapter(384) if use_iram else None
        if use_iram:
            with torch.no_grad():
                self.freq2.scale.fill_(1.0)
                self.freq3.scale.fill_(1.0)
        self.region_stream = NAMRegionStream() if use_nam else None
        self.fuse2 = EvidenceFusion(192 + 128 + (192 if use_iram else 0) + (128 if use_nam else 0), 192)
        self.fuse3 = EvidenceFusion(384 + 128 + (384 if use_iram else 0) + (128 if use_nam else 0), 384)

    def forward(self, features, mean_60=None):
        f1, f2, f3, f4 = features
        sal2, sal3, coarse = self.x_stream(features)
        streams2, streams3 = [f2], [f3]
        if self.freq2 is not None:
            streams2.append(self.freq2(f2))
            streams3.append(self.freq3(f3))
        streams2.append(sal2)
        streams3.append(sal3)
        if self.region_stream is not None:
            nam2, nam3 = self.region_stream(mean_60, f2, f3)
            streams2.append(nam2)
            streams3.append(nam3)
        return [f1, self.fuse2(streams2), self.fuse3(streams3), f4], coarse
