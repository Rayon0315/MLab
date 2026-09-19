"""C: parallel semantic, regional and structural task representations in a neck."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import ConvBNAct
from models.components.cfm_iram_strong_common import resize_like


class SaliencyAdapterStream(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.project = nn.ModuleList([ConvBNAct(c, 64, kernel_size=1)
                                      for c in (96, 192, 384, 768)])
        self.semantic = ConvBNAct(128, channels, kernel_size=3)
        self.region = nn.Sequential(
            ConvBNAct(64 + channels, channels, kernel_size=1),
            ConvBNAct(channels, channels, kernel_size=3),
        )
        self.structure = ConvBNAct(128, channels, kernel_size=3)
        self.reconstruct = nn.Sequential(
            ConvBNAct(3 * channels, channels, kernel_size=1),
            ConvBNAct(channels, channels, kernel_size=3),
        )
        self.coarse_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.saliency2 = ConvBNAct(channels, channels, kernel_size=3)
        self.saliency3 = ConvBNAct(channels, channels, kernel_size=3)

    def forward(self, features):
        p1, p2, p3, p4 = [project(f) for project, f in zip(self.project, features)]
        # One F2 working grid and two output adapters, not a new top-down decoder.
        semantic = self.semantic(torch.cat([resize_like(p3, p2), resize_like(p4, p2)], dim=1))
        region = self.region(torch.cat([
            p2, F.avg_pool2d(semantic, 7, stride=1, padding=3),
        ], dim=1))
        high1 = p1 - F.avg_pool2d(p1, 3, stride=1, padding=1)
        high2 = p2 - F.avg_pool2d(p2, 3, stride=1, padding=1)
        structure = self.structure(torch.cat([resize_like(high1, p2), high2], dim=1))
        task = self.reconstruct(torch.cat([semantic, region, structure], dim=1))
        return self.saliency2(task), self.saliency3(resize_like(task, p3)), self.coarse_head(task)
