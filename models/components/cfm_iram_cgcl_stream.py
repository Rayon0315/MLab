"""B: CGCL-like centroid-relative reconstruction, not a paper reproduction."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import ConvBNAct
from models.components.cfm_iram_strong_common import resize_like


def centroid_geometry(coarse_logits, output_size):
    """Differentiable soft centroid and nine center-relative spatial channels.

    Channels: dx, dy, squared distance, center, surround, four soft quadrants.
    Coordinates span [-1,1]; center/surround radius is 0.35 in that space.
    """
    with torch.autocast(device_type=coarse_logits.device.type, enabled=False):
        probability = coarse_logits.float().sigmoid()
        h, w = probability.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=probability.device),
            torch.linspace(-1, 1, w, device=probability.device), indexing="ij")
        mass = probability.sum(dim=(-2, -1)).clamp_min(1e-6)
        cx = (probability * xx).sum(dim=(-2, -1)) / mass
        cy = (probability * yy).sum(dim=(-2, -1)) / mass
        height, width = output_size
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, height, device=probability.device),
            torch.linspace(-1, 1, width, device=probability.device), indexing="ij")
        dx = xx[None, None] - cx[:, :, None, None]
        dy = yy[None, None] - cy[:, :, None, None]
        distance2 = dx.square() + dy.square()
        center = torch.exp(-distance2 / (2 * 0.35 ** 2))
        right, bottom = (4 * dx).sigmoid(), (4 * dy).sigmoid()
        geometry = torch.cat([
            dx, dy, distance2, center, 1 - center,
            (1 - right) * (1 - bottom), right * (1 - bottom),
            (1 - right) * bottom, right * bottom,
        ], dim=1)
        return geometry, torch.cat([cx, cy], dim=1)


class SpatialReconstruction(nn.Module):
    def __init__(self, in_channels, channels=128):
        super().__init__()
        self.local = ConvBNAct(in_channels, channels, kernel_size=1)
        self.position = ConvBNAct(9, 32, kernel_size=1)
        self.reconstruct = nn.Sequential(
            ConvBNAct(4 * channels + 32, channels, kernel_size=1),
            ConvBNAct(channels, channels, kernel_size=3),
        )

    @staticmethod
    def _pool(feature, weight):
        return (feature * weight).sum((-2, -1), keepdim=True) / weight.sum((-2, -1), keepdim=True).clamp_min(1e-6)

    def forward(self, feature, semantic, coarse):
        local = self.local(feature)
        semantic = resize_like(semantic, local)
        geometry, _ = centroid_geometry(coarse, local.shape[-2:])
        with torch.autocast(device_type=local.device.type, enabled=False):
            probability = resize_like(coarse.float().sigmoid(), local)
            center_weight = geometry[:, 3:4] * probability
            surround_weight = geometry[:, 4:5] * (1 - probability)
            center_context = self._pool(semantic.float(), center_weight)
            surround_context = self._pool(semantic.float(), surround_weight)
        center_context = center_context.to(local.dtype).expand_as(local)
        surround_context = surround_context.to(local.dtype).expand_as(local)
        position = self.position(geometry.to(local.dtype))
        return self.reconstruct(torch.cat([
            local, semantic, center_context, surround_context, position,
        ], dim=1))


class SpatialOrganizationStream(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.project2 = ConvBNAct(192, channels, kernel_size=1)
        self.project3 = ConvBNAct(384, channels, kernel_size=1)
        self.project4 = ConvBNAct(768, channels, kernel_size=1)
        self.aggregate = nn.Sequential(
            ConvBNAct(3 * channels, channels, kernel_size=1),
            ConvBNAct(channels, channels, kernel_size=3),
        )
        self.coarse_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.spatial2 = SpatialReconstruction(192, channels)
        self.spatial3 = SpatialReconstruction(384, channels)

    def forward(self, features):
        _, f2, f3, f4 = features
        semantic = self.aggregate(torch.cat([
            resize_like(self.project2(f2), f3), self.project3(f3),
            resize_like(self.project4(f4), f3),
        ], dim=1))
        coarse = self.coarse_head(semantic)
        return (self.spatial2(f2, semantic, coarse),
                self.spatial3(f3, semantic, coarse), coarse)
