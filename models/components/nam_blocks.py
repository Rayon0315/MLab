# models/components/nam_blocks.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
    SaliencyGuidedFusion,
)


class EdgeSeedBlock(nn.Module):
    """Use the deepest semantic feature to initialize the edge stream."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels),
            ResidualConvBlock(channels),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.block(feature)


class HierarchicalEdgeConstructionBlock(nn.Module):
    """
    Refine a coarse edge feature at a finer RGB feature scale.

    Inputs:
        rgb_feature: current-scale RGB feature
        coarse_edge: edge feature from the previous coarser scale
        nam_map: one NAMLab hierarchy map
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.rgb_refine = ResidualConvBlock(channels)
        self.coarse_refine = ResidualConvBlock(channels)
        self.nam_encoder = nn.Sequential(
            ConvNormAct(1, channels // 4),
            ConvNormAct(channels // 4, channels),
            ResidualConvBlock(channels),
        )
        self.fusion = nn.Sequential(
            ConvNormAct(channels * 3, channels),
            ResidualConvBlock(channels),
        )

    def forward(
        self,
        rgb_feature: torch.Tensor,
        coarse_edge: torch.Tensor,
        nam_map: torch.Tensor,
    ) -> torch.Tensor:
        target_size = rgb_feature.shape[-2:]

        coarse_edge = F.interpolate(
            coarse_edge,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        nam_map = F.adaptive_max_pool2d(
            nam_map,
            output_size=target_size,
        )

        rgb_feature = self.rgb_refine(rgb_feature)
        coarse_edge = self.coarse_refine(coarse_edge)
        nam_feature = self.nam_encoder(nam_map)

        refined_edge = self.fusion(
            torch.cat(
                [rgb_feature, coarse_edge, nam_feature],
                dim=1,
            )
        )
        return coarse_edge + refined_edge


class HierarchicalEdgeConstructionModule(nn.Module):
    """
    Progressive NAMLab hierarchy construction.

    hier_20 -> stage3, stride 16
    hier_40 -> stage2, stride 8
    hier_60 -> stage1, stride 4
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.seed4 = EdgeSeedBlock(channels)
        self.edge3 = HierarchicalEdgeConstructionBlock(channels)
        self.edge2 = HierarchicalEdgeConstructionBlock(channels)
        self.edge1 = HierarchicalEdgeConstructionBlock(channels)

    def forward(
        self,
        feature1: torch.Tensor,
        feature2: torch.Tensor,
        feature3: torch.Tensor,
        feature4: torch.Tensor,
        nam_20: torch.Tensor,
        nam_40: torch.Tensor,
        nam_60: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        edge4 = self.seed4(feature4)
        edge3 = self.edge3(
            rgb_feature=feature3,
            coarse_edge=edge4,
            nam_map=nam_20,
        )
        edge2 = self.edge2(
            rgb_feature=feature2,
            coarse_edge=edge3,
            nam_map=nam_40,
        )
        edge1 = self.edge1(
            rgb_feature=feature1,
            coarse_edge=edge2,
            nam_map=nam_60,
        )
        return edge1, edge2, edge3, edge4


class EdgeGuidedSaliencyFusion(nn.Module):
    """
    Run the baseline saliency-guided fusion first, then inject the edge
    feature through a gate generated jointly by saliency and edge semantics.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.saliency_fusion = SaliencyGuidedFusion(channels)
        self.edge_refine = ResidualConvBlock(channels)
        self.edge_gate = nn.Conv2d(
            channels * 2,
            channels,
            kernel_size=1,
        )
        self.output_fusion = nn.Sequential(
            ConvNormAct(channels * 2, channels),
            ResidualConvBlock(channels),
        )

    def forward(
        self,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
        guide_logits: torch.Tensor,
        edge_feature: torch.Tensor,
    ) -> torch.Tensor:
        saliency_feature = self.saliency_fusion(
            low_feature=low_feature,
            high_feature=high_feature,
            guide_logits=guide_logits,
        )

        edge_feature = F.interpolate(
            edge_feature,
            size=saliency_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        edge_feature = self.edge_refine(edge_feature)

        edge_gate = torch.sigmoid(
            self.edge_gate(
                torch.cat(
                    [saliency_feature, edge_feature],
                    dim=1,
                )
            )
        )
        guided_edge = edge_feature * edge_gate

        fused_feature = self.output_fusion(
            torch.cat(
                [saliency_feature, guided_edge],
                dim=1,
            )
        )
        return saliency_feature + fused_feature
