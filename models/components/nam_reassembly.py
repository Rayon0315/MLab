# models/components/nam_reassembly.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
)


class StructureGuidedReassembly(nn.Module):
    """
    Reassemble a coarse semantic feature at a finer resolution.

    The structure feature predicts multiple sampling offsets and their
    position-wise weights. Zero initialization makes the initial behavior
    equivalent to ordinary bilinear upsampling.
    """

    def __init__(
        self,
        channels: int,
        num_samples: int = 4,
        max_offset: float = 2.0,
    ) -> None:
        super().__init__()

        self.num_samples = num_samples
        self.max_offset = max_offset

        self.high_refine = ResidualConvBlock(channels)

        self.offset_head = nn.Conv2d(
            channels,
            num_samples * 2,
            kernel_size=3,
            padding=1,
        )
        self.weight_head = nn.Conv2d(
            channels,
            num_samples,
            kernel_size=3,
            padding=1,
        )

        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)

    def forward(
        self,
        high_feature: torch.Tensor,
        structure_feature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = high_feature.shape[0]
        target_height, target_width = structure_feature.shape[-2:]
        source_height, source_width = high_feature.shape[-2:]

        high_feature = self.high_refine(high_feature)

        offsets = torch.tanh(
            self.offset_head(structure_feature)
        )
        offsets = offsets * self.max_offset
        offsets = offsets.view(
            batch_size,
            self.num_samples,
            2,
            target_height,
            target_width,
        )
        offsets = offsets.permute(0, 1, 3, 4, 2)

        sample_weights = torch.softmax(
            self.weight_head(structure_feature),
            dim=1,
        )

        theta = high_feature.new_tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        theta = theta.unsqueeze(0).repeat(
            batch_size,
            1,
            1,
        )

        base_grid = F.affine_grid(
            theta,
            size=(
                batch_size,
                high_feature.shape[1],
                target_height,
                target_width,
            ),
            align_corners=False,
        )

        offset_x = offsets[..., 0] * (
            2.0 / source_width
        )
        offset_y = offsets[..., 1] * (
            2.0 / source_height
        )
        normalized_offsets = torch.stack(
            [offset_x, offset_y],
            dim=-1,
        )

        reassembled_feature = torch.zeros(
            (
                batch_size,
                high_feature.shape[1],
                target_height,
                target_width,
            ),
            device=high_feature.device,
            dtype=high_feature.dtype,
        )

        for sample_index in range(self.num_samples):
            sampling_grid = (
                base_grid
                + normalized_offsets[:, sample_index]
            )

            sampled_feature = F.grid_sample(
                high_feature,
                sampling_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

            reassembled_feature = (
                reassembled_feature
                + sampled_feature
                * sample_weights[
                    :, sample_index:sample_index + 1
                ]
            )

        return reassembled_feature


class HierarchicalStructureReassemblyFusion(nn.Module):
    """
    Fuse the current RGB feature with a structure-guided reassembly of the
    coarser semantic feature.

    NAM hierarchy and cumulative H-ECM features determine where the high-level
    semantic feature should be sampled, instead of only weighting an already
    interpolated feature.
    """

    def __init__(
        self,
        channels: int,
        num_samples: int = 4,
        max_offset: float = 2.0,
    ) -> None:
        super().__init__()

        self.low_refine = ResidualConvBlock(channels)
        self.edge_refine = ResidualConvBlock(channels)

        self.nam_encoder = nn.Sequential(
            ConvNormAct(1, channels // 4),
            ConvNormAct(channels // 4, channels),
            ResidualConvBlock(channels),
        )

        self.structure_fusion = nn.Sequential(
            ConvNormAct(channels * 3, channels),
            ResidualConvBlock(channels),
        )

        self.reassembly = StructureGuidedReassembly(
            channels=channels,
            num_samples=num_samples,
            max_offset=max_offset,
        )

        self.output_fusion = nn.Sequential(
            ConvNormAct(channels * 4, channels),
            ResidualConvBlock(channels),
        )

    def forward(
        self,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
        guide_logits: torch.Tensor,
        edge_feature: torch.Tensor,
        nam_map: torch.Tensor,
    ) -> torch.Tensor:
        target_size = low_feature.shape[-2:]

        low_feature = self.low_refine(low_feature)

        edge_feature = F.interpolate(
            edge_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        edge_feature = self.edge_refine(edge_feature)

        nam_map = F.adaptive_max_pool2d(
            nam_map,
            output_size=target_size,
        )
        nam_feature = self.nam_encoder(nam_map)

        structure_feature = self.structure_fusion(
            torch.cat(
                [
                    low_feature,
                    edge_feature,
                    nam_feature,
                ],
                dim=1,
            )
        )

        high_feature = self.reassembly(
            high_feature=high_feature,
            structure_feature=structure_feature,
        )

        guide_logits = F.interpolate(
            guide_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        guide = torch.sigmoid(guide_logits)
        uncertainty = 4.0 * guide * (1.0 - guide)

        foreground_feature = low_feature * (1.0 + guide)
        boundary_feature = low_feature * uncertainty

        fused_feature = self.output_fusion(
            torch.cat(
                [
                    foreground_feature,
                    boundary_feature,
                    high_feature,
                    structure_feature,
                ],
                dim=1,
            )
        )

        return high_feature + fused_feature