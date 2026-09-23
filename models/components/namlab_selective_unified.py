from __future__ import annotations

import torch
import torch.nn as nn

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
)


FeaturePyramid = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


class RegionVisualUnifiedFusion(nn.Module):
    """
    Same simple fusion used by the previous Unified Fusion experiment.

        concat(visual, region)
        -> 1x1 ConvNormAct
        -> ResidualConvBlock
        -> residual add
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            ConvNormAct(
                channels * 2,
                channels,
                kernel_size=1,
                padding=0,
            ),
            ResidualConvBlock(channels),
        )

    def forward(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction = self.fusion(
            torch.cat(
                [visual_feature, region_feature],
                dim=1,
            )
        )
        return visual_feature + reconstruction


class SelectiveUnifiedNAMLab(nn.Module):
    """
    Keep Stage1 RGB-M60 detail guidance always active.
    Selectively enable M60 guidance at Stage2/3/4.

    Enabled stage:
        M60 -> original HybridRegionScaleEncoder
            -> RegionVisualUnifiedFusion

    Disabled stage:
        raw backbone feature passes through unchanged.
    """

    def __init__(
        self,
        original_hybrid: nn.Module,
        stage_channels: tuple[int, int, int, int],
        enabled_stages: tuple[int, ...],
    ) -> None:
        super().__init__()

        invalid = [s for s in enabled_stages if s not in (2, 3, 4)]
        if invalid:
            raise ValueError(
                f'enabled_stages only accepts 2/3/4, got {invalid}'
            )

        self.enabled_stages = tuple(sorted(set(enabled_stages)))

        # Reuse parent-initialized modules for strict control.
        self.region_input = original_hybrid.region_input
        self.region_encoder = original_hybrid.region_encoder
        self.stage1_detail_reconstruction = (
            original_hybrid.stage1_detail_reconstruction
        )

        _, c2, c3, c4 = stage_channels

        self.stage2_fusion = (
            RegionVisualUnifiedFusion(c2)
            if 2 in self.enabled_stages
            else None
        )
        self.stage3_fusion = (
            RegionVisualUnifiedFusion(c3)
            if 3 in self.enabled_stages
            else None
        )
        self.stage4_fusion = (
            RegionVisualUnifiedFusion(c4)
            if 4 in self.enabled_stages
            else None
        )

    def forward(
        self,
        features: FeaturePyramid,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> FeaturePyramid:
        stage1, stage2, stage3, stage4 = features

        detail_region, mean_60_normalized = self.region_input(
            image=image,
            mean_60=mean_60,
        )

        # Stage1 detail path is always active.
        region1 = self.region_encoder.stage1_encoder(
            detail_region,
            target_size=stage1.shape[-2:],
        )
        stage1 = self.stage1_detail_reconstruction(
            visual_feature=stage1,
            detail_feature=region1,
        )

        # Compute M60 feature only when the stage is enabled.
        if self.stage2_fusion is not None:
            region2 = self.region_encoder.stage2_encoder(
                mean_60_normalized,
                target_size=stage2.shape[-2:],
            )
            stage2 = self.stage2_fusion(
                visual_feature=stage2,
                region_feature=region2,
            )

        if self.stage3_fusion is not None:
            region3 = self.region_encoder.stage3_encoder(
                mean_60_normalized,
                target_size=stage3.shape[-2:],
            )
            stage3 = self.stage3_fusion(
                visual_feature=stage3,
                region_feature=region3,
            )

        if self.stage4_fusion is not None:
            region4 = self.region_encoder.stage4_encoder(
                mean_60_normalized,
                target_size=stage4.shape[-2:],
            )
            stage4 = self.stage4_fusion(
                visual_feature=stage4,
                region_feature=region4,
            )

        return stage1, stage2, stage3, stage4
