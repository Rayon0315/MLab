# models/networks/mambavision_small_progressive_region_direct_hier60_region_hybrid_feature_upsample_sod.py

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import ConvNormAct

from models.networks.mambavision_small_progressive_region_direct_hier60_region_hybrid_sod import (
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD,
)
from models.networks.mambavision_small_progressive_region_direct_sod import (
    PRETRAINED_PATH,
)


class FeatureUpsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.projection = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

        self.local_refine = ConvNormAct(
            out_channels,
            out_channels,
            kernel_size=3,
            groups=out_channels,
        )

        self.channel_mix = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                out_channels,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.projection(
            feature
        )

        feature = F.interpolate(
            feature,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )

        refined = self.local_refine(
            feature
        )

        refined = self.channel_mix(
            refined
        )

        return (
            feature
            + refined
        )


class HighResolutionPredictionHead(nn.Module):
    """
    Original Hybrid:

        decoded1
        128 x 88 x 88
            ↓
        PredictionHead
            ↓
        1 x 88 x 88
            ↓
        bilinear x4

    Current:

        decoded1
        128 x 88 x 88
            ↓
        64 x 176 x 176
            ↓
        32 x 352 x 352
            ↓
        Prediction
            ↓
        1 x 352 x 352
    """

    def __init__(
        self,
        in_channels: int = 128,
    ) -> None:
        super().__init__()

        self.upsample1 = FeatureUpsampleBlock(
            in_channels=in_channels,
            out_channels=64,
        )

        self.upsample2 = FeatureUpsampleBlock(
            in_channels=64,
            out_channels=32,
        )

        self.refine = nn.Sequential(
            ConvNormAct(
                32,
                32,
                kernel_size=3,
                groups=32,
            ),
            ConvNormAct(
                32,
                32,
                kernel_size=1,
                padding=0,
            ),
        )

        self.prediction = nn.Conv2d(
            32,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.upsample1(
            feature
        )

        feature = self.upsample2(
            feature
        )

        feature = self.refine(
            feature
        )

        return self.prediction(
            feature
        )


class MambaVisionSmallProgressiveRegionDirectHier60RegionHybridFeatureUpsampleSOD(
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD
):
    """
    Region Hybrid Hier60 + feature-level final upsampling.

    Hybrid path remains unchanged:

        Stage1:
            RGB - M60
            original Stage1 encoder

        Stage2/3/4:
            M60
            hybrid region-preserving + spatial-context encoder

        region/visual interaction
        progressive decoder
        boundary refinement

    Only final pred1 is replaced.
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
        )

        self.pred1 = HighResolutionPredictionHead(
            in_channels=128,
        )


def build_model(
) -> MambaVisionSmallProgressiveRegionDirectHier60RegionHybridFeatureUpsampleSOD:
    return (
        MambaVisionSmallProgressiveRegionDirectHier60RegionHybridFeatureUpsampleSOD(
            pretrained_path=PRETRAINED_PATH,
        )
    )


if __name__ == "__main__":
    model = build_model()
    model.eval()

    image = torch.randn(
        1,
        3,
        352,
        352,
    )

    mean_60 = torch.rand(
        1,
        3,
        352,
        352,
    )

    with torch.no_grad():
        outputs = model(
            image=image,
            mean_60=mean_60,
        )

    print(
        "pred:",
        outputs["pred"].shape,
    )

    if "aux" in outputs:
        print(
            "aux:",
            [
                tensor.shape
                for tensor in outputs["aux"]
            ],
        )