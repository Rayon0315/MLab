# models/networks/mambavision_small_progressive_region_direct_hier60_feature_upsample_sod.py

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import ConvNormAct

from models.networks.mambavision_small_progressive_region_direct_hier60_sod import (
    MambaVisionSmallProgressiveRegionDirectHier60SOD,
)
from models.networks.mambavision_small_progressive_region_direct_sod import (
    PRETRAINED_PATH,
)


class FeatureUpsampleBlock(nn.Module):
    """
    Lightweight feature-space upsampling block.

    Process:
        low-resolution feature
            -> channel projection
            -> bilinear x2
            -> depthwise local refinement
            -> channel mixing
            -> residual fusion

    Bilinear interpolation only establishes the denser spatial grid.
    The recovered feature itself is refined in feature space before
    saliency prediction.
    """

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
    Feature-level high-resolution prediction head.

    Pure Hier60 original:
        decoded1
        128 x 88 x 88
            ->
        PredictionHead
            ->
        1 x 88 x 88
            ->
        bilinear x4

    This version:
        decoded1
        128 x 88 x 88
            ->
        64 x 176 x 176
            ->
        32 x 352 x 352
            ->
        high-resolution refinement
            ->
        1 x 352 x 352

    The saliency logits are therefore generated after spatial
    reconstruction instead of before it.
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

        prediction = self.prediction(
            feature
        )

        return prediction


class MambaVisionSmallProgressiveRegionDirectHier60FeatureUpsampleSOD(
    MambaVisionSmallProgressiveRegionDirectHier60SOD
):
    """
    Pure Hier60 with feature-level prediction upsampling.

    Everything before the final prediction head is inherited
    unchanged from Pure Hier60:

        Stage1 <- RGB - M60
        Stage2 <- M60
        Stage3 <- M60
        Stage4 <- M60

        region reconstruction
        progressive decoder
        boundary refinement

    Only the final prediction path changes.

    Original:
        decoded1
            -> PredictionHead
            -> low-resolution logits
            -> bilinear resize

    Current:
        decoded1
            -> feature x2
            -> feature x2
            -> high-resolution feature refinement
            -> prediction
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

        # decoded1 has 128 channels in the validated
        # Progressive / Pure Hier60 decoder.
        self.pred1 = HighResolutionPredictionHead(
            in_channels=128,
        )


def build_model(
) -> MambaVisionSmallProgressiveRegionDirectHier60FeatureUpsampleSOD:
    return (
        MambaVisionSmallProgressiveRegionDirectHier60FeatureUpsampleSOD(
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