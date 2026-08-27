# models/networks/mambavision_small_progressive_region_direct_hier60_region_preserving_sod.py

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.mambavision_small_progressive_region_direct_hier60_sod import (
    MambaVisionSmallProgressiveRegionDirectHier60SOD,
)
from models.networks.mambavision_small_progressive_region_direct_sod import (
    PRETRAINED_PATH,
)


class PixelLayerNorm(nn.Module):
    """
    LayerNorm over channels independently at every spatial position.

    Unlike BatchNorm / GroupNorm, this normalization does not aggregate
    statistics across different regions in the image.
    """

    def __init__(
        self,
        channels: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(channels)
        )
        self.bias = nn.Parameter(
            torch.zeros(channels)
        )
        self.eps = eps

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        feature = feature.permute(
            0,
            2,
            3,
            1,
        )

        feature = F.layer_norm(
            feature,
            normalized_shape=(
                feature.shape[-1],
            ),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )

        return feature.permute(
            0,
            3,
            1,
            2,
        ).contiguous()


class PointwiseRegionMLP(nn.Module):
    """
    Region-preserving channel encoder.

    There is deliberately no spatial convolution:
        - nearest resize preserves piecewise-constant region values;
        - 1x1 projections operate on each pixel independently;
        - PixelLayerNorm normalizes channels only;
        - identical pixels inside one NAM region therefore remain identical.

    The residual channel MLP starts from zero so the initial representation
    is the direct pointwise projection of the region mean map.
    """

    def __init__(
        self,
        out_channels: int,
        bottleneck_ratio: float = 0.5,
    ) -> None:
        super().__init__()

        hidden_channels = max(
            32,
            int(
                out_channels
                * bottleneck_ratio
            ),
        )

        self.input_projection = nn.Conv2d(
            3,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.norm = PixelLayerNorm(
            out_channels
        )

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(
                out_channels,
                hidden_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        self._init_residual()

    def _init_residual(
        self,
    ) -> None:
        last_projection = (
            self.channel_mlp[-1]
        )

        nn.init.zeros_(
            last_projection.weight
        )

        if (
            last_projection.bias
            is not None
        ):
            nn.init.zeros_(
                last_projection.bias
            )

    def forward(
        self,
        region: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        region = F.interpolate(
            region,
            size=target_size,
            mode="nearest",
        )

        base_feature = (
            self.input_projection(
                region
            )
        )

        residual_feature = (
            self.channel_mlp(
                self.norm(
                    base_feature
                )
            )
        )

        return (
            base_feature
            + residual_feature
        )


class RegionPreservingHier60PyramidEncoder(
    nn.Module
):
    """
    Controlled replacement of the H60 region pyramid encoder.

    Stage1:
        RGB - M60
        -> keep the original spatial convolution encoder unchanged.

    Stage2/3/4:
        M60
        -> nearest resize
        -> pointwise 1x1 projection
        -> channel-only residual MLP

    This isolates the hypothesis that spatial convolution / bilinear
    interpolation unnecessarily mixes neighboring NAM regions.
    """

    def __init__(
        self,
        stage1_encoder: nn.Module,
        stage2_channels: int,
        stage3_channels: int,
        stage4_channels: int,
    ) -> None:
        super().__init__()

        # Reuse the already-created original Stage1 encoder exactly.
        # No new initialization is introduced into the detail branch.
        self.stage1_encoder = (
            stage1_encoder
        )

        self.stage2_encoder = (
            PointwiseRegionMLP(
                out_channels=stage2_channels,
            )
        )

        self.stage3_encoder = (
            PointwiseRegionMLP(
                out_channels=stage3_channels,
            )
        )

        self.stage4_encoder = (
            PointwiseRegionMLP(
                out_channels=stage4_channels,
            )
        )

    def forward(
        self,
        detail_region: torch.Tensor,
        fine_region: torch.Tensor,
        middle_region: torch.Tensor,
        coarse_region: torch.Tensor,
        stage1_size: tuple[int, int],
        stage2_size: tuple[int, int],
        stage3_size: tuple[int, int],
        stage4_size: tuple[int, int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        region1 = (
            self.stage1_encoder(
                detail_region,
                target_size=stage1_size,
            )
        )

        region2 = (
            self.stage2_encoder(
                fine_region,
                target_size=stage2_size,
            )
        )

        region3 = (
            self.stage3_encoder(
                middle_region,
                target_size=stage3_size,
            )
        )

        region4 = (
            self.stage4_encoder(
                coarse_region,
                target_size=stage4_size,
            )
        )

        return (
            region1,
            region2,
            region3,
            region4,
        )


class MambaVisionSmallProgressiveRegionDirectHier60RegionPreservingSOD(
    MambaVisionSmallProgressiveRegionDirectHier60SOD
):
    """
    H60 with region-preserving encoding for the M60 branch.

    The validated H60 graph is constructed first. We then replace only
    Stage2/3/4 region encoders while retaining the original Stage1
    detail encoder.

    Unchanged:
        - MambaVision backbone
        - Direct H60 hierarchy
        - Stage1 RGB-M60 detail encoder
        - all region/visual interaction modules
        - progressive decoder
        - boundary refinement
        - prediction heads
        - loss and training protocol

    Changed:
        Stage2/3/4 M60 encoding

        old:
            bilinear resize
            -> spatial 3x3 Conv / residual Conv

        new:
            nearest resize
            -> 1x1 projection
            -> channel-only residual MLP
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
    ) -> None:
        # Build the validated H60 network first so all retained modules
        # keep the same initialization order under the same seed.
        super().__init__(
            pretrained_path=pretrained_path
        )

        old_region_encoder = (
            self.region_encoder
        )

        stage2_channels = (
            self.backbone.out_channels[1]
        )
        stage3_channels = (
            self.backbone.out_channels[2]
        )
        stage4_channels = (
            self.backbone.out_channels[3]
        )

        self.region_encoder = (
            RegionPreservingHier60PyramidEncoder(
                stage1_encoder=(
                    old_region_encoder
                    .stage1_encoder
                ),
                stage2_channels=(
                    stage2_channels
                ),
                stage3_channels=(
                    stage3_channels
                ),
                stage4_channels=(
                    stage4_channels
                ),
            )
        )


def build_model(
) -> (
    MambaVisionSmallProgressiveRegionDirectHier60RegionPreservingSOD
):
    return (
        MambaVisionSmallProgressiveRegionDirectHier60RegionPreservingSOD(
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
                for tensor
                in outputs["aux"]
            ],
        )
