from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.mambavision_small_progressive_region_direct_hier60_sod import (
    MambaVisionSmallProgressiveRegionDirectHier60SOD,
)
from models.networks.mambavision_small_progressive_region_direct_sod import PRETRAINED_PATH


class PixelLayerNorm(nn.Module):
    """LayerNorm over channels independently at each spatial position."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(
            x,
            (x.shape[-1],),
            self.weight,
            self.bias,
            self.eps,
        )
        return x.permute(0, 3, 1, 2).contiguous()


class HybridRegionScaleEncoder(nn.Module):
    """
    Region-preserving main branch + lightweight spatial context branch.

    Main:
        nearest resize -> 1x1 projection -> channel-only residual MLP

    Context:
        depthwise 3x3 -> pixel LayerNorm -> GELU -> 1x1 projection

    Output:
        preserve + alpha * context
    """

    def __init__(
        self,
        out_channels: int,
        bottleneck_ratio: float = 0.5,
        initial_context_scale: float = 0.1,
    ) -> None:
        super().__init__()

        hidden_channels = max(32, int(out_channels * bottleneck_ratio))

        self.input_projection = nn.Conv2d(
            3,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.preserve_norm = PixelLayerNorm(out_channels)

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

        # Same initialization as the previous region-preserving experiment.
        nn.init.zeros_(self.channel_mlp[-1].weight)
        nn.init.zeros_(self.channel_mlp[-1].bias)

        self.context_depthwise = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=out_channels,
            bias=False,
        )

        self.context_norm = PixelLayerNorm(out_channels)
        self.context_activation = nn.GELU()

        self.context_projection = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.context_scale = nn.Parameter(
            torch.tensor(float(initial_context_scale))
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

        base = self.input_projection(region)

        preserve = base + self.channel_mlp(
            self.preserve_norm(base)
        )

        context = self.context_depthwise(preserve)
        context = self.context_norm(context)
        context = self.context_activation(context)
        context = self.context_projection(context)

        return preserve + self.context_scale * context


class HybridHier60PyramidEncoder(nn.Module):
    """
    Stage1 keeps the original RGB-M60 encoder.
    Stage2/3/4 use the hybrid region encoder.
    """

    def __init__(
        self,
        stage1_encoder: nn.Module,
        stage2_channels: int,
        stage3_channels: int,
        stage4_channels: int,
        initial_context_scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.stage1_encoder = stage1_encoder

        self.stage2_encoder = HybridRegionScaleEncoder(
            stage2_channels,
            initial_context_scale=initial_context_scale,
        )
        self.stage3_encoder = HybridRegionScaleEncoder(
            stage3_channels,
            initial_context_scale=initial_context_scale,
        )
        self.stage4_encoder = HybridRegionScaleEncoder(
            stage4_channels,
            initial_context_scale=initial_context_scale,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        region1 = self.stage1_encoder(
            detail_region,
            target_size=stage1_size,
        )
        region2 = self.stage2_encoder(
            fine_region,
            target_size=stage2_size,
        )
        region3 = self.stage3_encoder(
            middle_region,
            target_size=stage3_size,
        )
        region4 = self.stage4_encoder(
            coarse_region,
            target_size=stage4_size,
        )

        return region1, region2, region3, region4


class MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD(
    MambaVisionSmallProgressiveRegionDirectHier60SOD
):
    """
    Controlled H60 ablation.

    Unchanged:
        backbone, H60 assignment, Stage1 detail encoder,
        region/visual interaction, progressive decoder,
        refinement, heads, loss, training protocol.

    Replaced:
        Stage2/3/4 M60 encoders.

    Previous region-preserving:
        nearest -> pointwise channel encoder

    Hybrid:
        nearest -> pointwise preserve branch
        + depthwise local context correction
    """

    input_keys = ("image", "mean_60")

    def __init__(
        self,
        pretrained_path: str | Path | None,
    ) -> None:
        super().__init__(pretrained_path=pretrained_path)

        old_region_encoder = self.region_encoder

        self.region_encoder = HybridHier60PyramidEncoder(
            stage1_encoder=old_region_encoder.stage1_encoder,
            stage2_channels=self.backbone.out_channels[1],
            stage3_channels=self.backbone.out_channels[2],
            stage4_channels=self.backbone.out_channels[3],
            initial_context_scale=0.1,
        )


def build_model() -> MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD:
    return MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD(
        pretrained_path=PRETRAINED_PATH,
    )


if __name__ == "__main__":
    model = build_model()
    model.eval()

    image = torch.randn(1, 3, 352, 352)
    mean_60 = torch.rand(1, 3, 352, 352)

    with torch.no_grad():
        outputs = model(
            image=image,
            mean_60=mean_60,
        )

    print("pred:", outputs["pred"].shape)

    if "aux" in outputs:
        print("aux:", [x.shape for x in outputs["aux"]])

    print(
        "context scales:",
        float(model.region_encoder.stage2_encoder.context_scale),
        float(model.region_encoder.stage3_encoder.context_scale),
        float(model.region_encoder.stage4_encoder.context_scale),
    )
