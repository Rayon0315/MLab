# models/networks/mambavision_small_progressive_region_direct_hier60_region_hybrid_cfm_sod.py

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.mambavision_small_progressive_region_direct_hier60_region_hybrid_sod import (
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD,
)
from models.networks.mambavision_small_progressive_region_direct_sod import (
    PRETRAINED_PATH,
)


# =========================================================
# CFMNet modules
# Adapted from:
# CFMNet: A Lightweight Backbone With Cooperative Feature
# Modeling for Remote Sensing Vision Tasks, TGRS 2026
# =========================================================


class BlurPool(nn.Module):
    """
    Lightweight anti-aliased downsampling used by SSRM.
    """

    def __init__(
        self,
        channels: int,
        stride: int = 3,
    ) -> None:
        super().__init__()

        kernel_1d = torch.tensor(
            [1.0, 2.0, 1.0]
        )

        kernel_2d = (
            kernel_1d[:, None]
            * kernel_1d[None, :]
        )

        kernel_2d = (
            kernel_2d
            / kernel_2d.sum()
        )

        self.register_buffer(
            "kernel",
            kernel_2d[
                None,
                None,
            ].repeat(
                channels,
                1,
                1,
                1,
            ),
        )

        self.channels = channels
        self.stride = stride

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        kernel = self.kernel.to(
            dtype=x.dtype
        )

        return F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=1,
            groups=self.channels,
        )


# =========================================================
# TCRM
# Texture-Aware Channel Recalibration Module
# =========================================================


class TCRM(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads

        self.temperature = nn.Parameter(
            torch.ones(
                num_heads,
                1,
                1,
            )
        )

        self.qkv = nn.Conv2d(
            dim,
            dim * 3,
            kernel_size=1,
            bias=True,
        )

        self.qkv_dwconv = nn.Conv2d(
            dim * 3,
            dim * 3,
            kernel_size=3,
            padding=1,
            groups=dim * 3,
            bias=True,
        )

        self.project_out = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=True,
        )

        self.gate = nn.Sequential(
            nn.Conv2d(
                dim,
                dim // 2,
                kernel_size=1,
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                dim // 2,
                1,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        # Same idea as CFMNet:
        # progressively activate channel recalibration.
        self.gamma = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        dtype = x.dtype

        # Official implementation performs the attention
        # calculation in float32.
        x = x.float()

        q, k, v = self.qkv_dwconv(
            self.qkv(x)
        ).chunk(
            3,
            dim=1,
        )

        (
            batch_size,
            channels,
            height,
            width,
        ) = q.shape

        head_dim = (
            channels
            // self.num_heads
        )

        q = q.reshape(
            batch_size,
            self.num_heads,
            head_dim,
            height * width,
        )

        k = k.reshape(
            batch_size,
            self.num_heads,
            head_dim,
            height * width,
        )

        v = v.reshape(
            batch_size,
            self.num_heads,
            head_dim,
            height * width,
        )

        q = F.normalize(
            q,
            dim=-1,
            eps=1e-6,
        )

        k = F.normalize(
            k,
            dim=-1,
            eps=1e-6,
        )

        # Scene-adaptive sparse Top-k.
        gate_mean = self.gate(
            x
        ).mean()

        dynamic_k = int(
            torch.clamp(
                head_dim
                * gate_mean,
                min=1.0,
                max=float(
                    head_dim
                ),
            )
            .detach()
            .item()
        )

        temperature = torch.clamp(
            self.temperature,
            min=0.01,
            max=10.0,
        )

        attention = (
            q
            @ k.transpose(
                -2,
                -1,
            )
        )

        attention = (
            attention
            * temperature
            / (
                head_dim
                ** 0.5
            )
        )

        topk_index = torch.topk(
            attention,
            k=dynamic_k,
            dim=-1,
            largest=True,
        ).indices

        mask = torch.zeros_like(
            attention
        )

        mask.scatter_(
            -1,
            topk_index,
            1.0,
        )

        attention = torch.where(
            mask > 0,
            attention,
            torch.full_like(
                attention,
                -1e4,
            ),
        )

        attention = torch.softmax(
            attention,
            dim=-1,
        )

        output = (
            attention
            @ v
        )

        output = output.reshape(
            batch_size,
            channels,
            height,
            width,
        )

        output = self.project_out(
            output
        )

        output = (
            residual.float()
            + self.gamma
            * output
        )

        return output.to(
            dtype
        )


# =========================================================
# SSRM
# Spatial Structure Reorganization Module
# =========================================================


class SSRM(nn.Module):
    def __init__(
        self,
        channels: int,
        att_kernel: int = 11,
    ) -> None:
        super().__init__()

        padding = (
            att_kernel // 2
        )

        self.max_pool = nn.MaxPool2d(
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.blur_pool = BlurPool(
            channels=channels,
            stride=3,
        )

        self.horizontal1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(
                att_kernel,
                3,
            ),
            padding=(
                padding,
                1,
            ),
            groups=channels,
            bias=False,
        )

        self.vertical1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(
                3,
                att_kernel,
            ),
            padding=(
                1,
                padding,
            ),
            groups=channels,
            bias=False,
        )

        self.horizontal2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(
                att_kernel,
                3,
            ),
            padding=(
                padding,
                1,
            ),
            groups=channels,
            bias=False,
        )

        self.vertical2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=(
                3,
                att_kernel,
            ),
            padding=(
                1,
                padding,
            ),
            groups=channels,
            bias=False,
        )

        self.norm = nn.BatchNorm2d(
            channels
        )

    @staticmethod
    def h_transform(
        x: torch.Tensor,
    ) -> torch.Tensor:
        shape = x.shape

        x = F.pad(
            x,
            (
                0,
                shape[-1],
            ),
        )

        x = x.reshape(
            shape[0],
            shape[1],
            -1,
        )

        x = x[
            ...,
            :-shape[-1],
        ]

        return x.reshape(
            shape[0],
            shape[1],
            shape[2],
            2 * shape[3] - 1,
        )

    @staticmethod
    def inv_h_transform(
        x: torch.Tensor,
    ) -> torch.Tensor:
        shape = x.shape

        x = x.reshape(
            shape[0],
            shape[1],
            -1,
        ).contiguous()

        x = F.pad(
            x,
            (
                0,
                shape[-2],
            ),
        )

        x = x.reshape(
            shape[0],
            shape[1],
            shape[-2],
            2 * shape[-2],
        )

        return x[
            ...,
            :shape[-2],
        ]

    @classmethod
    def v_transform(
        cls,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.permute(
            0,
            1,
            3,
            2,
        )

        x = cls.h_transform(
            x
        )

        return x.permute(
            0,
            1,
            3,
            2,
        )

    @classmethod
    def inv_v_transform(
        cls,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x.permute(
            0,
            1,
            3,
            2,
        )

        x = cls.inv_h_transform(
            x
        )

        return x.permute(
            0,
            1,
            3,
            2,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        low = self.max_pool(
            x
        )

        low = self.blur_pool(
            low
        )

        horizontal1 = (
            self.horizontal1(
                low
            )
        )

        vertical1 = (
            self.vertical1(
                low
            )
        )

        horizontal2 = (
            self.horizontal2(
                self.h_transform(
                    low
                )
            )
        )

        horizontal2 = (
            self.inv_h_transform(
                horizontal2
            )
        )

        vertical2 = (
            self.vertical2(
                self.v_transform(
                    low
                )
            )
        )

        vertical2 = (
            self.inv_v_transform(
                vertical2
            )
        )

        structure = (
            horizontal1
            + vertical1
            + horizontal2
            + vertical2
        )

        attention = torch.sigmoid(
            self.norm(
                structure
            )
        )

        attention = F.interpolate(
            attention,
            size=x.shape[-2:],
            mode="nearest",
        )

        return (
            x
            * attention
        )


# =========================================================
# LDFM
# Local Detail Feature Modulator
# =========================================================


class LDFM(nn.Module):
    def __init__(
        self,
        dim: int,
        growth_rate: float = 2.0,
    ) -> None:
        super().__init__()

        hidden_dim = int(
            dim
            * growth_rate
        )

        self.local = nn.Sequential(
            nn.Conv2d(
                dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=dim,
            ),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                dim,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.local(
            x
        )


# =========================================================
# EGPCM
# Efficient Global Perception Convolution Module
# =========================================================


def geo_ensemble(
    kernel: torch.Tensor,
) -> torch.Tensor:
    variants = [
        kernel,
        kernel.flip(
            3
        ),
        kernel.flip(
            2
        ),
        kernel.flip(
            2,
            3,
        ),
    ]

    rotated = torch.rot90(
        kernel,
        -1,
        dims=(
            2,
            3,
        ),
    )

    variants.extend(
        [
            rotated,
            rotated.flip(
                3
            ),
            rotated.flip(
                2
            ),
            rotated.flip(
                2,
                3,
            ),
        ]
    )

    return (
        sum(
            variants
        )
        / len(
            variants
        )
    )


class EGPCM(nn.Module):
    def __init__(
        self,
        channels: int,
        modeled_channels: int,
        small_kernel: int = 3,
        large_kernel: int = 13,
        kernel_scale: float = 0.5,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.modeled_channels = (
            modeled_channels
        )

        self.small_kernel = (
            small_kernel
        )

        self.large_kernel = (
            large_kernel
        )

        self.kernel_scale = (
            kernel_scale
        )

        hidden_channels = max(
            1,
            modeled_channels // 2,
        )

        self.dynamic_kernel_generator = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(
                    1
                ),
                nn.Conv2d(
                    modeled_channels,
                    hidden_channels,
                    kernel_size=1,
                ),
                nn.GELU(),
                nn.Conv2d(
                    hidden_channels,
                    modeled_channels
                    * small_kernel
                    * small_kernel,
                    kernel_size=1,
                ),
            )
        )

        nn.init.zeros_(
            self.dynamic_kernel_generator[
                -1
            ].weight
        )

        nn.init.zeros_(
            self.dynamic_kernel_generator[
                -1
            ].bias
        )

        large_kernel_weight = (
            torch.randn(
                modeled_channels,
                1,
                large_kernel,
                large_kernel,
            )
        )

        self.large_kernel_weight = (
            nn.Parameter(
                geo_ensemble(
                    large_kernel_weight
                )
            )
        )

        self.gamma = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        dtype = x.dtype

        x = x.float()

        modeled_feature = x[
            :,
            :self.modeled_channels,
        ]

        identity_feature = x[
            :,
            self.modeled_channels:,
        ]

        (
            batch_size,
            channels,
            height,
            width,
        ) = modeled_feature.shape

        dynamic_kernel = (
            self.dynamic_kernel_generator(
                modeled_feature
            )
        )

        dynamic_kernel = (
            torch.tanh(
                dynamic_kernel
            )
            * self.kernel_scale
        )

        dynamic_kernel = (
            dynamic_kernel.reshape(
                batch_size
                * channels,
                1,
                self.small_kernel,
                self.small_kernel,
            )
        )

        dynamic_input = (
            modeled_feature.reshape(
                1,
                batch_size
                * channels,
                height,
                width,
            )
        )

        dynamic_output = F.conv2d(
            dynamic_input,
            dynamic_kernel,
            padding=(
                self.small_kernel
                // 2
            ),
            groups=(
                batch_size
                * channels
            ),
        )

        dynamic_output = (
            dynamic_output.reshape(
                batch_size,
                channels,
                height,
                width,
            )
        )

        normalized_large_kernel = (
            self.large_kernel_weight
            / (
                self.large_kernel_weight.norm(
                    dim=(
                        2,
                        3,
                    ),
                    keepdim=True,
                )
                + 1e-6
            )
        )

        large_output = F.conv2d(
            modeled_feature,
            normalized_large_kernel,
            padding=(
                self.large_kernel
                // 2
            ),
            groups=channels,
        )

        modeled_output = (
            dynamic_output
            + large_output
        )

        output = torch.cat(
            [
                modeled_output,
                identity_feature,
            ],
            dim=1,
        )

        output = (
            residual.float()
            + self.gamma
            * (
                output
                - residual.float()
            )
        )

        return output.to(
            dtype
        )


# =========================================================
# Cooperative Feature Modeling Block
# =========================================================


class CFMBlock(nn.Module):
    """
    Split one decoder feature into four complementary
    channel subspaces:

        TCRM | SSRM | LDFM | EGPCM

    Then fuse them back to the original channel dimension.
    """

    def __init__(
        self,
        dim: int,
        stage: int,
        att_kernel: int = 11,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()

        branch_dim = (
            dim // 4
        )

        self.tcrm = TCRM(
            dim=branch_dim,
            num_heads=4,
        )

        self.ssrm = SSRM(
            channels=branch_dim,
            att_kernel=att_kernel,
        )

        self.ldfm = LDFM(
            dim=branch_dim,
            growth_rate=2.0,
        )

        # Original CFMNet stage ratios:
        #
        # Stage1 -> 1/8
        # Stage2 -> 1/4
        # Stage3 -> 1/2
        # Stage4 -> 1/2
        egpcm_ratios = (
            1 / 8,
            1 / 4,
            1 / 2,
            1 / 2,
        )

        modeled_channels = max(
            8,
            int(
                branch_dim
                * egpcm_ratios[
                    stage
                ]
            ),
        )

        self.egpcm = EGPCM(
            channels=branch_dim,
            modeled_channels=(
                modeled_channels
            ),
            small_kernel=3,
            large_kernel=13,
            kernel_scale=0.5,
        )

        hidden_channels = int(
            dim
            * mlp_ratio
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                dim,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                hidden_channels
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                hidden_channels,
                dim,
                kernel_size=1,
                bias=False,
            ),
        )

        self.output_norm = (
            nn.BatchNorm2d(
                dim
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        shortcut = x

        (
            tcrm_feature,
            ssrm_feature,
            ldfm_feature,
            egpcm_feature,
        ) = torch.chunk(
            x,
            chunks=4,
            dim=1,
        )

        tcrm_feature = self.tcrm(
            tcrm_feature
        )

        ssrm_feature = self.ssrm(
            ssrm_feature
        )

        ldfm_feature = self.ldfm(
            ldfm_feature
        )

        egpcm_feature = self.egpcm(
            egpcm_feature
        )

        fused = torch.cat(
            [
                tcrm_feature,
                ssrm_feature,
                ldfm_feature,
                egpcm_feature,
            ],
            dim=1,
        )

        fused = self.fusion(
            fused
        )

        return (
            shortcut
            + self.output_norm(
                fused
            )
        )


# =========================================================
# Adapter
# =========================================================


class CFMAdapter(nn.Module):
    """
    Add CFM to the existing Hybrid decoder without forcing
    an immediate distribution shift.

        output = x + alpha * (CFM(x) - x)

    alpha starts from 0.
    """

    def __init__(
        self,
        dim: int,
        stage: int,
    ) -> None:
        super().__init__()

        self.block = CFMBlock(
            dim=dim,
            stage=stage,
            att_kernel=11,
            mlp_ratio=2.0,
        )

        self.scale = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        enhanced = self.block(
            x
        )

        return (
            x
            + self.scale
            * (
                enhanced
                - x
            )
        )


class PostCFM(nn.Module):
    """
    Wrap an existing decoder module:

        old_module(...)
            ->
        CFMAdapter
    """

    def __init__(
        self,
        base_module: nn.Module,
        adapter: nn.Module,
    ) -> None:
        super().__init__()

        self.base_module = (
            base_module
        )

        self.adapter = (
            adapter
        )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        feature = (
            self.base_module(
                *args,
                **kwargs,
            )
        )

        return self.adapter(
            feature
        )


# =========================================================
# Hybrid Hier60 + CFM
# =========================================================


class MambaVisionSmallProgressiveRegionDirectHier60RegionHybridCFMSOD(
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD
):
    """
    Hybrid Hier60 remains unchanged.

    CFM is inserted at:

        decoded4 -> CFM4 -> pred4
                        -> Stage3

        decoded3 -> CFM3 -> pred3
                        -> Stage2

        decoded2 -> CFM2 -> pred2
                        -> Stage1

    Therefore CFM modifies both:
        1. auxiliary predictions
        2. the feature actually propagated downstream
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

        stage2_channels = (
            self.backbone.out_channels[
                1
            ]
        )

        stage3_channels = (
            self.backbone.out_channels[
                2
            ]
        )

        # -------------------------------------------------
        # decoded4
        #
        # Parent:
        # deep_projection -> context4 -> pred4
        #
        # New:
        # deep_projection -> context4 -> CFM4 -> pred4
        #
        # decoded4 enhanced by CFM4 also goes into fusion3.
        # -------------------------------------------------

        self.context4 = PostCFM(
            base_module=self.context4,
            adapter=CFMAdapter(
                dim=stage3_channels,
                stage=3,
            ),
        )

        # -------------------------------------------------
        # decoded3
        #
        # fusion3 -> CFM3 -> pred3
        #
        # enhanced decoded3 also goes into reduce3.
        # -------------------------------------------------

        self.fusion3 = PostCFM(
            base_module=self.fusion3,
            adapter=CFMAdapter(
                dim=stage3_channels,
                stage=2,
            ),
        )

        # -------------------------------------------------
        # decoded2
        #
        # fusion2 -> CFM2 -> pred2
        #
        # enhanced decoded2 also goes into reduce2.
        # -------------------------------------------------

        self.fusion2 = PostCFM(
            base_module=self.fusion2,
            adapter=CFMAdapter(
                dim=stage2_channels,
                stage=1,
            ),
        )


def build_model(
) -> MambaVisionSmallProgressiveRegionDirectHier60RegionHybridCFMSOD:
    return (
        MambaVisionSmallProgressiveRegionDirectHier60RegionHybridCFMSOD(
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

    print(
        "aux:",
        [
            prediction.shape
            for prediction
            in outputs["aux"]
        ],
    )

    print(
        "cfm scales:",
        float(
            model.context4.adapter.scale
        ),
        float(
            model.fusion3.adapter.scale
        ),
        float(
            model.fusion2.adapter.scale
        ),
    )