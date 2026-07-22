# models/components/nam_attention.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class LightResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=channels,
            ),
        )

        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.act(
            x + self.block(x)
        )


class NAMHierarchyEncoder(nn.Module):
    """
    Shared NAM pyramid encoder.

    Each NAM hierarchy is independently passed through the
    same encoder.

    Outputs:
        stride 4:  64 channels
        stride 8:  96 channels
        stride 16: 128 channels
        stride 32: 160 channels
    """

    out_channels = (
        64,
        96,
        128,
        160,
    )

    def __init__(self) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvGNAct(
                1,
                32,
                kernel_size=3,
                stride=2,
            ),
            ConvGNAct(
                32,
                64,
                kernel_size=3,
                stride=2,
            ),
            LightResidualBlock(64),
        )

        self.stage2 = nn.Sequential(
            ConvGNAct(
                64,
                96,
                kernel_size=3,
                stride=2,
            ),
            LightResidualBlock(96),
        )

        self.stage3 = nn.Sequential(
            ConvGNAct(
                96,
                128,
                kernel_size=3,
                stride=2,
            ),
            LightResidualBlock(128),
        )

        self.stage4 = nn.Sequential(
            ConvGNAct(
                128,
                160,
                kernel_size=3,
                stride=2,
            ),
            LightResidualBlock(160),
        )

    def forward(
        self,
        nam: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        feature1 = self.stem(nam)
        feature2 = self.stage2(feature1)
        feature3 = self.stage3(feature2)
        feature4 = self.stage4(feature3)

        return (
            feature1,
            feature2,
            feature3,
            feature4,
        )


class HierarchicalNAMAttention(nn.Module):
    """
    RGB-conditioned hierarchical NAM attention.

    Operations:
        1. Predict spatial weights for NAM20/40/60.
        2. Aggregate the three NAM hierarchy features.
        3. Generate channel and spatial attention.
        4. Build RGB-NAM interaction features.
        5. Inject the result into the RGB feature residually.
    """

    def __init__(
        self,
        rgb_channels: int,
        nam_channels: int,
    ) -> None:
        super().__init__()

        interaction_channels = min(
            128,
            max(
                32,
                rgb_channels // 4,
            ),
        )

        attention_channels = max(
            16,
            rgb_channels // 8,
        )

        self.rgb_context = ConvGNAct(
            rgb_channels,
            nam_channels,
            kernel_size=1,
            padding=0,
        )

        self.hierarchy_attention = nn.Sequential(
            ConvGNAct(
                nam_channels * 4,
                nam_channels,
                kernel_size=1,
                padding=0,
            ),
            LightResidualBlock(
                nam_channels
            ),
            nn.Conv2d(
                nam_channels,
                3,
                kernel_size=1,
            ),
        )

        self.nam_refine = nn.Sequential(
            LightResidualBlock(
                nam_channels
            ),
            LightResidualBlock(
                nam_channels
            ),
        )

        self.nam_to_rgb = ConvGNAct(
            nam_channels,
            rgb_channels,
            kernel_size=1,
            padding=0,
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                rgb_channels * 2,
                attention_channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                attention_channels,
                rgb_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                4,
                1,
                kernel_size=7,
                padding=3,
            ),
            nn.Sigmoid(),
        )

        self.rgb_reduce = ConvGNAct(
            rgb_channels,
            interaction_channels,
            kernel_size=1,
            padding=0,
        )

        self.nam_reduce = ConvGNAct(
            rgb_channels,
            interaction_channels,
            kernel_size=1,
            padding=0,
        )

        self.interaction_fusion = nn.Sequential(
            ConvGNAct(
                interaction_channels * 4,
                rgb_channels,
                kernel_size=1,
                padding=0,
            ),
            LightResidualBlock(
                rgb_channels
            ),
        )

        self.interaction_gate = nn.Sequential(
            nn.Conv2d(
                interaction_channels * 2,
                rgb_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.output_refine = LightResidualBlock(
            rgb_channels
        )

        self.interaction_scale = nn.Parameter(
            torch.tensor(1.0)
        )

        self.nam_scale = nn.Parameter(
            torch.tensor(0.5)
        )

    def forward(
        self,
        rgb_feature: torch.Tensor,
        nam_20: torch.Tensor,
        nam_40: torch.Tensor,
        nam_60: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        rgb_context = self.rgb_context(
            rgb_feature
        )

        hierarchy_logits = (
            self.hierarchy_attention(
                torch.cat(
                    [
                        rgb_context,
                        nam_20,
                        nam_40,
                        nam_60,
                    ],
                    dim=1,
                )
            )
        )

        hierarchy_weights = torch.softmax(
            hierarchy_logits,
            dim=1,
        )

        selected_nam = (
            hierarchy_weights[:, 0:1] * nam_20
            + hierarchy_weights[:, 1:2] * nam_40
            + hierarchy_weights[:, 2:3] * nam_60
        )

        selected_nam = self.nam_refine(
            selected_nam
        )

        nam_rgb = self.nam_to_rgb(
            selected_nam
        )

        channel_attention = (
            self.channel_attention(
                torch.cat(
                    [
                        rgb_feature,
                        nam_rgb,
                    ],
                    dim=1,
                )
            )
        )

        spatial_attention = (
            self.spatial_attention(
                torch.cat(
                    [
                        rgb_feature.mean(
                            dim=1,
                            keepdim=True,
                        ),
                        rgb_feature.amax(
                            dim=1,
                            keepdim=True,
                        ),
                        nam_rgb.mean(
                            dim=1,
                            keepdim=True,
                        ),
                        nam_rgb.amax(
                            dim=1,
                            keepdim=True,
                        ),
                    ],
                    dim=1,
                )
            )
        )

        rgb_reduced = self.rgb_reduce(
            rgb_feature
        )

        nam_reduced = self.nam_reduce(
            nam_rgb
        )

        interaction_feature = (
            self.interaction_fusion(
                torch.cat(
                    [
                        rgb_reduced,
                        nam_reduced,
                        rgb_reduced
                        * nam_reduced,
                        torch.abs(
                            rgb_reduced
                            - nam_reduced
                        ),
                    ],
                    dim=1,
                )
            )
        )

        interaction_gate = (
            self.interaction_gate(
                torch.cat(
                    [
                        rgb_reduced,
                        nam_reduced,
                    ],
                    dim=1,
                )
            )
        )

        attention = (
            1.0 + channel_attention
        ) * (
            1.0 + spatial_attention
        )

        output = (
            rgb_feature
            + self.interaction_scale
            * interaction_gate
            * interaction_feature
            * attention
            + self.nam_scale
            * nam_rgb
        )

        output = self.output_refine(
            output
        )

        return (
            output,
            selected_nam,
            hierarchy_weights,
        )


class NAMDecoderInjection(nn.Module):
    """
    Inject selected NAM features into one decoder scale.
    """

    def __init__(
        self,
        decoder_channels: int,
        nam_channels: int,
    ) -> None:
        super().__init__()

        attention_channels = max(
            16,
            decoder_channels // 8,
        )

        self.nam_projection = ConvGNAct(
            nam_channels,
            decoder_channels,
            kernel_size=1,
            padding=0,
        )

        self.saliency_refine = (
            LightResidualBlock(
                decoder_channels
            )
        )

        self.nam_refine = (
            LightResidualBlock(
                decoder_channels
            )
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                decoder_channels * 2,
                attention_channels,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                attention_channels,
                decoder_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                4,
                1,
                kernel_size=7,
                padding=3,
            ),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            ConvGNAct(
                decoder_channels * 3,
                decoder_channels,
                kernel_size=1,
                padding=0,
            ),
            LightResidualBlock(
                decoder_channels
            ),
            LightResidualBlock(
                decoder_channels
            ),
        )

        self.output_refine = (
            LightResidualBlock(
                decoder_channels
            )
        )

        self.fusion_scale = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(
        self,
        saliency_feature: torch.Tensor,
        selected_nam: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        target_size = (
            saliency_feature.shape[-2:]
        )

        selected_nam = F.interpolate(
            selected_nam,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        selected_nam = (
            self.nam_projection(
                selected_nam
            )
        )

        saliency_feature = (
            self.saliency_refine(
                saliency_feature
            )
        )

        selected_nam = self.nam_refine(
            selected_nam
        )

        channel_attention = (
            self.channel_attention(
                torch.cat(
                    [
                        saliency_feature,
                        selected_nam,
                    ],
                    dim=1,
                )
            )
        )

        spatial_attention = (
            self.spatial_attention(
                torch.cat(
                    [
                        saliency_feature.mean(
                            dim=1,
                            keepdim=True,
                        ),
                        saliency_feature.amax(
                            dim=1,
                            keepdim=True,
                        ),
                        selected_nam.mean(
                            dim=1,
                            keepdim=True,
                        ),
                        selected_nam.amax(
                            dim=1,
                            keepdim=True,
                        ),
                    ],
                    dim=1,
                )
            )
        )

        fusion_feature = self.fusion(
            torch.cat(
                [
                    saliency_feature,
                    selected_nam,
                    saliency_feature
                    * selected_nam,
                ],
                dim=1,
            )
        )

        attention = (
            1.0 + channel_attention
        ) * (
            1.0 + spatial_attention
        )

        output = (
            saliency_feature
            + self.fusion_scale
            * fusion_feature
            * attention
        )

        output = self.output_refine(
            output
        )

        return (
            output,
            selected_nam,
        )