# models/components/sod_blocks.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
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


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.conv1 = ConvNormAct(
            channels,
            channels,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
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
            x + self.conv2(
                self.conv1(x)
            )
        )


class PyramidContextBlock(nn.Module):
    """
    在最高层特征上提取局部和多尺度全局上下文。
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        branch_channels = channels // 4

        self.local_branch = ConvNormAct(
            channels,
            branch_channels,
            kernel_size=1,
            padding=0,
        )

        self.pool_sizes = (
            1,
            2,
            4,
        )

        self.pool_branches = nn.ModuleList(
            [
                ConvNormAct(
                    channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0,
                )
                for _ in self.pool_sizes
            ]
        )

        self.fusion = nn.Sequential(
            ConvNormAct(
                channels,
                channels,
            ),
            ResidualConvBlock(
                channels
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        target_size = x.shape[-2:]

        features = [
            self.local_branch(x)
        ]

        for pool_size, branch in zip(
            self.pool_sizes,
            self.pool_branches,
        ):
            pooled = F.adaptive_avg_pool2d(
                x,
                output_size=pool_size,
            )

            pooled = branch(pooled)

            pooled = F.interpolate(
                pooled,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

            features.append(pooled)

        context = torch.cat(
            features,
            dim=1,
        )

        return x + self.fusion(context)


class SaliencyGuidedFusion(nn.Module):
    """
    使用高层粗显著图引导当前尺度的低层特征。

    guide 用于增强前景区域；
    uncertainty 用于强调目标边界和不确定区域。
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.low_refine = ResidualConvBlock(
            channels
        )

        self.high_refine = ResidualConvBlock(
            channels
        )

        self.fusion = nn.Sequential(
            ConvNormAct(
                channels * 3,
                channels,
            ),
            ResidualConvBlock(
                channels
            ),
        )

    def forward(
        self,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
        guide_logits: torch.Tensor,
    ) -> torch.Tensor:
        target_size = low_feature.shape[-2:]

        high_feature = F.interpolate(
            high_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        guide_logits = F.interpolate(
            guide_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        low_feature = self.low_refine(
            low_feature
        )

        high_feature = self.high_refine(
            high_feature
        )

        guide = torch.sigmoid(
            guide_logits
        )

        uncertainty = 4.0 * guide * (
            1.0 - guide
        )

        foreground_feature = low_feature * (
            1.0 + guide
        )

        boundary_feature = (
            low_feature * uncertainty
        )

        fused = torch.cat(
            [
                foreground_feature,
                boundary_feature,
                high_feature,
            ],
            dim=1,
        )

        return high_feature + self.fusion(
            fused
        )


class BoundaryRefinementBlock(nn.Module):
    """
    使用 stride 4 浅层特征和 stride 8 语义特征，
    对最终显著性特征进行边界细化。

    当前不单独输出边缘图，边界分支通过最终显著图
    的监督端到端学习。
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.boundary_fusion = nn.Sequential(
            ConvNormAct(
                channels * 3,
                channels,
            ),
            ResidualConvBlock(
                channels
            ),
        )

        self.boundary_gate = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
        )

        self.output_fusion = nn.Sequential(
            ConvNormAct(
                channels * 2,
                channels,
            ),
            ResidualConvBlock(
                channels
            ),
        )

    def forward(
        self,
        shallow_feature: torch.Tensor,
        semantic_feature: torch.Tensor,
        saliency_feature: torch.Tensor,
    ) -> torch.Tensor:
        target_size = shallow_feature.shape[-2:]

        semantic_feature = F.interpolate(
            semantic_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        saliency_feature = F.interpolate(
            saliency_feature,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        local_average = F.avg_pool2d(
            shallow_feature,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        detail_feature = torch.abs(
            shallow_feature - local_average
        )

        boundary_feature = (
            self.boundary_fusion(
                torch.cat(
                    [
                        shallow_feature,
                        semantic_feature,
                        detail_feature,
                    ],
                    dim=1,
                )
            )
        )

        boundary_attention = torch.sigmoid(
            self.boundary_gate(
                boundary_feature
            )
        )

        boundary_feature = boundary_feature * (
            1.0 + boundary_attention
        )

        return self.output_fusion(
            torch.cat(
                [
                    saliency_feature,
                    boundary_feature,
                ],
                dim=1,
            )
        )


class PredictionHead(nn.Module):
    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.head = nn.Sequential(
            ConvNormAct(
                channels,
                channels // 2,
            ),
            nn.Conv2d(
                channels // 2,
                1,
                kernel_size=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(x)