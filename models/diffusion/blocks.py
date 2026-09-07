from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embedding for integer diffusion steps."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(
            half,
            dtype=torch.float32,
            device=timesteps.device,
        )
        / max(half, 1)
    )
    arguments = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat(
        [torch.cos(arguments), torch.sin(arguments)],
        dim=-1,
    )
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])],
            dim=-1,
        )
    return embedding


class TimeResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        groups: int = 8,
    ) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(
            min(groups, in_channels),
            in_channels,
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2),
        )

        self.norm2 = nn.GroupNorm(
            min(groups, out_channels),
            out_channels,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        )

        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(
        self,
        feature: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.skip(feature)

        x = self.conv1(
            F.silu(
                self.norm1(feature)
            )
        )

        scale, shift = self.time_projection(
            time_embedding
        ).chunk(2, dim=1)

        x = self.norm2(x)
        x = (
            x
            * (1.0 + scale[:, :, None, None])
            + shift[:, :, None, None]
        )
        x = self.conv2(F.silu(x))

        return residual + x


class SpatialSelfAttention(nn.Module):
    """Full attention used only at the 11x11 bottleneck."""

    def __init__(
        self,
        channels: int,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")

        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1,
            bias=False,
        )
        self.projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        qkv = self.qkv(self.norm(feature))
        q, k, v = qkv.chunk(3, dim=1)

        def reshape(x: torch.Tensor) -> torch.Tensor:
            return (
                x.reshape(
                    batch,
                    self.heads,
                    self.head_dim,
                    height * width,
                )
                .transpose(2, 3)
            )

        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        attention = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) * self.scale
        attention = attention.softmax(dim=-1)

        x = torch.matmul(attention, v)
        x = (
            x.transpose(2, 3)
            .contiguous()
            .reshape(batch, channels, height, width)
        )
        return feature + self.projection(x)


class PriorConditionBlock(nn.Module):
    """
    Inject one hierarchical prior with both spatial residual and
    channel-wise affine modulation.
    """

    def __init__(
        self,
        prior_channels: int,
        feature_channels: int,
    ) -> None:
        super().__init__()

        self.prior_projection = nn.Sequential(
            nn.Conv2d(
                prior_channels,
                feature_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(8, feature_channels),
            nn.SiLU(),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
                groups=feature_channels,
                bias=False,
            ),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=1,
            ),
        )

        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                feature_channels,
                feature_channels * 2,
                kernel_size=1,
            ),
        )

        nn.init.zeros_(self.spatial[-1].weight)
        if self.spatial[-1].bias is not None:
            nn.init.zeros_(self.spatial[-1].bias)
        nn.init.zeros_(self.channel[-1].weight)
        if self.channel[-1].bias is not None:
            nn.init.zeros_(self.channel[-1].bias)

    def forward(
        self,
        feature: torch.Tensor,
        prior: torch.Tensor,
    ) -> torch.Tensor:
        if prior.shape[-2:] != feature.shape[-2:]:
            prior = F.interpolate(
                prior,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        prior = self.prior_projection(prior)
        gamma, beta = self.channel(prior).chunk(2, dim=1)
        gamma = 0.5 * torch.tanh(gamma)
        beta = 0.5 * torch.tanh(beta)

        return (
            feature * (1.0 + gamma)
            + beta
            + self.spatial(prior)
        )


class InformationPerturbation(nn.Module):
    """IPDiff-style training-time feature corruption."""

    def __init__(self, probability: float = 0.2) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability <= 0.0:
            return feature

        keep = torch.rand(
            feature.shape[0],
            1,
            feature.shape[2],
            feature.shape[3],
            device=feature.device,
            dtype=feature.dtype,
        ) >= self.probability

        return feature * keep
