from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    """Convolution block used by MambaCVC figures: Conv + BN + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class GCTransferUnit(nn.Module):
    """Algorithm 1: Global Correlation Transfer Unit (GCTU).

    Fig. 3 annotates Q/K/V as H x W x C/2. We preserve that layout
    for the matrix multiplications in Algorithm 1, producing a W x W
    correlation matrix independently for each H position. This avoids
    inventing a spatial flattening step that is not present in the paper.
    """

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("GCTU channels must be even")

        reduced = channels // 2
        self.align_high = ConvBNReLU(channels, channels, 1)

        self.q1 = ConvBNReLU(channels, reduced, 1)
        self.k1 = ConvBNReLU(channels, reduced, 1)
        self.v1 = ConvBNReLU(channels, reduced, 1)
        self.q2 = ConvBNReLU(channels, reduced, 1)
        self.k2 = ConvBNReLU(channels, reduced, 1)
        self.v2 = ConvBNReLU(channels, reduced, 1)

        # Fig.3 explicitly marks the final 1x1 projection as shared.
        self.shared_output = ConvBNReLU(reduced, channels, 1)

    def _correlation(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        # Fig. 3 explicitly annotates q/k/v as H x W x C/2.
        # Preserve that layout for q @ k^T instead of flattening H*W:
        # [B,H,W,C/2] @ [B,H,C/2,W] -> [B,H,W,W].
        q_hwc = q.permute(0, 2, 3, 1)
        k_hwc = k.permute(0, 2, 3, 1)
        scores = torch.matmul(
            q_hwc,
            k_hwc.transpose(-1, -2),
        )
        return torch.softmax(scores, dim=-1)

    @staticmethod
    def _apply_correlation(
        scores: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        value_hwc = value.permute(0, 2, 3, 1)
        out_hwc = torch.matmul(scores, value_hwc)
        return out_hwc.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        low_feature: torch.Tensor,
        high_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        high_aligned = F.interpolate(
            high_feature,
            size=low_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high_aligned = self.align_high(high_aligned)

        q1 = self.q1(low_feature)
        k1 = self.k1(low_feature)
        v1 = self.v1(low_feature)
        q2 = self.q2(high_aligned)
        k2 = self.k2(high_aligned)
        v2 = self.v2(high_aligned)

        s11 = self._correlation(q1, k1)
        s12 = self._correlation(q1, k2)
        s21 = self._correlation(q2, k1)
        s22 = self._correlation(q2, k2)

        # Algorithm 1, Step 4 is followed literally:
        #   f_e_hat = s11 * v1 + s21 * v1
        #   f_n_hat = s12 * v2 + s22 * v2
        low_hat = (
            self._apply_correlation(s11, v1)
            + self._apply_correlation(s21, v1)
        )
        high_hat = (
            self._apply_correlation(s12, v2)
            + self._apply_correlation(s22, v2)
        )

        low_gct = self.shared_output(low_hat) + low_feature
        high_gct = self.shared_output(high_hat) + high_aligned
        return low_gct, high_gct, high_aligned


class VarianceCompensationUnit(nn.Module):
    """Equations (2)-(6): shared dual-baseline variance compensation."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        # The same c3x3 symbol is used on both symmetric branches in Fig.3.
        self.baseline_conv = ConvBNReLU(channels, channels, 3)
        self.deviation_conv = ConvBNReLU(channels, channels, 3)
        self.fuse_conv = ConvBNReLU(channels, channels, 3)

    def forward(
        self,
        branch1: torch.Tensor,
        branch2: torch.Tensor,
        residual1: torch.Tensor,
        residual2: torch.Tensor,
    ) -> torch.Tensor:
        base1 = self.baseline_conv(branch1)
        base2 = self.baseline_conv(branch2)

        f_mul = base1 * base2

        f_vcu1 = self.deviation_conv(base2 - f_mul) + base1
        f_vcu2 = self.deviation_conv(base1 - f_mul) + base2

        compensated = self.fuse_conv(f_vcu1 + f_vcu2)
        return compensated + residual1 + residual2


class GCVCM(nn.Module):
    """Global Correlation Variance Compensation Module."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.gctu = GCTransferUnit(channels)
        self.vcu = VarianceCompensationUnit(channels)

    def forward(
        self,
        encoder_feature: torch.Tensor,
        deeper_feature: torch.Tensor,
    ) -> torch.Tensor:
        gct_low, gct_high, deeper_aligned = self.gctu(
            encoder_feature,
            deeper_feature,
        )
        return self.vcu(
            gct_low,
            gct_high,
            encoder_feature,
            deeper_aligned,
        )


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    if c % groups != 0:
        raise ValueError("channels must be divisible by groups")
    x = x.reshape(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.reshape(b, c, h, w)


class SharedChannelSpatialAttention(nn.Module):
    """Shared CA + SA used by both LCTU branches.

    The paper specifies shared channel and spatial attention but does not
    define their inner formula. We use the canonical CBAM-style CA/SA.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        maxv = F.adaptive_max_pool2d(x, 1)
        channel_weight = torch.sigmoid(
            self.channel_mlp(avg) + self.channel_mlp(maxv)
        )
        x = x * channel_weight

        spatial_input = torch.cat(
            [x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)],
            dim=1,
        )
        spatial_weight = torch.sigmoid(self.spatial(spatial_input))
        return x * spatial_weight


class LCTransferUnit(nn.Module):
    """Algorithm 2: Local Correlation Transfer Unit (LCTU)."""

    def __init__(self, channels: int = 32, groups: int = 4) -> None:
        super().__init__()
        if channels % groups != 0:
            raise ValueError("LCTU channels must be divisible by groups")
        if groups != 4:
            raise ValueError("MambaCVC LCTU uses exactly 4 groups")

        self.channels = channels
        self.groups = groups
        group_channels = channels // groups

        self.low_input = ConvBNReLU(channels, channels, 1)
        self.high_input = ConvBNReLU(channels, channels, 1)

        def make_branch() -> nn.ModuleList:
            return nn.ModuleList(
                [
                    ConvBNReLU(group_channels, group_channels, 1),
                    ConvBNReLU(group_channels, group_channels, 3),
                    ConvBNReLU(group_channels, group_channels, 5, dilation=2),
                    ConvBNReLU(group_channels, group_channels, 7, dilation=3),
                ]
            )

        self.low_convs = make_branch()
        self.high_convs = make_branch()
        self.attention = SharedChannelSpatialAttention(channels)

    @staticmethod
    def _progressive_branch(
        chunks: tuple[torch.Tensor, ...],
        convs: nn.ModuleList,
    ) -> tuple[torch.Tensor, ...]:
        c1 = convs[0](chunks[0])
        c2 = convs[1](chunks[1] + c1)
        c3 = convs[2](chunks[2] + c2)
        c4 = convs[3](chunks[3] + c3)
        return c1, c2, c3, c4

    def forward(
        self,
        global_feature: torch.Tensor,
        deeper_decoder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        deeper_aligned = F.interpolate(
            deeper_decoder,
            size=global_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low_tilde = self.low_input(global_feature)
        high_tilde = self.high_input(deeper_aligned)

        low_shuffled = channel_shuffle(low_tilde, self.groups)
        high_shuffled = channel_shuffle(high_tilde, self.groups)
        low_chunks = torch.chunk(low_shuffled, self.groups, dim=1)
        high_chunks = torch.chunk(high_shuffled, self.groups, dim=1)

        low_parts = self._progressive_branch(low_chunks, self.low_convs)
        high_parts = self._progressive_branch(high_chunks, self.high_convs)

        f_sum = torch.stack((*low_parts, *high_parts), dim=0).sum(dim=0)
        low_interacted = torch.cat([part * f_sum for part in low_parts], dim=1)
        high_interacted = torch.cat([part * f_sum for part in high_parts], dim=1)

        # One shared CA+SA module is applied to both branches (Algorithm 2).
        low_lct = self.attention(low_interacted) + low_tilde
        high_lct = self.attention(high_interacted) + high_tilde
        return low_lct, high_lct, low_tilde, high_tilde


class LCVCM(nn.Module):
    """Local Correlation Variance Compensation Module."""

    def __init__(self, channels: int = 32, groups: int = 4) -> None:
        super().__init__()
        self.lctu = LCTransferUnit(channels, groups)
        self.vcu = VarianceCompensationUnit(channels)

    def forward(
        self,
        global_feature: torch.Tensor,
        deeper_decoder: torch.Tensor,
    ) -> torch.Tensor:
        lct_low, lct_high, low_tilde, high_tilde = self.lctu(
            global_feature,
            deeper_decoder,
        )
        return self.vcu(
            lct_low,
            lct_high,
            low_tilde,
            high_tilde,
        )


class DecoderBlock(nn.Sequential):
    """Dec-i in Fig.2: three consecutive 3x3 Conv + BN + ReLU blocks."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__(
            ConvBNReLU(channels, channels, 3),
            ConvBNReLU(channels, channels, 3),
            ConvBNReLU(channels, channels, 3),
        )


class SaliencyHead(nn.Module):
    """Blue prediction operator in Fig.2: one 3x3 convolution to 1 logit."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
