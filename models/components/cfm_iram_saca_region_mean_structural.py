"""Region-Mean Structural Field for CFMNet + SACA.

Treat `mean_60` as a continuous region-level structural field, never as a
binary edge map.

Shared representation:
  Preserve   = piecewise region-mean RGB field
  Transition = four-direction continuous region differences
  Context    = local/broad region context + center-context discrepancy

Two uses:
  A) F1 structural reconstruction
  B) late structural evidence after complete IRAM+SACA
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import ConvBNAct


class RegionMeanStructuralEncoder(nn.Module):
    def __init__(self, channels: int = 96) -> None:
        super().__init__()
        if channels % 3 != 0:
            raise ValueError("channels must be divisible by 3")

        branch_channels = channels // 3

        self.preserve = nn.Sequential(
            ConvBNAct(3, branch_channels, kernel_size=1),
            ConvBNAct(
                branch_channels,
                branch_channels,
                kernel_size=3,
                groups=branch_channels,
            ),
        )

        self.transition = nn.Sequential(
            ConvBNAct(4, branch_channels, kernel_size=1),
            ConvBNAct(
                branch_channels,
                branch_channels,
                kernel_size=3,
                groups=branch_channels,
            ),
        )

        self.context = nn.Sequential(
            ConvBNAct(12, branch_channels, kernel_size=1),
            ConvBNAct(
                branch_channels,
                branch_channels,
                kernel_size=3,
                groups=branch_channels,
            ),
        )

        self.fuse = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=1),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
            ),
            ConvBNAct(channels, channels, kernel_size=1),
        )

    @staticmethod
    def _directional_transition(mean_60: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(
            mean_60[:, :, :, 1:] - mean_60[:, :, :, :-1]
        ).mean(dim=1, keepdim=True)

        dy = torch.abs(
            mean_60[:, :, 1:, :] - mean_60[:, :, :-1, :]
        ).mean(dim=1, keepdim=True)

        left = F.pad(dx, (1, 0, 0, 0), value=0.0)
        right = F.pad(dx, (0, 1, 0, 0), value=0.0)
        up = F.pad(dy, (0, 0, 1, 0), value=0.0)
        down = F.pad(dy, (0, 0, 0, 1), value=0.0)

        return torch.cat([left, right, up, down], dim=1)

    def forward(
        self,
        mean_60: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        layout = F.interpolate(
            mean_60,
            size=target_size,
            mode="nearest",
        )

        preserve_feature = self.preserve(layout)

        transition = self._directional_transition(mean_60)
        transition = F.adaptive_max_pool2d(
            transition,
            output_size=target_size,
        )
        transition_feature = self.transition(transition)

        local_context = F.avg_pool2d(
            layout,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        broad_context = F.avg_pool2d(
            layout,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        context_descriptor = torch.cat(
            [
                local_context,
                broad_context,
                torch.abs(layout - local_context),
                torch.abs(layout - broad_context),
            ],
            dim=1,
        )
        context_feature = self.context(context_descriptor)

        structural = torch.cat(
            [preserve_feature, transition_feature, context_feature],
            dim=1,
        )
        return structural + self.fuse(structural)


class F1StructuralReconstruction(nn.Module):
    def __init__(
        self,
        f1_channels: int = 96,
        structural_channels: int = 96,
    ) -> None:
        super().__init__()

        self.f1_query = ConvBNAct(
            f1_channels,
            structural_channels,
            kernel_size=1,
        )
        self.structural_query = ConvBNAct(
            structural_channels,
            structural_channels,
            kernel_size=1,
        )

        self.interaction = nn.Sequential(
            ConvBNAct(
                structural_channels * 4,
                structural_channels,
                kernel_size=1,
            ),
            ConvBNAct(
                structural_channels,
                structural_channels,
                kernel_size=3,
                groups=structural_channels,
            ),
            ConvBNAct(
                structural_channels,
                structural_channels,
                kernel_size=1,
            ),
        )

        self.support = nn.Sequential(
            nn.Conv2d(
                structural_channels,
                1,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.delta = nn.Conv2d(
            structural_channels,
            f1_channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        f1: torch.Tensor,
        structural: torch.Tensor,
    ) -> torch.Tensor:
        q_f1 = self.f1_query(f1)
        q_struct = self.structural_query(structural)

        relation = self.interaction(
            torch.cat(
                [
                    q_f1,
                    q_struct,
                    q_f1 * q_struct,
                    torch.abs(q_f1 - q_struct),
                ],
                dim=1,
            )
        )

        return f1 + self.support(structural) * self.delta(relation)


class LateStructuralCorrection(nn.Module):
    def __init__(
        self,
        host_channels: int,
        structural_channels: int = 96,
    ) -> None:
        super().__init__()

        self.host_query = ConvBNAct(
            host_channels,
            structural_channels,
            kernel_size=1,
        )
        self.structural_query = ConvBNAct(
            structural_channels,
            structural_channels,
            kernel_size=1,
        )

        self.interaction = nn.Sequential(
            ConvBNAct(
                structural_channels * 4,
                structural_channels,
                kernel_size=1,
            ),
            ConvBNAct(
                structural_channels,
                structural_channels,
                kernel_size=3,
                groups=structural_channels,
            ),
            ConvBNAct(
                structural_channels,
                structural_channels,
                kernel_size=1,
            ),
        )

        self.support = nn.Sequential(
            nn.Conv2d(
                structural_channels,
                1,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.delta = nn.Conv2d(
            structural_channels,
            host_channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        host_feature: torch.Tensor,
        structural: torch.Tensor,
    ) -> torch.Tensor:
        q_host = self.host_query(host_feature)
        q_struct = self.structural_query(structural)

        relation = self.interaction(
            torch.cat(
                [
                    q_host,
                    q_struct,
                    q_host * q_struct,
                    torch.abs(q_host - q_struct),
                ],
                dim=1,
            )
        )

        return (
            host_feature
            + self.support(structural)
            * self.delta(relation)
        )
