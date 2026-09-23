"""NAM structural evidence for two controlled SACA integration schemes.

Shared premise
--------------
Only NAMLab region mean M60 is used. RGB-M60 is intentionally excluded.

M60 is encoded as a saliency-agnostic structural-aware evidence:
    * region layout: piecewise region-mean image preserved with nearest resize
    * region transition: local RGB range on M60

Scheme A: late evidence interaction after complete SACA.
Scheme B: prediction-only structural refinement after complete decoder logits.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import ConvBNAct
from models.components.cfm_iram_saca import SACAStream
from models.components.cfm_iram_saca_v2 import SACAv2Stream
from models.components.cfm_iram_strong_common import IRAMStrongNeck


class NAMStructuralEvidenceEncoder(nn.Module):
    """Encode M60 into a structural-aware evidence tensor at a target grid."""

    def __init__(
        self,
        out_channels: int = 128,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()

        self.layout_encoder = nn.Sequential(
            ConvBNAct(3, hidden_channels, kernel_size=1),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                groups=hidden_channels,
            ),
        )

        self.transition_encoder = nn.Sequential(
            ConvBNAct(1, hidden_channels, kernel_size=3),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                groups=hidden_channels,
            ),
        )

        self.fuse = nn.Sequential(
            ConvBNAct(
                hidden_channels * 2,
                out_channels,
                kernel_size=1,
            ),
            ConvBNAct(
                out_channels,
                out_channels,
                kernel_size=3,
                groups=out_channels,
            ),
            ConvBNAct(
                out_channels,
                out_channels,
                kernel_size=1,
            ),
        )

    @staticmethod
    def _transition_map(mean_60: torch.Tensor) -> torch.Tensor:
        local_max = F.max_pool2d(
            mean_60,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        local_min = -F.max_pool2d(
            -mean_60,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        rgb_range = (local_max - local_min).clamp_min(0.0)
        return rgb_range.mean(dim=1, keepdim=True)

    def forward(
        self,
        mean_60: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        # Preserve piecewise region geometry.
        region_layout = F.interpolate(
            mean_60,
            size=target_size,
            mode="nearest",
        )

        # Preserve sparse transitions while reducing resolution.
        transition = self._transition_map(mean_60)
        transition = F.adaptive_max_pool2d(
            transition,
            output_size=target_size,
        )

        layout_feature = self.layout_encoder(region_layout)
        transition_feature = self.transition_encoder(transition)

        return self.fuse(
            torch.cat(
                [layout_feature, transition_feature],
                dim=1,
            )
        )


class NAMLateEvidenceInteraction(nn.Module):
    """Residual cooperation between complete SACA task evidence and NAM."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()

        self.task_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.nam_projection = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, channels),
        )

        self.interaction = nn.Sequential(
            ConvBNAct(channels * 4, channels, kernel_size=1),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
            ),
        )

        self.delta = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        task: torch.Tensor,
        nam: torch.Tensor,
    ) -> torch.Tensor:
        task_query = self.task_projection(task)
        nam_query = self.nam_projection(nam)

        interaction = self.interaction(
            torch.cat(
                [
                    task_query,
                    nam_query,
                    task_query * nam_query,
                    torch.abs(task_query - nam_query),
                ],
                dim=1,
            )
        )

        return task + self.delta(interaction)


class SACANAMLateInteractionStream(nn.Module):
    """Complete SACA-v1 first, NAM structural evidence second."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.saca = SACAStream(channels=channels, ablation="full")
        self.nam2 = NAMStructuralEvidenceEncoder(out_channels=channels)
        self.nam3 = NAMStructuralEvidenceEncoder(out_channels=channels)
        self.interact2 = NAMLateEvidenceInteraction(channels=channels)
        self.interact3 = NAMLateEvidenceInteraction(channels=channels)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        mean_60: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task2, task3, coarse = self.saca(features)

        nam2 = self.nam2(mean_60, target_size=task2.shape[-2:])
        nam3 = self.nam3(mean_60, target_size=task3.shape[-2:])

        return (
            self.interact2(task2, nam2),
            self.interact3(task3, nam3),
            coarse,
        )


class SACAv2NAMLateInteractionStream(nn.Module):
    """Complete SACA-v2 first, NAM structural evidence second."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.saca = SACAv2Stream(channels=channels, ablation="full")
        self.nam2 = NAMStructuralEvidenceEncoder(out_channels=channels)
        self.nam3 = NAMStructuralEvidenceEncoder(out_channels=channels)
        self.interact2 = NAMLateEvidenceInteraction(channels=channels)
        self.interact3 = NAMLateEvidenceInteraction(channels=channels)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        mean_60: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        task2, task3, coarse = self.saca(features)

        nam2 = self.nam2(mean_60, target_size=task2.shape[-2:])
        nam3 = self.nam3(mean_60, target_size=task3.shape[-2:])

        return (
            self.interact2(task2, nam2),
            self.interact3(task3, nam3),
            coarse,
        )


class IRAMNAMLateInteractionNeck(IRAMStrongNeck):
    """Original IRAM/ThreeStreamFusion with an M60-aware task stream."""

    def forward(
        self,
        features: Sequence[torch.Tensor],
        mean_60: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        f1, f2, f3, f4 = features

        task2, task3, coarse = self.x_stream(
            features,
            mean_60=mean_60,
        )

        out2 = self.fuse2(f2, self.freq2(f2), task2)
        out3 = self.fuse3(f3, self.freq3(f3), task3)

        return [f1, out2, out3, f4], coarse


class NAMPredictionStructureRefiner(nn.Module):
    """Use M60 only to make a residual correction to final saliency logits."""

    def __init__(
        self,
        nam_channels: int = 128,
        hidden_channels: int = 64,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        if reduction < 1:
            raise ValueError("reduction must be >= 1")

        self.reduction = reduction
        self.nam_encoder = NAMStructuralEvidenceEncoder(
            out_channels=nam_channels,
            hidden_channels=64,
        )
        self.nam_projection = ConvBNAct(
            nam_channels,
            hidden_channels,
            kernel_size=1,
        )

        self.prediction_encoder = nn.Sequential(
            ConvBNAct(3, hidden_channels, kernel_size=3),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                groups=hidden_channels,
            ),
        )

        self.interaction = nn.Sequential(
            ConvBNAct(
                hidden_channels * 4,
                hidden_channels,
                kernel_size=1,
            ),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                groups=hidden_channels,
            ),
        )

        self.delta_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    @staticmethod
    def _prediction_boundary(probability: torch.Tensor) -> torch.Tensor:
        local_max = F.max_pool2d(
            probability,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        local_min = -F.max_pool2d(
            -probability,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        return (local_max - local_min).clamp(0.0, 1.0)

    def forward(
        self,
        logits: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        boundary = self._prediction_boundary(probability)
        uncertainty = (
            4.0 * probability * (1.0 - probability)
        ).clamp(0.0, 1.0)

        height, width = logits.shape[-2:]
        target_size = (
            max(1, height // self.reduction),
            max(1, width // self.reduction),
        )

        nam_feature = self.nam_projection(
            self.nam_encoder(mean_60, target_size=target_size)
        )

        probability_low = F.interpolate(
            probability,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        boundary_low = F.adaptive_max_pool2d(
            boundary,
            output_size=target_size,
        )
        uncertainty_low = F.interpolate(
            uncertainty,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        prediction_feature = self.prediction_encoder(
            torch.cat(
                [probability_low, boundary_low, uncertainty_low],
                dim=1,
            )
        )

        interaction = self.interaction(
            torch.cat(
                [
                    prediction_feature,
                    nam_feature,
                    prediction_feature * nam_feature,
                    torch.abs(prediction_feature - nam_feature),
                ],
                dim=1,
            )
        )

        delta_low = self.delta_head(interaction)
        delta = F.interpolate(
            delta_low,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        # NAM may correct only prediction boundary / uncertainty regions.
        support = (boundary + uncertainty).clamp(0.0, 1.0)
        return logits + support * delta
