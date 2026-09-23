"""M60 region-relation evidence for SACA-v1 / SACA-v2.

Design
------
M60 is not treated as a fourth generic feature stream.

At the F2 working grid:
    1. Resize region-mean M60 with nearest interpolation.
    2. Build a non-learned local 5x5 affinity from M60 RGB differences.
    3. Use that affinity to reconstruct the learned SACA Region feature R,
       producing an M60-consistent region representation R_m60.
    4. Compare R and R_m60 explicitly:
           agreement    = q(R) * q(R_m60)
           disagreement = |q(R) - q(R_m60)|
    5. Reconstruct a residual delta_R from:
           [semantic, R, R_m60, agreement, disagreement]
    6. Feed R* = R + delta_R into the ORIGINAL SACA cooperation block.

Important:
    * Semantic construction is unchanged.
    * Region construction is unchanged.
    * Structure construction uses the ORIGINAL learned Region R.
    * M60 modifies only the Region state immediately before cooperation.
    * Outer Base / IRAM / Task ThreeStreamFusion is unchanged.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import ConvBNAct
from models.components.cfm_iram_saca import SACAStream
from models.components.cfm_iram_saca_v2 import SACAv2Stream
from models.components.cfm_iram_strong_common import (
    IRAMStrongNeck,
    resize_like,
)


class M60RegionRelationBridge(nn.Module):
    """Inject explicit M60 region relations into a learned Region state.

    M60 is loaded by SODDataset in RGB [0, 1].  We preserve its piecewise
    region-mean structure by using nearest interpolation and direct RGB
    distances, instead of first turning M60 into another learned feature
    stream.

    The local distance scale is normalized independently at each F2 position:
        d_ij / mean_j(d_ij)

    Therefore the affinity has no tuned temperature hyperparameter.
    """

    def __init__(
        self,
        channels: int = 128,
        window_size: int = 5,
    ) -> None:
        super().__init__()

        if window_size < 1 or window_size % 2 == 0:
            raise ValueError(
                "window_size must be a positive odd integer."
            )

        self.channels = channels
        self.window_size = window_size
        self.radius = window_size // 2

        # Shared comparison space for R and its M60-consistent reconstruction.
        # GroupNorm has no running statistics, so the same module can safely
        # process both tensors.
        self.compare_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.compare_norm = nn.GroupNorm(
            8,
            channels,
        )

        # semantic + R + R_m60 + agreement + disagreement -> delta_R
        self.reconstruct = nn.Sequential(
            ConvBNAct(
                channels * 5,
                channels,
                kernel_size=1,
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=1,
            ),
        )

    def _build_affinity(
        self,
        mean_60: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """Return local M60 affinity [B, K*K, H, W]."""
        mean_f2 = F.interpolate(
            mean_60,
            size=target_size,
            mode="nearest",
        )

        height, width = target_size

        padded = F.pad(
            mean_f2,
            (
                self.radius,
                self.radius,
                self.radius,
                self.radius,
            ),
            mode="replicate",
        )

        distances: list[torch.Tensor] = []

        for dy in range(self.window_size):
            for dx in range(self.window_size):
                neighbor = padded[
                    :,
                    :,
                    dy:dy + height,
                    dx:dx + width,
                ]

                # Mean absolute RGB distance between two region-mean pixels.
                distance = torch.mean(
                    torch.abs(
                        neighbor - mean_f2
                    ),
                    dim=1,
                )
                distances.append(distance)

        distance_tensor = torch.stack(
            distances,
            dim=1,
        )

        # Local self-normalization makes the relation contrast-adaptive and
        # avoids introducing a hand-tuned affinity temperature.
        local_scale = (
            distance_tensor
            .mean(
                dim=1,
                keepdim=True,
            )
            .clamp_min(1e-4)
        )

        normalized_distance = (
            distance_tensor / local_scale
        )

        return torch.softmax(
            -normalized_distance,
            dim=1,
        )

    def _region_consistent_reconstruction(
        self,
        region: torch.Tensor,
        affinity: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate R using only the local relation supplied by M60."""
        height, width = region.shape[-2:]

        padded = F.pad(
            region,
            (
                self.radius,
                self.radius,
                self.radius,
                self.radius,
            ),
            mode="replicate",
        )

        reconstructed = torch.zeros_like(
            region
        )

        index = 0

        for dy in range(self.window_size):
            for dx in range(self.window_size):
                neighbor = padded[
                    :,
                    :,
                    dy:dy + height,
                    dx:dx + width,
                ]

                reconstructed = (
                    reconstructed
                    + neighbor
                    * affinity[
                        :,
                        index:index + 1,
                    ]
                )

                index += 1

        return reconstructed

    def _comparison_feature(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        return self.compare_norm(
            self.compare_projection(
                feature
            )
        )

    def forward(
        self,
        semantic: torch.Tensor,
        region: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        affinity = self._build_affinity(
            mean_60=mean_60,
            target_size=region.shape[-2:],
        )

        region_m60 = (
            self._region_consistent_reconstruction(
                region=region,
                affinity=affinity,
            )
        )

        region_query = self._comparison_feature(
            region
        )
        m60_query = self._comparison_feature(
            region_m60
        )

        agreement = (
            region_query * m60_query
        )
        disagreement = torch.abs(
            region_query - m60_query
        )

        delta_region = self.reconstruct(
            torch.cat(
                [
                    semantic,
                    region,
                    region_m60,
                    agreement,
                    disagreement,
                ],
                dim=1,
            )
        )

        # Deliberately no learnable residual scale in the first experiment.
        region_refined = (
            region + delta_region
        )

        return (
            region_refined,
            region_m60,
        )


class SACAM60RegionRelationStream(SACAStream):
    """SACA-v1 + M60 region relation, before original shared cooperation."""

    def __init__(
        self,
        channels: int = 128,
        window_size: int = 5,
    ) -> None:
        super().__init__(
            channels=channels,
            ablation="full",
        )

        self.m60_region_relation = (
            M60RegionRelationBridge(
                channels=channels,
                window_size=window_size,
            )
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        mean_60: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if len(features) != 4:
            raise ValueError(
                "SACAM60RegionRelationStream expects "
                f"four CFMNet stages, got {len(features)}"
            )

        p1, p2, p3, p4 = [
            project(feature)
            for project, feature in zip(
                self.project,
                features,
            )
        ]

        semantic = self.semantic_anchor(
            p2=p2,
            p3=p3,
            p4=p4,
        )

        # Original SACA-v1 Region construction.
        region = self.region_reconstruction(
            p2=p2,
            semantic=semantic,
            use_semantic_guidance=True,
        )

        # Original SACA-v1 Structure construction still sees the ORIGINAL R.
        structure = self.structure_reconstruction(
            p1=p1,
            p2=p2,
            semantic=semantic,
            region=region,
            use_guidance=True,
        )

        # New evidence is injected only after A/R/T are fully constructed.
        region_with_m60, _ = (
            self.m60_region_relation(
                semantic=semantic,
                region=region,
                mean_60=mean_60,
            )
        )

        # Original SACA-v1 shared Region/Structure cooperation.
        (
            region_refined,
            structure_refined,
            cooperative,
        ) = self.cooperation(
            semantic=semantic,
            region=region_with_m60,
            structure=structure,
            enabled=True,
        )

        task = self.task_reconstruction(
            torch.cat(
                [
                    semantic,
                    region_refined,
                    structure_refined,
                    cooperative,
                ],
                dim=1,
            )
        )

        return (
            self.saliency2(task),
            self.saliency3(
                resize_like(
                    task,
                    p3,
                )
            ),
            self.coarse_head(
                semantic
            ),
        )


class SACAv2M60RegionRelationStream(SACAv2Stream):
    """SACA-v2 + M60 region relation, before original directional cooperation."""

    def __init__(
        self,
        channels: int = 128,
        window_size: int = 5,
    ) -> None:
        super().__init__(
            channels=channels,
            ablation="full",
        )

        self.m60_region_relation = (
            M60RegionRelationBridge(
                channels=channels,
                window_size=window_size,
            )
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        mean_60: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if len(features) != 4:
            raise ValueError(
                "SACAv2M60RegionRelationStream expects "
                f"four CFMNet stages, got {len(features)}"
            )

        p1, p2, p3, p4 = [
            project(feature)
            for project, feature in zip(
                self.project,
                features,
            )
        ]

        # Original SACA-v2 semantic front-end.
        semantic_seed = self.semantic_anchor(
            p2=p2,
            p3=p3,
            p4=p4,
        )

        coarse_logits = self.coarse_head(
            semantic_seed
        )

        (
            semantic,
            probability,
            relevance,
        ) = self.semantic_relevance(
            semantic_seed=semantic_seed,
            coarse_logits=coarse_logits,
            enabled=True,
        )

        # Original SACA-v2 Region construction.
        region = self.region_reconstruction(
            p2=p2,
            semantic=semantic,
            probability=probability,
            relevance=relevance,
        )

        _, boundary_feature = (
            self.semantic_boundary(
                coarse_logits
            )
        )

        # Original Guided Structure still sees the ORIGINAL R.
        structure = self.structure_reconstruction(
            p1=p1,
            p2=p2,
            semantic=semantic,
            region=region,
            boundary_feature=boundary_feature,
            use_boundary=True,
        )

        # Same M60 bridge as v1.
        region_with_m60, _ = (
            self.m60_region_relation(
                semantic=semantic,
                region=region,
                mean_60=mean_60,
            )
        )

        # Original full SACA-v2 directional cooperation.
        (
            region_refined,
            structure_refined,
            cooperative,
        ) = self.cooperation(
            semantic=semantic,
            region=region_with_m60,
            structure=structure,
            enabled=True,
        )

        task = self.task_reconstruction(
            torch.cat(
                [
                    semantic,
                    region_refined,
                    structure_refined,
                    cooperative,
                ],
                dim=1,
            )
        )

        return (
            self.saliency2(task),
            self.saliency3(
                resize_like(
                    task,
                    p3,
                )
            ),
            coarse_logits,
        )


class IRAMM60StrongNeck(IRAMStrongNeck):
    """The original IRAM strong neck with an M60-aware task stream.

    All IRAM / ThreeStreamFusion modules are inherited unchanged.
    The only interface change is:
        x_stream(features, mean_60)
    """

    def forward(
        self,
        features: Sequence[torch.Tensor],
        mean_60: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        torch.Tensor,
    ]:
        f1, f2, f3, f4 = features

        x2, x3, coarse = self.x_stream(
            features,
            mean_60,
        )

        out2 = self.fuse2(
            f2,
            self.freq2(f2),
            x2,
        )

        out3 = self.fuse3(
            f3,
            self.freq3(f3),
            x3,
        )

        return (
            [f1, out2, out3, f4],
            coarse,
        )
