"""Semantic-Anchored Cooperative Adapter (SACA) for CFMNet + IRAM.

SACA is a task-adaptation stream built from the evidence ordering suggested by
recent leave-one-out experiments:

    semantic anchor -> semantic-conditioned region -> guided structure
                     -> region/structure cooperative refinement

The outer neck stays unchanged:

    [CFMNet base, IRAM frequency, SACA task] -> existing ThreeStreamFusion.

The ablation switches intentionally preserve the module graph, channel widths,
and parameter count. The selected pathway is silenced with zeros only at the
point where it should stop influencing the task representation.
"""
from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import ConvBNAct
from models.components.cfm_iram_strong_common import resize_like


SACAAblation = Literal[
    "full",
    "no_semantic_region_guidance",
    "no_guided_structure",
    "no_region_structure_cooperation",
]


class SemanticAnchor(nn.Module):
    """Build a sample-adaptive multi-scale saliency semantic anchor at F2 scale.

    F2/F3/F4 are first projected to the same width. Global descriptors predict
    three sample-wise scale weights, then the weighted feature is reconstructed
    into the task width. This is deliberately feature reconstruction rather than
    a spatial gate.
    """

    def __init__(
        self,
        projected_channels: int = 64,
        out_channels: int = 128,
    ) -> None:
        super().__init__()

        self.scale_score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                projected_channels * 3,
                projected_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU6(inplace=True),
            nn.Conv2d(
                projected_channels,
                3,
                kernel_size=1,
                bias=True,
            ),
        )

        self.reconstruct = nn.Sequential(
            ConvBNAct(
                projected_channels,
                out_channels,
                kernel_size=3,
            ),
            ConvBNAct(
                out_channels,
                out_channels,
                kernel_size=3,
            ),
        )

    def forward(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
    ) -> torch.Tensor:
        p3_up = resize_like(p3, p2)
        p4_up = resize_like(p4, p2)

        stacked = torch.cat(
            [p2, p3_up, p4_up],
            dim=1,
        )

        weights = torch.softmax(
            self.scale_score(stacked),
            dim=1,
        )

        semantic_seed = (
            p2 * weights[:, 0:1]
            + p3_up * weights[:, 1:2]
            + p4_up * weights[:, 2:3]
        )

        return self.reconstruct(semantic_seed)


class RegionReconstruction(nn.Module):
    """Construct a semantic-conditioned region representation.

    Inputs:
        * F2-local region seed
        * semantic anchor
        * 3x3 pooled semantic context
        * 7x7 pooled semantic context

    The four tensors are fused into a new region representation. This is not a
    multiplicative mask and therefore gives region evidence its own feature
    capacity.
    """

    def __init__(
        self,
        projected_channels: int = 64,
        channels: int = 128,
    ) -> None:
        super().__init__()

        self.region_seed = nn.Sequential(
            ConvBNAct(
                projected_channels,
                channels,
                kernel_size=3,
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
            ),
        )

        self.reduce = ConvBNAct(
            channels * 4,
            channels,
            kernel_size=1,
        )

        self.context = nn.Sequential(
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

    def forward(
        self,
        p2: torch.Tensor,
        semantic: torch.Tensor,
        use_semantic_guidance: bool,
    ) -> torch.Tensor:
        seed = self.region_seed(p2)

        semantic_direct = semantic
        semantic_local = F.avg_pool2d(
            semantic,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        semantic_broad = F.avg_pool2d(
            semantic,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        if not use_semantic_guidance:
            semantic_direct = torch.zeros_like(semantic_direct)
            semantic_local = torch.zeros_like(semantic_local)
            semantic_broad = torch.zeros_like(semantic_broad)

        region = self.reduce(
            torch.cat(
                [
                    seed,
                    semantic_direct,
                    semantic_local,
                    semantic_broad,
                ],
                dim=1,
            )
        )

        return region + self.context(region)


class StructureReconstruction(nn.Module):
    """Turn raw high-frequency candidates into salient structure evidence.

    Raw structure is extracted from F1/F2 high-pass residuals. In the full
    model, semantic and region representations jointly reconstruct that raw
    structure so that high-frequency background texture is less likely to be
    treated as salient shape.
    """

    def __init__(
        self,
        projected_channels: int = 64,
        channels: int = 128,
    ) -> None:
        super().__init__()

        self.structure_seed = nn.Sequential(
            ConvBNAct(
                projected_channels * 2,
                channels,
                kernel_size=3,
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
            ),
        )

        self.reduce = ConvBNAct(
            channels * 3,
            channels,
            kernel_size=1,
        )

        self.refine = nn.Sequential(
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

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        semantic: torch.Tensor,
        region: torch.Tensor,
        use_guidance: bool,
    ) -> torch.Tensor:
        high1 = p1 - F.avg_pool2d(
            p1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        high2 = p2 - F.avg_pool2d(
            p2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        raw_structure = self.structure_seed(
            torch.cat(
                [
                    resize_like(high1, p2),
                    high2,
                ],
                dim=1,
            )
        )

        semantic_input = semantic
        region_input = region

        if not use_guidance:
            semantic_input = torch.zeros_like(semantic_input)
            region_input = torch.zeros_like(region_input)

        structure = self.reduce(
            torch.cat(
                [
                    raw_structure,
                    semantic_input,
                    region_input,
                ],
                dim=1,
            )
        )

        return structure + self.refine(structure)


class RegionStructureCooperation(nn.Module):
    """One explicit bidirectional correction round between region and structure.

    A shared semantic-conditioned interaction representation produces two
    different corrections. The full version also emits a cooperative feature
    used by final task reconstruction. The ablation computes all modules but
    zeros their influence, preserving parameter count and almost identical cost.
    """

    def __init__(
        self,
        channels: int = 128,
    ) -> None:
        super().__init__()

        self.shared = ConvBNAct(
            channels * 3,
            channels,
            kernel_size=1,
        )

        self.region_delta = nn.Sequential(
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

        self.structure_delta = nn.Sequential(
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

        self.cooperative_feature = nn.Sequential(
            ConvBNAct(
                channels * 2,
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

    def forward(
        self,
        semantic: torch.Tensor,
        region: torch.Tensor,
        structure: torch.Tensor,
        enabled: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        shared = self.shared(
            torch.cat(
                [semantic, region, structure],
                dim=1,
            )
        )

        delta_region = self.region_delta(shared)
        delta_structure = self.structure_delta(shared)

        region_refined = region + delta_region
        structure_refined = structure + delta_structure

        cooperative = self.cooperative_feature(
            torch.cat(
                [region_refined, structure_refined],
                dim=1,
            )
        )

        if not enabled:
            region_refined = region + torch.zeros_like(delta_region)
            structure_refined = structure + torch.zeros_like(delta_structure)
            cooperative = torch.zeros_like(cooperative)

        return (
            region_refined,
            structure_refined,
            cooperative,
        )


class SACAStream(nn.Module):
    """Full Semantic-Anchored Cooperative Adapter and its clean ablations.

    Interface is intentionally identical to the validated SaliencyAdapterStream:

        forward(F1..F4) -> (task_F2, task_F3, coarse_saliency)

    Therefore the existing IRAMStrongNeck / ThreeStreamFusion / UNetFormer path
    can be reused without modification.
    """

    VALID_ABLATIONS = {
        "full",
        "no_semantic_region_guidance",
        "no_guided_structure",
        "no_region_structure_cooperation",
    }

    def __init__(
        self,
        channels: int = 128,
        ablation: SACAAblation = "full",
    ) -> None:
        super().__init__()

        if ablation not in self.VALID_ABLATIONS:
            raise ValueError(
                f"ablation must be one of {sorted(self.VALID_ABLATIONS)}, "
                f"got {ablation!r}"
            )

        self.channels = channels
        self.ablation = ablation

        self.project = nn.ModuleList(
            [
                ConvBNAct(
                    in_channels,
                    64,
                    kernel_size=1,
                )
                for in_channels in (96, 192, 384, 768)
            ]
        )

        self.semantic_anchor = SemanticAnchor(
            projected_channels=64,
            out_channels=channels,
        )

        self.region_reconstruction = RegionReconstruction(
            projected_channels=64,
            channels=channels,
        )

        self.structure_reconstruction = StructureReconstruction(
            projected_channels=64,
            channels=channels,
        )

        self.cooperation = RegionStructureCooperation(
            channels=channels,
        )

        self.task_reconstruction = nn.Sequential(
            ConvBNAct(
                channels * 4,
                channels,
                kernel_size=1,
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
            ),
        )

        # Explicitly supervise the semantic anchor rather than the final task
        # reconstruction. This keeps the anchor task-aware and gives region /
        # structure branches a stable saliency reference.
        self.coarse_head = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
        )

        self.saliency2 = ConvBNAct(
            channels,
            channels,
            kernel_size=3,
        )
        self.saliency3 = ConvBNAct(
            channels,
            channels,
            kernel_size=3,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if len(features) != 4:
            raise ValueError(
                f"SACAStream expects four CFMNet stages, got {len(features)}"
            )

        p1, p2, p3, p4 = [
            project(feature)
            for project, feature in zip(self.project, features)
        ]

        semantic = self.semantic_anchor(
            p2=p2,
            p3=p3,
            p4=p4,
        )

        region = self.region_reconstruction(
            p2=p2,
            semantic=semantic,
            use_semantic_guidance=(
                self.ablation != "no_semantic_region_guidance"
            ),
        )

        structure = self.structure_reconstruction(
            p1=p1,
            p2=p2,
            semantic=semantic,
            region=region,
            use_guidance=(
                self.ablation != "no_guided_structure"
            ),
        )

        (
            region_refined,
            structure_refined,
            cooperative,
        ) = self.cooperation(
            semantic=semantic,
            region=region,
            structure=structure,
            enabled=(
                self.ablation != "no_region_structure_cooperation"
            ),
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
                resize_like(task, p3)
            ),
            self.coarse_head(semantic),
        )
