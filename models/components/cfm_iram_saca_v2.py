"""SACA-v2: closed-loop semantic-anchored saliency task adaptation.

This module upgrades the validated SACA task stream while preserving the outer
model contract:

    CFMNet fused pyramid
        ├─ IRAM frequency stream
        └─ SACA-v2 task stream
    [base, frequency, task] -> existing ThreeStreamFusion -> UNetFormer

SACA-v2 turns the coarse saliency prediction into an internal reasoning signal:

    multi-scale semantic anchor A0
        -> coarse saliency M
        -> foreground/background semantic relevance
        -> refined semantic anchor A
        -> semantic-conditioned region R
        -> saliency-boundary-guided structure T
        -> directional T->R and R->T cooperation
        -> task representation

The ablation switches preserve module widths and parameter count. Disabled
mechanisms are still computed and are silenced only at their influence point.
"""
from __future__ import annotations

from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import ConvBNAct
from models.components.cfm_iram_strong_common import resize_like


SACAv2Ablation = Literal[
    "full",
    "no_semantic_relevance",
    "no_semantic_boundary",
    "no_directional_cooperation",
]


class SemanticAnchor(nn.Module):
    """Build the validated sample-adaptive multi-scale semantic anchor at F2."""

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


class SemanticRelevanceRefinement(nn.Module):
    """Refine semantic features with supervised soft FG/BG relevance.

    The coarse saliency probability softly pools foreground/background
    prototypes from the semantic anchor. Cosine relation maps then reconstruct
    a new semantic feature; they are not used as a scalar gate over IRAM or
    backbone features.
    """

    def __init__(
        self,
        channels: int = 128,
    ) -> None:
        super().__init__()

        self.relation_projection = ConvBNAct(
            1,
            channels,
            kernel_size=3,
        )

        self.reconstruct = nn.Sequential(
            ConvBNAct(
                channels * 4,
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

    @staticmethod
    def _weighted_prototype(
        feature: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        mass = weight.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(1e-6)

        return (
            feature * weight
        ).sum(
            dim=(2, 3),
            keepdim=True,
        ) / mass

    def forward(
        self,
        semantic_seed: torch.Tensor,
        coarse_logits: torch.Tensor,
        enabled: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        probability = torch.sigmoid(
            coarse_logits
        )

        foreground_prototype = self._weighted_prototype(
            semantic_seed,
            probability,
        )
        background_prototype = self._weighted_prototype(
            semantic_seed,
            1.0 - probability,
        )

        normalized_feature = F.normalize(
            semantic_seed,
            p=2,
            dim=1,
            eps=1e-6,
        )
        normalized_foreground = F.normalize(
            foreground_prototype,
            p=2,
            dim=1,
            eps=1e-6,
        )
        normalized_background = F.normalize(
            background_prototype,
            p=2,
            dim=1,
            eps=1e-6,
        )

        similarity_foreground = (
            normalized_feature
            * normalized_foreground
        ).sum(
            dim=1,
            keepdim=True,
        )
        similarity_background = (
            normalized_feature
            * normalized_background
        ).sum(
            dim=1,
            keepdim=True,
        )

        relevance = (
            similarity_foreground
            - similarity_background
        )

        relation_feature = self.relation_projection(
            relevance
        )

        semantic_delta = self.reconstruct(
            torch.cat(
                [
                    semantic_seed,
                    semantic_seed * similarity_foreground,
                    semantic_seed * similarity_background,
                    relation_feature,
                ],
                dim=1,
            )
        )

        if enabled:
            semantic = semantic_seed + semantic_delta
            relevance_output = relevance
        else:
            semantic = semantic_seed + torch.zeros_like(
                semantic_delta
            )
            relevance_output = torch.zeros_like(
                relevance
            )

        return (
            semantic,
            probability,
            relevance_output,
        )


class SemanticRegionReconstruction(nn.Module):
    """Build salient-region evidence from F2 plus semantic task signals."""

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

        self.mask_projection = ConvBNAct(
            1,
            channels,
            kernel_size=3,
        )
        self.relevance_projection = ConvBNAct(
            1,
            channels,
            kernel_size=3,
        )

        self.reduce = ConvBNAct(
            channels * 6,
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
        probability: torch.Tensor,
        relevance: torch.Tensor,
    ) -> torch.Tensor:
        seed = self.region_seed(p2)

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

        mask_feature = self.mask_projection(
            probability
        )
        relevance_feature = self.relevance_projection(
            relevance
        )

        region = self.reduce(
            torch.cat(
                [
                    seed,
                    semantic,
                    semantic_local,
                    semantic_broad,
                    mask_feature,
                    relevance_feature,
                ],
                dim=1,
            )
        )

        return region + self.context(region)


class SemanticBoundaryPrior(nn.Module):
    """Convert the coarse saliency probability into a soft task boundary prior."""

    def __init__(
        self,
        channels: int = 128,
    ) -> None:
        super().__init__()

        self.encode = nn.Sequential(
            ConvBNAct(
                1,
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

    def forward(
        self,
        coarse_logits: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        probability = torch.sigmoid(
            coarse_logits
        )

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

        boundary = (
            local_max - local_min
        ).clamp_min(0.0)

        return boundary, self.encode(boundary)


class SalientStructureReconstruction(nn.Module):
    """Reconstruct salient structure from high-frequency and task evidence."""

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
            channels * 4,
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
        boundary_feature: torch.Tensor,
        use_boundary: bool,
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

        if not use_boundary:
            boundary_feature = torch.zeros_like(
                boundary_feature
            )

        structure = self.reduce(
            torch.cat(
                [
                    raw_structure,
                    semantic,
                    region,
                    boundary_feature,
                ],
                dim=1,
            )
        )

        return structure + self.refine(structure)


class DirectionalRegionStructureCooperation(nn.Module):
    """Role-specific bidirectional refinement between region and structure.

    Structure -> Region uses pair product/difference to constrain region shape.
    Region -> Structure uses a region-derived channel-spatial support tensor to
    suppress structure responses that are inconsistent with salient regions.
    """

    def __init__(
        self,
        channels: int = 128,
    ) -> None:
        super().__init__()

        self.structure_to_region = nn.Sequential(
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

        self.region_support = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        self.region_to_structure = nn.Sequential(
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

        self.cooperative_feature = nn.Sequential(
            ConvBNAct(
                channels * 4,
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
        pair_product = region * structure
        pair_difference = torch.abs(
            region - structure
        )

        delta_region = self.structure_to_region(
            torch.cat(
                [
                    region,
                    structure,
                    semantic,
                    pair_product,
                    pair_difference,
                ],
                dim=1,
            )
        )

        region_support = torch.sigmoid(
            self.region_support(region)
        )
        supported_structure = (
            structure * region_support
        )

        delta_structure = self.region_to_structure(
            torch.cat(
                [
                    structure,
                    region,
                    semantic,
                    supported_structure,
                    pair_difference,
                ],
                dim=1,
            )
        )

        region_refined_candidate = (
            region + delta_region
        )
        structure_refined_candidate = (
            structure + delta_structure
        )

        cooperative = self.cooperative_feature(
            torch.cat(
                [
                    region_refined_candidate,
                    structure_refined_candidate,
                    region_refined_candidate
                    * structure_refined_candidate,
                    torch.abs(
                        region_refined_candidate
                        - structure_refined_candidate
                    ),
                ],
                dim=1,
            )
        )

        if enabled:
            region_refined = region_refined_candidate
            structure_refined = structure_refined_candidate
            cooperative_output = cooperative
        else:
            region_refined = region + torch.zeros_like(
                delta_region
            )
            structure_refined = structure + torch.zeros_like(
                delta_structure
            )
            cooperative_output = torch.zeros_like(
                cooperative
            )

        return (
            region_refined,
            structure_refined,
            cooperative_output,
        )


class SACAv2Stream(nn.Module):
    """Closed-loop SACA-v2 task stream and clean mechanism ablations.

    Interface remains identical to the validated SaliencyAdapterStream/SACA:

        forward(F1..F4) -> (task_F2, task_F3, coarse_saliency_logits)
    """

    VALID_ABLATIONS = {
        "full",
        "no_semantic_relevance",
        "no_semantic_boundary",
        "no_directional_cooperation",
    }

    def __init__(
        self,
        channels: int = 128,
        ablation: SACAv2Ablation = "full",
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
                for in_channels in (
                    96,
                    192,
                    384,
                    768,
                )
            ]
        )

        self.semantic_anchor = SemanticAnchor(
            projected_channels=64,
            out_channels=channels,
        )

        # The initial coarse head is explicitly supervised by the existing aux
        # loss and also drives the internal relevance/boundary reasoning loop.
        self.coarse_head = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
        )

        self.semantic_relevance = SemanticRelevanceRefinement(
            channels=channels,
        )

        self.region_reconstruction = SemanticRegionReconstruction(
            projected_channels=64,
            channels=channels,
        )

        self.semantic_boundary = SemanticBoundaryPrior(
            channels=channels,
        )

        self.structure_reconstruction = SalientStructureReconstruction(
            projected_channels=64,
            channels=channels,
        )

        self.cooperation = DirectionalRegionStructureCooperation(
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
                f"SACAv2Stream expects four CFMNet stages, got {len(features)}"
            )

        p1, p2, p3, p4 = [
            project(feature)
            for project, feature in zip(
                self.project,
                features,
            )
        ]

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
            enabled=(
                self.ablation
                != "no_semantic_relevance"
            ),
        )

        region = self.region_reconstruction(
            p2=p2,
            semantic=semantic,
            probability=probability,
            relevance=relevance,
        )

        _, boundary_feature = self.semantic_boundary(
            coarse_logits
        )

        structure = self.structure_reconstruction(
            p1=p1,
            p2=p2,
            semantic=semantic,
            region=region,
            boundary_feature=boundary_feature,
            use_boundary=(
                self.ablation
                != "no_semantic_boundary"
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
                self.ablation
                != "no_directional_cooperation"
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
            coarse_logits,
        )
