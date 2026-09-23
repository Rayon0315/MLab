"""NAMLab saliency-agnostic structure prior for SACA-v1 / SACA-v2.

Design
------
NAMLab is used only as a structure prior.

    M60 -> region-transition boundary cue
    |RGB - M60| -> intra-region residual cue

    [boundary cue, residual cue]
        -> NAMStructurePriorBuilder
        -> P_NAM

P_NAM is independent from all saliency predictions/features. It is introduced
only inside Structure Reconstruction.

SACA-v1:
    raw_structure + semantic + region + P_NAM -> salient structure

SACA-v2:
    raw_structure + semantic + region + semantic_boundary + P_NAM
        -> salient structure
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from models.components.cfm_external_necks import ConvBNAct
from models.components.cfm_iram_saca import (
    SACAStream,
    StructureReconstruction,
)
from models.components.cfm_iram_saca_v2 import (
    SACAv2Stream,
    SalientStructureReconstruction,
)
from models.components.cfm_iram_strong_common import (
    IRAMStrongNeck,
    resize_like,
)


class NAMStructurePriorBuilder(nn.Module):
    """Build a saliency-agnostic NAMLab structure prior at the F2 grid."""

    def __init__(
        self,
        out_channels: int = 128,
        branch_channels: int = 64,
    ) -> None:
        super().__init__()

        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

        self.boundary_encoder = nn.Sequential(
            ConvBNAct(
                1,
                branch_channels,
                kernel_size=3,
            ),
            ConvBNAct(
                branch_channels,
                branch_channels,
                kernel_size=3,
                groups=branch_channels,
            ),
        )

        self.residual_encoder = nn.Sequential(
            ConvBNAct(
                3,
                branch_channels,
                kernel_size=3,
            ),
            ConvBNAct(
                branch_channels,
                branch_channels,
                kernel_size=3,
                groups=branch_channels,
            ),
        )

        self.fuse = nn.Sequential(
            ConvBNAct(
                branch_channels * 2,
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

    def _denormalize_image(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        return (
            image * self.image_std
            + self.image_mean
        ).clamp(0.0, 1.0)

    @staticmethod
    def _m60_boundary(
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
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

        return (
            local_max - local_min
        ).clamp_min(0.0).mean(
            dim=1,
            keepdim=True,
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        image_rgb = self._denormalize_image(
            image
        )

        boundary = self._m60_boundary(
            mean_60
        )

        residual = torch.abs(
            image_rgb - mean_60
        )

        boundary_f2 = F.adaptive_max_pool2d(
            boundary,
            output_size=target_size,
        )
        residual_f2 = F.adaptive_avg_pool2d(
            residual,
            output_size=target_size,
        )

        boundary_feature = self.boundary_encoder(
            boundary_f2
        )
        residual_feature = self.residual_encoder(
            residual_f2
        )

        return self.fuse(
            torch.cat(
                [
                    boundary_feature,
                    residual_feature,
                ],
                dim=1,
            )
        )


class NAMGuidedStructureReconstructionV1(
    StructureReconstruction
):
    """Original SACA-v1 Structure Reconstruction + independent NAM prior."""

    def __init__(
        self,
        projected_channels: int = 64,
        channels: int = 128,
    ) -> None:
        super().__init__(
            projected_channels=projected_channels,
            channels=channels,
        )

        self.reduce = ConvBNAct(
            channels * 4,
            channels,
            kernel_size=1,
        )

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        semantic: torch.Tensor,
        region: torch.Tensor,
        nam_prior: torch.Tensor,
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
            semantic_input = torch.zeros_like(
                semantic_input
            )
            region_input = torch.zeros_like(
                region_input
            )

        structure = self.reduce(
            torch.cat(
                [
                    raw_structure,
                    semantic_input,
                    region_input,
                    nam_prior,
                ],
                dim=1,
            )
        )

        return structure + self.refine(
            structure
        )


class NAMGuidedStructureReconstructionV2(
    SalientStructureReconstruction
):
    """Original SACA-v2 Structure Reconstruction + independent NAM prior."""

    def __init__(
        self,
        projected_channels: int = 64,
        channels: int = 128,
    ) -> None:
        super().__init__(
            projected_channels=projected_channels,
            channels=channels,
        )

        self.reduce = ConvBNAct(
            channels * 5,
            channels,
            kernel_size=1,
        )

    def forward(
        self,
        p1: torch.Tensor,
        p2: torch.Tensor,
        semantic: torch.Tensor,
        region: torch.Tensor,
        boundary_feature: torch.Tensor,
        nam_prior: torch.Tensor,
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

        boundary_input = boundary_feature
        if not use_boundary:
            boundary_input = torch.zeros_like(
                boundary_input
            )

        structure = self.reduce(
            torch.cat(
                [
                    raw_structure,
                    semantic,
                    region,
                    boundary_input,
                    nam_prior,
                ],
                dim=1,
            )
        )

        return structure + self.refine(
            structure
        )


class SACANAMStructurePriorStream(
    SACAStream
):
    """Full SACA-v1 with NAMLab used only for Structure Reconstruction."""

    def __init__(
        self,
        channels: int = 128,
    ) -> None:
        super().__init__(
            channels=channels,
            ablation="full",
        )

        self.nam_structure_prior = (
            NAMStructurePriorBuilder(
                out_channels=channels,
            )
        )
        self.structure_reconstruction = (
            NAMGuidedStructureReconstructionV1(
                projected_channels=64,
                channels=channels,
            )
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if len(features) != 4:
            raise ValueError(
                "SACANAMStructurePriorStream expects "
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

        region = self.region_reconstruction(
            p2=p2,
            semantic=semantic,
            use_semantic_guidance=True,
        )

        nam_prior = self.nam_structure_prior(
            image=image,
            mean_60=mean_60,
            target_size=p2.shape[-2:],
        )

        structure = self.structure_reconstruction(
            p1=p1,
            p2=p2,
            semantic=semantic,
            region=region,
            nam_prior=nam_prior,
            use_guidance=True,
        )

        (
            region_refined,
            structure_refined,
            cooperative,
        ) = self.cooperation(
            semantic=semantic,
            region=region,
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
                resize_like(task, p3)
            ),
            self.coarse_head(
                semantic
            ),
        )


class SACAv2NAMStructurePriorStream(
    SACAv2Stream
):
    """Full SACA-v2 with NAMLab used only for Structure Reconstruction."""

    def __init__(
        self,
        channels: int = 128,
    ) -> None:
        super().__init__(
            channels=channels,
            ablation="full",
        )

        self.nam_structure_prior = (
            NAMStructurePriorBuilder(
                out_channels=channels,
            )
        )
        self.structure_reconstruction = (
            NAMGuidedStructureReconstructionV2(
                projected_channels=64,
                channels=channels,
            )
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if len(features) != 4:
            raise ValueError(
                "SACAv2NAMStructurePriorStream expects "
                f"four CFMNet stages, got {len(features)}"
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
            enabled=True,
        )

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

        nam_prior = self.nam_structure_prior(
            image=image,
            mean_60=mean_60,
            target_size=p2.shape[-2:],
        )

        structure = self.structure_reconstruction(
            p1=p1,
            p2=p2,
            semantic=semantic,
            region=region,
            boundary_feature=boundary_feature,
            nam_prior=nam_prior,
            use_boundary=True,
        )

        (
            region_refined,
            structure_refined,
            cooperative,
        ) = self.cooperation(
            semantic=semantic,
            region=region,
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
                resize_like(task, p3)
            ),
            coarse_logits,
        )


class IRAMNAMStructurePriorNeck(
    IRAMStrongNeck
):
    """Original IRAM strong neck; only task-stream call gains image/M60."""

    def forward(
        self,
        features: Sequence[torch.Tensor],
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        torch.Tensor,
    ]:
        f1, f2, f3, f4 = features

        task2, task3, coarse = (
            self.x_stream(
                features,
                image=image,
                mean_60=mean_60,
            )
        )

        out2 = self.fuse2(
            f2,
            self.freq2(f2),
            task2,
        )
        out3 = self.fuse3(
            f3,
            self.freq3(f3),
            task3,
        )

        return (
            [f1, out2, out3, f4],
            coarse,
        )
