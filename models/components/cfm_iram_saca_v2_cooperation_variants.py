"""SACA-v2 cooperation variants.

Keep the validated SACA-v2 front-end unchanged:

    Semantic Anchor
      -> coarse saliency
      -> Semantic Relevance
      -> Semantic-conditioned Region
      -> Semantic Boundary
      -> Guided Structure

Only the Region/Structure cooperation stage is changed.

Experiments:
    1. structure_to_region_only : T -> R ON, R -> T OFF
    2. region_to_structure_only : T -> R OFF, R -> T ON
    3. shared_v1                : SACA-v2 front-end + SACA-v1 shared cooperation
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from models.components.cfm_external_necks import ConvBNAct
from models.components.cfm_iram_saca import RegionStructureCooperation
from models.components.cfm_iram_saca_v2 import SACAv2Stream


DirectionalMode = Literal[
    "structure_to_region_only",
    "region_to_structure_only",
]


class DirectionalOneWayRegionStructureCooperation(nn.Module):
    """One-way ablations of the original SACA-v2 directional cooperation.

    Important:
        Both directional branches are still computed.
        The disabled direction is silenced only at its influence point.

    Therefore T->R-only and R->T-only keep the same directional module
    graph and parameter count as the original SACA-v2 directional block.
    """

    def __init__(
        self,
        channels: int = 128,
        mode: DirectionalMode = "structure_to_region_only",
    ) -> None:
        super().__init__()

        if mode not in {
            "structure_to_region_only",
            "region_to_structure_only",
        }:
            raise ValueError(
                "mode must be 'structure_to_region_only' or "
                f"'region_to_structure_only', got {mode!r}"
            )

        self.mode = mode

        # Exactly the same T -> R branch as the original SACA-v2.
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

        # Exactly the same region support used by the original R -> T path.
        self.region_support = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True,
        )

        # Exactly the same R -> T branch as the original SACA-v2.
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

        # Keep the original cooperative feature construction.
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

        # T -> R: unchanged from SACA-v2.
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

        # R -> T: unchanged from SACA-v2.
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

        # Silence only the selected influence path.
        if self.mode == "structure_to_region_only":
            region_refined_candidate = (
                region + delta_region
            )
            structure_refined_candidate = (
                structure
                + torch.zeros_like(delta_structure)
            )
        else:
            region_refined_candidate = (
                region
                + torch.zeros_like(delta_region)
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

        # Keep the same enabled contract as SACAv2Stream.
        if enabled:
            region_refined = region_refined_candidate
            structure_refined = structure_refined_candidate
            cooperative_output = cooperative
        else:
            region_refined = region
            structure_refined = structure
            cooperative_output = torch.zeros_like(
                cooperative
            )

        return (
            region_refined,
            structure_refined,
            cooperative_output,
        )


class SACAv2StructureToRegionOnlyStream(SACAv2Stream):
    """SACA-v2 front-end + T -> R only."""

    def __init__(
        self,
        channels: int = 128,
    ) -> None:
        super().__init__(
            channels=channels,
            ablation="full",
        )

        self.cooperation = (
            DirectionalOneWayRegionStructureCooperation(
                channels=channels,
                mode="structure_to_region_only",
            )
        )


class SACAv2RegionToStructureOnlyStream(SACAv2Stream):
    """SACA-v2 front-end + R -> T only."""

    def __init__(
        self,
        channels: int = 128,
    ) -> None:
        super().__init__(
            channels=channels,
            ablation="full",
        )

        self.cooperation = (
            DirectionalOneWayRegionStructureCooperation(
                channels=channels,
                mode="region_to_structure_only",
            )
        )


class SACAv2SharedV1CooperationStream(SACAv2Stream):
    """SACA-v2 front-end + validated SACA-v1 shared cooperation."""

    def __init__(
        self,
        channels: int = 128,
    ) -> None:
        super().__init__(
            channels=channels,
            ablation="full",
        )

        # Directly reuse the original SACA-v1 shared cooperation:
        # [semantic, region, structure] -> shared -> delta_R / delta_T.
        self.cooperation = RegionStructureCooperation(
            channels=channels,
        )
