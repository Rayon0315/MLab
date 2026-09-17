from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
)


class SemanticHardBackgroundVerifier(nn.Module):
    """
    CFPS-Match: local counter-evidence + semantic matching counter-evidence.

    Local branch:
        raw Stage2 / Stage3 before NAMLab
        -> local evidence

    Semantic branch:
        raw Stage4 before NAMLab
        + detached coarse saliency
        -> image-specific foreground/background semantic anchors
        -> cosine matching on raw Stage2 / Stage3
        -> semantic counter-evidence

    The verifier remains one-way:
        refined_logits = coarse_logits - suppression

    The coarse saliency map is detached when it is used to construct
    semantic anchors. It is only a guide for gathering semantic evidence;
    the semantic branch cannot train the coarse prediction to make its own
    matching problem easier.
    """

    def __init__(
        self,
        stage2_channels: int,
        stage3_channels: int,
        stage4_channels: int,
        evidence_channels: int = 64,
        semantic_channels: int = 64,
        semantic_evidence_channels: int = 32,
        hidden_channels: int = 128,
        confidence_power: float = 2.0,
        anchor_power: float = 2.0,
        max_suppression: float = 4.0,
        initial_suppression_scale: float = 0.25,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        self.confidence_power = confidence_power
        self.anchor_power = anchor_power
        self.max_suppression = max_suppression
        self.epsilon = epsilon

        # ------------------------------------------------------------
        # Local counter-evidence branch: keep the original CFPS source.
        # ------------------------------------------------------------
        self.local_stage2_projection = ConvNormAct(
            stage2_channels,
            evidence_channels,
            kernel_size=1,
            padding=0,
        )

        self.local_stage3_projection = ConvNormAct(
            stage3_channels,
            evidence_channels,
            kernel_size=1,
            padding=0,
        )

        self.local_evidence_fusion = nn.Sequential(
            ConvNormAct(
                evidence_channels * 2,
                hidden_channels,
                kernel_size=3,
            ),
            ResidualConvBlock(
                hidden_channels
            ),
        )

        # ------------------------------------------------------------
        # Semantic matching branch.
        # Stage4 constructs image-specific semantic anchors.
        # Stage2 / Stage3 are independently projected for matching.
        # ------------------------------------------------------------
        self.semantic_stage2_projection = ConvNormAct(
            stage2_channels,
            semantic_channels,
            kernel_size=1,
            padding=0,
        )

        self.semantic_stage3_projection = ConvNormAct(
            stage3_channels,
            semantic_channels,
            kernel_size=1,
            padding=0,
        )

        self.semantic_stage4_projection = ConvNormAct(
            stage4_channels,
            semantic_channels,
            kernel_size=1,
            padding=0,
        )

        # Each scale receives:
        #   foreground cosine similarity
        #   background cosine similarity
        #   semantic margin = bg - fg
        #   detached coarse foreground probability
        self.semantic_stage2_encoder = nn.Sequential(
            ConvNormAct(
                4,
                semantic_evidence_channels,
                kernel_size=3,
            ),
            ResidualConvBlock(
                semantic_evidence_channels
            ),
        )

        self.semantic_stage3_encoder = nn.Sequential(
            ConvNormAct(
                4,
                semantic_evidence_channels,
                kernel_size=3,
            ),
            ResidualConvBlock(
                semantic_evidence_channels
            ),
        )

        self.semantic_evidence_fusion = nn.Sequential(
            ConvNormAct(
                semantic_evidence_channels * 2,
                hidden_channels,
                kernel_size=3,
            ),
            ResidualConvBlock(
                hidden_channels
            ),
        )

        # Preserve the two evidence streams until the final verifier.
        # +1 channel is the detached coarse foreground probability.
        self.verifier = nn.Sequential(
            ConvNormAct(
                hidden_channels * 2 + 1,
                hidden_channels,
                kernel_size=3,
            ),
            ResidualConvBlock(
                hidden_channels
            ),
        )

        self.fp_risk_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
        )

        self.suppression_strength_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
        )

        # Same conservative initialization as the original CFPS.
        nn.init.zeros_(
            self.fp_risk_head.weight
        )
        nn.init.constant_(
            self.fp_risk_head.bias,
            -2.0,
        )

        nn.init.zeros_(
            self.suppression_strength_head.weight
        )
        nn.init.zeros_(
            self.suppression_strength_head.bias
        )

        ratio = (
            initial_suppression_scale
            / max_suppression
        )
        ratio = min(
            max(ratio, 1e-4),
            1.0 - 1e-4,
        )

        initial_logit = math.log(
            ratio / (1.0 - ratio)
        )

        self.suppression_scale_logit = nn.Parameter(
            torch.tensor(
                initial_logit,
                dtype=torch.float32,
            )
        )

    def _build_semantic_anchors(
        self,
        semantic_stage4: torch.Tensor,
        coarse_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        guide = F.interpolate(
            coarse_probability,
            size=semantic_stage4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        foreground_weight = guide.pow(
            self.anchor_power
        )
        background_weight = (
            1.0 - guide
        ).pow(
            self.anchor_power
        )

        foreground_anchor = (
            semantic_stage4
            * foreground_weight
        ).sum(
            dim=(2, 3),
            keepdim=True,
        ) / foreground_weight.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(
            self.epsilon
        )

        background_anchor = (
            semantic_stage4
            * background_weight
        ).sum(
            dim=(2, 3),
            keepdim=True,
        ) / background_weight.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(
            self.epsilon
        )

        foreground_anchor = F.normalize(
            foreground_anchor,
            dim=1,
            eps=self.epsilon,
        )

        background_anchor = F.normalize(
            background_anchor,
            dim=1,
            eps=self.epsilon,
        )

        return (
            foreground_anchor,
            background_anchor,
        )

    def _semantic_match(
        self,
        feature: torch.Tensor,
        foreground_anchor: torch.Tensor,
        background_anchor: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        feature = F.normalize(
            feature,
            dim=1,
            eps=self.epsilon,
        )

        foreground_similarity = (
            feature * foreground_anchor
        ).sum(
            dim=1,
            keepdim=True,
        )

        background_similarity = (
            feature * background_anchor
        ).sum(
            dim=1,
            keepdim=True,
        )

        # Positive margin means stronger background semantic evidence.
        semantic_margin = (
            background_similarity
            - foreground_similarity
        )

        return (
            foreground_similarity,
            background_similarity,
            semantic_margin,
        )

    def forward(
        self,
        raw_stage2: torch.Tensor,
        raw_stage3: torch.Tensor,
        raw_stage4: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_size = coarse_logits.shape[-2:]

        # The main-path prediction is only evidence for the verifier.
        coarse_probability = torch.sigmoid(
            coarse_logits.detach()
        )

        # ------------------------------------------------------------
        # 1. Original local counter-evidence.
        # ------------------------------------------------------------
        local2 = self.local_stage2_projection(
            raw_stage2
        )

        local3 = self.local_stage3_projection(
            raw_stage3
        )

        local3 = F.interpolate(
            local3,
            size=local2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        local_evidence = self.local_evidence_fusion(
            torch.cat(
                [
                    local2,
                    local3,
                ],
                dim=1,
            )
        )

        # ------------------------------------------------------------
        # 2. High-level semantic anchors from raw Stage4.
        # ------------------------------------------------------------
        semantic2 = self.semantic_stage2_projection(
            raw_stage2
        )

        semantic3 = self.semantic_stage3_projection(
            raw_stage3
        )

        semantic4 = self.semantic_stage4_projection(
            raw_stage4
        )

        (
            foreground_anchor,
            background_anchor,
        ) = self._build_semantic_anchors(
            semantic_stage4=semantic4,
            coarse_probability=coarse_probability,
        )

        (
            foreground_similarity2,
            background_similarity2,
            semantic_margin2,
        ) = self._semantic_match(
            feature=semantic2,
            foreground_anchor=foreground_anchor,
            background_anchor=background_anchor,
        )

        (
            foreground_similarity3,
            background_similarity3,
            semantic_margin3,
        ) = self._semantic_match(
            feature=semantic3,
            foreground_anchor=foreground_anchor,
            background_anchor=background_anchor,
        )

        coarse2 = F.interpolate(
            coarse_probability,
            size=semantic2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        coarse3 = F.interpolate(
            coarse_probability,
            size=semantic3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        semantic_evidence2 = self.semantic_stage2_encoder(
            torch.cat(
                [
                    foreground_similarity2,
                    background_similarity2,
                    semantic_margin2,
                    coarse2,
                ],
                dim=1,
            )
        )

        semantic_evidence3 = self.semantic_stage3_encoder(
            torch.cat(
                [
                    foreground_similarity3,
                    background_similarity3,
                    semantic_margin3,
                    coarse3,
                ],
                dim=1,
            )
        )

        semantic_evidence3 = F.interpolate(
            semantic_evidence3,
            size=semantic_evidence2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        semantic_evidence = self.semantic_evidence_fusion(
            torch.cat(
                [
                    semantic_evidence2,
                    semantic_evidence3,
                ],
                dim=1,
            )
        )

        # ------------------------------------------------------------
        # 3. Evidence fusion and one-way false-positive suppression.
        # ------------------------------------------------------------
        local_evidence = F.interpolate(
            local_evidence,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        semantic_evidence = F.interpolate(
            semantic_evidence,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        verifier_feature = self.verifier(
            torch.cat(
                [
                    local_evidence,
                    semantic_evidence,
                    coarse_probability,
                ],
                dim=1,
            )
        )

        fp_risk_logits = self.fp_risk_head(
            verifier_feature
        )

        fp_risk = torch.sigmoid(
            fp_risk_logits
        )

        suppression_strength = torch.sigmoid(
            self.suppression_strength_head(
                verifier_feature
            )
        )

        suppression_scale = (
            self.max_suppression
            * torch.sigmoid(
                self.suppression_scale_logit
            )
        )

        confidence_gate = coarse_probability.pow(
            self.confidence_power
        )

        suppression = (
            suppression_scale
            * confidence_gate
            * fp_risk
            * suppression_strength
        )

        refined_logits = (
            coarse_logits
            - suppression
        )

        # Diagnostics: average Stage2/Stage3 matching evidence.
        foreground_similarity3_up = F.interpolate(
            foreground_similarity3,
            size=foreground_similarity2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        background_similarity3_up = F.interpolate(
            background_similarity3,
            size=background_similarity2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        semantic_margin3_up = F.interpolate(
            semantic_margin3,
            size=semantic_margin2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        foreground_similarity = 0.5 * (
            foreground_similarity2
            + foreground_similarity3_up
        )
        background_similarity = 0.5 * (
            background_similarity2
            + background_similarity3_up
        )
        semantic_margin = 0.5 * (
            semantic_margin2
            + semantic_margin3_up
        )

        foreground_similarity = F.interpolate(
            foreground_similarity,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        background_similarity = F.interpolate(
            background_similarity,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        semantic_margin = F.interpolate(
            semantic_margin,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "refined_logits": refined_logits,
            "fp_risk_logits": fp_risk_logits,
            "suppression": suppression,
            "suppression_scale": suppression_scale,
            "fg_similarity": foreground_similarity,
            "bg_similarity": background_similarity,
            "semantic_margin": semantic_margin,
        }
