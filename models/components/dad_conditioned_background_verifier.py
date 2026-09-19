from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
)


class DADConditionedBackgroundVerifier(nn.Module):
    """
    Evidence-conditioned false-positive verifier (ECFV).

    Role:
        DAD-v2 remains the main FG/BG relational evidence generator.
        This verifier does not learn an independent competing decision
        space. Instead, it reads the current DAD state and checks whether
        a high-confidence foreground prediction has counter-evidence.

    Inputs:
        raw_stage2/raw_stage3:
            visual counter-evidence from the pre-NAM backbone.
            They are detached here so FP supervision cannot reshape the
            shared backbone / DAD feature geometry.

        coarse_logits:
            detached coarse foreground confidence.

        dad_semantic_margin:
            q_fg - q_bg from DAD-v2. Detached.

        dad_guide_logits:
            prediction3 used to build DAD prototypes. Detached.

    DAD conditioning:
        ambiguity:
            U = 1 - |margin|

        agreement:
            A = margin * (2 * sigmoid(guide) - 1)

        U is invariant to swapping the two prototype branches.
        A describes whether prototype matching and the DAD guide point
        in the same direction.

    The verifier input is:
        [visual counter-evidence, coarse p, U, A]

    Suppression is intentionally kept identical to the original CFPS:
        scale * p^gamma * fp_risk * suppression_strength

    Gradient isolation:
        verifier-side tensors from the main network are detached before
        entering the ECFV path. Therefore FP-risk supervision trains only
        this verifier branch, while the main DAD path keeps its own feature
        geometry.
    """

    def __init__(
        self,
        stage2_channels: int,
        stage3_channels: int,
        evidence_channels: int = 64,
        hidden_channels: int = 128,
        confidence_power: float = 2.0,
        max_suppression: float = 4.0,
        initial_suppression_scale: float = 0.25,
    ) -> None:
        super().__init__()

        self.confidence_power = float(
            confidence_power
        )
        self.max_suppression = float(
            max_suppression
        )

        self.stage2_projection = ConvNormAct(
            stage2_channels,
            evidence_channels,
            kernel_size=1,
            padding=0,
        )

        self.stage3_projection = ConvNormAct(
            stage3_channels,
            evidence_channels,
            kernel_size=1,
            padding=0,
        )

        self.evidence_fusion = nn.Sequential(
            ConvNormAct(
                evidence_channels * 2,
                hidden_channels,
                kernel_size=3,
            ),
            ResidualConvBlock(
                hidden_channels
            ),
        )

        # +3 conditioning channels:
        #   coarse foreground probability
        #   DAD ambiguity
        #   DAD-guide agreement
        self.verifier = nn.Sequential(
            ConvNormAct(
                hidden_channels + 3,
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

        # Keep original CFPS initialization.
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

    @staticmethod
    def _resize(
        tensor: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        if tensor.shape[-2:] == size:
            return tensor

        return F.interpolate(
            tensor,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        raw_stage2: torch.Tensor,
        raw_stage3: torch.Tensor,
        coarse_logits: torch.Tensor,
        dad_semantic_margin: torch.Tensor,
        dad_guide_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_size = coarse_logits.shape[-2:]

        # Critical change versus the original CFPS:
        # FP supervision cannot alter shared backbone / DAD geometry.
        raw_stage2 = raw_stage2.detach()
        raw_stage3 = raw_stage3.detach()

        evidence2 = self.stage2_projection(
            raw_stage2
        )
        evidence3 = self.stage3_projection(
            raw_stage3
        )

        evidence3 = self._resize(
            evidence3,
            evidence2.shape[-2:],
        )

        visual_evidence = self.evidence_fusion(
            torch.cat(
                [
                    evidence2,
                    evidence3,
                ],
                dim=1,
            )
        )

        visual_evidence = self._resize(
            visual_evidence,
            target_size,
        )

        coarse_probability = torch.sigmoid(
            coarse_logits.detach()
        )

        margin = self._resize(
            dad_semantic_margin.detach(),
            target_size,
        ).clamp(
            -1.0,
            1.0,
        )

        guide_probability = torch.sigmoid(
            self._resize(
                dad_guide_logits.detach(),
                target_size,
            )
        )

        guide_direction = (
            2.0 * guide_probability
            - 1.0
        )

        dad_ambiguity = (
            1.0 - margin.abs()
        ).clamp(
            0.0,
            1.0,
        )

        dad_agreement = (
            margin
            * guide_direction
        ).clamp(
            -1.0,
            1.0,
        )

        verifier_feature = self.verifier(
            torch.cat(
                [
                    visual_evidence,
                    coarse_probability,
                    dad_ambiguity,
                    dad_agreement,
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

        suppression_strength_logits = (
            self.suppression_strength_head(
                verifier_feature
            )
        )
        suppression_strength = torch.sigmoid(
            suppression_strength_logits
        )

        suppression_scale = (
            self.max_suppression
            * torch.sigmoid(
                self.suppression_scale_logit
            )
        )

        confidence_gate = (
            coarse_probability.pow(
                self.confidence_power
            )
        )

        # Keep the original CFPS suppression formula unchanged
        # for the first clean ECFV experiment.
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

        return {
            "refined_logits": refined_logits,
            "fp_risk_logits": fp_risk_logits,
            "suppression": suppression,
            "suppression_scale": suppression_scale,
            "suppression_strength": suppression_strength,
            "dad_ambiguity": dad_ambiguity,
            "dad_agreement": dad_agreement,
            "guide_probability": guide_probability,
        }
