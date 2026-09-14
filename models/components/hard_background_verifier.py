from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
)


class HardBackgroundVerifier(nn.Module):
    """
    Counter-evidence verifier for high-confidence false positives.

    Evidence source:
        raw Stage2 / Stage3 features BEFORE NAMLab.

    Main-path signal:
        detached coarse foreground probability only.

    The verifier is intentionally one-way:
        it can only subtract from the coarse saliency logit.

    This keeps its role explicit:
        detect hard background that the main saliency branch
        has already mistaken for foreground.
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

        self.confidence_power = confidence_power
        self.max_suppression = max_suppression

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

        # +1 channel for the detached coarse foreground probability.
        self.verifier = nn.Sequential(
            ConvNormAct(
                hidden_channels + 1,
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

        # Start conservatively:
        #   q_fp ~= sigmoid(-2) = 0.119
        # so the new branch initially perturbs the baseline only mildly.
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

    def forward(
        self,
        raw_stage2: torch.Tensor,
        raw_stage3: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_size = coarse_logits.shape[-2:]

        evidence2 = self.stage2_projection(
            raw_stage2
        )

        evidence3 = self.stage3_projection(
            raw_stage3
        )

        evidence3 = F.interpolate(
            evidence3,
            size=evidence2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        evidence = self.evidence_fusion(
            torch.cat(
                [
                    evidence2,
                    evidence3,
                ],
                dim=1,
            )
        )

        evidence = F.interpolate(
            evidence,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        coarse_probability = torch.sigmoid(
            coarse_logits.detach()
        )

        verifier_feature = self.verifier(
            torch.cat(
                [
                    evidence,
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

        return {
            "refined_logits": refined_logits,
            "fp_risk_logits": fp_risk_logits,
            "suppression": suppression,
            "suppression_scale": suppression_scale,
        }
