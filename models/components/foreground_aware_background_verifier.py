from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
)


class ForegroundAwareBackgroundVerifier(nn.Module):
    """
    Asymmetric background counter-evidence verifier.

    Foreground and background do NOT share a complementary probability space.

    The main branch already produced:
        - coarse foreground probability p
        - foreground prototype support g_fg

    The verifier asks a different question:
        "Given that the main model predicts foreground, is there independent
         visual counter-evidence that this location is actually a false
         positive?"

    Inputs:
        detached raw Stage2 / Stage3 evidence
        detached coarse foreground probability
        detached foreground support
        mismatch = p * (1 - g_fg)

    mismatch is evidence, not a hard gate. Therefore the verifier can still
    learn high risk when both foreground support and background counter-
    evidence are high (an explicit evidence-conflict state).

    FP auxiliary gradients are isolated from the shared backbone, decoder and
    foreground metric space by detaching all main-network inputs here.
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

        # +3:
        #   coarse foreground probability
        #   foreground support
        #   prediction-support mismatch
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
        fg_support: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_size = coarse_logits.shape[-2:]

        # Strict gradient isolation for counter-evidence supervision.
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

        visual_counter_evidence = (
            self.evidence_fusion(
                torch.cat(
                    [
                        evidence2,
                        evidence3,
                    ],
                    dim=1,
                )
            )
        )

        visual_counter_evidence = (
            self._resize(
                visual_counter_evidence,
                target_size,
            )
        )

        coarse_probability = torch.sigmoid(
            coarse_logits.detach()
        )

        fg_support = self._resize(
            fg_support.detach(),
            target_size,
        ).clamp(
            0.0,
            1.0,
        )

        prediction_support_mismatch = (
            coarse_probability
            * (1.0 - fg_support)
        )

        verifier_feature = self.verifier(
            torch.cat(
                [
                    visual_counter_evidence,
                    coarse_probability,
                    fg_support,
                    prediction_support_mismatch,
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

        # BG counter-evidence has veto power only.
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
            "suppression_strength": suppression_strength,
            "suppression_strength_logits": (
                suppression_strength_logits
            ),
            "suppression": suppression,
            "suppression_scale": suppression_scale,
            "coarse_probability": coarse_probability,
            "fg_support": fg_support,
            "prediction_support_mismatch": (
                prediction_support_mismatch
            ),
        }
