# models/networks/vmamba_small_progressive_region_direct_hier60_region_hybrid_dictionary_disagreement_agreement_refinement_sod.py

from __future__ import annotations

from pathlib import Path

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
    ResidualConvBlock,
)
from models.networks.vmamba_small_progressive_region_direct_hier60_region_hybrid_dictionary_routing_sod import (
    PRETRAINED_PATH,
    VMambaSmallProgressiveRegionDirectHier60RegionHybridDictionaryRoutingSOD,
)


class AgreementAwareFGBGRefinement(nn.Module):
    """
    Lightweight residual refinement for a strong coarse saliency prediction.

    Belief:
        coarse_logit / decoded1

    Evidence:
        Stage1 detail-semantic feature
        Stage2 semantic feature

    The evidence branch builds image-level foreground/background prototypes
    from the current prediction, then checks whether each pixel's visual
    evidence agrees with the prediction.

    Two mechanisms are used:

    1. Disagreement correction
        uncertain pixels or prediction/evidence conflicts receive
        residual correction.

    2. Agreement amplification
        only reliable pixels where prediction and visual evidence agree
        receive a small signed logit-margin boost.

    The agreement branch is detached from the evidence path so it cannot
    manufacture agreement by changing the evidence representation.
    """

    def __init__(
        self,
        channels: int = 128,
        signal_channels: int = 32,
        prototype_gamma: float = 4.0,
        evidence_temperature: float = 0.2,
        uncertainty_weight: float = 0.5,
        disagreement_weight: float = 1.0,
        initial_output_scale: float = 0.1,
        initial_agreement_scale: float = 0.05,
        max_agreement_scale: float = 0.5,
    ) -> None:
        super().__init__()

        if prototype_gamma <= 0.0:
            raise ValueError(
                "prototype_gamma must be positive."
            )

        if evidence_temperature <= 0.0:
            raise ValueError(
                "evidence_temperature must be positive."
            )

        if max_agreement_scale <= 0.0:
            raise ValueError(
                "max_agreement_scale must be positive."
            )

        if not (
            0.0
            < initial_agreement_scale
            < max_agreement_scale
        ):
            raise ValueError(
                "initial_agreement_scale must be in "
                "(0, max_agreement_scale)."
            )

        self.prototype_gamma = prototype_gamma
        self.evidence_temperature = evidence_temperature
        self.uncertainty_weight = uncertainty_weight
        self.disagreement_weight = disagreement_weight

        # Visual evidence is deliberately separated from decoded1.
        self.evidence_encoder = nn.Sequential(
            ConvNormAct(
                channels * 2,
                channels,
                kernel_size=1,
                padding=0,
            ),
            ResidualConvBlock(
                channels
            ),
        )

        # p, 1-p, uncertainty, sim_fg, sim_bg,
        # signed_evidence, disagreement
        self.signal_encoder = nn.Sequential(
            ConvNormAct(
                7,
                signal_channels,
                kernel_size=3,
            ),
            ResidualConvBlock(
                signal_channels
            ),
        )

        self.correction_fusion = nn.Sequential(
            ConvNormAct(
                channels
                + channels
                + signal_channels,
                channels,
                kernel_size=1,
                padding=0,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(
                8,
                channels,
            ),
            nn.GELU(),
            ResidualConvBlock(
                channels
            ),
        )

        self.delta_head = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
            bias=True,
        )

        # Start exactly from the validated Dictionary model.
        nn.init.zeros_(
            self.delta_head.weight
        )
        nn.init.zeros_(
            self.delta_head.bias
        )

        self.output_scale = nn.Parameter(
            torch.tensor(
                float(
                    initial_output_scale
                ),
                dtype=torch.float32,
            )
        )

        self.max_agreement_scale = float(
            max_agreement_scale
        )

        # Positive, bounded agreement amplification.
        #
        # agreement_scale =
        #     max_agreement_scale * sigmoid(raw_agreement_scale)
        #
        # Default:
        #     initial = 0.05
        #     maximum = 0.50
        #
        # This keeps the initial perturbation very small while still
        # allowing the model to increase the margin when useful.
        initial_ratio = (
            initial_agreement_scale
            / max_agreement_scale
        )

        raw_agreement_scale = math.log(
            initial_ratio
            / (
                1.0
                - initial_ratio
            )
        )

        self.raw_agreement_scale = nn.Parameter(
            torch.tensor(
                raw_agreement_scale,
                dtype=torch.float32,
            )
        )

    def get_agreement_scale(
        self,
    ) -> torch.Tensor:
        return (
            self.max_agreement_scale
            * torch.sigmoid(
                self.raw_agreement_scale
            )
        )

    @staticmethod
    def _weighted_prototype(
        feature: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        numerator = (
            feature
            * weight
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        denominator = weight.sum(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(
            1e-6
        )

        return (
            numerator
            / denominator
        )

    def forward(
        self,
        saliency_feature: torch.Tensor,
        shallow_feature: torch.Tensor,
        semantic_feature: torch.Tensor,
        coarse_logit: torch.Tensor,
    ) -> torch.Tensor:
        target_size = (
            saliency_feature.shape[-2:]
        )

        if (
            shallow_feature.shape[-2:]
            != target_size
        ):
            shallow_feature = (
                F.interpolate(
                    shallow_feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        if (
            semantic_feature.shape[-2:]
            != target_size
        ):
            semantic_feature = (
                F.interpolate(
                    semantic_feature,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        if (
            coarse_logit.shape[-2:]
            != target_size
        ):
            coarse_logit = (
                F.interpolate(
                    coarse_logit,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        # -------------------------------------------------
        # Evidence
        # -------------------------------------------------

        evidence_feature = (
            self.evidence_encoder(
                torch.cat(
                    [
                        shallow_feature,
                        semantic_feature,
                    ],
                    dim=1,
                )
            )
        )

        # -------------------------------------------------
        # Coarse belief
        # -------------------------------------------------
        # Detach the selection probability so this branch cannot
        # improve prototype selection simply by changing confidence.
        # -------------------------------------------------

        probability = (
            torch.sigmoid(
                coarse_logit
            )
            .detach()
        )

        uncertainty = (
            4.0
            * probability
            * (
                1.0
                - probability
            )
        )

        # -------------------------------------------------
        # Soft FG/BG prototypes
        # -------------------------------------------------
        # gamma=4 emphasizes confident pixels without hard thresholds.
        # -------------------------------------------------

        foreground_weight = (
            probability.pow(
                self.prototype_gamma
            )
        )

        background_weight = (
            (
                1.0
                - probability
            ).pow(
                self.prototype_gamma
            )
        )

        foreground_prototype = (
            self._weighted_prototype(
                evidence_feature,
                foreground_weight,
            )
        )

        background_prototype = (
            self._weighted_prototype(
                evidence_feature,
                background_weight,
            )
        )

        normalized_evidence = (
            F.normalize(
                evidence_feature,
                p=2,
                dim=1,
                eps=1e-6,
            )
        )

        normalized_foreground = (
            F.normalize(
                foreground_prototype,
                p=2,
                dim=1,
                eps=1e-6,
            )
        )

        normalized_background = (
            F.normalize(
                background_prototype,
                p=2,
                dim=1,
                eps=1e-6,
            )
        )

        similarity_foreground = (
            normalized_evidence
            * normalized_foreground
        ).sum(
            dim=1,
            keepdim=True,
        )

        similarity_background = (
            normalized_evidence
            * normalized_background
        ).sum(
            dim=1,
            keepdim=True,
        )

        similarity_margin = (
            similarity_foreground
            - similarity_background
        )

        signed_evidence = torch.tanh(
            similarity_margin
            / self.evidence_temperature
        )

        # -------------------------------------------------
        # Prediction-evidence disagreement
        # -------------------------------------------------

        prediction_signed = (
            2.0
            * probability
            - 1.0
        )

        disagreement = F.relu(
            -prediction_signed
            * signed_evidence
        )

        # -------------------------------------------------
        # Prediction-evidence agreement
        # -------------------------------------------------
        #
        # When prediction and evidence have the same sign:
        #
        #     agreement = |P| * |E|
        #
        # We multiply once more by |P| so weak/ambiguous coarse
        # predictions are not aggressively sharpened.
        #
        # This gives:
        #
        #     reliable_agreement = |P|^2 * |E|
        #
        # The signed evidence then determines whether the margin
        # is pushed toward foreground (+) or background (-).
        # -------------------------------------------------

        agreement = F.relu(
            prediction_signed
            * signed_evidence
        )

        prediction_confidence = (
            prediction_signed.abs()
        )

        reliable_agreement = (
            agreement
            * prediction_confidence
        )

        agreement_boost = (
            reliable_agreement
            * signed_evidence
        ).detach()

        # -------------------------------------------------
        # Correction gate
        # -------------------------------------------------

        correction_gate = torch.clamp(
            self.uncertainty_weight
            * uncertainty
            + self.disagreement_weight
            * disagreement,
            min=0.0,
            max=1.0,
        )

        signals = torch.cat(
            [
                probability,
                1.0 - probability,
                uncertainty,
                similarity_foreground,
                similarity_background,
                signed_evidence,
                disagreement,
            ],
            dim=1,
        )

        signal_feature = (
            self.signal_encoder(
                signals
            )
        )

        correction_feature = (
            self.correction_fusion(
                torch.cat(
                    [
                        saliency_feature,
                        evidence_feature,
                        signal_feature,
                    ],
                    dim=1,
                )
            )
        )

        delta_logit = (
            self.delta_head(
                correction_feature
            )
        )

        agreement_scale = (
            self.get_agreement_scale()
        )

        return (
            coarse_logit
            + self.output_scale
            * correction_gate
            * delta_logit
            + agreement_scale
            * agreement_boost
        )


class VMambaSmallProgressiveRegionDirectHier60RegionHybridDictionaryDisagreementAgreementRefinementSOD(
    VMambaSmallProgressiveRegionDirectHier60RegionHybridDictionaryRoutingSOD
):
    """
    Controlled comparison:

        VMamba-S
        + Hier60 Hybrid Region Direct
        + Dictionary Routing
        + Disagreement-Aware FG/BG Refinement
        + Agreement-Aware Margin Amplification

    Everything before the coarse Stage1 prediction follows the validated
    Dictionary model. The existing disagreement correction is preserved,
    and one lightweight agreement-margin branch is added.
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
        num_prototypes: int = 12,
        dictionary_dim: int = 128,
        dictionary_temperature: float = 0.2,
        routing_strength: float = 0.5,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path,
            num_prototypes=num_prototypes,
            dictionary_dim=dictionary_dim,
            dictionary_temperature=dictionary_temperature,
            routing_strength=routing_strength,
        )

        self.disagreement_refinement = (
            AgreementAwareFGBGRefinement(
                channels=128,
                signal_channels=32,
                prototype_gamma=4.0,
                evidence_temperature=0.2,
                uncertainty_weight=0.5,
                disagreement_weight=1.0,
                initial_output_scale=0.1,
                initial_agreement_scale=0.05,
                max_agreement_scale=0.5,
            )
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor | list[torch.Tensor],
    ]:
        input_size = (
            image.shape[-2:]
        )

        # -------------------------------------------------
        # VMamba backbone
        # -------------------------------------------------

        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = self.backbone(
            image
        )

        raw_stage2 = stage2
        raw_stage3 = stage3
        raw_stage4 = stage4

        # -------------------------------------------------
        # Dictionary
        # -------------------------------------------------

        type_field = (
            self.latent_type_dictionary(
                stage2=raw_stage2,
                stage3=raw_stage3,
                stage4=raw_stage4,
            )
        )

        # -------------------------------------------------
        # Hier60 region hierarchy
        # -------------------------------------------------

        (
            detail_region,
            fine_region,
            middle_region,
            coarse_region,
        ) = self.region_hierarchy(
            image=image,
            mean_20=mean_60,
            mean_40=mean_60,
            mean_60=mean_60,
        )

        (
            region1,
            region2,
            region3,
            region4,
        ) = self.region_encoder(
            detail_region=detail_region,
            fine_region=fine_region,
            middle_region=middle_region,
            coarse_region=coarse_region,
            stage1_size=(
                stage1.shape[-2:]
            ),
            stage2_size=(
                stage2.shape[-2:]
            ),
            stage3_size=(
                stage3.shape[-2:]
            ),
            stage4_size=(
                stage4.shape[-2:]
            ),
        )

        # -------------------------------------------------
        # Region-conditioned reconstruction
        # -------------------------------------------------

        stage4 = (
            self.stage4_region_interaction(
                visual_feature=stage4,
                region_feature=region4,
            )
        )

        stage3 = (
            self.stage3_region_interaction(
                visual_feature=stage3,
                region_feature=region3,
            )
        )

        stage2 = (
            self.stage2_region_reconstruction(
                visual_feature=stage2,
                region_feature=region2,
            )
        )

        stage1 = (
            self.stage1_detail_reconstruction(
                visual_feature=stage1,
                detail_feature=region1,
            )
        )

        # -------------------------------------------------
        # Stage4
        # -------------------------------------------------

        decoded4 = (
            self.deep_projection(
                stage4
            )
        )

        decoded4 = (
            self.context4(
                decoded4
            )
        )

        prediction4 = (
            self.pred4(
                decoded4
            )
        )

        # -------------------------------------------------
        # Stage3 + Dictionary Routing
        # -------------------------------------------------

        global3 = self.global3(
            stage4,
            target_size=(
                stage3.shape[-2:]
            ),
        )

        (
            routed_stage3,
            routed_decoded4,
            routed_global3,
        ) = self.dictionary_router3(
            low_feature=stage3,
            high_feature=decoded4,
            global_feature=global3,
            type_field=type_field,
        )

        decoded3 = self.fusion3(
            low_feature=routed_stage3,
            high_feature=routed_decoded4,
            global_feature=routed_global3,
        )

        prediction3 = (
            self.pred3(
                decoded3
            )
        )

        decoded3_reduced = (
            self.reduce3(
                decoded3
            )
        )

        # -------------------------------------------------
        # Stage2 + Dictionary Routing
        # -------------------------------------------------

        global2 = self.global2(
            stage4,
            target_size=(
                stage2.shape[-2:]
            ),
        )

        (
            routed_stage2,
            routed_decoded3,
            routed_global2,
        ) = self.dictionary_router2(
            low_feature=stage2,
            high_feature=decoded3_reduced,
            global_feature=global2,
            type_field=type_field,
        )

        decoded2 = self.fusion2(
            low_feature=routed_stage2,
            high_feature=routed_decoded3,
            global_feature=routed_global2,
        )

        prediction2 = (
            self.pred2(
                decoded2
            )
        )

        decoded2_reduced = (
            self.reduce2(
                decoded2
            )
        )

        # -------------------------------------------------
        # Stage1 + Dictionary Routing
        # -------------------------------------------------

        stage1_feature = (
            self.stage1_adapter(
                stage1
            )
        )

        global1 = self.global1(
            stage4,
            target_size=(
                stage1.shape[-2:]
            ),
        )

        (
            routed_stage1,
            routed_decoded2,
            routed_global1,
        ) = self.dictionary_router1(
            low_feature=stage1_feature,
            high_feature=decoded2_reduced,
            global_feature=global1,
            type_field=type_field,
        )

        decoded1 = self.fusion1(
            low_feature=routed_stage1,
            high_feature=routed_decoded2,
            global_feature=routed_global1,
        )

        # -------------------------------------------------
        # Existing boundary refinement
        # -------------------------------------------------

        stage2_boundary = (
            self.stage2_boundary_adapter(
                stage2
            )
        )

        decoded1 = (
            self.boundary_refinement(
                shallow_feature=stage1_feature,
                semantic_feature=stage2_boundary,
                saliency_feature=decoded1,
            )
        )

        # -------------------------------------------------
        # Validated Dictionary coarse prediction
        # -------------------------------------------------

        coarse_prediction = (
            self.pred1(
                decoded1
            )
        )

        # -------------------------------------------------
        # NEW refinement
        # -------------------------------------------------

        refined_prediction = (
            self.disagreement_refinement(
                saliency_feature=decoded1,
                shallow_feature=stage1_feature,
                semantic_feature=stage2_boundary,
                coarse_logit=coarse_prediction,
            )
        )

        refined_prediction = (
            F.interpolate(
                refined_prediction,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )
        )

        # Keep exactly the same aux set as the validated Dictionary model.
        return {
            "pred": refined_prediction,
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model(
) -> VMambaSmallProgressiveRegionDirectHier60RegionHybridDictionaryDisagreementAgreementRefinementSOD:
    return (
        VMambaSmallProgressiveRegionDirectHier60RegionHybridDictionaryDisagreementAgreementRefinementSOD(
            pretrained_path=PRETRAINED_PATH,
            num_prototypes=12,
            dictionary_dim=128,
            dictionary_temperature=0.2,
            routing_strength=0.5,
        )
    )


if __name__ == "__main__":
    model = build_model()
    model.eval()

    image = torch.randn(
        1,
        3,
        352,
        352,
    )

    mean_60 = torch.rand(
        1,
        3,
        352,
        352,
    )

    with torch.no_grad():
        outputs = model(
            image=image,
            mean_60=mean_60,
        )

    print(
        "pred:",
        outputs["pred"].shape,
    )

    print(
        "aux:",
        [
            tensor.shape
            for tensor
            in outputs["aux"]
        ],
    )

    print(
        "dictionary:",
        model.latent_type_dictionary.prototypes.shape,
    )

    print(
        "refinement output scale:",
        float(
            model.disagreement_refinement.output_scale
        ),
    )

    print(
        "agreement scale:",
        float(
            model.disagreement_refinement
            .get_agreement_scale()
        ),
    )
