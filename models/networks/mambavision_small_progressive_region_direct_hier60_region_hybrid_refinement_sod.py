from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import ConvNormAct, PredictionHead, ResidualConvBlock
from models.networks.mambavision_small_progressive_region_direct_hier60_region_hybrid_sod import (
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD,
)
from models.networks.mambavision_small_progressive_region_direct_sod import PRETRAINED_PATH


class CoarseGuidanceEncoder(nn.Module):
    """Encode foreground/background confidence and uncertainty from Stage-1 logits."""

    def __init__(self, out_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvNormAct(3, out_channels, kernel_size=3),
            ResidualConvBlock(out_channels),
        )

    def forward(
        self,
        coarse_logit: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        coarse_logit = F.interpolate(
            coarse_logit,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        p = torch.sigmoid(coarse_logit)
        guidance = torch.cat(
            [
                p,
                1.0 - p,
                4.0 * p * (1.0 - p),
            ],
            dim=1,
        )
        return self.encoder(guidance)


class RegionGuidedRefinementBlock(nn.Module):
    """One scale of the second-stage top-down refinement decoder."""

    def __init__(
        self,
        visual_channels: int,
        region_channels: int,
        refine_channels: int = 128,
        region_reduced: int = 64,
        guidance_channels: int = 32,
        has_high_feature: bool = True,
        initial_correction_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.has_high_feature = has_high_feature

        self.visual_projection = ConvNormAct(
            visual_channels,
            refine_channels,
            kernel_size=1,
            padding=0,
        )
        self.region_projection = ConvNormAct(
            region_channels,
            region_reduced,
            kernel_size=1,
            padding=0,
        )
        self.guidance_encoder = CoarseGuidanceEncoder(guidance_channels)

        if has_high_feature:
            self.high_projection = ConvNormAct(
                refine_channels,
                refine_channels,
                kernel_size=1,
                padding=0,
            )

        fusion_channels = refine_channels + region_reduced + guidance_channels
        if has_high_feature:
            fusion_channels += refine_channels

        self.fusion = nn.Sequential(
            ConvNormAct(
                fusion_channels,
                refine_channels,
                kernel_size=1,
                padding=0,
            ),
            nn.Conv2d(
                refine_channels,
                refine_channels,
                kernel_size=3,
                padding=1,
                groups=refine_channels,
                bias=False,
            ),
            nn.GroupNorm(8, refine_channels),
            nn.GELU(),
            ResidualConvBlock(refine_channels),
        )

        self.refinement_gate = nn.Sequential(
            nn.Conv2d(
                region_reduced + guidance_channels,
                refine_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.correction_scale = nn.Parameter(
            torch.tensor(float(initial_correction_scale), dtype=torch.float32)
        )

    def forward(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
        coarse_logit: torch.Tensor,
        high_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_size = visual_feature.shape[-2:]

        visual = self.visual_projection(visual_feature)
        region = self.region_projection(region_feature)
        guidance = self.guidance_encoder(coarse_logit, target_size)

        fusion_inputs = [visual, region, guidance]

        if self.has_high_feature:
            if high_feature is None:
                raise ValueError("high_feature is required for this refinement block")
            high_feature = F.interpolate(
                high_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            fusion_inputs.append(self.high_projection(high_feature))

        correction = self.fusion(torch.cat(fusion_inputs, dim=1))
        gate = self.refinement_gate(torch.cat([region, guidance], dim=1))

        return visual + self.correction_scale * gate * correction


class RegionGuidedRefinementStage(nn.Module):
    """
    Full second-stage refinement path:

        decoded4 -> refine4
                    ↓
        decoded3 -> refine3
                    ↓
        decoded2 -> refine2
                    ↓
        decoded1 -> refine1 -> delta logit

    Every scale is conditioned by the Stage-1 coarse prediction and the
    corresponding Hybrid region feature.
    """

    def __init__(
        self,
        decoded4_channels: int,
        decoded3_channels: int,
        decoded2_channels: int,
        decoded1_channels: int,
        region4_channels: int,
        region3_channels: int,
        region2_channels: int,
        region1_channels: int,
        refine_channels: int = 128,
        initial_correction_scale: float = 0.1,
        initial_output_scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.refine4 = RegionGuidedRefinementBlock(
            decoded4_channels,
            region4_channels,
            refine_channels=refine_channels,
            has_high_feature=False,
            initial_correction_scale=initial_correction_scale,
        )
        self.refine3 = RegionGuidedRefinementBlock(
            decoded3_channels,
            region3_channels,
            refine_channels=refine_channels,
            has_high_feature=True,
            initial_correction_scale=initial_correction_scale,
        )
        self.refine2 = RegionGuidedRefinementBlock(
            decoded2_channels,
            region2_channels,
            refine_channels=refine_channels,
            has_high_feature=True,
            initial_correction_scale=initial_correction_scale,
        )
        self.refine1 = RegionGuidedRefinementBlock(
            decoded1_channels,
            region1_channels,
            refine_channels=refine_channels,
            has_high_feature=True,
            initial_correction_scale=initial_correction_scale,
        )

        self.delta_head = PredictionHead(refine_channels)
        self.output_scale = nn.Parameter(
            torch.tensor(float(initial_output_scale), dtype=torch.float32)
        )

    def forward(
        self,
        decoded1: torch.Tensor,
        decoded2: torch.Tensor,
        decoded3: torch.Tensor,
        decoded4: torch.Tensor,
        region1: torch.Tensor,
        region2: torch.Tensor,
        region3: torch.Tensor,
        region4: torch.Tensor,
        coarse_logit: torch.Tensor,
    ) -> torch.Tensor:
        refined4 = self.refine4(
            decoded4,
            region4,
            coarse_logit,
        )
        refined3 = self.refine3(
            decoded3,
            region3,
            coarse_logit,
            high_feature=refined4,
        )
        refined2 = self.refine2(
            decoded2,
            region2,
            coarse_logit,
            high_feature=refined3,
        )
        refined1 = self.refine1(
            decoded1,
            region1,
            coarse_logit,
            high_feature=refined2,
        )

        delta_logit = self.delta_head(refined1)
        if delta_logit.shape[-2:] != coarse_logit.shape[-2:]:
            delta_logit = F.interpolate(
                delta_logit,
                size=coarse_logit.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return coarse_logit + self.output_scale * delta_logit


class MambaVisionSmallProgressiveRegionDirectHier60RegionHybridRefinementSOD(
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD
):
    """Hybrid H60 Stage-1 + a full region-guided Stage-2 refinement decoder."""

    input_keys = ("image", "mean_60")

    def __init__(self, pretrained_path: str | Path | None) -> None:
        super().__init__(pretrained_path=pretrained_path)

        stage1_channels = self.backbone.out_channels[0]
        stage2_channels = self.backbone.out_channels[1]
        stage3_channels = self.backbone.out_channels[2]
        stage4_channels = self.backbone.out_channels[3]

        # Stage-1 decoded widths:
        # decoded4 = stage3_channels
        # decoded3 = stage3_channels
        # decoded2 = stage2_channels
        # decoded1 = 128
        self.refinement_stage = RegionGuidedRefinementStage(
            decoded4_channels=stage3_channels,
            decoded3_channels=stage3_channels,
            decoded2_channels=stage2_channels,
            decoded1_channels=128,
            region4_channels=stage4_channels,
            region3_channels=stage3_channels,
            region2_channels=stage2_channels,
            region1_channels=stage1_channels,
            refine_channels=128,
            initial_correction_scale=0.1,
            initial_output_scale=0.1,
        )

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        input_size = image.shape[-2:]

        # ==============================================================
        # Stage 1: backbone
        # ==============================================================
        stage1, stage2, stage3, stage4 = self.backbone(image)

        # ==============================================================
        # Stage 1: H60 region hierarchy
        # ==============================================================
        detail_region, fine_region, middle_region, coarse_region = (
            self.region_hierarchy(
                image=image,
                mean_20=mean_60,
                mean_40=mean_60,
                mean_60=mean_60,
            )
        )

        # ==============================================================
        # Stage 1: Hybrid region encoder
        # ==============================================================
        region1, region2, region3, region4 = self.region_encoder(
            detail_region=detail_region,
            fine_region=fine_region,
            middle_region=middle_region,
            coarse_region=coarse_region,
            stage1_size=stage1.shape[-2:],
            stage2_size=stage2.shape[-2:],
            stage3_size=stage3.shape[-2:],
            stage4_size=stage4.shape[-2:],
        )

        # ==============================================================
        # Stage 1: region-conditioned backbone reconstruction
        # ==============================================================
        stage4 = self.stage4_region_interaction(
            visual_feature=stage4,
            region_feature=region4,
        )
        stage3 = self.stage3_region_interaction(
            visual_feature=stage3,
            region_feature=region3,
        )
        stage2 = self.stage2_region_reconstruction(
            visual_feature=stage2,
            region_feature=region2,
        )
        stage1 = self.stage1_detail_reconstruction(
            visual_feature=stage1,
            detail_feature=region1,
        )

        # ==============================================================
        # Stage 1: progressive decoder
        # ==============================================================
        decoded4 = self.context4(self.deep_projection(stage4))
        prediction4 = self.pred4(decoded4)

        global3 = self.global3(stage4, target_size=stage3.shape[-2:])
        decoded3 = self.fusion3(
            low_feature=stage3,
            high_feature=decoded4,
            global_feature=global3,
        )
        prediction3 = self.pred3(decoded3)
        decoded3_reduced = self.reduce3(decoded3)

        global2 = self.global2(stage4, target_size=stage2.shape[-2:])
        decoded2 = self.fusion2(
            low_feature=stage2,
            high_feature=decoded3_reduced,
            global_feature=global2,
        )
        prediction2 = self.pred2(decoded2)
        decoded2_reduced = self.reduce2(decoded2)

        stage1_feature = self.stage1_adapter(stage1)
        global1 = self.global1(stage4, target_size=stage1.shape[-2:])
        decoded1 = self.fusion1(
            low_feature=stage1_feature,
            high_feature=decoded2_reduced,
            global_feature=global1,
        )

        stage2_boundary = self.stage2_boundary_adapter(stage2)
        decoded1 = self.boundary_refinement(
            shallow_feature=stage1_feature,
            semantic_feature=stage2_boundary,
            saliency_feature=decoded1,
        )

        coarse_prediction = self.pred1(decoded1)

        # ==============================================================
        # Stage 2: region-guided progressive refinement
        # ==============================================================
        refined_prediction = self.refinement_stage(
            decoded1=decoded1,
            decoded2=decoded2,
            decoded3=decoded3,
            decoded4=decoded4,
            region1=region1,
            region2=region2,
            region3=region3,
            region4=region4,
            coarse_logit=coarse_prediction,
        )

        refined_prediction = F.interpolate(
            refined_prediction,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        # Existing SODLoss can remain unchanged:
        # final refined prediction is the main output;
        # Stage-1 coarse prediction becomes an additional auxiliary output.
        return {
            "pred": refined_prediction,
            "aux": [
                coarse_prediction,
                prediction2,
                prediction3,
                prediction4,
            ],
        }


def build_model() -> MambaVisionSmallProgressiveRegionDirectHier60RegionHybridRefinementSOD:
    return MambaVisionSmallProgressiveRegionDirectHier60RegionHybridRefinementSOD(
        pretrained_path=PRETRAINED_PATH,
    )


if __name__ == "__main__":
    model = build_model()
    model.eval()

    image = torch.randn(1, 3, 352, 352)
    mean_60 = torch.rand(1, 3, 352, 352)

    with torch.no_grad():
        outputs = model(image=image, mean_60=mean_60)

    print("pred:", outputs["pred"].shape)
    print("aux:", [x.shape for x in outputs["aux"]])
    print("refinement output scale:", float(model.refinement_stage.output_scale))
    print(
        "refinement correction scales:",
        [
            float(model.refinement_stage.refine4.correction_scale),
            float(model.refinement_stage.refine3.correction_scale),
            float(model.refinement_stage.refine2.correction_scale),
            float(model.refinement_stage.refine1.correction_scale),
        ],
    )
