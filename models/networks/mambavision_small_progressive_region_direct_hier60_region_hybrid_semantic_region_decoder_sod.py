# models/networks/mambavision_small_progressive_region_direct_hier60_region_hybrid_semantic_region_decoder_sod.py

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.sod_blocks import (
    ConvNormAct,
    PredictionHead,
    ResidualConvBlock,
)
from models.networks.mambavision_small_progressive_region_direct_hier60_region_hybrid_cfm_sod import (
    CFMAdapter,
)
from models.networks.mambavision_small_progressive_region_direct_hier60_region_hybrid_sod import (
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD,
)
from models.networks.mambavision_small_progressive_region_direct_sod import (
    PRETRAINED_PATH,
)


class DirectionalCompletionBlock(nn.Module):
    """
    Lightweight structure completion block.

    3x3 depthwise:
        ordinary local continuity

    1x7 / 7x1 depthwise:
        elongated and thin structures

    It is used only in the structure stream. The semantic stream does not
    depend on this block for pixel-level reconstruction.
    """

    def __init__(
        self,
        channels: int,
        directional_kernel: int = 7,
    ) -> None:
        super().__init__()

        padding = directional_kernel // 2

        self.pre = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
            padding=0,
        )

        self.local = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.horizontal = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, directional_kernel),
                padding=(0, padding),
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.vertical = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(directional_kernel, 1),
                padding=(padding, 0),
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            ConvNormAct(
                channels * 3,
                channels,
                kernel_size=1,
                padding=0,
            ),
            ResidualConvBlock(channels),
        )

    def forward(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        base = self.pre(feature)

        local = self.local(base)
        horizontal = self.horizontal(base)
        vertical = self.vertical(base)

        correction = self.fusion(
            torch.cat(
                [
                    local,
                    horizontal,
                    vertical,
                ],
                dim=1,
            )
        )

        return base + correction


class SemanticObjectStream(nn.Module):
    """
    Object-level semantic discovery.

    Unlike the old progressive chain, this stream makes one explicit
    high-level object hypothesis from Stage4 + Stage3.

    CFM is used only here:
        - it can improve semantic discrimination / object recognition;
        - its spatial artifacts are not propagated through every decoder stage.

    The CFMAdapter is the same implementation used by the existing
    Hybrid-CFM experiment, so this experiment changes the role/location of CFM
    rather than silently changing the CFM implementation itself.
    """

    def __init__(
        self,
        stage4_channels: int,
        semantic_channels: int,
    ) -> None:
        super().__init__()

        self.stage4_projection = ConvNormAct(
            stage4_channels,
            semantic_channels,
            kernel_size=1,
            padding=0,
        )

        self.stage3_projection = ConvNormAct(
            semantic_channels,
            semantic_channels,
            kernel_size=1,
            padding=0,
        )

        self.semantic_fusion = nn.Sequential(
            ConvNormAct(
                semantic_channels * 2,
                semantic_channels,
                kernel_size=1,
                padding=0,
            ),
            ResidualConvBlock(
                semantic_channels
            ),
        )

        # Same CFM adapter design as the current Hybrid-CFM network.
        # Stage3-level semantic feature corresponds to CFM stage index 2.
        self.cfm = CFMAdapter(
            dim=semantic_channels,
            stage=2,
        )

        self.semantic_refine = ResidualConvBlock(
            semantic_channels
        )

        self.semantic_head = PredictionHead(
            semantic_channels
        )

    def forward(
        self,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        target_size = stage3.shape[-2:]

        deep = self.stage4_projection(
            stage4
        )
        deep = F.interpolate(
            deep,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        middle = self.stage3_projection(
            stage3
        )

        semantic = self.semantic_fusion(
            torch.cat(
                [
                    middle,
                    deep,
                ],
                dim=1,
            )
        )

        semantic = self.cfm(
            semantic
        )

        semantic = self.semantic_refine(
            semantic
        )

        semantic_logit = self.semantic_head(
            semantic
        )

        return semantic, semantic_logit


class RegionStructureStage2(nn.Module):
    """
    First spatial reconstruction stage.

    Inputs:
        Stage2 visual-region reconstructed feature
        Stage2 region feature
        semantic object feature

    The semantic feature is guidance, not a hard mask. Region/local evidence
    still has its own path, so a semantic miss does not permanently suppress
    the structure branch.
    """

    def __init__(
        self,
        stage2_channels: int,
        semantic_channels: int,
        initial_correction_scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.semantic_projection = ConvNormAct(
            semantic_channels,
            stage2_channels,
            kernel_size=1,
            padding=0,
        )

        self.region_projection = ConvNormAct(
            stage2_channels,
            stage2_channels,
            kernel_size=1,
            padding=0,
        )

        self.joint_gate = nn.Sequential(
            ConvNormAct(
                stage2_channels * 3,
                stage2_channels,
                kernel_size=1,
                padding=0,
            ),
            nn.Conv2d(
                stage2_channels,
                stage2_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            ConvNormAct(
                stage2_channels * 3,
                stage2_channels,
                kernel_size=1,
                padding=0,
            ),
            ResidualConvBlock(
                stage2_channels
            ),
        )

        self.completion = DirectionalCompletionBlock(
            stage2_channels,
            directional_kernel=7,
        )

        self.correction_scale = nn.Parameter(
            torch.tensor(
                float(initial_correction_scale),
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        stage2: torch.Tensor,
        region2: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        target_size = stage2.shape[-2:]

        semantic = self.semantic_projection(
            semantic
        )
        semantic = F.interpolate(
            semantic,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        region = self.region_projection(
            region2
        )

        gate = self.joint_gate(
            torch.cat(
                [
                    stage2,
                    region,
                    semantic,
                ],
                dim=1,
            )
        )

        # 0.5 + gate gives the semantic branch a bounded residual influence
        # without turning the gate into a binary pass/reject mask.
        semantic_guidance = (
            semantic
            * (0.5 + gate)
        )

        correction = self.fusion(
            torch.cat(
                [
                    stage2,
                    region,
                    semantic_guidance,
                ],
                dim=1,
            )
        )

        correction = self.completion(
            correction
        )

        return (
            stage2
            + self.correction_scale
            * correction
        )


class RegionStructureStage1(nn.Module):
    """
    High-resolution object completion.

    It combines:
        Stage1 local appearance
        RGB-M60 region detail
        completed Stage2 structure
        object-level semantic feature

    This stage is responsible for filling, connecting and trimming the mask.
    It is intentionally not another top-down saliency decoder.
    """

    def __init__(
        self,
        stage1_channels: int,
        stage2_channels: int,
        semantic_channels: int,
        output_channels: int = 128,
        initial_correction_scale: float = 0.1,
    ) -> None:
        super().__init__()

        self.stage1_projection = ConvNormAct(
            stage1_channels,
            output_channels,
            kernel_size=1,
            padding=0,
        )

        self.region_projection = ConvNormAct(
            stage1_channels,
            output_channels,
            kernel_size=1,
            padding=0,
        )

        self.stage2_projection = ConvNormAct(
            stage2_channels,
            output_channels,
            kernel_size=1,
            padding=0,
        )

        self.semantic_projection = ConvNormAct(
            semantic_channels,
            output_channels,
            kernel_size=1,
            padding=0,
        )

        self.semantic_gate = nn.Sequential(
            ConvNormAct(
                output_channels * 3,
                output_channels,
                kernel_size=1,
                padding=0,
            ),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.fusion = nn.Sequential(
            ConvNormAct(
                output_channels * 4,
                output_channels,
                kernel_size=1,
                padding=0,
            ),
            ResidualConvBlock(
                output_channels
            ),
        )

        self.completion = DirectionalCompletionBlock(
            output_channels,
            directional_kernel=7,
        )

        self.output_refine = ResidualConvBlock(
            output_channels
        )

        self.correction_scale = nn.Parameter(
            torch.tensor(
                float(initial_correction_scale),
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        stage1: torch.Tensor,
        region1: torch.Tensor,
        stage2_structure: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        target_size = stage1.shape[-2:]

        local = self.stage1_projection(
            stage1
        )

        region = self.region_projection(
            region1
        )

        structure = self.stage2_projection(
            stage2_structure
        )
        structure = F.interpolate(
            structure,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        semantic = self.semantic_projection(
            semantic
        )
        semantic = F.interpolate(
            semantic,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        gate = self.semantic_gate(
            torch.cat(
                [
                    local,
                    region,
                    semantic,
                ],
                dim=1,
            )
        )

        semantic_guidance = (
            semantic
            * (0.5 + gate)
        )

        correction = self.fusion(
            torch.cat(
                [
                    local,
                    region,
                    structure,
                    semantic_guidance,
                ],
                dim=1,
            )
        )

        correction = self.completion(
            correction
        )

        completed = (
            local
            + self.correction_scale
            * correction
        )

        return self.output_refine(
            completed
        )


class SemanticRegionCooperativeDecoder(nn.Module):
    """
    Semantic-region cooperative decoder.

    Stream A:
        Stage4 + Stage3
        -> semantic object discovery
        -> coarse object logit

    Stream B:
        Stage2 + Stage1 + region features
        -> region/structure completion
        -> completion residual

    Final:
        final_logit = semantic_logit + completion_scale * completion_delta

    Only the semantic logit is used as auxiliary supervision. We no longer
    force every intermediate scale to independently produce a full saliency
    mask.
    """

    def __init__(
        self,
        stage1_channels: int,
        stage2_channels: int,
        stage3_channels: int,
        stage4_channels: int,
        completion_channels: int = 128,
        initial_structure_scale: float = 0.1,
        initial_completion_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.semantic_stream = SemanticObjectStream(
            stage4_channels=stage4_channels,
            semantic_channels=stage3_channels,
        )

        self.structure_stage2 = RegionStructureStage2(
            stage2_channels=stage2_channels,
            semantic_channels=stage3_channels,
            initial_correction_scale=initial_structure_scale,
        )

        self.structure_stage1 = RegionStructureStage1(
            stage1_channels=stage1_channels,
            stage2_channels=stage2_channels,
            semantic_channels=stage3_channels,
            output_channels=completion_channels,
            initial_correction_scale=initial_structure_scale,
        )

        self.completion_head = PredictionHead(
            completion_channels
        )

        self.completion_scale = nn.Parameter(
            torch.tensor(
                float(initial_completion_scale),
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        stage1: torch.Tensor,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
        region1: torch.Tensor,
        region2: torch.Tensor,
        input_size: tuple[int, int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        semantic, semantic_logit = (
            self.semantic_stream(
                stage3=stage3,
                stage4=stage4,
            )
        )

        structure2 = self.structure_stage2(
            stage2=stage2,
            region2=region2,
            semantic=semantic,
        )

        completion_feature = (
            self.structure_stage1(
                stage1=stage1,
                region1=region1,
                stage2_structure=structure2,
                semantic=semantic,
            )
        )

        completion_delta = self.completion_head(
            completion_feature
        )

        semantic_full = F.interpolate(
            semantic_logit,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        completion_full = F.interpolate(
            completion_delta,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        final_logit = (
            semantic_full
            + self.completion_scale
            * completion_full
        )

        return (
            final_logit,
            semantic_logit,
            completion_delta,
        )


class MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSemanticRegionDecoderSOD(
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD
):
    """
    Hybrid H60 backbone/region path with a replaced decoder.

    Retained:
        MambaVision-S backbone
        H60 assignment
        Hybrid Region Mean encoder
        Stage4/Stage3 region-visual interaction
        Stage2 region-conditioned reconstruction
        Stage1 RGB-M60 detail reconstruction

    Removed:
        old decoded4 -> decoded3 -> decoded2 -> decoded1 progressive chain
        persistent GAP-only global stream
        four-stage auxiliary prediction design
        old boundary refinement head

    Added:
        semantic object stream with a single CFM location
        region/structure completion stream
        logit-level semantic + completion residual prediction
        one semantic auxiliary prediction
    """

    input_keys = (
        "image",
        "mean_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
    ) -> None:
        super().__init__(
            pretrained_path=pretrained_path
        )

        stage1_channels = (
            self.backbone.out_channels[0]
        )
        stage2_channels = (
            self.backbone.out_channels[1]
        )
        stage3_channels = (
            self.backbone.out_channels[2]
        )
        stage4_channels = (
            self.backbone.out_channels[3]
        )

        # The parent network is used to construct exactly the validated
        # backbone + Hybrid region path. Remove its old progressive decoder
        # so unused decoder parameters do not remain in the model.
        old_decoder_modules = (
            "deep_projection",
            "context4",
            "pred4",
            "global3",
            "global2",
            "global1",
            "fusion3",
            "pred3",
            "reduce3",
            "fusion2",
            "pred2",
            "reduce2",
            "stage1_adapter",
            "fusion1",
            "stage2_boundary_adapter",
            "boundary_refinement",
            "pred1",
        )

        for module_name in old_decoder_modules:
            delattr(
                self,
                module_name,
            )

        self.decoder = (
            SemanticRegionCooperativeDecoder(
                stage1_channels=stage1_channels,
                stage2_channels=stage2_channels,
                stage3_channels=stage3_channels,
                stage4_channels=stage4_channels,
                completion_channels=128,
                initial_structure_scale=0.1,
                initial_completion_scale=1.0,
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
        input_size = image.shape[-2:]

        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = self.backbone(
            image
        )

        # H60 assignment:
        #   Stage1 <- RGB - M60
        #   Stage2 <- M60
        #   Stage3 <- M60
        #   Stage4 <- M60
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
            stage1_size=stage1.shape[-2:],
            stage2_size=stage2.shape[-2:],
            stage3_size=stage3.shape[-2:],
            stage4_size=stage4.shape[-2:],
        )

        # Keep the validated Hybrid region-conditioned backbone
        # reconstruction exactly where it was.
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

        (
            prediction,
            semantic_prediction,
            completion_delta,
        ) = self.decoder(
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
            region1=region1,
            region2=region2,
            input_size=input_size,
        )

        return {
            "pred": prediction,
            # Only object-level semantic discovery is independently
            # supervised. The structure stream learns through final pred.
            "aux": [
                semantic_prediction,
            ],
            # Extra outputs are ignored by the current SODLoss/test path,
            # but are useful for later qualitative diagnostics.
            "semantic_pred": semantic_prediction,
            "completion_delta": completion_delta,
        }


def build_model(
) -> MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSemanticRegionDecoderSOD:
    return (
        MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSemanticRegionDecoderSOD(
            pretrained_path=PRETRAINED_PATH,
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
            for tensor in outputs["aux"]
        ],
    )

    print(
        "semantic_pred:",
        outputs["semantic_pred"].shape,
    )

    print(
        "completion_delta:",
        outputs["completion_delta"].shape,
    )

    print(
        "completion scale:",
        float(model.decoder.completion_scale),
    )

    print(
        "CFM scale:",
        float(
            model.decoder
            .semantic_stream
            .cfm
            .scale
        ),
    )
