# models/networks/mambavision_small_nam_oasis_sod.py
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.mambavision import (
    MambaVisionBackbone,
    mamba_vision_small,
)
from models.components.sod_blocks import (
    BoundaryRefinementBlock,
    ConvNormAct,
    PredictionHead,
    PyramidContextBlock,
    ResidualConvBlock,
    SaliencyGuidedFusion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "mambavision"
    / "mambavision_small_1k.pth.tar"
)


class GuidanceFeatureEnhancement(nn.Module):
    """
    Enhance multi-scale features with a spatial guidance map.

    The operation follows the OASIS-style residual modulation:

        output = feature * (1 + factor * guidance)
    """

    def __init__(
        self,
        factor: float,
        guidance_is_logits: bool,
    ) -> None:
        super().__init__()

        self.factor = factor
        self.guidance_is_logits = guidance_is_logits

    def forward(
        self,
        features: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        guidance: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.guidance_is_logits:
            guidance = torch.sigmoid(
                guidance
            )
        else:
            guidance = (
                guidance
                .float()
                .clamp(0.0, 1.0)
            )

        enhanced_features: list[
            torch.Tensor
        ] = []

        for feature in features:
            resized_guidance = F.interpolate(
                guidance,
                size=feature.shape[-2:],
                mode="nearest",
            ).to(
                dtype=feature.dtype
            )

            enhanced_feature = (
                feature
                * (
                    1.0
                    + self.factor
                    * resized_guidance
                )
            )

            enhanced_features.append(
                enhanced_feature
            )

        return tuple(
            enhanced_features
        )


class OASISStructureDecoder(nn.Module):
    """
    Generate a target-aware structure map.

    Inputs:
        Multi-scale features enhanced by the rough NAM edge.
        Deep semantic context used as the static-image
        replacement for OASIS object memory.

    Output:
        Structure logits at stride 4.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.semantic_projection = (
            ConvNormAct(
                channels,
                channels,
                kernel_size=1,
                padding=0,
            )
        )

        self.decode3 = nn.Sequential(
            ConvNormAct(
                channels * 2,
                channels,
            ),
            ResidualConvBlock(
                channels
            ),
        )

        self.decode2 = nn.Sequential(
            ConvNormAct(
                channels * 2,
                channels,
            ),
            ResidualConvBlock(
                channels
            ),
        )

        self.decode1 = nn.Sequential(
            ConvNormAct(
                channels * 2,
                channels,
            ),
            ResidualConvBlock(
                channels
            ),
        )

        self.structure_head = nn.Sequential(
            ConvNormAct(
                channels,
                channels // 2,
            ),
            nn.Conv2d(
                channels // 2,
                1,
                kernel_size=1,
            ),
        )

    @staticmethod
    def _resize(
        feature: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.interpolate(
            feature,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        features: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        semantic_feature: torch.Tensor,
    ) -> torch.Tensor:
        (
            feature1,
            feature2,
            feature3,
            feature4,
        ) = features

        decoded4 = (
            feature4
            + self.semantic_projection(
                semantic_feature
            )
        )

        decoded3 = self.decode3(
            torch.cat(
                [
                    self._resize(
                        decoded4,
                        feature3,
                    ),
                    feature3,
                ],
                dim=1,
            )
        )

        decoded2 = self.decode2(
            torch.cat(
                [
                    self._resize(
                        decoded3,
                        feature2,
                    ),
                    feature2,
                ],
                dim=1,
            )
        )

        decoded1 = self.decode1(
            torch.cat(
                [
                    self._resize(
                        decoded2,
                        feature1,
                    ),
                    feature1,
                ],
                dim=1,
            )
        )

        structure_logits = (
            self.structure_head(
                decoded1
            )
        )

        return structure_logits


class MambaVisionSmallNAMOASISSOD(
    nn.Module
):
    """
    MambaVision-S SOD with OASIS-style structure refinement.

    Backbone:
        stage1: stride 4,  channels 96
        stage2: stride 8,  channels 192
        stage3: stride 16, channels 384
        stage4: stride 32, channels 768

    Structure refinement:
        NAM rough-edge enhancement
        -> semantic structure decoder
        -> target-aware structure map
        -> multi-scale feature re-enhancement

    Saliency decoder:
        The original MambaVision-S SOD decoder is preserved.
    """

    input_keys = (
        "image",
        "nam_60",
    )

    edge_target_key = "nam_60"

    def __init__(
        self,
        pretrained_path: (
            str | Path | None
        ),
        decoder_channels: int = 128,
        rough_edge_factor: float = 0.5,
        structure_factor: float = 1.0,
    ) -> None:
        super().__init__()

        self.backbone: (
            MambaVisionBackbone
        ) = mamba_vision_small(
            pretrained_path=pretrained_path,
        )

        self.projections = nn.ModuleList(
            [
                ConvNormAct(
                    in_channels,
                    decoder_channels,
                    kernel_size=1,
                    padding=0,
                )
                for in_channels
                in self.backbone.out_channels
            ]
        )

        # OASIS first enhancement:
        # rough global edge prior.
        self.rough_edge_enhancement = (
            GuidanceFeatureEnhancement(
                factor=rough_edge_factor,
                guidance_is_logits=False,
            )
        )

        # Static-image replacement for
        # the object memory used by OASIS.
        self.structure_semantic = (
            nn.Sequential(
                PyramidContextBlock(
                    decoder_channels
                ),
                ResidualConvBlock(
                    decoder_channels
                ),
            )
        )

        self.structure_decoder = (
            OASISStructureDecoder(
                decoder_channels
            )
        )

        # OASIS second enhancement:
        # target-aware structure map.
        self.structure_enhancement = (
            GuidanceFeatureEnhancement(
                factor=structure_factor,
                guidance_is_logits=True,
            )
        )

        # Original baseline decoder.
        self.context4 = PyramidContextBlock(
            decoder_channels
        )

        self.pred4 = PredictionHead(
            decoder_channels
        )

        self.fusion3 = SaliencyGuidedFusion(
            decoder_channels
        )

        self.pred3 = PredictionHead(
            decoder_channels
        )

        self.fusion2 = SaliencyGuidedFusion(
            decoder_channels
        )

        self.pred2 = PredictionHead(
            decoder_channels
        )

        self.fusion1 = SaliencyGuidedFusion(
            decoder_channels
        )

        self.boundary_refinement = (
            BoundaryRefinementBlock(
                decoder_channels
            )
        )

        self.pred1 = PredictionHead(
            decoder_channels
        )

    def forward(
        self,
        image: torch.Tensor,
        nam_60: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor
        | list[torch.Tensor],
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

        original_features = tuple(
            projection(feature)
            for projection, feature in zip(
                self.projections,
                (
                    stage1,
                    stage2,
                    stage3,
                    stage4,
                ),
            )
        )

        # Step 1:
        # Highlight rough global boundaries
        # using NAMLab hier_60.
        rough_edge_features = (
            self.rough_edge_enhancement(
                features=original_features,
                guidance=nam_60,
            )
        )

        # Step 2:
        # Extract target semantics from
        # the deepest enhanced feature.
        structure_semantic = (
            self.structure_semantic(
                rough_edge_features[3]
            )
        )

        # Step 3:
        # Fuse rough boundaries and target
        # semantics into a structure map.
        structure_logits = (
            self.structure_decoder(
                features=rough_edge_features,
                semantic_feature=(
                    structure_semantic
                ),
            )
        )

        # Step 4:
        # Apply the target-aware structure
        # map to the original features.
        (
            feature1,
            feature2,
            feature3,
            feature4,
        ) = self.structure_enhancement(
            features=original_features,
            guidance=structure_logits,
        )

        # Original baseline decoder.
        decoded4 = self.context4(
            feature4
        )

        prediction4 = self.pred4(
            decoded4
        )

        decoded3 = self.fusion3(
            low_feature=feature3,
            high_feature=decoded4,
            guide_logits=prediction4,
        )

        prediction3 = self.pred3(
            decoded3
        )

        decoded2 = self.fusion2(
            low_feature=feature2,
            high_feature=decoded3,
            guide_logits=prediction3,
        )

        prediction2 = self.pred2(
            decoded2
        )

        decoded1 = self.fusion1(
            low_feature=feature1,
            high_feature=decoded2,
            guide_logits=prediction2,
        )

        decoded1 = (
            self.boundary_refinement(
                shallow_feature=feature1,
                semantic_feature=feature2,
                saliency_feature=decoded1,
            )
        )

        prediction1 = self.pred1(
            decoded1
        )

        prediction1 = F.interpolate(
            prediction1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": prediction1,
            "aux": [
                prediction2,
                prediction3,
                prediction4,
            ],
            "edge": structure_logits,
        }


def build_model(
) -> MambaVisionSmallNAMOASISSOD:
    return MambaVisionSmallNAMOASISSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
        rough_edge_factor=0.5,
        structure_factor=1.0,
    )