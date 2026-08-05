# models/networks/mambavision_small_enf_nam_sod.py
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
    SaliencyGuidedFusion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "mambavision"
    / "mambavision_small_1k.pth.tar"
)


class NAMConditionNetwork(nn.Module):
    """
    Encode the NAMLab hier_60 map into a shared condition feature.

    The NAM map is first resized to the shallowest backbone feature size.
    Four convolution blocks are then used following the condition-network
    design of ENFNet.
    """

    def __init__(
        self,
        condition_channels: int = 64,
    ) -> None:
        super().__init__()

        hidden_channels = (
            condition_channels // 2
        )

        self.encoder = nn.Sequential(
            ConvNormAct(
                1,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            ConvNormAct(
                hidden_channels,
                condition_channels,
                kernel_size=3,
                padding=1,
            ),
            ConvNormAct(
                condition_channels,
                condition_channels,
                kernel_size=3,
                padding=1,
            ),
            ConvNormAct(
                condition_channels,
                condition_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(
        self,
        nam_60: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        nam_60 = F.interpolate(
            nam_60,
            size=output_size,
            mode="nearest",
        )

        return self.encoder(
            nam_60
        )


class ENFParameterGenerator(nn.Module):
    """
    Generate spatial gamma and beta parameters from the NAM condition.

    gamma starts from one and beta starts from zero, so the initial feature
    transform is an identity mapping.
    """

    def __init__(
        self,
        condition_channels: int,
        feature_channels: int,
        affine_strength: float = 0.5,
    ) -> None:
        super().__init__()

        self.affine_strength = float(
            affine_strength
        )

        self.gamma_branch = nn.Sequential(
            ConvNormAct(
                condition_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

        self.beta_branch = nn.Sequential(
            ConvNormAct(
                condition_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(
                feature_channels,
                feature_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )

        nn.init.zeros_(
            self.gamma_branch[-1].weight
        )
        nn.init.zeros_(
            self.gamma_branch[-1].bias
        )

        nn.init.zeros_(
            self.beta_branch[-1].weight
        )
        nn.init.zeros_(
            self.beta_branch[-1].bias
        )

    def forward(
        self,
        condition: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        gamma_delta = self.gamma_branch(
            condition
        )

        beta_delta = self.beta_branch(
            condition
        )

        gamma = (
            1.0
            + self.affine_strength
            * torch.tanh(
                gamma_delta
            )
        )

        beta = (
            self.affine_strength
            * torch.tanh(
                beta_delta
            )
        )

        return gamma, beta


class ENFGuidanceBlock(nn.Module):
    """
    Apply ENFNet-style NAM-conditioned affine transformation.

    guided = feature * gamma + beta
    contrast = guided - AvgPool3x3(guided)

    The local contrast feature is fused back through a residual branch.
    """

    def __init__(
        self,
        feature_channels: int,
        condition_channels: int,
        affine_strength: float = 0.5,
    ) -> None:
        super().__init__()

        self.parameter_generator = (
            ENFParameterGenerator(
                condition_channels=(
                    condition_channels
                ),
                feature_channels=(
                    feature_channels
                ),
                affine_strength=(
                    affine_strength
                ),
            )
        )

        self.contrast_fusion = (
            nn.Sequential(
                ConvNormAct(
                    feature_channels * 2,
                    feature_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.Conv2d(
                    feature_channels,
                    feature_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
            )
        )

        nn.init.zeros_(
            self.contrast_fusion[-1].weight
        )
        nn.init.zeros_(
            self.contrast_fusion[-1].bias
        )

    def forward(
        self,
        feature: torch.Tensor,
        shared_condition: torch.Tensor,
    ) -> torch.Tensor:
        condition = F.interpolate(
            shared_condition,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        gamma, beta = (
            self.parameter_generator(
                condition
            )
        )

        guided_feature = (
            feature * gamma + beta
        )

        local_average = F.avg_pool2d(
            guided_feature,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        contrast_feature = (
            guided_feature
            - local_average
        )

        refinement = (
            self.contrast_fusion(
                torch.cat(
                    [
                        guided_feature,
                        contrast_feature,
                    ],
                    dim=1,
                )
            )
        )

        return (
            guided_feature
            + refinement
        )


class MambaVisionSmallENFNAMSOD(nn.Module):
    """
    MambaVision-S SOD with ENFNet-style NAM guidance.

    Only NAMLab hier_60 is used so that this experiment can be directly
    compared with the previous NESS/NAM experiment.
    """

    input_keys = (
        "image",
        "nam_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decoder_channels: int = 128,
        condition_channels: int = 64,
        affine_strength: float = 0.5,
    ) -> None:
        super().__init__()

        self.backbone: MambaVisionBackbone = (
            mamba_vision_small(
                pretrained_path=(
                    pretrained_path
                ),
            )
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

        self.nam_condition = (
            NAMConditionNetwork(
                condition_channels=(
                    condition_channels
                ),
            )
        )

        self.guidance_blocks = (
            nn.ModuleList(
                [
                    ENFGuidanceBlock(
                        feature_channels=(
                            decoder_channels
                        ),
                        condition_channels=(
                            condition_channels
                        ),
                        affine_strength=(
                            affine_strength
                        ),
                    )
                    for _ in range(4)
                ]
            )
        )

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
        torch.Tensor | list[torch.Tensor],
    ]:
        input_size = (
            image.shape[-2:]
        )

        stage1, stage2, stage3, stage4 = (
            self.backbone(
                image
            )
        )

        projected_features = [
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
        ]

        shared_condition = (
            self.nam_condition(
                nam_60=nam_60,
                output_size=(
                    projected_features[
                        0
                    ].shape[-2:]
                ),
            )
        )

        guided_features = [
            guidance_block(
                feature=feature,
                shared_condition=(
                    shared_condition
                ),
            )
            for guidance_block, feature
            in zip(
                self.guidance_blocks,
                projected_features,
            )
        ]

        (
            feature1,
            feature2,
            feature3,
            feature4,
        ) = guided_features

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
        }


def build_model() -> (
    MambaVisionSmallENFNAMSOD
):
    return MambaVisionSmallENFNAMSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
        condition_channels=64,
        affine_strength=0.5,
    )