# models/networks/mambavision_small_nam_aggressive_sod.py

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.mambavision import (
    MambaVisionBackbone,
    mamba_vision_small,
)
from models.components.nam_attention import (
    NAMDecoderInjection,
    NAMHierarchyEncoder,
    HierarchicalNAMAttention,
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


class MambaVisionSmallNAMAggressiveSOD(
    nn.Module
):
    """
    Aggressive NAMLab fusion network.

    Fusion paths:
        1. RGB + NAM20 + NAM40 + NAM60 input concat.
        2. Shared multi-scale NAM hierarchy encoder.
        3. RGB-conditioned hierarchy attention at all
           four MambaVision stages.
        4. Fused stage feature is passed into the next
           backbone stage.
        5. Selected NAM feature is injected into all
           four decoder stages.
        6. NAM-aware edge prediction with edge loss.
    """

    input_keys = (
        "image",
        "nam_20",
        "nam_40",
        "nam_60",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()

        self.backbone: MambaVisionBackbone = (
            mamba_vision_small(
                pretrained_path=pretrained_path,
            )
        )

        self._expand_input_convolution()

        self.nam_encoder = (
            NAMHierarchyEncoder()
        )

        nam_channels = (
            self.nam_encoder.out_channels
        )

        self.encoder_nam_fusions = (
            nn.ModuleList(
                [
                    HierarchicalNAMAttention(
                        rgb_channels=rgb_channels,
                        nam_channels=nam_channels_at_scale,
                    )
                    for (
                        rgb_channels,
                        nam_channels_at_scale,
                    ) in zip(
                        self.backbone.out_channels,
                        nam_channels,
                    )
                ]
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

        self.context4 = PyramidContextBlock(
            decoder_channels
        )

        self.decoder_nam4 = (
            NAMDecoderInjection(
                decoder_channels,
                nam_channels[3],
            )
        )

        self.pred4 = PredictionHead(
            decoder_channels
        )

        self.fusion3 = (
            SaliencyGuidedFusion(
                decoder_channels
            )
        )

        self.decoder_nam3 = (
            NAMDecoderInjection(
                decoder_channels,
                nam_channels[2],
            )
        )

        self.pred3 = PredictionHead(
            decoder_channels
        )

        self.fusion2 = (
            SaliencyGuidedFusion(
                decoder_channels
            )
        )

        self.decoder_nam2 = (
            NAMDecoderInjection(
                decoder_channels,
                nam_channels[1],
            )
        )

        self.pred2 = PredictionHead(
            decoder_channels
        )

        self.fusion1 = (
            SaliencyGuidedFusion(
                decoder_channels
            )
        )

        self.boundary_refinement = (
            BoundaryRefinementBlock(
                decoder_channels
            )
        )

        self.decoder_nam1 = (
            NAMDecoderInjection(
                decoder_channels,
                nam_channels[0],
            )
        )

        self.final_refinement = nn.Sequential(
            ResidualConvBlock(
                decoder_channels
            ),
            ResidualConvBlock(
                decoder_channels
            ),
        )

        self.pred1 = PredictionHead(
            decoder_channels
        )

        self.edge_fusion = nn.Sequential(
            ConvNormAct(
                decoder_channels * 2,
                decoder_channels,
            ),
            ResidualConvBlock(
                decoder_channels
            ),
        )

        self.edge_head = PredictionHead(
            decoder_channels
        )

    def _expand_input_convolution(
        self,
    ) -> None:
        rgb_conv: nn.Conv2d = (
            self.backbone
            .patch_embed
            .conv_down[0]
        )

        input_conv = nn.Conv2d(
            in_channels=6,
            out_channels=rgb_conv.out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            input_conv.weight[:, :3].copy_(
                rgb_conv.weight
            )

            input_conv.weight[:, 3:].zero_()

        self.backbone.patch_embed.conv_down[0] = (
            input_conv
        )

    def _forward_backbone(
        self,
        model_input: torch.Tensor,
        nam_20_features: tuple[
            torch.Tensor,
            ...,
        ],
        nam_40_features: tuple[
            torch.Tensor,
            ...,
        ],
        nam_60_features: tuple[
            torch.Tensor,
            ...,
        ],
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        x = self.backbone.patch_embed(
            model_input
        )

        stage_features = []
        selected_nam_features = []
        hierarchy_weights = []

        for stage_index, (
            level,
            nam_fusion,
            nam_20_feature,
            nam_40_feature,
            nam_60_feature,
        ) in enumerate(
            zip(
                self.backbone.levels,
                self.encoder_nam_fusions,
                nam_20_features,
                nam_40_features,
                nam_60_features,
            )
        ):
            stage_feature = (
                level.forward_blocks(x)
            )

            if (
                stage_index
                == len(
                    self.backbone.levels
                ) - 1
            ):
                stage_feature = (
                    self.backbone.norm(
                        stage_feature
                    )
                )

            (
                stage_feature,
                selected_nam,
                stage_hierarchy_weights,
            ) = nam_fusion(
                rgb_feature=stage_feature,
                nam_20=nam_20_feature,
                nam_40=nam_40_feature,
                nam_60=nam_60_feature,
            )

            stage_features.append(
                stage_feature
            )

            selected_nam_features.append(
                selected_nam
            )

            hierarchy_weights.append(
                stage_hierarchy_weights
            )

            if level.downsample is not None:
                x = level.downsample(
                    stage_feature
                )
            else:
                x = stage_feature

        return (
            tuple(stage_features),
            tuple(selected_nam_features),
            tuple(hierarchy_weights),
        )

    def forward(
        self,
        image: torch.Tensor,
        nam_20: torch.Tensor,
        nam_40: torch.Tensor,
        nam_60: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor
        | list[torch.Tensor],
    ]:
        input_size = image.shape[-2:]

        model_input = torch.cat(
            [
                image,
                nam_20,
                nam_40,
                nam_60,
            ],
            dim=1,
        )

        nam_20_features = self.nam_encoder(
            nam_20
        )

        nam_40_features = self.nam_encoder(
            nam_40
        )

        nam_60_features = self.nam_encoder(
            nam_60
        )

        (
            stage_features,
            selected_nam_features,
            _,
        ) = self._forward_backbone(
            model_input=model_input,
            nam_20_features=(
                nam_20_features
            ),
            nam_40_features=(
                nam_40_features
            ),
            nam_60_features=(
                nam_60_features
            ),
        )

        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = stage_features

        (
            selected_nam1,
            selected_nam2,
            selected_nam3,
            selected_nam4,
        ) = selected_nam_features

        (
            feature1,
            feature2,
            feature3,
            feature4,
        ) = [
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

        decoded4 = self.context4(
            feature4
        )

        (
            decoded4,
            decoder_nam4,
        ) = self.decoder_nam4(
            saliency_feature=decoded4,
            selected_nam=selected_nam4,
        )

        prediction4 = self.pred4(
            decoded4
        )

        decoded3 = self.fusion3(
            low_feature=feature3,
            high_feature=decoded4,
            guide_logits=prediction4,
        )

        (
            decoded3,
            decoder_nam3,
        ) = self.decoder_nam3(
            saliency_feature=decoded3,
            selected_nam=selected_nam3,
        )

        prediction3 = self.pred3(
            decoded3
        )

        decoded2 = self.fusion2(
            low_feature=feature2,
            high_feature=decoded3,
            guide_logits=prediction3,
        )

        (
            decoded2,
            decoder_nam2,
        ) = self.decoder_nam2(
            saliency_feature=decoded2,
            selected_nam=selected_nam2,
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

        (
            decoded1,
            decoder_nam1,
        ) = self.decoder_nam1(
            saliency_feature=decoded1,
            selected_nam=selected_nam1,
        )

        decoded1 = self.final_refinement(
            decoded1
        )

        prediction1 = self.pred1(
            decoded1
        )

        edge_feature = self.edge_fusion(
            torch.cat(
                [
                    decoded1,
                    decoder_nam1,
                ],
                dim=1,
            )
        )

        edge_prediction = self.edge_head(
            edge_feature
        )

        prediction1 = F.interpolate(
            prediction1,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        edge_prediction = F.interpolate(
            edge_prediction,
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
            "edge": edge_prediction,
        }


def build_model(
) -> MambaVisionSmallNAMAggressiveSOD:
    return MambaVisionSmallNAMAggressiveSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )