# models/networks/mambavision_small_nam_stage_attention_sod.py

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


class NAMConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class NAMDepthwiseResidualBlock(nn.Module):
    """
    轻量残差块。

    使用 depthwise convolution 提取局部边缘结构，
    再使用 pointwise convolution 完成通道混合。
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=channels,
            ),
        )

        self.act = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.act(
            x + self.block(x)
        )


class SharedNAMPyramidEncoder(nn.Module):
    """
    三个 NAM hierarchy 共享同一组编码器参数。

    输入：
        [B, 1, H, W]

    输出：
        feature1: [B, 48,  H/4,  W/4]
        feature2: [B, 64,  H/8,  W/8]
        feature3: [B, 96,  H/16, W/16]
        feature4: [B, 128, H/32, W/32]
    """

    out_channels = (
        48,
        64,
        96,
        128,
    )

    def __init__(self) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            NAMConvNormAct(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            NAMConvNormAct(
                in_channels=32,
                out_channels=48,
                kernel_size=3,
                stride=2,
            ),
            NAMDepthwiseResidualBlock(
                channels=48,
            ),
        )

        self.stage2 = nn.Sequential(
            NAMConvNormAct(
                in_channels=48,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            NAMDepthwiseResidualBlock(
                channels=64,
            ),
        )

        self.stage3 = nn.Sequential(
            NAMConvNormAct(
                in_channels=64,
                out_channels=96,
                kernel_size=3,
                stride=2,
            ),
            NAMDepthwiseResidualBlock(
                channels=96,
            ),
        )

        self.stage4 = nn.Sequential(
            NAMConvNormAct(
                in_channels=96,
                out_channels=128,
                kernel_size=3,
                stride=2,
            ),
            NAMDepthwiseResidualBlock(
                channels=128,
            ),
        )

    def forward(
        self,
        nam: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        feature1 = self.stem(nam)
        feature2 = self.stage2(feature1)
        feature3 = self.stage3(feature2)
        feature4 = self.stage4(feature3)

        return (
            feature1,
            feature2,
            feature3,
            feature4,
        )


class RGBConditionedNAMStageAttention(nn.Module):
    """
    RGB 条件多层级 NAM stage attention。

    每个 stage 执行：

        1. 使用当前 RGB feature 生成 NAM 选择上下文；
        2. 对 NAM20、NAM40、NAM60 生成逐像素 softmax 权重；
        3. 得到当前 stage 的动态 NAM feature；
        4. 计算 RGB-NAM channel attention；
        5. 计算 RGB-NAM spatial attention；
        6. 建模 RGB、NAM、乘积和差异特征；
        7. 通过残差方式写回当前 stage。

    融合后的 stage feature 会继续送入下一 stage。
    """

    def __init__(
        self,
        rgb_channels: int,
        nam_channels: int,
        initial_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()

        interaction_channels = (
            rgb_channels // 2
        )

        channel_hidden = max(
            16,
            rgb_channels // 4,
        )

        self.rgb_context = (
            NAMConvNormAct(
                in_channels=rgb_channels,
                out_channels=nam_channels,
                kernel_size=1,
                padding=0,
            )
        )

        self.hierarchy_context = nn.Sequential(
            NAMConvNormAct(
                in_channels=nam_channels * 4,
                out_channels=nam_channels,
                kernel_size=1,
                padding=0,
            ),
            NAMDepthwiseResidualBlock(
                channels=nam_channels,
            ),
        )

        self.hierarchy_logits = nn.Conv2d(
            nam_channels,
            3,
            kernel_size=1,
        )

        nn.init.zeros_(
            self.hierarchy_logits.weight
        )
        nn.init.zeros_(
            self.hierarchy_logits.bias
        )

        self.selected_nam_refine = (
            NAMDepthwiseResidualBlock(
                channels=nam_channels,
            )
        )

        self.nam_to_rgb = (
            NAMConvNormAct(
                in_channels=nam_channels,
                out_channels=rgb_channels,
                kernel_size=1,
                padding=0,
            )
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                rgb_channels * 2,
                channel_hidden,
                kernel_size=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                channel_hidden,
                rgb_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(
                4,
                1,
                kernel_size=7,
                padding=3,
            ),
            nn.Sigmoid(),
        )

        self.rgb_reduce = (
            NAMConvNormAct(
                in_channels=rgb_channels,
                out_channels=interaction_channels,
                kernel_size=1,
                padding=0,
            )
        )

        self.nam_reduce = (
            NAMConvNormAct(
                in_channels=rgb_channels,
                out_channels=interaction_channels,
                kernel_size=1,
                padding=0,
            )
        )

        self.interaction_fusion = nn.Sequential(
            NAMConvNormAct(
                in_channels=interaction_channels * 4,
                out_channels=rgb_channels,
                kernel_size=1,
                padding=0,
            ),
            NAMDepthwiseResidualBlock(
                channels=rgb_channels,
            ),
        )

        self.residual_scale = nn.Parameter(
            torch.tensor(
                initial_residual_scale,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        rgb_feature: torch.Tensor,
        nam_20: torch.Tensor,
        nam_40: torch.Tensor,
        nam_60: torch.Tensor,
    ) -> torch.Tensor:
        rgb_context = self.rgb_context(
            rgb_feature
        )

        hierarchy_context = (
            self.hierarchy_context(
                torch.cat(
                    [
                        rgb_context,
                        nam_20,
                        nam_40,
                        nam_60,
                    ],
                    dim=1,
                )
            )
        )

        hierarchy_weights = torch.softmax(
            self.hierarchy_logits(
                hierarchy_context
            ),
            dim=1,
        )

        selected_nam = (
            hierarchy_weights[:, 0:1]
            * nam_20
            + hierarchy_weights[:, 1:2]
            * nam_40
            + hierarchy_weights[:, 2:3]
            * nam_60
        )

        selected_nam = (
            self.selected_nam_refine(
                selected_nam
            )
        )

        nam_feature = self.nam_to_rgb(
            selected_nam
        )

        channel_attention = (
            self.channel_attention(
                torch.cat(
                    [
                        rgb_feature,
                        nam_feature,
                    ],
                    dim=1,
                )
            )
        )

        spatial_attention = (
            self.spatial_attention(
                torch.cat(
                    [
                        rgb_feature.mean(
                            dim=1,
                            keepdim=True,
                        ),
                        rgb_feature.amax(
                            dim=1,
                            keepdim=True,
                        ),
                        nam_feature.mean(
                            dim=1,
                            keepdim=True,
                        ),
                        nam_feature.amax(
                            dim=1,
                            keepdim=True,
                        ),
                    ],
                    dim=1,
                )
            )
        )

        rgb_reduced = self.rgb_reduce(
            rgb_feature
        )

        nam_reduced = self.nam_reduce(
            nam_feature
        )

        interaction_feature = (
            self.interaction_fusion(
                torch.cat(
                    [
                        rgb_reduced,
                        nam_reduced,
                        rgb_reduced
                        * nam_reduced,
                        torch.abs(
                            rgb_reduced
                            - nam_reduced
                        ),
                    ],
                    dim=1,
                )
            )
        )

        attention_feature = (
            interaction_feature
            * (1.0 + channel_attention)
            * (1.0 + spatial_attention)
        )

        residual_scale = torch.tanh(
            self.residual_scale
        )

        return (
            rgb_feature
            + residual_scale
            * attention_feature
        )


class MambaVisionSmallNAMStageAttentionSOD(
    nn.Module
):
    """
    MambaVision-Small + NAM stage attention。

    RGB：
        保持标准三通道输入；
        PatchEmbed 完整继承 ImageNet 预训练权重。

    NAM：
        NAM20、NAM40、NAM60 分别经过共享金字塔编码器。

    Encoder：
        四个 MambaVision stage 后均加入
        RGB-conditioned hierarchy attention。

        每个 stage 的融合结果继续进入下一 stage。

    Decoder：
        与 mambavision_small_sod baseline 完全一致。

    输出：
        pred + aux

    不包含：
        六通道输入；
        decoder NAM injection；
        edge prediction；
        edge loss。
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

        self.nam_encoder = (
            SharedNAMPyramidEncoder()
        )

        self.stage_attentions = nn.ModuleList(
            [
                RGBConditionedNAMStageAttention(
                    rgb_channels=rgb_channels,
                    nam_channels=nam_channels,
                    initial_residual_scale=0.1,
                )
                for (
                    rgb_channels,
                    nam_channels,
                ) in zip(
                    self.backbone.out_channels,
                    self.nam_encoder.out_channels,
                )
            ]
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

    def _forward_backbone(
        self,
        image: torch.Tensor,
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
    ) -> tuple[torch.Tensor, ...]:
        x = self.backbone.patch_embed(
            image
        )

        stage_features = []

        for stage_index, (
            level,
            stage_attention,
            nam_20_feature,
            nam_40_feature,
            nam_60_feature,
        ) in enumerate(
            zip(
                self.backbone.levels,
                self.stage_attentions,
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

            stage_feature = stage_attention(
                rgb_feature=stage_feature,
                nam_20=nam_20_feature,
                nam_40=nam_40_feature,
                nam_60=nam_60_feature,
            )

            stage_features.append(
                stage_feature
            )

            if level.downsample is not None:
                x = level.downsample(
                    stage_feature
                )
            else:
                x = stage_feature

        return tuple(stage_features)

    def forward(
        self,
        image: torch.Tensor,
        nam_20: torch.Tensor,
        nam_40: torch.Tensor,
        nam_60: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor | list[torch.Tensor],
    ]:
        input_size = image.shape[-2:]

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
            stage1,
            stage2,
            stage3,
            stage4,
        ) = self._forward_backbone(
            image=image,
            nam_20_features=nam_20_features,
            nam_40_features=nam_40_features,
            nam_60_features=nam_60_features,
        )

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


def build_model(
) -> MambaVisionSmallNAMStageAttentionSOD:
    return (
        MambaVisionSmallNAMStageAttentionSOD(
            pretrained_path=PRETRAINED_PATH,
            decoder_channels=128,
        )
    )