# models/networks/mambavision_small_sk23_sod.py
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


class SemanticKernelGenerator(nn.Module):
    """
    Generate image-dependent semantic kernels from Stage4.

    SeaNet-style SKC:
        Stage4
        -> depthwise separable convolution
        -> channel projection
        -> adaptive average pooling
        -> semantic kernel
    """

    def __init__(
        self,
        in_channels: int,
        kernel_channels: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size

        self.depthwise = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size=3,
            groups=in_channels,
        )

        self.pointwise = ConvNormAct(
            in_channels,
            kernel_channels,
            kernel_size=1,
            padding=0,
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (kernel_size, kernel_size)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.pool(x)

        return x


def dynamic_depthwise_conv2d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    dilation: int,
) -> torch.Tensor:
    """
    Per-image, per-channel dynamic depthwise convolution.

    Args:
        x:
            [B, C, H, W]

        kernel:
            [B, C, K, K]

        dilation:
            Dilation rate.

    Returns:
        [B, C, H, W]
    """

    batch_size, channels, height, width = x.shape

    kernel_size = kernel.shape[-1]

    if kernel.shape[:2] != (
        batch_size,
        channels,
    ):
        raise ValueError(
            "Dynamic kernel shape does not match input feature shape."
        )

    padding = (
        dilation
        * (kernel_size - 1)
        // 2
    )

    grouped_input = x.reshape(
        1,
        batch_size * channels,
        height,
        width,
    )

    grouped_kernel = kernel.reshape(
        batch_size * channels,
        1,
        kernel_size,
        kernel_size,
    )

    output = F.conv2d(
        grouped_input,
        grouped_kernel,
        bias=None,
        stride=1,
        padding=padding,
        dilation=dilation,
        groups=batch_size * channels,
    )

    output = output.reshape(
        batch_size,
        channels,
        height,
        width,
    )

    return output


class DynamicSemanticMatching(nn.Module):
    """
    SeaNet-inspired spatial semantic matching.

    The semantic kernel generated from Stage4 is used as
    the depthwise convolution kernel of the current stage.

    Three receptive fields are used:
        dilation = 1
        dilation = 2
        dilation = 3

    The matched feature is fused and added back to the
    original backbone feature through a residual connection.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.dilations = (
            1,
            2,
            3,
        )

        self.fusion = ConvNormAct(
            channels,
            channels,
            kernel_size=1,
            padding=0,
        )

    def forward(
        self,
        feature: torch.Tensor,
        semantic_kernel: torch.Tensor,
    ) -> torch.Tensor:
        matched_features = [
            dynamic_depthwise_conv2d(
                feature,
                semantic_kernel,
                dilation=dilation,
            )
            for dilation in self.dilations
        ]

        matched = torch.stack(
            matched_features,
            dim=0,
        ).sum(
            dim=0
        )

        matched = self.fusion(
            matched
        )

        return feature + matched


class MambaVisionSmallSK23SOD(nn.Module):
    """
    MambaVision-S with Stage4-driven semantic matching.

    Backbone:
        Stage1:
            stride 4
            channels 96
            unchanged

        Stage2:
            stride 8
            channels 192
            enhanced by Stage4 semantic kernel K2

        Stage3:
            stride 16
            channels 384
            enhanced by Stage4 semantic kernel K3

        Stage4:
            stride 32
            channels 768
            generates K2 and K3

    Decoder:
        Same as the current MambaVision-S baseline.
    """

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

        self.kernel_generator2 = (
            SemanticKernelGenerator(
                in_channels=stage4_channels,
                kernel_channels=stage2_channels,
                kernel_size=5,
            )
        )

        self.kernel_generator3 = (
            SemanticKernelGenerator(
                in_channels=stage4_channels,
                kernel_channels=stage3_channels,
                kernel_size=5,
            )
        )

        self.semantic_matching2 = (
            DynamicSemanticMatching(
                channels=stage2_channels,
            )
        )

        self.semantic_matching3 = (
            DynamicSemanticMatching(
                channels=stage3_channels,
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
                in (
                    stage1_channels,
                    stage2_channels,
                    stage3_channels,
                    stage4_channels,
                )
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

    def forward(
        self,
        image: torch.Tensor,
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
        ) = self.backbone(image)

        # -----------------------------------------------
        # Stage4 -> semantic kernels
        # -----------------------------------------------

        kernel2 = self.kernel_generator2(
            stage4
        )

        kernel3 = self.kernel_generator3(
            stage4
        )

        # -----------------------------------------------
        # Stage4-conditioned semantic matching
        #
        # Stage2:
        #   [B, 192, H/8, H/8]
        #   <- K2 [B, 192, 5, 5]
        #
        # Stage3:
        #   [B, 384, H/16, H/16]
        #   <- K3 [B, 384, 5, 5]
        # -----------------------------------------------

        stage2 = self.semantic_matching2(
            feature=stage2,
            semantic_kernel=kernel2,
        )

        stage3 = self.semantic_matching3(
            feature=stage3,
            semantic_kernel=kernel3,
        )

        # -----------------------------------------------
        # Original baseline decoder starts here.
        # -----------------------------------------------

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

        decoded1 = self.boundary_refinement(
            shallow_feature=feature1,
            semantic_feature=feature2,
            saliency_feature=decoded1,
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


def build_model() -> MambaVisionSmallSK23SOD:
    return MambaVisionSmallSK23SOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )