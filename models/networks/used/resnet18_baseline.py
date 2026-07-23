# models/networks/resnet18_baseline.py

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class ConvBNReLU(nn.Sequential):
    """3×3 convolution followed by BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResNet18Baseline(nn.Module):
    """
    Minimal ResNet18-based RGB SOD network.

    Input:
        image: Tensor[B, 3, H, W]

    Output:
        {
            "pred": Tensor[B, 1, H, W]
        }

    `pred` contains logits. Sigmoid is not applied inside the network.
    """

    def __init__(self) -> None:
        super().__init__()

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT,
        )

        # ResNet stem: output resolution is approximately 1/4 of input.
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )

        # Feature channels:
        # layer1: 64,  resolution 1/4
        # layer2: 128, resolution 1/8
        # layer3: 256, resolution 1/16
        # layer4: 512, resolution 1/32
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        decoder_channels = 64

        # Convert all ResNet feature levels to the same channel count.
        self.lateral1 = nn.Conv2d(
            64,
            decoder_channels,
            kernel_size=1,
        )
        self.lateral2 = nn.Conv2d(
            128,
            decoder_channels,
            kernel_size=1,
        )
        self.lateral3 = nn.Conv2d(
            256,
            decoder_channels,
            kernel_size=1,
        )
        self.lateral4 = nn.Conv2d(
            512,
            decoder_channels,
            kernel_size=1,
        )

        # Refine fused features after top-down addition.
        self.fuse3 = ConvBNReLU(
            decoder_channels,
            decoder_channels,
        )
        self.fuse2 = ConvBNReLU(
            decoder_channels,
            decoder_channels,
        )
        self.fuse1 = ConvBNReLU(
            decoder_channels,
            decoder_channels,
        )

        self.prediction = nn.Conv2d(
            decoder_channels,
            1,
            kernel_size=1,
        )

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        input_size = image.shape[-2:]

        stem_feature = self.stem(image)

        feature1 = self.layer1(stem_feature)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)

        pyramid4 = self.lateral4(feature4)

        pyramid3 = self.lateral3(feature3)
        pyramid3 = pyramid3 + F.interpolate(
            pyramid4,
            size=feature3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pyramid3 = self.fuse3(pyramid3)

        pyramid2 = self.lateral2(feature2)
        pyramid2 = pyramid2 + F.interpolate(
            pyramid3,
            size=feature2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pyramid2 = self.fuse2(pyramid2)

        pyramid1 = self.lateral1(feature1)
        pyramid1 = pyramid1 + F.interpolate(
            pyramid2,
            size=feature1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pyramid1 = self.fuse1(pyramid1)

        prediction = self.prediction(pyramid1)

        prediction = F.interpolate(
            prediction,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": prediction,
        }


def build_model() -> nn.Module:
    """Build and return the complete network."""
    return ResNet18Baseline()