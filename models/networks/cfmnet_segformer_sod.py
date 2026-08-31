# models/networks/cfmnet_segformer_sod.py

from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
    build_cfmnet,
)
from models.components.segmentation_heads import (
    SegFormerHead,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "cfmnet"
    / "cfmnet_imagenet1k.pth"
)


class CFMNetSegFormerSOD(nn.Module):
    """
    CFMNet + SegFormer Head baseline for EORSSD.

    Backbone:
        official canonical CFMNet
        channels [96, 192, 384, 768]
        strides  [4, 8, 16, 32]

    Decoder:
        standard SegFormer-style parallel multilevel head

    This first reproduction baseline intentionally has:
        - RGB input only
        - no NAMLab / Region Mean
        - no Progressive Decoder
        - no refinement stage
        - no auxiliary saliency heads
    """

    input_keys = (
        "image",
    )

    def __init__(
        self,
        pretrained_path: str | Path | None,
        decoder_channels: int = 256,
    ) -> None:
        super().__init__()

        self.backbone = build_cfmnet(
            pretrained_path=(
                pretrained_path
            )
        )

        self.decode_head = (
            SegFormerHead(
                in_channels=(
                    CFMNET_OUT_CHANNELS
                ),
                embed_dim=(
                    decoder_channels
                ),
                dropout_ratio=0.1,
                num_classes=1,
            )
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor,
    ]:
        input_size = (
            image.shape[-2:]
        )

        features = self.backbone(
            image
        )

        prediction = (
            self.decode_head(
                features
            )
        )

        prediction = F.interpolate(
            prediction,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred": prediction,
        }


def build_model(
) -> CFMNetSegFormerSOD:
    pretrained_path = (
        PRETRAINED_PATH
        if PRETRAINED_PATH.exists()
        else None
    )

    if pretrained_path is None:
        warnings.warn(
            "CFMNet ImageNet-1K checkpoint was not found at "
            f"{PRETRAINED_PATH}. The backbone will train from scratch. "
            "The official downstream CFMNet experiments use ImageNet-1K "
            "pretraining.",
            RuntimeWarning,
        )

    return CFMNetSegFormerSOD(
        pretrained_path=(
            pretrained_path
        ),
        decoder_channels=256,
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

    with torch.no_grad():
        output = model(
            image=image
        )

    print(
        "pred:",
        output["pred"].shape,
    )

    with torch.no_grad():
        features = model.backbone(
            image
        )

    print(
        "features:",
        [
            tuple(
                feature.shape
            )
            for feature
            in features
        ],
    )
