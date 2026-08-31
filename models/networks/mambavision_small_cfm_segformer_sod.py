# models/networks/mambavision_small_cfm_segformer_sod.py
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.mambavision import (
    MambaVisionBackbone,
    mamba_vision_small,
)
from models.backbones.cfmnet import CFMBlock
from models.components.segmentation_heads import SegFormerHead


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "mambavision"
    / "mambavision_small_1k.pth.tar"
)


def _init_cfm_weights(module: nn.Module) -> None:
    """Match the initialization used by the canonical CFMNet backbone."""
    if isinstance(
        module,
        (nn.Linear, nn.Conv1d, nn.Conv2d),
    ):
        nn.init.trunc_normal_(
            module.weight,
            std=0.02,
        )
        if module.bias is not None:
            nn.init.constant_(
                module.bias,
                0.0,
            )
    elif isinstance(
        module,
        (nn.LayerNorm, nn.GroupNorm),
    ):
        if module.bias is not None:
            nn.init.constant_(
                module.bias,
                0.0,
            )
        if module.weight is not None:
            nn.init.constant_(
                module.weight,
                1.0,
            )


class MambaVisionSmallCFMSegFormerSOD(nn.Module):
    """
    RGB-only MambaVision-S + CFM feature modeling + SegFormer head.

    Backbone:
        MambaVision-S ImageNet-1K pretrained
        stage1:  96 channels, stride 4
        stage2: 192 channels, stride 8
        stage3: 384 channels, stride 16
        stage4: 768 channels, stride 32

    CFM integration:
        one CFMBlock is applied to each native MambaVision stage output.
        No channel projection is inserted before CFM, so each block works
        directly on the native backbone representation.

    Decoder:
        standard SegFormer-style four-level parallel fusion head.

    Intentionally excluded:
        Region Mean / NAMLab
        Progressive decoder
        Dictionary / prototype routing
        refinement / boundary / edge branches
        auxiliary prediction heads
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

        self.backbone: MambaVisionBackbone = (
            mamba_vision_small(
                pretrained_path=pretrained_path,
            )
        )

        self.cfm_blocks = nn.ModuleList(
            [
                CFMBlock(
                    dim=channels,
                    stage=stage_index,
                    att_kernel=11,
                    mlp_ratio=2.0,
                    drop_path=0.0,
                    act_layer=nn.ReLU,
                    norm_layer=nn.BatchNorm2d,
                )
                for stage_index, channels
                in enumerate(
                    self.backbone.out_channels
                )
            ]
        )

        # CFMNet applies trunc-normal initialization to its convolutional
        # blocks. Because these four blocks are instantiated outside the
        # CFMNet backbone here, initialize only them explicitly.
        self.cfm_blocks.apply(
            _init_cfm_weights
        )

        self.decode_head = SegFormerHead(
            in_channels=self.backbone.out_channels,
            embed_dim=decoder_channels,
            dropout_ratio=0.1,
            num_classes=1,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]

        features = self.backbone(
            image
        )

        features = tuple(
            cfm_block(feature)
            for cfm_block, feature
            in zip(
                self.cfm_blocks,
                features,
            )
        )

        prediction = self.decode_head(
            features
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


def build_model() -> MambaVisionSmallCFMSegFormerSOD:
    return MambaVisionSmallCFMSegFormerSOD(
        pretrained_path=PRETRAINED_PATH,
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
        outputs = model(
            image=image
        )

    print(
        "pred:",
        outputs["pred"].shape,
    )
