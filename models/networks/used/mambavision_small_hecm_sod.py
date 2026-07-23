# models/networks/mambavision_small_hecm_sod.py
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.mambavision import (
    MambaVisionBackbone,
    mamba_vision_small,
)
from models.components.nam_blocks import (
    EdgeGuidedSaliencyFusion,
    HierarchicalEdgeConstructionModule,
)
from models.components.sod_blocks import (
    ConvNormAct,
    PredictionHead,
    PyramidContextBlock,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "mambavision"
    / "mambavision_small_1k.pth.tar"
)


class MambaVisionSmallHECMSOD(nn.Module):
    """
    MambaVision-Small SOD with Hierarchical Edge Construction Module.

    RGB features:
        feature1: stride 4
        feature2: stride 8
        feature3: stride 16
        feature4: stride 32

    NAM hierarchy assignment:
        hier_20 -> edge3, stride 16
        hier_40 -> edge2, stride 8
        hier_60 -> edge1, stride 4
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

        self.backbone: MambaVisionBackbone = mamba_vision_small(
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
                for in_channels in self.backbone.out_channels
            ]
        )

        self.edge_construction = HierarchicalEdgeConstructionModule(
            decoder_channels
        )
        self.edge_head = PredictionHead(decoder_channels)

        self.context4 = PyramidContextBlock(decoder_channels)
        self.pred4 = PredictionHead(decoder_channels)

        self.fusion3 = EdgeGuidedSaliencyFusion(decoder_channels)
        self.pred3 = PredictionHead(decoder_channels)

        self.fusion2 = EdgeGuidedSaliencyFusion(decoder_channels)
        self.pred2 = PredictionHead(decoder_channels)

        self.fusion1 = EdgeGuidedSaliencyFusion(decoder_channels)
        self.pred1 = PredictionHead(decoder_channels)

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

        stage1, stage2, stage3, stage4 = self.backbone(image)
        feature1, feature2, feature3, feature4 = [
            projection(feature)
            for projection, feature in zip(
                self.projections,
                (stage1, stage2, stage3, stage4),
            )
        ]

        edge1, edge2, edge3, _ = self.edge_construction(
            feature1=feature1,
            feature2=feature2,
            feature3=feature3,
            feature4=feature4,
            nam_20=nam_20,
            nam_40=nam_40,
            nam_60=nam_60,
        )

        decoded4 = self.context4(feature4)
        prediction4 = self.pred4(decoded4)

        decoded3 = self.fusion3(
            low_feature=feature3,
            high_feature=decoded4,
            guide_logits=prediction4,
            edge_feature=edge3,
        )
        prediction3 = self.pred3(decoded3)

        decoded2 = self.fusion2(
            low_feature=feature2,
            high_feature=decoded3,
            guide_logits=prediction3,
            edge_feature=edge2,
        )
        prediction2 = self.pred2(decoded2)

        decoded1 = self.fusion1(
            low_feature=feature1,
            high_feature=decoded2,
            guide_logits=prediction2,
            edge_feature=edge1,
        )
        prediction1 = self.pred1(decoded1)
        edge_prediction = self.edge_head(edge1)

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


def build_model() -> MambaVisionSmallHECMSOD:
    return MambaVisionSmallHECMSOD(
        pretrained_path=PRETRAINED_PATH,
        decoder_channels=128,
    )
