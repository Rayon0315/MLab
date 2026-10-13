from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn

from models.backbones.fremanet_mobilevit import (
    FREMANET_MOBILEVIT_OUT_CHANNELS,
    build_fremanet_mobilevit_s,
)
from models.components.fremanet_blocks import FreMaNetHead


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "fremanet"
    / "mobilevit_s.pt"
)


class FreMaNetMobileViTSOD(nn.Module):
    input_keys = ("image",)

    def __init__(
        self,
        pretrained_path: str | Path | None = PRETRAINED_PATH,
        backbone: nn.Module | None = None,
        channel: int = 32,
    ) -> None:
        super().__init__()
        if backbone is None:
            resolved_path: str | Path | None = pretrained_path
            if (
                resolved_path is not None
                and not Path(resolved_path).is_file()
            ):
                warnings.warn(
                    (
                        "FreMaNet MobileViT-S checkpoint was not found at "
                        f"{resolved_path}; using random initialization."
                    ),
                    RuntimeWarning,
                )
                resolved_path = None
            backbone = build_fremanet_mobilevit_s(
                pretrained_path=resolved_path,
            )

        self.backbone = backbone
        self.head = FreMaNetHead(
            in_channels=FREMANET_MOBILEVIT_OUT_CHANNELS,
            channel=channel,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        prediction = self.head(
            features,
            output_size=image.shape[-2:],
        )
        return {"pred": prediction}


def build_model() -> FreMaNetMobileViTSOD:
    return FreMaNetMobileViTSOD()
