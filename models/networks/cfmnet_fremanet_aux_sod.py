from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn

from models.components.fremanet_blocks import FreMaNetAuxHead


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "cfmnet"
    / "cfmnet_imagenet1k.pth"
)
CFMNET_OUT_CHANNELS = (96, 192, 384, 768)


class CFMNetFreMaNetAuxSOD(nn.Module):
    input_keys = ("image",)

    def __init__(
        self,
        pretrained_path: str | Path | None = PRETRAINED_PATH,
        backbone: nn.Module | None = None,
        channel: int = 32,
    ) -> None:
        super().__init__()
        if backbone is None:
            from models.backbones.cfmnet import build_cfmnet

            resolved_path: str | Path | None = pretrained_path
            if (
                resolved_path is not None
                and not Path(resolved_path).is_file()
            ):
                warnings.warn(
                    (
                        "CFMNet ImageNet-1K checkpoint was not found at "
                        f"{resolved_path}; using random initialization."
                    ),
                    RuntimeWarning,
                )
                resolved_path = None

            backbone = build_cfmnet(
                pretrained_path=resolved_path,
            )

        self.backbone = backbone
        self.head = FreMaNetAuxHead(
            in_channels=CFMNET_OUT_CHANNELS,
            channel=channel,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        features = self.backbone(image)
        prediction, auxiliary = self.head(
            features,
            output_size=image.shape[-2:],
        )

        return {
            "pred": prediction,
            "aux": auxiliary,
        }


def build_model() -> CFMNetFreMaNetAuxSOD:
    return CFMNetFreMaNetAuxSOD()
