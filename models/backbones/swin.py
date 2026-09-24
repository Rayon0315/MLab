from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import swin_t


class SwinTinyBackbone(nn.Module):
    """
    Swin-T backbone adapted to MLab's four-stage dense-feature interface.

    Output:
        stage1: [B,  96, H/4,  W/4]
        stage2: [B, 192, H/8,  W/8]
        stage3: [B, 384, H/16, W/16]
        stage4: [B, 768, H/32, W/32]
    """

    def __init__(
        self,
        pretrained_path: str | Path | None = None,
    ) -> None:
        super().__init__()

        classification_model = swin_t(
            weights=None
        )

        if pretrained_path is not None:
            pretrained_path = Path(
                pretrained_path
            )

            if not pretrained_path.is_file():
                raise FileNotFoundError(
                    "Swin-T pretrained checkpoint "
                    f"not found: {pretrained_path}"
                )

            checkpoint = torch.load(
                pretrained_path,
                map_location="cpu",
                weights_only=True,
            )

            if (
                isinstance(checkpoint, dict)
                and "model" in checkpoint
                and isinstance(
                    checkpoint["model"],
                    dict,
                )
            ):
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            classification_model.load_state_dict(
                state_dict,
                strict=True,
            )

        self.features = classification_model.features

        self.out_channels = (
            96,
            192,
            384,
            768,
        )

        self.out_strides = (
            4,
            8,
            16,
            32,
        )

    @staticmethod
    def _to_nchw(
        feature: torch.Tensor,
    ) -> torch.Tensor:
        return feature.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x = self.features[0](x)

        x = self.features[1](x)
        stage1 = self._to_nchw(x)

        x = self.features[2](x)
        x = self.features[3](x)
        stage2 = self._to_nchw(x)

        x = self.features[4](x)
        x = self.features[5](x)
        stage3 = self._to_nchw(x)

        x = self.features[6](x)
        x = self.features[7](x)
        stage4 = self._to_nchw(x)

        return (
            stage1,
            stage2,
            stage3,
            stage4,
        )


def swin_tiny(
    pretrained_path: str | Path | None = None,
    **_: Any,
) -> SwinTinyBackbone:
    return SwinTinyBackbone(
        pretrained_path=pretrained_path
    )
