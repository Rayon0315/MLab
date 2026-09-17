# models/networks/cfmnet_iram_x_base.py

from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn

from models.networks.cfmnet_unetformer_sod import (
    CFMNetUNetFormerSOD,
    PRETRAINED_PATH,
)


class CFMNetIRAMXUNetFormerSOD(
    CFMNetUNetFormerSOD
):
    """
    Canonical CFMNet fused pyramid
        -> one IRAM + X neck
        -> unchanged canonical UNetFormer decoder
    """

    def __init__(
        self,
        pretrained_path: str
        | Path
        | None,
        neck: nn.Module,
        decode_channels: int = 64,
        window_size: int = 8,
    ) -> None:
        super().__init__(
            pretrained_path=(
                pretrained_path
            ),
            decode_channels=(
                decode_channels
            ),
            window_size=(
                window_size
            ),
        )

        self.neck = neck

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[
        str,
        torch.Tensor
        | list[
            torch.Tensor
        ],
    ]:
        output_size = (
            image.shape[-2:]
        )

        features = list(
            self.backbone(
                image
            )
        )

        features = self.neck(
            features
        )

        (
            prediction,
            auxiliary,
        ) = self.decoder(
            features=features,
            output_size=output_size,
        )

        outputs: dict[
            str,
            torch.Tensor
            | list[
                torch.Tensor
            ],
        ] = {
            "pred": prediction,
        }

        if auxiliary is not None:
            outputs[
                "aux"
            ] = [
                auxiliary
            ]

        return outputs


def resolve_pretrained_path(
) -> Path | None:
    if PRETRAINED_PATH.exists():
        return PRETRAINED_PATH

    warnings.warn(
        (
            "CFMNet ImageNet-1K checkpoint "
            "was not found at "
            f"{PRETRAINED_PATH}. "
            "The backbone will train "
            "from scratch."
        ),
        RuntimeWarning,
    )

    return None
