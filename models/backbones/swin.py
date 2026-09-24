from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import swin_t


class SwinTinyBackbone(nn.Module):
    """
    Swin-T adapted to MLab's four-stage dense-prediction interface.

    Dense-output convention:
        - extract the output of each Swin stage before the next PatchMerging
        - apply a dedicated LayerNorm to every exported stage
        - return NCHW feature maps

    This follows the dense-prediction Swin convention used by the official
    object-detection implementation, where norm0/norm1/norm2/norm3 are
    applied to the four exported stage features.

    Input:
        [B, 3, H, W]

    Output:
        stage1: [B,  96, H/4,  W/4]
        stage2: [B, 192, H/8,  W/8]
        stage3: [B, 384, H/16, W/16]
        stage4: [B, 768, H/32, W/32]
    """

    out_channels = (
        96,
        192,
        384,
        768,
    )

    out_strides = (
        4,
        8,
        16,
        32,
    )

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
                    "Swin-T pretrained checkpoint not found: "
                    f"{pretrained_path}"
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

        # torchvision Swin feature layout:
        #   0 patch embedding
        #   1 stage1
        #   2 patch merging
        #   3 stage2
        #   4 patch merging
        #   5 stage3
        #   6 patch merging
        #   7 stage4
        self.features = (
            classification_model.features
        )

        # Dense-prediction output norms.
        # Stage1~3 have no classification counterpart, so they start as
        # identity LayerNorms (weight=1, bias=0), exactly the standard
        # initialization for LayerNorm and are fine-tuned by the SOD task.
        #
        # Stage4 DOES have a classification counterpart:
        # torchvision's classification forward applies model.norm after
        # the last Swin stage. Copy those pretrained parameters so the
        # deepest dense feature starts from the pretrained representation.
        self.output_norms = nn.ModuleList(
            [
                nn.LayerNorm(
                    96,
                    eps=1e-5,
                ),
                nn.LayerNorm(
                    192,
                    eps=1e-5,
                ),
                nn.LayerNorm(
                    384,
                    eps=1e-5,
                ),
                nn.LayerNorm(
                    768,
                    eps=1e-5,
                ),
            ]
        )

        with torch.no_grad():
            self.output_norms[3].weight.copy_(
                classification_model.norm.weight
            )
            self.output_norms[3].bias.copy_(
                classification_model.norm.bias
            )

    @staticmethod
    def _to_nchw(
        feature: torch.Tensor,
    ) -> torch.Tensor:
        # torchvision Swin intermediate tensors are NHWC.
        return feature.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

    def _export_stage(
        self,
        feature: torch.Tensor,
        stage_index: int,
    ) -> torch.Tensor:
        feature = self.output_norms[
            stage_index
        ](
            feature
        )

        return self._to_nchw(
            feature
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Patch embedding.
        x = self.features[0](x)

        # Stage 1.
        x = self.features[1](x)
        stage1 = self._export_stage(
            x,
            stage_index=0,
        )

        # Merge -> Stage 2.
        x = self.features[2](x)
        x = self.features[3](x)
        stage2 = self._export_stage(
            x,
            stage_index=1,
        )

        # Merge -> Stage 3.
        x = self.features[4](x)
        x = self.features[5](x)
        stage3 = self._export_stage(
            x,
            stage_index=2,
        )

        # Merge -> Stage 4.
        x = self.features[6](x)
        x = self.features[7](x)
        stage4 = self._export_stage(
            x,
            stage_index=3,
        )

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
