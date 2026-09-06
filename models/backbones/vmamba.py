# models/backbones/vmamba.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.backbones.vmamba_official import Backbone_VSSM


class VMambaSmallBackbone(Backbone_VSSM):
    """
    VMamba-S [s2l15] backbone adapted to MLab.

    Input:
        [B, 3, H, W]

    Output:
        (
            stage1,  # [B,  96, H/4,  W/4]
            stage2,  # [B, 192, H/8,  W/8]
            stage3,  # [B, 384, H/16, W/16]
            stage4,  # [B, 768, H/32, W/32]
        )
    """

    def __init__(
        self,
        pretrained_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        if pretrained_path is not None:
            pretrained_path = Path(pretrained_path)

            if not pretrained_path.is_file():
                raise FileNotFoundError(
                    f"VMamba pretrained checkpoint not found: "
                    f"{pretrained_path}"
                )

            pretrained = str(pretrained_path)
        else:
            pretrained = None

        super().__init__(
            # dense prediction outputs
            out_indices=(0, 1, 2, 3),

            # official VMamba-S [s2l15]
            dims=96,
            depths=[2, 2, 15, 2],

            patch_size=4,
            in_chans=3,

            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",

            forward_type="v05_noz",

            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,

            drop_path_rate=0.3,

            patch_norm=True,
            norm_layer="ln2d",

            downsample_version="v3",
            patchembed_version="v2",

            use_checkpoint=False,
            posembed=False,
            imgsize=224,

            pretrained=pretrained,

            **kwargs,
        )

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

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        features = super().forward(x)

        return tuple(features)


def vmamba_small(
    pretrained_path: str | Path | None = None,
    **kwargs: Any,
) -> VMambaSmallBackbone:
    return VMambaSmallBackbone(
        pretrained_path=pretrained_path,
        **kwargs,
    )


vmamba_S = vmamba_small