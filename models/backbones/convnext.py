# models/backbones/convnext.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(normalized_shape)
        )
        self.bias = nn.Parameter(
            torch.zeros(normalized_shape)
        )

        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (
            normalized_shape,
        )

        if data_format not in (
            "channels_last",
            "channels_first",
        ):
            raise ValueError(
                f"Unsupported data format: {data_format}"
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )

        mean = x.mean(
            dim=1,
            keepdim=True,
        )

        variance = (
            x - mean
        ).pow(2).mean(
            dim=1,
            keepdim=True,
        )

        x = (
            x - mean
        ) / torch.sqrt(
            variance + self.eps
        )

        return (
            self.weight[:, None, None] * x
            + self.bias[:, None, None]
        )


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )

        self.norm = LayerNorm(
            dim,
            eps=1e-6,
        )

        self.pwconv1 = nn.Linear(
            dim,
            4 * dim,
        )

        self.act = nn.GELU()

        self.pwconv2 = nn.Linear(
            4 * dim,
            dim,
        )

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value
                * torch.ones(dim)
            )
        else:
            self.gamma = None

        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.0
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)

        x = x.permute(
            0,
            2,
            3,
            1,
        )

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(
            0,
            3,
            1,
            2,
        )

        return residual + self.drop_path(x)


def unwrap_state_dict(
    checkpoint: dict,
) -> dict[str, torch.Tensor]:
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    prefixes = (
        "module.",
        "model.",
        "encoder.",
        "backbone.",
    )

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        cleaned_key = key

        prefix_removed = True

        while prefix_removed:
            prefix_removed = False

            for prefix in prefixes:
                if cleaned_key.startswith(prefix):
                    cleaned_key = cleaned_key[
                        len(prefix):
                    ]
                    prefix_removed = True
                    break

        cleaned_state_dict[
            cleaned_key
        ] = value

    return cleaned_state_dict


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt backbone for dense prediction.

    Outputs:
        stage1: stride 4
        stage2: stride 8
        stage3: stride 16
        stage4: stride 32
    """

    def __init__(
        self,
        depths: list[int],
        dims: list[int],
        in_chans: int = 3,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()

        if len(depths) != 4:
            raise ValueError(
                "ConvNeXtBackbone expects four stages."
            )

        if len(dims) != 4:
            raise ValueError(
                "ConvNeXtBackbone expects four channel dimensions."
            )

        self.out_channels = tuple(dims)

        self.out_strides = (
            4,
            8,
            16,
            32,
        )

        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(
                in_chans,
                dims[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm(
                dims[0],
                eps=1e-6,
                data_format="channels_first",
            ),
        )

        self.downsample_layers.append(
            stem
        )

        for stage_index in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(
                    dims[stage_index],
                    eps=1e-6,
                    data_format="channels_first",
                ),
                nn.Conv2d(
                    dims[stage_index],
                    dims[stage_index + 1],
                    kernel_size=2,
                    stride=2,
                ),
            )

            self.downsample_layers.append(
                downsample_layer
            )

        drop_path_values = [
            value.item()
            for value in torch.linspace(
                0,
                drop_path_rate,
                sum(depths),
            )
        ]

        self.stages = nn.ModuleList()

        depth_offset = 0

        for stage_index in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[stage_index],
                        drop_path=drop_path_values[
                            depth_offset + block_index
                        ],
                        layer_scale_init_value=(
                            layer_scale_init_value
                        ),
                    )
                    for block_index
                    in range(depths[stage_index])
                ]
            )

            self.stages.append(stage)

            depth_offset += depths[
                stage_index
            ]

        # Kept for compatibility with the official
        # ImageNet classification checkpoint.
        self.norm = nn.LayerNorm(
            dims[-1],
            eps=1e-6,
        )

        self.apply(
            self._init_weights
        )

    @staticmethod
    def _init_weights(
        module: nn.Module,
    ) -> None:
        if isinstance(
            module,
            (nn.Conv2d, nn.Linear),
        ):
            nn.init.trunc_normal_(
                module.weight,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        features = []

        for stage_index in range(4):
            x = self.downsample_layers[
                stage_index
            ](x)

            x = self.stages[
                stage_index
            ](x)

            features.append(x)

        return tuple(features)

    def load_pretrained(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "ConvNeXt pretrained checkpoint "
                f"not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = unwrap_state_dict(
            checkpoint
        )

        state_dict = {
            key: value
            for key, value
            in state_dict.items()
            if not key.startswith("head.")
        }

        self.load_state_dict(
            state_dict,
            strict=True,
        )


def convnext_small(
    pretrained_path: str | Path | None = None,
    **kwargs: Any,
) -> ConvNeXtBackbone:
    model = ConvNeXtBackbone(
        depths=kwargs.pop(
            "depths",
            [3, 3, 27, 3],
        ),
        dims=kwargs.pop(
            "dims",
            [96, 192, 384, 768],
        ),
        drop_path_rate=kwargs.pop(
            "drop_path_rate",
            0.4,
        ),
        layer_scale_init_value=kwargs.pop(
            "layer_scale_init_value",
            1e-6,
        ),
        **kwargs,
    )

    if pretrained_path is not None:
        model.load_pretrained(
            checkpoint_path=pretrained_path,
        )

    return model