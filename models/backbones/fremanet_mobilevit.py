from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


FREMANET_MOBILEVIT_OUT_CHANNELS = (64, 96, 128, 160)


def _make_divisible(
    value: int | float,
    divisor: int = 8,
    min_value: int | None = None,
) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(
        min_value,
        int(value + divisor / 2) // divisor * divisor,
    )
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = False,
        use_norm: bool = True,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential()
        self.block.add_module(
            "conv",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups,
                padding=padding,
                bias=bias,
            ),
        )
        if use_norm:
            self.block.add_module(
                "norm",
                nn.BatchNorm2d(
                    out_channels,
                    momentum=0.1,
                ),
            )
        if use_act:
            self.block.add_module("act", nn.SiLU())

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int | float,
        skip_connection: bool = True,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        hidden_dim = _make_divisible(
            int(round(in_channels * expand_ratio)),
            8,
        )
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module(
                "exp_1x1",
                ConvLayer(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                ),
            )
        self.block.add_module(
            "conv_3x3",
            ConvLayer(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=stride,
                groups=hidden_dim,
            ),
        )
        self.block.add_module(
            "red_1x1",
            ConvLayer(
                hidden_dim,
                out_channels,
                kernel_size=1,
                use_act=False,
            ),
        )
        self.use_residual = (
            stride == 1
            and in_channels == out_channels
            and skip_connection
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.block(x)
        if self.use_residual:
            return x + output
        return output


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(
            embed_dim,
            embed_dim * 3,
            bias=True,
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv_proj(x).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.transpose(1, 3).contiguous()
        query, key, value = (
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
        )
        attention = torch.matmul(
            query * self.scale,
            key.transpose(-1, -2),
        )
        attention = self.attn_dropout(
            self.softmax(attention)
        )
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(
            batch,
            tokens,
            channels,
        )
        return self.out_proj(output)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
    ) -> None:
        super().__init__()
        self.pre_norm_mha = nn.Sequential(
            nn.LayerNorm(embed_dim),
            MultiHeadAttention(
                embed_dim,
                num_heads,
                attn_dropout=attn_dropout,
            ),
            nn.Dropout(dropout),
        )
        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, ffn_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_dim, embed_dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pre_norm_mha(x)
        return x + self.pre_norm_ffn(x)


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        transformer_blocks: int,
        num_heads: int,
        patch_h: int = 2,
        patch_w: int = 2,
        dropout: float = 0.1,
        ffn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.local_rep = nn.Sequential()
        self.local_rep.add_module(
            "conv_3x3",
            ConvLayer(
                in_channels,
                in_channels,
                kernel_size=3,
            ),
        )
        self.local_rep.add_module(
            "conv_1x1",
            ConvLayer(
                in_channels,
                transformer_dim,
                kernel_size=1,
                use_norm=False,
                use_act=False,
            ),
        )
        global_layers: list[nn.Module] = [
            TransformerEncoder(
                transformer_dim,
                ffn_dim,
                num_heads,
                attn_dropout,
                dropout,
                ffn_dropout,
            )
            for _ in range(transformer_blocks)
        ]
        global_layers.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_layers)
        self.conv_proj = ConvLayer(
            transformer_dim,
            in_channels,
            kernel_size=1,
        )
        self.fusion = ConvLayer(
            in_channels * 2,
            in_channels,
            kernel_size=3,
        )
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = patch_h * patch_w

    def _unfold(
        self,
        x: Tensor,
    ) -> tuple[Tensor, dict[str, object]]:
        batch, channels, orig_h, orig_w = x.shape
        new_h = int(
            math.ceil(orig_h / self.patch_h)
            * self.patch_h
        )
        new_w = int(
            math.ceil(orig_w / self.patch_w)
            * self.patch_w
        )
        interpolate = (
            new_h != orig_h
            or new_w != orig_w
        )
        if interpolate:
            x = F.interpolate(
                x,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )

        patches_h = new_h // self.patch_h
        patches_w = new_w // self.patch_w
        total_patches = patches_h * patches_w
        x = x.reshape(
            batch * channels * patches_h,
            self.patch_h,
            patches_w,
            self.patch_w,
        )
        x = x.transpose(1, 2)
        x = x.reshape(
            batch,
            channels,
            total_patches,
            self.patch_area,
        )
        x = x.transpose(1, 3)
        x = x.reshape(
            batch * self.patch_area,
            total_patches,
            channels,
        )
        info: dict[str, object] = {
            "orig_size": (orig_h, orig_w),
            "batch": batch,
            "interpolate": interpolate,
            "total_patches": total_patches,
            "patches_h": patches_h,
            "patches_w": patches_w,
        }
        return x, info

    def _fold(
        self,
        x: Tensor,
        info: dict[str, object],
    ) -> Tensor:
        batch = int(info["batch"])
        total_patches = int(info["total_patches"])
        patches_h = int(info["patches_h"])
        patches_w = int(info["patches_w"])
        x = x.contiguous().view(
            batch,
            self.patch_area,
            total_patches,
            -1,
        )
        channels = x.shape[-1]
        x = x.transpose(1, 3)
        x = x.reshape(
            batch * channels * patches_h,
            patches_w,
            self.patch_h,
            self.patch_w,
        )
        x = x.transpose(1, 2)
        x = x.reshape(
            batch,
            channels,
            patches_h * self.patch_h,
            patches_w * self.patch_w,
        )
        if bool(info["interpolate"]):
            x = F.interpolate(
                x,
                size=info["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        feature = self.local_rep(x)
        patches, info = self._unfold(feature)
        patches = self.global_rep(patches)
        feature = self._fold(patches, info)
        feature = self.conv_proj(feature)
        return self.fusion(
            torch.cat((residual, feature), dim=1)
        )


class FreMaNetMobileViTS(nn.Module):
    """MobileViT-S feature extractor matching FreMaNet's official stage contract."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = ConvLayer(
            3,
            16,
            kernel_size=3,
            stride=2,
        )
        self.layer_1 = self._make_mobilenet_layer(
            16,
            32,
            num_blocks=1,
            stride=1,
            expand_ratio=4,
        )
        self.layer_2 = self._make_mobilenet_layer(
            32,
            64,
            num_blocks=3,
            stride=2,
            expand_ratio=4,
        )
        self.layer_3 = self._make_mobilevit_layer(
            64,
            96,
            transformer_dim=144,
            ffn_dim=288,
            transformer_blocks=2,
            stride=2,
        )
        self.layer_4 = self._make_mobilevit_layer(
            96,
            128,
            transformer_dim=192,
            ffn_dim=384,
            transformer_blocks=4,
            stride=2,
        )
        self.layer_5 = self._make_mobilevit_layer(
            128,
            160,
            transformer_dim=240,
            ffn_dim=480,
            transformer_blocks=3,
            stride=2,
        )
        self.apply(self._init_parameters)

    @staticmethod
    def _make_mobilenet_layer(
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        expand_ratio: int,
    ) -> nn.Sequential:
        blocks: list[nn.Module] = []
        current = in_channels
        for index in range(num_blocks):
            block_stride = stride if index == 0 else 1
            blocks.append(
                InvertedResidual(
                    current,
                    out_channels,
                    stride=block_stride,
                    expand_ratio=expand_ratio,
                )
            )
            current = out_channels
        return nn.Sequential(*blocks)

    @staticmethod
    def _make_mobilevit_layer(
        in_channels: int,
        out_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        transformer_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            InvertedResidual(
                in_channels,
                out_channels,
                stride=stride,
                expand_ratio=4,
            ),
            MobileViTBlock(
                out_channels,
                transformer_dim,
                ffn_dim,
                transformer_blocks,
                num_heads=4,
                patch_h=2,
                patch_w=2,
                dropout=0.1,
                ffn_dropout=0.0,
                attn_dropout=0.0,
            ),
        )

    @staticmethod
    def _init_parameters(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            if module.weight is not None:
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(
            module,
            (nn.LayerNorm, nn.BatchNorm2d),
        ):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            if module.weight is not None:
                nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        stage1 = x
        x = self.layer_3(x)
        stage2 = x
        x = self.layer_4(x)
        stage3 = x
        x = self.layer_5(x)
        stage4 = x
        return [stage1, stage2, stage3, stage4]


def _extract_state_dict(checkpoint: object) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "MobileViT checkpoint must be a state-dict-like object."
        )

    for key in (
        "state_dict",
        "model",
        "model_state_dict",
    ):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break

    state_dict: dict[str, Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, Tensor):
            continue
        normalized_key = str(key)
        for prefix in (
            "module.",
            "model.",
            "backbone.",
        ):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        state_dict[normalized_key] = value
    return state_dict


def build_fremanet_mobilevit_s(
    pretrained_path: str | Path | None = None,
) -> FreMaNetMobileViTS:
    model = FreMaNetMobileViTS()
    if pretrained_path is None:
        return model

    path = Path(pretrained_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"FreMaNet MobileViT-S checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = _extract_state_dict(checkpoint)
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state
        and value.shape == model_state[key].shape
    }
    model.load_state_dict(compatible, strict=False)
    return model


__all__ = [
    "FREMANET_MOBILEVIT_OUT_CHANNELS",
    "FreMaNetMobileViTS",
    "build_fremanet_mobilevit_s",
]
