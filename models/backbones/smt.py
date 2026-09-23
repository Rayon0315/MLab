# Adapted for MLab from the official SMT implementation:
# https://github.com/AFeng-x/SMT
# Original project is released under the MIT License.

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Per-sample stochastic depth, equivalent to timm DropPath for this use."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def _init_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        fan_out = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.out_channels
        )
        fan_out //= module.groups
        nn.init.normal_(
            module.weight,
            mean=0.0,
            std=math.sqrt(2.0 / fan_out),
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DWConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size, _, channels = x.shape
        x = x.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()

        out_features = out_features or in_features

        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
        )
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(
            hidden_features,
            out_features,
        )
        self.drop = nn.Dropout(drop)

        self.apply(_init_module)

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(
            x + self.dwconv(x, height, width)
        )
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        ca_num_heads: int,
        sa_num_heads: int,
        qkv_bias: bool,
        ca_attention: int,
        expand_ratio: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        self.ca_attention = int(ca_attention)
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.sa_num_heads = sa_num_heads

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.ca_attention == 1:
            if dim % ca_num_heads != 0:
                raise ValueError(
                    f"dim={dim} must be divisible by ca_num_heads={ca_num_heads}."
                )

            self.split_groups = dim // ca_num_heads
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
            self.s = nn.Linear(dim, dim, bias=qkv_bias)

            for head_index in range(ca_num_heads):
                kernel_size = 3 + 2 * head_index
                local_conv = nn.Conv2d(
                    self.split_groups,
                    self.split_groups,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    stride=1,
                    groups=self.split_groups,
                )
                setattr(
                    self,
                    f"local_conv_{head_index + 1}",
                    local_conv,
                )

            self.proj0 = nn.Conv2d(
                dim,
                dim * expand_ratio,
                kernel_size=1,
                groups=self.split_groups,
            )
            self.bn = nn.BatchNorm2d(
                dim * expand_ratio
            )
            self.proj1 = nn.Conv2d(
                dim * expand_ratio,
                dim,
                kernel_size=1,
            )
        else:
            if dim % sa_num_heads != 0:
                raise ValueError(
                    f"dim={dim} must be divisible by sa_num_heads={sa_num_heads}."
                )

            head_dim = dim // sa_num_heads
            self.scale = head_dim ** -0.5
            self.q = nn.Linear(
                dim,
                dim,
                bias=qkv_bias,
            )
            self.kv = nn.Linear(
                dim,
                dim * 2,
                bias=qkv_bias,
            )
            self.attn_drop = nn.Dropout(
                attn_drop
            )
            self.local_conv = nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
            )

        self.apply(_init_module)

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size, token_count, channels = x.shape

        if self.ca_attention == 1:
            value = self.v(x)
            spatial = self.s(x).reshape(
                batch_size,
                height,
                width,
                self.ca_num_heads,
                channels // self.ca_num_heads,
            ).permute(3, 0, 4, 1, 2)

            spatial_outputs: list[torch.Tensor] = []

            for head_index in range(
                self.ca_num_heads
            ):
                local_conv = getattr(
                    self,
                    f"local_conv_{head_index + 1}",
                )
                spatial_head = local_conv(
                    spatial[head_index]
                ).reshape(
                    batch_size,
                    self.split_groups,
                    -1,
                    height,
                    width,
                )
                spatial_outputs.append(
                    spatial_head
                )

            spatial_out = torch.cat(
                spatial_outputs,
                dim=2,
            ).reshape(
                batch_size,
                channels,
                height,
                width,
            )

            spatial_out = self.proj1(
                self.act(
                    self.bn(
                        self.proj0(
                            spatial_out
                        )
                    )
                )
            )

            spatial_out = spatial_out.reshape(
                batch_size,
                channels,
                token_count,
            ).permute(0, 2, 1)

            x = spatial_out * value
        else:
            query = self.q(x).reshape(
                batch_size,
                token_count,
                self.sa_num_heads,
                channels // self.sa_num_heads,
            ).permute(0, 2, 1, 3)

            key_value = self.kv(x).reshape(
                batch_size,
                token_count,
                2,
                self.sa_num_heads,
                channels // self.sa_num_heads,
            ).permute(2, 0, 3, 1, 4)

            key, value = key_value[0], key_value[1]

            attention = (
                query
                @ key.transpose(-2, -1)
            ) * self.scale
            attention = attention.softmax(
                dim=-1
            )
            attention = self.attn_drop(
                attention
            )

            x = (
                attention @ value
            ).transpose(1, 2).reshape(
                batch_size,
                token_count,
                channels,
            )

            local_value = value.transpose(
                1,
                2,
            ).reshape(
                batch_size,
                token_count,
                channels,
            ).transpose(
                1,
                2,
            ).reshape(
                batch_size,
                channels,
                height,
                width,
            )

            local_value = self.local_conv(
                local_value
            ).reshape(
                batch_size,
                channels,
                token_count,
            ).transpose(1, 2)

            x = x + local_value

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        ca_num_heads: int,
        sa_num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        ca_attention: int,
        expand_ratio: int,
        drop: float,
        attn_drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(
            dim,
            eps=1e-6,
        )
        self.attn = Attention(
            dim=dim,
            ca_num_heads=ca_num_heads,
            sa_num_heads=sa_num_heads,
            qkv_bias=qkv_bias,
            ca_attention=ca_attention,
            expand_ratio=expand_ratio,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0.0
            else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(
            dim,
            eps=1e-6,
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(
                dim * mlp_ratio
            ),
            drop=drop,
        )

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0

        self.apply(_init_module)

    def forward(
        self,
        x: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        x = x + self.drop_path(
            self.gamma_1
            * self.attn(
                self.norm1(x),
                height,
                width,
            )
        )
        x = x + self.drop_path(
            self.gamma_2
            * self.mlp(
                self.norm2(x),
                height,
                width,
            )
        )
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.norm = nn.LayerNorm(
            embed_dim,
            eps=1e-6,
        )

        self.apply(_init_module)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class Head(nn.Module):
    def __init__(
        self,
        dim: int,
        head_conv: int = 3,
    ) -> None:
        super().__init__()

        padding = 3 if head_conv == 7 else 1

        self.conv = nn.Sequential(
            nn.Conv2d(
                3,
                dim,
                kernel_size=head_conv,
                stride=2,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=2,
                stride=2,
            ),
        )
        self.norm = nn.LayerNorm(
            dim,
            eps=1e-6,
        )

        self.apply(_init_module)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        x = self.conv(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, height, width


class SMTTinyBackbone(nn.Module):
    """
    SMT-T adapted as a four-stage dense-prediction backbone for MLab.

    Input:
        [B, 3, H, W]

    Output:
        (
            stage1,  # [B,  64, H/4,  W/4]
            stage2,  # [B, 128, H/8,  W/8]
            stage3,  # [B, 256, H/16, W/16]
            stage4,  # [B, 512, H/32, W/32]
        )

    This uses the official SMT-T architecture/configuration:
        embed_dims   = [64, 128, 256, 512]
        depths       = [2, 2, 8, 1]
        ca_attentions= [1, 1, 1, 0]
        drop_path    = 0.1

    The ImageNet classification head is intentionally omitted.
    """

    out_channels = (
        64,
        128,
        256,
        512,
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
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        embed_dims = [64, 128, 256, 512]
        ca_num_heads = [4, 4, 4, -1]
        sa_num_heads = [-1, -1, 8, 16]
        mlp_ratios = [4, 4, 4, 2]
        depths = [2, 2, 8, 1]
        ca_attentions = [1, 1, 1, 0]
        qkv_bias = True
        expand_ratio = 2

        total_depth = sum(depths)
        drop_path_values = torch.linspace(
            0.0,
            drop_path_rate,
            total_depth,
        ).tolist()

        current_depth = 0

        for stage_index in range(4):
            if stage_index == 0:
                patch_embed: nn.Module = Head(
                    dim=embed_dims[stage_index],
                    head_conv=3,
                )
            else:
                patch_embed = OverlapPatchEmbed(
                    in_channels=embed_dims[
                        stage_index - 1
                    ],
                    embed_dim=embed_dims[
                        stage_index
                    ],
                )

            blocks = nn.ModuleList()

            for block_index in range(
                depths[stage_index]
            ):
                # Official SMT alternates CA/SA inside stage 3:
                # blocks 0,2,4,6 use CA; blocks 1,3,5,7 use SA.
                if (
                    stage_index == 2
                    and block_index % 2 == 1
                ):
                    ca_attention = 0
                else:
                    ca_attention = (
                        ca_attentions[
                            stage_index
                        ]
                    )

                blocks.append(
                    Block(
                        dim=embed_dims[
                            stage_index
                        ],
                        ca_num_heads=(
                            ca_num_heads[
                                stage_index
                            ]
                        ),
                        sa_num_heads=(
                            sa_num_heads[
                                stage_index
                            ]
                        ),
                        mlp_ratio=(
                            mlp_ratios[
                                stage_index
                            ]
                        ),
                        qkv_bias=qkv_bias,
                        ca_attention=(
                            ca_attention
                        ),
                        expand_ratio=(
                            expand_ratio
                        ),
                        drop=0.0,
                        attn_drop=0.0,
                        drop_path=(
                            drop_path_values[
                                current_depth
                                + block_index
                            ]
                        ),
                    )
                )

            current_depth += depths[
                stage_index
            ]

            norm = nn.LayerNorm(
                embed_dims[stage_index],
                eps=1e-6,
            )

            setattr(
                self,
                f"patch_embed{stage_index + 1}",
                patch_embed,
            )
            setattr(
                self,
                f"block{stage_index + 1}",
                blocks,
            )
            setattr(
                self,
                f"norm{stage_index + 1}",
                norm,
            )

        self.apply(_init_module)

        if pretrained_path is not None:
            self.load_pretrained(
                pretrained_path
            )

    def load_pretrained(
        self,
        pretrained_path: str | Path,
    ) -> None:
        pretrained_path = Path(
            pretrained_path
        )

        if not pretrained_path.is_file():
            raise FileNotFoundError(
                "SMT-T pretrained checkpoint not found: "
                f"{pretrained_path}"
            )

        # This checkpoint is the official SMT release. PyTorch 2.6+ defaults
        # to weights_only=True, but the release may contain non-tensor metadata.
        checkpoint: Any = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False,
        )

        if (
            isinstance(checkpoint, dict)
            and "model" in checkpoint
        ):
            state_dict = checkpoint[
                "model"
            ]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise TypeError(
                "Unsupported SMT checkpoint format. Expected a state_dict "
                "or a dict containing checkpoint['model']."
            )

        normalized_state_dict: dict[
            str,
            torch.Tensor,
        ] = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]

            # Classification head is intentionally absent in this backbone.
            if key.startswith("head."):
                continue

            normalized_state_dict[key] = value

        incompatible = self.load_state_dict(
            normalized_state_dict,
            strict=False,
        )

        if (
            incompatible.missing_keys
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "SMT-T checkpoint did not map cleanly to the MLab backbone.\n"
                f"Missing keys: {incompatible.missing_keys}\n"
                f"Unexpected keys: {incompatible.unexpected_keys}"
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        batch_size = x.shape[0]
        outputs: list[torch.Tensor] = []

        for stage_index in range(4):
            patch_embed = getattr(
                self,
                f"patch_embed{stage_index + 1}",
            )
            blocks = getattr(
                self,
                f"block{stage_index + 1}",
            )
            norm = getattr(
                self,
                f"norm{stage_index + 1}",
            )

            tokens, height, width = (
                patch_embed(x)
            )

            for block in blocks:
                tokens = block(
                    tokens,
                    height,
                    width,
                )

            tokens = norm(tokens)

            feature = tokens.reshape(
                batch_size,
                height,
                width,
                -1,
            ).permute(
                0,
                3,
                1,
                2,
            ).contiguous()

            outputs.append(feature)
            x = feature

        return tuple(outputs)


def smt_tiny(
    pretrained_path: str | Path | None = None,
    **kwargs: Any,
) -> SMTTinyBackbone:
    return SMTTinyBackbone(
        pretrained_path=pretrained_path,
        **kwargs,
    )


# Short alias matching the paper notation SMT-T.
smt_t = smt_tiny
