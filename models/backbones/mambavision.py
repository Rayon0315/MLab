# models/backbones/mambavision.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.layers import DropPath, Mlp


def window_partition(
    x: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """
    将 BCHW 特征划分为局部窗口。

    Args:
        x: [B, C, H, W]
        window_size: 窗口边长

    Returns:
        [B * num_windows, window_size * window_size, C]
    """
    batch_size, channels, height, width = x.shape

    x = x.reshape(
        batch_size,
        channels,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
    )

    windows = x.permute(
        0, 2, 4, 3, 5, 1
    ).reshape(
        -1,
        window_size * window_size,
        channels,
    )

    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    将窗口序列恢复为 BCHW 特征。

    Args:
        windows:
            [B * num_windows, window_size * window_size, C]
        window_size: 窗口边长
        height: padding 后的高度
        width: padding 后的宽度

    Returns:
        [B, C, H, W]
    """
    windows_per_image = (
        height // window_size
    ) * (
        width // window_size
    )

    batch_size = windows.shape[0] // windows_per_image
    channels = windows.shape[-1]

    x = windows.reshape(
        batch_size,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )

    x = x.permute(
        0, 5, 1, 3, 2, 4
    ).reshape(
        batch_size,
        channels,
        height,
        width,
    )

    return x


def unwrap_state_dict(
    checkpoint: dict,
) -> dict[str, torch.Tensor]:
    """
    从常见 checkpoint 格式中提取 state_dict，
    并移除常见前缀。
    """
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
                    cleaned_key = cleaned_key[len(prefix):]
                    prefix_removed = True
                    break

        cleaned_state_dict[cleaned_key] = value

    return cleaned_state_dict


class Downsample(nn.Module):
    """
    使用 stride=2 的卷积进行下采样。
    """

    def __init__(
        self,
        dim: int,
        keep_dim: bool = False,
    ) -> None:
        super().__init__()

        out_dim = dim if keep_dim else dim * 2

        self.reduction = nn.Sequential(
            nn.Conv2d(
                dim,
                out_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.reduction(x)


class PatchEmbed(nn.Module):
    """
    两次 stride=2 卷积，将输入降采样到 stride 4。
    """

    def __init__(
        self,
        in_chans: int = 3,
        in_dim: int = 32,
        dim: int = 80,
    ) -> None:
        super().__init__()

        self.proj = nn.Identity()

        self.conv_down = nn.Sequential(
            nn.Conv2d(
                in_chans,
                in_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                in_dim,
                eps=1e-4,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_dim,
                dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                dim,
                eps=1e-4,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.conv_down(x)

        return x


class ConvBlock(nn.Module):
    """
    MambaVision 前两个 stage 使用的卷积残差块。
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale: float | None = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.norm1 = nn.BatchNorm2d(
            dim,
            eps=1e-5,
        )
        self.act1 = nn.GELU(
            approximate="tanh",
        )

        self.conv2 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.norm2 = nn.BatchNorm2d(
            dim,
            eps=1e-5,
        )

        self.use_layer_scale = isinstance(
            layer_scale,
            (int, float),
        )

        if self.use_layer_scale:
            self.gamma = nn.Parameter(
                float(layer_scale) * torch.ones(dim)
            )

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.use_layer_scale:
            x = x * self.gamma.reshape(
                1, -1, 1, 1
            )

        x = residual + self.drop_path(x)

        return x


class MambaVisionMixer(nn.Module):
    """
    MambaVision 中使用的 Mamba Mixer。

    输入输出格式均为：
        [B, L, C]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.d_inner = int(
            self.expand * self.d_model
        )

        if dt_rank == "auto":
            self.dt_rank = math.ceil(
                self.d_model / 16
            )
        else:
            self.dt_rank = int(dt_rank)

        if self.d_inner % 2 != 0:
            raise ValueError(
                "MambaVisionMixer requires an even inner dimension."
            )

        inner_half = self.d_inner // 2

        self.in_proj = nn.Linear(
            self.d_model,
            self.d_inner,
            bias=bias,
            **factory_kwargs,
        )

        self.x_proj = nn.Linear(
            inner_half,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )

        self.dt_proj = nn.Linear(
            self.dt_rank,
            inner_half,
            bias=True,
            **factory_kwargs,
        )

        dt_init_std = (
            self.dt_rank ** -0.5
        ) * dt_scale

        if dt_init == "constant":
            nn.init.constant_(
                self.dt_proj.weight,
                dt_init_std,
            )
        elif dt_init == "random":
            nn.init.uniform_(
                self.dt_proj.weight,
                -dt_init_std,
                dt_init_std,
            )
        else:
            raise ValueError(
                f"Unsupported dt_init: {dt_init}"
            )

        dt = torch.exp(
            torch.rand(
                inner_half,
                **factory_kwargs,
            )
            * (
                math.log(dt_max)
                - math.log(dt_min)
            )
            + math.log(dt_min)
        ).clamp(
            min=dt_init_floor
        )

        inv_dt = dt + torch.log(
            -torch.expm1(-dt)
        )

        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(
                1,
                self.d_state + 1,
                dtype=torch.float32,
                device=device,
            ),
            "n -> d n",
            d=inner_half,
        ).contiguous()

        self.A_log = nn.Parameter(
            torch.log(A)
        )
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(
            torch.ones(
                inner_half,
                device=device,
            )
        )
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(
            self.d_inner,
            self.d_model,
            bias=bias,
            **factory_kwargs,
        )

        # 保持官方 v1.2.0 的参数结构。
        #
        # 官方代码使用 bias=conv_bias // 2。
        # 当 conv_bias=True 时，True // 2 等于 0，
        # 因此官方预训练模型中这两个卷积没有 bias 参数。
        official_conv_bias = bool(
            conv_bias // 2
        )

        self.conv1d_x = nn.Conv1d(
            in_channels=inner_half,
            out_channels=inner_half,
            kernel_size=d_conv,
            groups=inner_half,
            bias=official_conv_bias,
            **factory_kwargs,
        )

        self.conv1d_z = nn.Conv1d(
            in_channels=inner_half,
            out_channels=inner_half,
            kernel_size=d_conv,
            groups=inner_half,
            bias=official_conv_bias,
            **factory_kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _, sequence_length, _ = hidden_states.shape

        inner_half = self.d_inner // 2

        xz = self.in_proj(hidden_states)
        xz = rearrange(
            xz,
            "b l d -> b d l",
        )

        x, z = xz.chunk(
            2,
            dim=1,
        )

        A = -torch.exp(
            self.A_log.float()
        )

        x = F.silu(
            F.conv1d(
                x,
                weight=self.conv1d_x.weight,
                bias=self.conv1d_x.bias,
                padding="same",
                groups=inner_half,
            )
        )

        z = F.silu(
            F.conv1d(
                z,
                weight=self.conv1d_z.weight,
                bias=self.conv1d_z.bias,
                padding="same",
                groups=inner_half,
            )
        )

        x_dbl = self.x_proj(
            rearrange(
                x,
                "b d l -> (b l) d",
            )
        )

        dt, B, C = torch.split(
            x_dbl,
            [
                self.dt_rank,
                self.d_state,
                self.d_state,
            ],
            dim=-1,
        )

        dt = rearrange(
            self.dt_proj(dt),
            "(b l) d -> b d l",
            l=sequence_length,
        )

        B = rearrange(
            B,
            "(b l) dstate -> b dstate l",
            l=sequence_length,
        ).contiguous()

        C = rearrange(
            C,
            "(b l) dstate -> b dstate l",
            l=sequence_length,
        ).contiguous()

        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

        y = torch.cat(
            [y, z],
            dim=1,
        )

        y = rearrange(
            y,
            "b d l -> b l d",
        )

        y = self.out_proj(y)

        return y


class Attention(nn.Module):
    """
    Window Attention。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                "dim must be divisible by num_heads."
            )

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=qkv_bias,
        )

        if qk_norm:
            self.q_norm = norm_layer(
                self.head_dim
            )
            self.k_norm = norm_layer(
                self.head_dim
            )
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_drop = nn.Dropout(
            attn_drop
        )

        self.proj = nn.Linear(
            dim,
            dim,
        )

        self.proj_drop = nn.Dropout(
            proj_drop
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape

        qkv = self.qkv(x).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )

        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )

        q, k, v = qkv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.training:
            dropout_p = self.attn_drop.p
        else:
            dropout_p = 0.0

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            scale=self.scale,
        )

        x = x.transpose(
            1, 2
        ).reshape(
            batch_size,
            sequence_length,
            channels,
        )

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Stage 3/4 中使用的 Mamba 或 Attention Block。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        counter: int,
        transformer_blocks: list[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        layer_scale: float | None = None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        if counter in transformer_blocks:
            self.mixer = Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                norm_layer=norm_layer,
            )
        else:
            self.mixer = MambaVisionMixer(
                d_model=dim,
                d_state=8,
                d_conv=3,
                expand=1,
            )

        if drop_path > 0.0:
            self.drop_path = DropPath(
                drop_path
            )
        else:
            self.drop_path = nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(
            dim * mlp_ratio
        )

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        use_layer_scale = isinstance(
            layer_scale,
            (int, float),
        )

        if use_layer_scale:
            self.gamma_1: torch.Tensor | float = nn.Parameter(
                float(layer_scale) * torch.ones(dim)
            )
            self.gamma_2: torch.Tensor | float = nn.Parameter(
                float(layer_scale) * torch.ones(dim)
            )
        else:
            self.gamma_1 = 1.0
            self.gamma_2 = 1.0

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.drop_path(
            self.gamma_1
            * self.mixer(
                self.norm1(x)
            )
        )

        x = x + self.drop_path(
            self.gamma_2
            * self.mlp(
                self.norm2(x)
            )
        )

        return x


class MambaVisionLayer(nn.Module):
    """
    一个完整 stage。

    Stage 1/2 使用 ConvBlock。
    Stage 3/4 使用 Mamba + Attention Block。
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        conv: bool = False,
        downsample: bool = True,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: bool | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        layer_scale: float | None = None,
        layer_scale_conv: float | None = None,
        transformer_blocks: list[int] | None = None,
    ) -> None:
        super().__init__()

        if transformer_blocks is None:
            transformer_blocks = []

        self.transformer_block = not conv
        self.window_size = window_size

        if conv:
            self.blocks = nn.ModuleList(
                [
                    ConvBlock(
                        dim=dim,
                        drop_path=(
                            drop_path[index]
                            if isinstance(drop_path, list)
                            else drop_path
                        ),
                        layer_scale=layer_scale_conv,
                    )
                    for index in range(depth)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=dim,
                        counter=index,
                        transformer_blocks=transformer_blocks,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=bool(qk_scale),
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=(
                            drop_path[index]
                            if isinstance(drop_path, list)
                            else drop_path
                        ),
                        layer_scale=layer_scale,
                    )
                    for index in range(depth)
                ]
            )

        if downsample:
            self.downsample = Downsample(dim)
        else:
            self.downsample = None

    def forward_blocks(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        只运行当前 stage，不执行末尾下采样。

        这个接口用于取得 SOD 所需的多尺度特征。
        """
        _, _, height, width = x.shape

        if self.transformer_block:
            pad_right = (
                self.window_size
                - width % self.window_size
            ) % self.window_size

            pad_bottom = (
                self.window_size
                - height % self.window_size
            ) % self.window_size

            if pad_right > 0 or pad_bottom > 0:
                x = F.pad(
                    x,
                    (
                        0,
                        pad_right,
                        0,
                        pad_bottom,
                    ),
                )

            padded_height, padded_width = x.shape[-2:]

            x = window_partition(
                x,
                self.window_size,
            )

        for block in self.blocks:
            x = block(x)

        if self.transformer_block:
            x = window_reverse(
                x,
                self.window_size,
                padded_height,
                padded_width,
            )

            if pad_right > 0 or pad_bottom > 0:
                x = x[
                    :,
                    :,
                    :height,
                    :width,
                ].contiguous()

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        保留官方语义：
        stage 计算完成后继续执行下采样。
        """
        x = self.forward_blocks(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class MambaVisionBackbone(nn.Module):
    """
    面向密集预测任务的 MambaVision Backbone。

    输入：
        [B, 3, H, W]

    输出：
        (
            stage1,  # stride 4
            stage2,  # stride 8
            stage3,  # stride 16
            stage4,  # stride 32
        )
    """

    def __init__(
        self,
        dim: int,
        in_dim: int,
        depths: list[int],
        window_size: list[int],
        mlp_ratio: float,
        num_heads: list[int],
        drop_path_rate: float = 0.2,
        in_chans: int = 3,
        qkv_bias: bool = True,
        qk_scale: bool | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        layer_scale: float | None = None,
        layer_scale_conv: float | None = None,
    ) -> None:
        super().__init__()

        if not (
            len(depths)
            == len(window_size)
            == len(num_heads)
            == 4
        ):
            raise ValueError(
                "MambaVisionBackbone expects four stages."
            )

        self.out_channels = tuple(
            int(
                dim * 2 ** stage_index
            )
            for stage_index in range(4)
        )

        self.out_strides = (
            4,
            8,
            16,
            32,
        )

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            in_dim=in_dim,
            dim=dim,
        )

        drop_path_values = [
            value.item()
            for value in torch.linspace(
                0,
                drop_path_rate,
                sum(depths),
            )
        ]

        self.levels = nn.ModuleList()

        depth_offset = 0

        for stage_index, depth in enumerate(depths):
            stage_dim = int(
                dim * 2 ** stage_index
            )

            is_conv_stage = stage_index < 2

            if depth % 2 != 0:
                transformer_blocks = list(
                    range(
                        depth // 2 + 1,
                        depth,
                    )
                )
            else:
                transformer_blocks = list(
                    range(
                        depth // 2,
                        depth,
                    )
                )

            level = MambaVisionLayer(
                dim=stage_dim,
                depth=depth,
                num_heads=num_heads[stage_index],
                window_size=window_size[stage_index],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=is_conv_stage,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_values[
                    depth_offset:
                    depth_offset + depth
                ],
                downsample=stage_index < 3,
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=transformer_blocks,
            )

            self.levels.append(level)

            depth_offset += depth

        # 官方分类模型在池化前对最后一级特征执行 BN。
        # 保留这一层可以直接加载对应的官方预训练参数。
        self.norm = nn.BatchNorm2d(
            self.out_channels[-1]
        )

        self.apply(
            self._init_weights
        )

    @staticmethod
    def _init_weights(
        module: nn.Module,
    ) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(
                module.bias
            )
            nn.init.ones_(
                module.weight
            )

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(
                module.weight
            )
            nn.init.zeros_(
                module.bias
            )

    @torch.jit.ignore
    def no_weight_decay_keywords(
        self,
    ) -> set[str]:
        return {"rpb"}

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        x = self.patch_embed(x)

        features = []

        for stage_index, level in enumerate(self.levels):
            # 先得到当前尺度特征。
            stage_feature = level.forward_blocks(x)

            # 官方分类模型会对最后一级特征执行 norm。
            if stage_index == len(self.levels) - 1:
                stage_feature = self.norm(
                    stage_feature
                )

            features.append(
                stage_feature
            )

            # 再执行下采样，为下一个 stage 提供输入。
            if level.downsample is not None:
                x = level.downsample(
                    stage_feature
                )
            else:
                x = stage_feature

        return tuple(features)

    # models/backbones/mambavision.py
    def load_pretrained(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MambaVision pretrained checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = unwrap_state_dict(checkpoint)

        # 当前模型是 backbone，不包含 ImageNet 分类头。
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("head.")
        }

        # 已经确认官方权重除分类头外与当前 backbone 完全匹配，
        # 因此这里使用 strict=True，避免静默漏载参数。
        self.load_state_dict(
            state_dict,
            strict=True,
        )


# models/backbones/mambavision.py
def mamba_vision_tiny(
    pretrained_path: str | Path | None = None,
    **kwargs: Any,
) -> MambaVisionBackbone:
    model = MambaVisionBackbone(
        depths=kwargs.pop(
            "depths",
            [1, 3, 8, 4],
        ),
        num_heads=kwargs.pop(
            "num_heads",
            [2, 4, 8, 16],
        ),
        window_size=kwargs.pop(
            "window_size",
            [8, 8, 14, 7],
        ),
        dim=kwargs.pop(
            "dim",
            80,
        ),
        in_dim=kwargs.pop(
            "in_dim",
            32,
        ),
        mlp_ratio=kwargs.pop(
            "mlp_ratio",
            4.0,
        ),
        drop_path_rate=kwargs.pop(
            "drop_path_rate",
            0.2,
        ),
        **kwargs,
    )

    if pretrained_path is not None:
        model.load_pretrained(
            checkpoint_path=pretrained_path,
        )

    return model

def mamba_vision_small(
    pretrained_path: str | Path | None = None,
    **kwargs: Any,
) -> MambaVisionBackbone:
    model = MambaVisionBackbone(
        depths=kwargs.pop(
            "depths",
            [3, 3, 7, 5],
        ),
        num_heads=kwargs.pop(
            "num_heads",
            [2, 4, 8, 16],
        ),
        window_size=kwargs.pop(
            "window_size",
            [8, 8, 14, 7],
        ),
        dim=kwargs.pop(
            "dim",
            96,
        ),
        in_dim=kwargs.pop(
            "in_dim",
            64,
        ),
        mlp_ratio=kwargs.pop(
            "mlp_ratio",
            4.0,
        ),
        drop_path_rate=kwargs.pop(
            "drop_path_rate",
            0.2,
        ),
        **kwargs,
    )

    if pretrained_path is not None:
        model.load_pretrained(
            checkpoint_path=pretrained_path,
        )

    return model


mamba_vision_T = mamba_vision_tiny
mamba_vision_S = mamba_vision_small