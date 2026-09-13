# models/networks/cfmnet_unetformer_sod.py

from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_

from models.backbones.cfmnet import (
    CFMNET_OUT_CHANNELS,
    build_cfmnet,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETRAINED_PATH = (
    PROJECT_ROOT
    / "pretrained"
    / "cfmnet"
    / "cfmnet_imagenet1k.pth"
)


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        bias: bool = False,
    ) -> None:
        padding = (
            (stride - 1)
            + dilation * (kernel_size - 1)
        ) // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True),
        )


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        bias: bool = False,
    ) -> None:
        padding = (
            (stride - 1)
            + dilation * (kernel_size - 1)
        ) // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            norm_layer(out_channels),
        )


class Conv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = False,
    ) -> None:
        padding = (
            (stride - 1)
            + dilation * (kernel_size - 1)
        ) // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=bias,
            )
        )


class SeparableConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        padding = (
            (stride - 1)
            + dilation * (kernel_size - 1)
        ) // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            norm_layer(in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
        )


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.ReLU6,
        drop: float = 0.0,
    ) -> None:
        super().__init__()

        out_features = (
            out_features
            if out_features is not None
            else in_features
        )

        hidden_features = (
            hidden_features
            if hidden_features is not None
            else in_features
        )

        self.fc1 = nn.Conv2d(
            in_features,
            hidden_features,
            kernel_size=1,
            bias=True,
        )

        self.act = act_layer()

        self.fc2 = nn.Conv2d(
            hidden_features,
            out_features,
            kernel_size=1,
            bias=True,
        )

        self.drop = nn.Dropout(
            drop,
            inplace=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    """
    UNetFormer global-local attention block.

    Local path:
        3x3 conv + 1x1 conv.

    Global path:
        window attention + strip pooling.

    This is kept close to the decoder used by the official
    CFMNet semantic-segmentation implementation.
    """

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 8,
        qkv_bias: bool = False,
        window_size: int = 8,
        relative_pos_embedding: bool = True,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by "
                f"num_heads={num_heads}"
            )

        self.num_heads = num_heads
        self.ws = window_size

        head_dim = (
            dim
            // num_heads
        )

        self.scale = (
            head_dim ** -0.5
        )

        self.qkv = Conv(
            dim,
            3 * dim,
            kernel_size=1,
            bias=qkv_bias,
        )

        self.local1 = ConvBN(
            dim,
            dim,
            kernel_size=3,
        )

        self.local2 = ConvBN(
            dim,
            dim,
            kernel_size=1,
        )

        self.proj = SeparableConvBN(
            dim,
            dim,
            kernel_size=window_size,
        )

        self.attn_x = nn.AvgPool2d(
            kernel_size=(
                window_size,
                1,
            ),
            stride=1,
            padding=(
                window_size // 2 - 1,
                0,
            ),
        )

        self.attn_y = nn.AvgPool2d(
            kernel_size=(
                1,
                window_size,
            ),
            stride=1,
            padding=(
                0,
                window_size // 2 - 1,
            ),
        )

        self.relative_pos_embedding = (
            relative_pos_embedding
        )

        if self.relative_pos_embedding:
            bias_count = (
                (2 * window_size - 1)
                * (2 * window_size - 1)
            )

            self.relative_position_bias_table = (
                nn.Parameter(
                    torch.zeros(
                        bias_count,
                        num_heads,
                    )
                )
            )

            coords_h = torch.arange(
                self.ws
            )

            coords_w = torch.arange(
                self.ws
            )

            coords = torch.stack(
                torch.meshgrid(
                    coords_h,
                    coords_w,
                    indexing="ij",
                )
            )

            coords_flatten = (
                torch.flatten(
                    coords,
                    1,
                )
            )

            relative_coords = (
                coords_flatten[
                    :,
                    :,
                    None,
                ]
                - coords_flatten[
                    :,
                    None,
                    :,
                ]
            )

            relative_coords = (
                relative_coords
                .permute(
                    1,
                    2,
                    0,
                )
                .contiguous()
            )

            relative_coords[
                :,
                :,
                0,
            ] += (
                self.ws
                - 1
            )

            relative_coords[
                :,
                :,
                1,
            ] += (
                self.ws
                - 1
            )

            relative_coords[
                :,
                :,
                0,
            ] *= (
                2 * self.ws
                - 1
            )

            relative_position_index = (
                relative_coords.sum(
                    dim=-1
                )
            )

            self.register_buffer(
                "relative_position_index",
                relative_position_index,
            )

            trunc_normal_(
                self.relative_position_bias_table,
                std=0.02,
            )

    @staticmethod
    def _pad_to_window(
        x: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        _, _, height, width = (
            x.shape
        )

        if (
            width
            % window_size
            != 0
        ):
            x = F.pad(
                x,
                (
                    0,
                    window_size
                    - width
                    % window_size,
                ),
                mode="reflect",
            )

        if (
            height
            % window_size
            != 0
        ):
            x = F.pad(
                x,
                (
                    0,
                    0,
                    0,
                    window_size
                    - height
                    % window_size,
                ),
                mode="reflect",
            )

        return x

    @staticmethod
    def _pad_out(
        x: torch.Tensor,
    ) -> torch.Tensor:
        return F.pad(
            x,
            (
                0,
                1,
                0,
                1,
            ),
            mode="reflect",
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        (
            batch_size,
            channels,
            height,
            width,
        ) = x.shape

        local = (
            self.local2(x)
            + self.local1(x)
        )

        padded = self._pad_to_window(
            x,
            self.ws,
        )

        (
            _,
            _,
            padded_height,
            padded_width,
        ) = padded.shape

        qkv = self.qkv(
            padded
        )

        q, k, v = rearrange(
            qkv,
            (
                "b (qkv head d) "
                "(hh ws1) (ww ws2) -> "
                "qkv (b hh ww) head "
                "(ws1 ws2) d"
            ),
            qkv=3,
            head=self.num_heads,
            d=(
                channels
                // self.num_heads
            ),
            hh=(
                padded_height
                // self.ws
            ),
            ww=(
                padded_width
                // self.ws
            ),
            ws1=self.ws,
            ws2=self.ws,
        )

        attention = (
            q
            @ k.transpose(
                -2,
                -1,
            )
        ) * self.scale

        if (
            self.relative_pos_embedding
        ):
            relative_bias = (
                self.relative_position_bias_table[
                    self.relative_position_index
                    .view(-1)
                ]
            )

            relative_bias = (
                relative_bias.view(
                    self.ws
                    * self.ws,
                    self.ws
                    * self.ws,
                    -1,
                )
            )

            relative_bias = (
                relative_bias
                .permute(
                    2,
                    0,
                    1,
                )
                .contiguous()
            )

            attention = (
                attention
                + relative_bias.unsqueeze(
                    0
                )
            )

        attention = (
            attention.softmax(
                dim=-1
            )
        )

        attention = (
            attention
            @ v
        )

        attention = rearrange(
            attention,
            (
                "(b hh ww) head "
                "(ws1 ws2) d -> "
                "b (head d) "
                "(hh ws1) (ww ws2)"
            ),
            b=batch_size,
            head=self.num_heads,
            d=(
                channels
                // self.num_heads
            ),
            hh=(
                padded_height
                // self.ws
            ),
            ww=(
                padded_width
                // self.ws
            ),
            ws1=self.ws,
            ws2=self.ws,
        )

        attention = attention[
            :,
            :,
            :height,
            :width,
        ]

        attention_x = self.attn_x(
            F.pad(
                attention,
                (
                    0,
                    0,
                    0,
                    1,
                ),
                mode="reflect",
            )
        )

        attention_y = self.attn_y(
            F.pad(
                attention,
                (
                    0,
                    1,
                    0,
                    0,
                ),
                mode="reflect",
            )
        )

        out = (
            attention_x
            + attention_y
            + local
        )

        out = self._pad_out(
            out
        )

        out = self.proj(
            out
        )

        return out[
            :,
            :,
            :height,
            :width,
        ]


class GlobalLocalBlock(nn.Module):
    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.ReLU6,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        window_size: int = 8,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(
            dim
        )

        self.attn = (
            GlobalLocalAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                window_size=window_size,
            )
        )

        self.drop_path = (
            DropPath(
                drop_path
            )
            if drop_path > 0.0
            else nn.Identity()
        )

        hidden_dim = int(
            dim
            * mlp_ratio
        )

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.norm2 = norm_layer(
            dim
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = (
            x
            + self.drop_path(
                self.attn(
                    self.norm1(x)
                )
            )
        )

        x = (
            x
            + self.drop_path(
                self.mlp(
                    self.norm2(x)
                )
            )
        )

        return x


class WeightedFusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        decode_channels: int,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()

        self.pre_conv = Conv(
            in_channels,
            decode_channels,
            kernel_size=1,
        )

        self.weights = nn.Parameter(
            torch.ones(
                2,
                dtype=torch.float32,
            )
        )

        self.epsilon = epsilon

        self.post_conv = (
            ConvBNReLU(
                decode_channels,
                decode_channels,
                kernel_size=3,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        weights = F.relu(
            self.weights
        )

        weights = (
            weights
            / (
                weights.sum()
                + self.epsilon
            )
        )

        x = (
            weights[0]
            * self.pre_conv(
                residual
            )
            + weights[1]
            * x
        )

        return self.post_conv(
            x
        )


class FeatureRefinementHead(
    nn.Module
):
    def __init__(
        self,
        in_channels: int,
        decode_channels: int,
    ) -> None:
        super().__init__()

        self.pre_conv = Conv(
            in_channels,
            decode_channels,
            kernel_size=1,
        )

        self.weights = nn.Parameter(
            torch.ones(
                2,
                dtype=torch.float32,
            )
        )

        self.epsilon = 1e-8

        self.post_conv = (
            ConvBNReLU(
                decode_channels,
                decode_channels,
                kernel_size=3,
            )
        )

        self.pixel_attention = (
            nn.Sequential(
                nn.Conv2d(
                    decode_channels,
                    decode_channels,
                    kernel_size=3,
                    padding=1,
                    groups=decode_channels,
                ),
                nn.Sigmoid(),
            )
        )

        channel_hidden = max(
            decode_channels
            // 16,
            1,
        )

        self.channel_attention = (
            nn.Sequential(
                nn.AdaptiveAvgPool2d(
                    1
                ),
                Conv(
                    decode_channels,
                    channel_hidden,
                    kernel_size=1,
                ),
                nn.ReLU6(
                    inplace=True
                ),
                Conv(
                    channel_hidden,
                    decode_channels,
                    kernel_size=1,
                ),
                nn.Sigmoid(),
            )
        )

        self.shortcut = ConvBN(
            decode_channels,
            decode_channels,
            kernel_size=1,
        )

        self.proj = (
            SeparableConvBN(
                decode_channels,
                decode_channels,
                kernel_size=3,
            )
        )

        self.act = nn.ReLU6(
            inplace=True
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        weights = F.relu(
            self.weights
        )

        weights = (
            weights
            / (
                weights.sum()
                + self.epsilon
            )
        )

        x = (
            weights[0]
            * self.pre_conv(
                residual
            )
            + weights[1]
            * x
        )

        x = self.post_conv(
            x
        )

        shortcut = self.shortcut(
            x
        )

        pixel = (
            self.pixel_attention(x)
            * x
        )

        channel = (
            self.channel_attention(x)
            * x
        )

        x = (
            pixel
            + channel
        )

        x = (
            self.proj(x)
            + shortcut
        )

        return self.act(
            x
        )


class AuxHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        self.conv = ConvBNReLU(
            in_channels,
            in_channels,
        )

        self.drop = nn.Dropout(
            0.1
        )

        self.conv_out = Conv(
            in_channels,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        x = self.conv(x)
        x = self.drop(x)
        x = self.conv_out(x)

        return F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class CFMNetUNetFormerDecoder(
    nn.Module
):
    """
    CFMNet paper-style dense prediction decoder adapted to binary SOD.

    Encoder:
        [96, 192, 384, 768]

    Decoder:
        Stage4 -> GlobalLocalBlock
        Stage3 -> weighted top-down fusion -> GlobalLocalBlock
        Stage2 -> weighted top-down fusion -> GlobalLocalBlock
        Stage1 -> feature refinement head
        -> binary saliency logits

    During training an auxiliary prediction is produced from
    Stage4/Stage3/Stage2 decoder features, matching the role
    of the auxiliary head in the original CFMNet segmentation setup.
    """

    def __init__(
        self,
        encoder_channels: tuple[
            int,
            int,
            int,
            int,
        ] = CFMNET_OUT_CHANNELS,
        decode_channels: int = 64,
        dropout: float = 0.1,
        window_size: int = 8,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        self.pre_conv = ConvBN(
            encoder_channels[-1],
            decode_channels,
            kernel_size=1,
        )

        self.block4 = (
            GlobalLocalBlock(
                dim=decode_channels,
                num_heads=8,
                window_size=window_size,
            )
        )

        self.fuse3 = WeightedFusion(
            encoder_channels[-2],
            decode_channels,
        )

        self.block3 = (
            GlobalLocalBlock(
                dim=decode_channels,
                num_heads=8,
                window_size=window_size,
            )
        )

        self.fuse2 = WeightedFusion(
            encoder_channels[-3],
            decode_channels,
        )

        self.block2 = (
            GlobalLocalBlock(
                dim=decode_channels,
                num_heads=8,
                window_size=window_size,
            )
        )

        self.refine1 = (
            FeatureRefinementHead(
                encoder_channels[-4],
                decode_channels,
            )
        )

        self.segmentation_head = (
            nn.Sequential(
                ConvBNReLU(
                    decode_channels,
                    decode_channels,
                ),
                nn.Dropout2d(
                    p=dropout,
                    inplace=True,
                ),
                Conv(
                    decode_channels,
                    num_classes,
                    kernel_size=1,
                ),
            )
        )

        self.aux_head = AuxHead(
            decode_channels,
            num_classes,
        )

    def forward(
        self,
        features: list[
            torch.Tensor
        ],
        output_size: tuple[
            int,
            int,
        ],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        (
            stage1,
            stage2,
            stage3,
            stage4,
        ) = features

        x4 = self.block4(
            self.pre_conv(
                stage4
            )
        )

        x3 = self.block3(
            self.fuse3(
                x4,
                stage3,
            )
        )

        x2 = self.block2(
            self.fuse2(
                x3,
                stage2,
            )
        )

        x1 = self.refine1(
            x2,
            stage1,
        )

        prediction = (
            self.segmentation_head(
                x1
            )
        )

        prediction = F.interpolate(
            prediction,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

        auxiliary = None

        if self.training:
            aux_size = (
                x2.shape[-2:]
            )

            aux4 = F.interpolate(
                x4,
                size=aux_size,
                mode="bilinear",
                align_corners=False,
            )

            aux3 = F.interpolate(
                x3,
                size=aux_size,
                mode="bilinear",
                align_corners=False,
            )

            auxiliary = (
                self.aux_head(
                    aux4
                    + aux3
                    + x2,
                    output_size=(
                        output_size
                    ),
                )
            )

        return (
            prediction,
            auxiliary,
        )


class CFMNetUNetFormerSOD(
    nn.Module
):
    input_keys = (
        "image",
    )

    def __init__(
        self,
        pretrained_path: str
        | Path
        | None,
        decode_channels: int = 64,
        window_size: int = 8,
    ) -> None:
        super().__init__()

        self.backbone = (
            build_cfmnet(
                pretrained_path=(
                    pretrained_path
                )
            )
        )

        self.decoder = (
            CFMNetUNetFormerDecoder(
                encoder_channels=(
                    CFMNET_OUT_CHANNELS
                ),
                decode_channels=(
                    decode_channels
                ),
                dropout=0.1,
                window_size=(
                    window_size
                ),
                num_classes=1,
            )
        )

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

        features = (
            self.backbone(
                image
            )
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


def build_model(
) -> CFMNetUNetFormerSOD:
    pretrained_path = (
        PRETRAINED_PATH
        if PRETRAINED_PATH.exists()
        else None
    )

    if pretrained_path is None:
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

    return CFMNetUNetFormerSOD(
        pretrained_path=(
            pretrained_path
        ),
        decode_channels=64,
        window_size=8,
    )


if __name__ == "__main__":
    model = build_model()

    image = torch.randn(
        2,
        3,
        352,
        352,
    )

    model.train()

    outputs = model(
        image=image
    )

    print(
        "train pred:",
        tuple(
            outputs[
                "pred"
            ].shape
        ),
    )

    print(
        "train aux:",
        [
            tuple(
                tensor.shape
            )
            for tensor
            in outputs.get(
                "aux",
                []
            )
        ],
    )

    model.eval()

    with torch.no_grad():
        outputs = model(
            image=image
        )

    print(
        "eval pred:",
        tuple(
            outputs[
                "pred"
            ].shape
        ),
    )

    with torch.no_grad():
        features = (
            model.backbone(
                image
            )
        )

    print(
        "features:",
        [
            tuple(
                feature.shape
            )
            for feature
            in features
        ],
    )
