from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNPReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        dilation: int = 1,
        use_bn: bool = True,
        use_activation: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_activation:
            layers.append(nn.PReLU(out_channels))
        super().__init__(*layers)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        use_bn: bool = True,
        use_activation: bool = False,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNPReLU(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=in_channels,
                dilation=dilation,
            ),
            ConvBNPReLU(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                use_bn=use_bn,
                use_activation=use_activation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthwiseSeparableConvAct(DepthwiseSeparableConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dilation=dilation,
            use_bn=True,
            use_activation=True,
        )


class FreSA(nn.Module):
    """Frequency-domain self-attention used by the official FreMaNet code."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 8, 1)

        self.to_qkv = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1,
        )
        self.control = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.weight = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)
        self.project_out = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        input_dtype = x.dtype

        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            q_fft = torch.fft.fft2(q.float())
            k_fft = torch.fft.fft2(k.float())
            v_fft = torch.fft.fft2(v.float())
            attention = q_fft * k_fft

        weight = self.weight(attention.real.to(dtype=input_dtype))

        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            frequency_output = (
                weight.float()
                * v_fft
            )
            spatial_output = torch.abs(
                torch.fft.ifft2(frequency_output)
            )

        spatial_output = spatial_output.to(dtype=input_dtype)
        spatial_output = self.activation(
            self.norm(spatial_output)
        )

        return self.project_out(
            x
            + torch.sigmoid(self.control)
            * spatial_output
        )


class MutualAssistanceChannelAttention(nn.Module):
    """Official FreMaNet MaCA behavior for three adjacent fused levels."""

    def __init__(
        self,
        channels: int,
        ratio: int = 8,
    ) -> None:
        super().__init__()
        hidden = max(channels // ratio, 1)

        self.pool = nn.AdaptiveMaxPool2d(1)

        def assistance_gate() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels * 2, hidden, kernel_size=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1),
                nn.Sigmoid(),
            )

        def self_gate() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1),
                nn.Sigmoid(),
            )

        self.assistance_gates = nn.ModuleList(
            [assistance_gate() for _ in range(3)]
        )
        self.self_gates = nn.ModuleList(
            [self_gate() for _ in range(3)]
        )
        self.controls = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, dtype=torch.float32))
                for _ in range(3)
            ]
        )

    def forward(
        self,
        x3: torch.Tensor,
        x2: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = [x3, x2, x1]
        descriptors = [self.pool(feature) for feature in features]
        outputs: list[torch.Tensor] = []

        for index, feature in enumerate(features):
            others = [
                descriptors[j]
                for j in range(3)
                if j != index
            ]
            assistance = self.assistance_gates[index](
                torch.cat(others, dim=1)
            )
            self_weight = self.self_gates[index](
                descriptors[index]
            )
            alpha = torch.sigmoid(self.controls[index])
            gate = (
                (1.0 - alpha) * self_weight
                + alpha * assistance
            )
            outputs.append(feature * gate + feature)

        return outputs[0], outputs[1], outputs[2]


class PDecoder(nn.Module):
    """Progressive decoder from the official FreMaNet implementation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_upsample1 = DepthwiseSeparableConv(channels, channels)
        self.conv_upsample2 = DepthwiseSeparableConv(channels, channels)
        self.conv_upsample3 = DepthwiseSeparableConv(channels, channels)
        self.conv_upsample4 = DepthwiseSeparableConv(channels, channels)
        self.conv_upsample5 = DepthwiseSeparableConv(
            channels * 2,
            channels * 2,
        )
        self.conv_concat2 = DepthwiseSeparableConv(
            channels * 2,
            channels * 2,
        )
        self.conv_concat3 = DepthwiseSeparableConv(
            channels * 3,
            channels * 3,
        )
        self.conv4 = DepthwiseSeparableConv(
            channels * 3,
            channels * 3,
        )
        self.prediction = nn.Conv2d(
            channels * 3,
            1,
            kernel_size=1,
        )

    @staticmethod
    def upsample(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=True,
        )

    def forward_with_intermediate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x1_1 = x1
        x2_1 = self.conv_upsample1(
            self.upsample(x1)
        ) * x2
        x3_1 = (
            self.conv_upsample2(
                self.upsample(self.upsample(x1))
            )
            * self.conv_upsample3(
                self.upsample(x2)
            )
            * x3
        )

        x2_2 = self.conv_concat2(
            torch.cat(
                (
                    x2_1,
                    self.conv_upsample4(
                        self.upsample(x1_1)
                    ),
                ),
                dim=1,
            )
        )

        x3_2 = self.conv_concat3(
            torch.cat(
                (
                    x3_1,
                    self.conv_upsample5(
                        self.upsample(x2_2)
                    ),
                ),
                dim=1,
            )
        )

        prediction = self.prediction(
            self.conv4(x3_2)
        )

        return prediction, x2_2

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
    ) -> torch.Tensor:
        prediction, _ = self.forward_with_intermediate(
            x1=x1,
            x2=x2,
            x3=x3,
        )
        return prediction


class FreMaNetHead(nn.Module):
    """FreSA + MaCA + PDecoder, independent of the backbone implementation."""

    def __init__(
        self,
        in_channels: Sequence[int],
        channel: int = 32,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(
                "FreMaNetHead expects four backbone stages."
            )

        self.normalizers = nn.ModuleList(
            [
                DepthwiseSeparableConvAct(
                    int(stage_channels),
                    channel,
                )
                for stage_channels in in_channels
            ]
        )
        self.fresa = nn.ModuleList(
            [FreSA(channel) for _ in range(4)]
        )
        self.maca = MutualAssistanceChannelAttention(channel)
        self.decoder = PDecoder(channel)

    @staticmethod
    def _upsample_to(
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

    def _prepare_decoder_features(
        self,
        features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(features) != 4:
            raise ValueError(
                "FreMaNetHead expects exactly four feature tensors."
            )

        normalized = [
            layer(feature)
            for layer, feature in zip(
                self.normalizers,
                features,
                strict=True,
            )
        ]
        enhanced = [
            layer(feature)
            for layer, feature in zip(
                self.fresa,
                normalized,
                strict=True,
            )
        ]
        x1, x2, x3, x4 = enhanced

        return self.maca(
            self._upsample_to(x4, x3) + x3,
            self._upsample_to(x3, x2) + x2,
            self._upsample_to(x2, x1) + x1,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        x3_maca, x2_maca, x1_maca = (
            self._prepare_decoder_features(features)
        )

        prediction = self.decoder(
            x3_maca,
            x2_maca,
            x1_maca,
        )

        return F.interpolate(
            prediction,
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )


class FreMaNetAuxHead(FreMaNetHead):
    """CFMNet research head with MLab deep supervision.

    The original FreMaNet main path stays unchanged. Auxiliary logits are
    attached to two semantically distinct decoder states:
      - mid decoder fusion x2_2 (stride 8),
      - coarse MaCA feature x3_maca (stride 16).
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        channel: int = 32,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            channel=channel,
        )
        self.aux_mid = nn.Conv2d(
            channel * 2,
            1,
            kernel_size=1,
        )
        self.aux_coarse = nn.Conv2d(
            channel,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: tuple[int, int],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x3_maca, x2_maca, x1_maca = (
            self._prepare_decoder_features(features)
        )

        prediction, decoder_mid = (
            self.decoder.forward_with_intermediate(
                x1=x3_maca,
                x2=x2_maca,
                x3=x1_maca,
            )
        )

        prediction = F.interpolate(
            prediction,
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )

        auxiliary = [
            self.aux_mid(decoder_mid),
            self.aux_coarse(x3_maca),
        ]

        return prediction, auxiliary
