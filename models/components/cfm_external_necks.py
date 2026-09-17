# models/components/cfm_external_necks.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2

        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU6(
                inplace=True
            ),
        )


class ESMAdapter(nn.Module):
    """
    Neck-adapted structural residual module inspired by the
    S²AM ESM/SCLD structural modeling idea.

    For one fused CFMNet feature:
        R_k = F - AvgPool_k(F), k in {3, 5, 7}

    Multi-scale structural residuals are fused, while an
    independently predicted consistency gate decides where the
    residual correction should be injected.

    The scalar residual scale starts at zero, so the initial
    network is exactly the original CFMNet + UNetFormer baseline.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.residual_fuse = nn.Sequential(
            ConvBNAct(
                channels * 3,
                channels,
                kernel_size=1,
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels
            ),
        )

        self.consistency = nn.Sequential(
            ConvBNAct(
                channels * 3,
                channels,
                kernel_size=1,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.scale = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residuals = []

        for kernel_size in (
            3,
            5,
            7,
        ):
            smooth = F.avg_pool2d(
                x,
                kernel_size=kernel_size,
                stride=1,
                padding=(
                    kernel_size // 2
                ),
            )

            residuals.append(
                x - smooth
            )

        residual_stack = torch.cat(
            residuals,
            dim=1,
        )

        consistency_stack = torch.cat(
            [
                residual.abs()
                for residual
                in residuals
            ],
            dim=1,
        )

        delta = self.residual_fuse(
            residual_stack
        )

        gate = self.consistency(
            consistency_stack
        )

        return (
            x
            + self.scale
            * gate
            * delta
        )


class ESMNeck(nn.Module):
    def __init__(
        self,
        channels: int,
        stage_index: int = 1,
    ) -> None:
        super().__init__()

        self.stage_index = (
            stage_index
        )

        self.adapter = ESMAdapter(
            channels
        )

    def forward(
        self,
        features: list[
            torch.Tensor
        ]
        | tuple[
            torch.Tensor,
            ...,
        ],
    ) -> list[
        torch.Tensor
    ]:
        output = list(
            features
        )

        output[
            self.stage_index
        ] = self.adapter(
            output[
                self.stage_index
            ]
        )

        return output


class SDRAdapter(nn.Module):
    """
    Neck-adapted smooth/detail recalibration inspired by SPLG-Mamba SDR.

    A fused stage feature is decomposed as:
        smooth = AvgPool(F)
        detail = F - smooth

    The module predicts a content-dependent gate for the detail
    component, then learns a residual correction from
    [smooth, recalibrated_detail].
    """

    def __init__(
        self,
        channels: int,
        smooth_kernel: int = 3,
    ) -> None:
        super().__init__()

        self.smooth_kernel = (
            smooth_kernel
        )

        self.detail_gate = nn.Sequential(
            ConvBNAct(
                channels * 2,
                channels,
                kernel_size=1,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.correction = nn.Sequential(
            ConvBNAct(
                channels * 2,
                channels,
                kernel_size=1,
            ),
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels
            ),
        )

        self.scale = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        smooth = F.avg_pool2d(
            x,
            kernel_size=(
                self.smooth_kernel
            ),
            stride=1,
            padding=(
                self.smooth_kernel
                // 2
            ),
        )

        detail = (
            x - smooth
        )

        relation = torch.cat(
            [
                smooth,
                detail,
            ],
            dim=1,
        )

        detail_gate = (
            self.detail_gate(
                relation
            )
        )

        clean_detail = (
            detail
            * detail_gate
        )

        delta = self.correction(
            torch.cat(
                [
                    smooth,
                    clean_detail,
                ],
                dim=1,
            )
        )

        return (
            x
            + self.scale
            * delta
        )


class SDRNeck(nn.Module):
    def __init__(
        self,
        channels: int,
        stage_index: int = 1,
    ) -> None:
        super().__init__()

        self.stage_index = (
            stage_index
        )

        self.adapter = SDRAdapter(
            channels
        )

    def forward(
        self,
        features: list[
            torch.Tensor
        ]
        | tuple[
            torch.Tensor,
            ...,
        ],
    ) -> list[
        torch.Tensor
    ]:
        output = list(
            features
        )

        output[
            self.stage_index
        ] = self.adapter(
            output[
                self.stage_index
            ]
        )

        return output


class ChannelMLP(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        hidden = max(
            channels // reduction,
            16,
        )

        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU6(
                inplace=True
            ),
            nn.Conv2d(
                hidden,
                channels,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


class MutualAssistanceChannelAdapter(
    nn.Module
):
    """
    Neck-adapted MaCA-style mutual channel assistance.

    Stage3 and Stage4 keep their own fused CFMNet representations.

    Each stage receives the neighboring feature after channel projection
    and spatial alignment. Parallel channel descriptors are computed
    from:
        1) the stage itself
        2) the assisting neighbor

    Their channel gates modulate self and assistance information before
    a lightweight residual correction is produced.
    """

    def __init__(
        self,
        stage3_channels: int,
        stage4_channels: int,
    ) -> None:
        super().__init__()

        self.stage4_to_stage3 = (
            nn.Conv2d(
                stage4_channels,
                stage3_channels,
                kernel_size=1,
                bias=False,
            )
        )

        self.stage3_to_stage4 = (
            nn.Conv2d(
                stage3_channels,
                stage4_channels,
                kernel_size=1,
                bias=False,
            )
        )

        self.self_attn3 = ChannelMLP(
            stage3_channels
        )

        self.assist_attn3 = (
            ChannelMLP(
                stage3_channels
            )
        )

        self.self_attn4 = ChannelMLP(
            stage4_channels
        )

        self.assist_attn4 = (
            ChannelMLP(
                stage4_channels
            )
        )

        self.refine3 = nn.Sequential(
            ConvBNAct(
                stage3_channels,
                stage3_channels,
                kernel_size=3,
                groups=stage3_channels,
            ),
            nn.Conv2d(
                stage3_channels,
                stage3_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                stage3_channels
            ),
        )

        self.refine4 = nn.Sequential(
            ConvBNAct(
                stage4_channels,
                stage4_channels,
                kernel_size=3,
                groups=stage4_channels,
            ),
            nn.Conv2d(
                stage4_channels,
                stage4_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                stage4_channels
            ),
        )

        self.scale3 = nn.Parameter(
            torch.zeros(1)
        )

        self.scale4 = nn.Parameter(
            torch.zeros(1)
        )

    @staticmethod
    def _channel_descriptor(
        x: torch.Tensor,
    ) -> torch.Tensor:
        return F.adaptive_avg_pool2d(
            x,
            output_size=1,
        )

    def forward(
        self,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        assist3 = (
            self.stage4_to_stage3(
                stage4
            )
        )

        assist3 = F.interpolate(
            assist3,
            size=(
                stage3.shape[-2:]
            ),
            mode="bilinear",
            align_corners=False,
        )

        assist4 = F.interpolate(
            stage3,
            size=(
                stage4.shape[-2:]
            ),
            mode="bilinear",
            align_corners=False,
        )

        assist4 = (
            self.stage3_to_stage4(
                assist4
            )
        )

        self_gate3 = torch.sigmoid(
            self.self_attn3(
                self._channel_descriptor(
                    stage3
                )
            )
        )

        assist_gate3 = torch.sigmoid(
            self.assist_attn3(
                self._channel_descriptor(
                    assist3
                )
            )
        )

        self_gate4 = torch.sigmoid(
            self.self_attn4(
                self._channel_descriptor(
                    stage4
                )
            )
        )

        assist_gate4 = torch.sigmoid(
            self.assist_attn4(
                self._channel_descriptor(
                    assist4
                )
            )
        )

        mixed3 = (
            self_gate3
            * stage3
            + assist_gate3
            * assist3
        )

        mixed4 = (
            self_gate4
            * stage4
            + assist_gate4
            * assist4
        )

        delta3 = self.refine3(
            mixed3
        )

        delta4 = self.refine4(
            mixed4
        )

        return (
            stage3
            + self.scale3
            * delta3,
            stage4
            + self.scale4
            * delta4,
        )


class MaCANeck(nn.Module):
    def __init__(
        self,
        stage3_channels: int,
        stage4_channels: int,
        stage3_index: int = 2,
        stage4_index: int = 3,
    ) -> None:
        super().__init__()

        self.stage3_index = (
            stage3_index
        )

        self.stage4_index = (
            stage4_index
        )

        self.adapter = (
            MutualAssistanceChannelAdapter(
                stage3_channels=(
                    stage3_channels
                ),
                stage4_channels=(
                    stage4_channels
                ),
            )
        )

    def forward(
        self,
        features: list[
            torch.Tensor
        ]
        | tuple[
            torch.Tensor,
            ...,
        ],
    ) -> list[
        torch.Tensor
    ]:
        output = list(
            features
        )

        (
            output[
                self.stage3_index
            ],
            output[
                self.stage4_index
            ],
        ) = self.adapter(
            output[
                self.stage3_index
            ],
            output[
                self.stage4_index
            ],
        )

        return output


class FrequencyReconstructionAdapter(
    nn.Module
):
    """
    Neck-adapted IRAM-style frequency reconstruction.

    The fused feature is transformed with FFT and split by a radial
    low-frequency mask. Low/high spatial reconstructions are weighted
    per channel, reconstructed, spatially attended, and returned as a
    residual correction.

    FFT is forced to float32 for AMP safety, then converted back to the
    feature dtype before learnable convolutions.
    """

    def __init__(
        self,
        channels: int,
        low_frequency_radius: float = 0.25,
    ) -> None:
        super().__init__()

        self.channels = (
            channels
        )

        self.low_frequency_radius = (
            low_frequency_radius
        )

        hidden = max(
            channels // 4,
            16,
        )

        self.frequency_weight = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU6(
                inplace=True
            ),
            nn.Conv2d(
                hidden,
                channels * 2,
                kernel_size=1,
                bias=True,
            ),
        )

        self.spatial_attention = (
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    1,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                nn.Sigmoid(),
            )
        )

        self.refine = nn.Sequential(
            ConvBNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
            ),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                channels
            ),
        )

        self.scale = nn.Parameter(
            torch.zeros(1)
        )

    def _frequency_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        fy = torch.fft.fftfreq(
            height,
            device=device,
            dtype=torch.float32,
        )

        fx = torch.fft.fftfreq(
            width,
            device=device,
            dtype=torch.float32,
        )

        grid_y, grid_x = (
            torch.meshgrid(
                fy,
                fx,
                indexing="ij",
            )
        )

        radius = torch.sqrt(
            grid_x.square()
            + grid_y.square()
        )

        return (
            radius
            <= self.low_frequency_radius
        ).float().view(
            1,
            1,
            height,
            width,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype

        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            x_float = x.float()

            spectrum = torch.fft.fft2(
                x_float,
                norm="ortho",
            )

            mask = self._frequency_mask(
                height=x.shape[-2],
                width=x.shape[-1],
                device=x.device,
            )

            low = torch.fft.ifft2(
                spectrum
                * mask,
                norm="ortho",
            ).real

            high = torch.fft.ifft2(
                spectrum
                * (1.0 - mask),
                norm="ortho",
            ).real

        low = low.to(
            dtype=input_dtype
        )

        high = high.to(
            dtype=input_dtype
        )

        stats = torch.cat(
            [
                F.adaptive_avg_pool2d(
                    low.abs(),
                    1,
                ),
                F.adaptive_avg_pool2d(
                    high.abs(),
                    1,
                ),
            ],
            dim=1,
        )

        weights = (
            self.frequency_weight(
                stats
            )
        )

        weights = weights.view(
            x.shape[0],
            2,
            self.channels,
            1,
            1,
        )

        weights = F.softmax(
            weights,
            dim=1,
        )

        reconstructed = (
            weights[:, 0]
            * low
            + weights[:, 1]
            * high
        )

        spatial_gate = (
            self.spatial_attention(
                reconstructed
            )
        )

        delta = self.refine(
            reconstructed
            * spatial_gate
        )

        return (
            x
            + self.scale
            * delta
        )


class IRAMNeck(nn.Module):
    def __init__(
        self,
        stage2_channels: int,
        stage3_channels: int,
        stage2_index: int = 1,
        stage3_index: int = 2,
    ) -> None:
        super().__init__()

        self.stage2_index = (
            stage2_index
        )

        self.stage3_index = (
            stage3_index
        )

        self.stage2_adapter = (
            FrequencyReconstructionAdapter(
                stage2_channels
            )
        )

        self.stage3_adapter = (
            FrequencyReconstructionAdapter(
                stage3_channels
            )
        )

    def forward(
        self,
        features: list[
            torch.Tensor
        ]
        | tuple[
            torch.Tensor,
            ...,
        ],
    ) -> list[
        torch.Tensor
    ]:
        output = list(
            features
        )

        output[
            self.stage2_index
        ] = self.stage2_adapter(
            output[
                self.stage2_index
            ]
        )

        output[
            self.stage3_index
        ] = self.stage3_adapter(
            output[
                self.stage3_index
            ]
        )

        return output
