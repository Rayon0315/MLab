# models/components/cfm_iram_selective_necks.py

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


class IRAMResidualCore(nn.Module):
    """
    Frequency-domain residual generator.

    This keeps the same experimental idea as the previous IRAM neck:
      1. FFT
      2. radial low/high-frequency split
      3. channel-wise adaptive low/high weighting
      4. spatial attention
      5. lightweight reconstruction

    It returns only delta_F. The outer selective adapter decides where
    that residual is allowed to modify the fused CFMNet feature.
    """

    def __init__(
        self,
        channels: int,
        low_frequency_radius: float = 0.25,
    ) -> None:
        super().__init__()

        self.channels = channels
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

        grid_y, grid_x = torch.meshgrid(
            fy,
            fx,
            indexing="ij",
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

        statistics = torch.cat(
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
                statistics
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

        return self.refine(
            reconstructed
            * spatial_gate
        )


class SelectiveIRAMAdapter(nn.Module):
    """
    Apply an IRAM residual only where an external gate allows it.

        F' = F + alpha * gate * delta_F_IRAM

    alpha starts from zero, so every new experiment starts exactly from
    the canonical CFMNet + UNetFormer feature interface.
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()

        self.iram = IRAMResidualCore(
            channels
        )

        self.scale = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        delta = self.iram(
            x
        )

        gate = F.interpolate(
            gate,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(
            min=0.0,
            max=1.0,
        )

        return (
            x
            + self.scale
            * gate
            * delta
        )


class PrototypeDiscriminationGate(nn.Module):
    """
    Use Stage4 as a semantic anchor.

    A learned coarse saliency score separates Stage4 tokens into soft
    foreground/background sets. Their weighted feature means form
    FG/BG prototypes.

    For each target stage:
        d = cos(F, p_fg) - cos(F, p_bg)

    We use the absolute discrimination confidence as the residual gate.
    High confidence in either foreground or background means the IRAM
    evidence is semantically interpretable; ambiguous locations receive
    weaker IRAM injection.

    gate = sigmoid(temperature * |d|)

    The minimum initial gate is 0.5 rather than 0, which avoids a dead
    path while the outer residual scale is still learning from zero.
    """

    def __init__(
        self,
        stage2_channels: int,
        stage3_channels: int,
        stage4_channels: int,
        embedding_channels: int = 128,
        temperature: float = 3.0,
    ) -> None:
        super().__init__()

        self.temperature = (
            temperature
        )

        self.semantic_score = nn.Conv2d(
            stage4_channels,
            1,
            kernel_size=1,
            bias=True,
        )

        nn.init.zeros_(
            self.semantic_score.weight
        )

        nn.init.zeros_(
            self.semantic_score.bias
        )

        self.anchor_proj = nn.Conv2d(
            stage4_channels,
            embedding_channels,
            kernel_size=1,
            bias=False,
        )

        self.stage2_proj = nn.Conv2d(
            stage2_channels,
            embedding_channels,
            kernel_size=1,
            bias=False,
        )

        self.stage3_proj = nn.Conv2d(
            stage3_channels,
            embedding_channels,
            kernel_size=1,
            bias=False,
        )

    @staticmethod
    def _weighted_prototype(
        feature: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        numerator = (
            feature
            * weight
        ).sum(
            dim=(-2, -1),
            keepdim=True,
        )

        denominator = weight.sum(
            dim=(-2, -1),
            keepdim=True,
        )

        return (
            numerator
            / (
                denominator
                + epsilon
            )
        )

    def _target_gate(
        self,
        target: torch.Tensor,
        foreground_prototype: torch.Tensor,
        background_prototype: torch.Tensor,
        projection: nn.Module,
    ) -> torch.Tensor:
        embedded = projection(
            target
        )

        embedded = F.normalize(
            embedded,
            dim=1,
            eps=1e-6,
        )

        foreground_prototype = (
            F.normalize(
                foreground_prototype,
                dim=1,
                eps=1e-6,
            )
        )

        background_prototype = (
            F.normalize(
                background_prototype,
                dim=1,
                eps=1e-6,
            )
        )

        foreground_similarity = (
            embedded
            * foreground_prototype
        ).sum(
            dim=1,
            keepdim=True,
        )

        background_similarity = (
            embedded
            * background_prototype
        ).sum(
            dim=1,
            keepdim=True,
        )

        difference = (
            foreground_similarity
            - background_similarity
        )

        return torch.sigmoid(
            self.temperature
            * difference.abs()
        )

    def forward(
        self,
        stage2: torch.Tensor,
        stage3: torch.Tensor,
        stage4: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        anchor = self.anchor_proj(
            stage4
        )

        probability = torch.sigmoid(
            self.semantic_score(
                stage4
            )
        )

        foreground_prototype = (
            self._weighted_prototype(
                anchor,
                probability,
            )
        )

        background_prototype = (
            self._weighted_prototype(
                anchor,
                1.0 - probability,
            )
        )

        gate2 = self._target_gate(
            stage2,
            foreground_prototype,
            background_prototype,
            self.stage2_proj,
        )

        gate3 = self._target_gate(
            stage3,
            foreground_prototype,
            background_prototype,
            self.stage3_proj,
        )

        return (
            gate2,
            gate3,
        )


class CoarseSemanticScore(nn.Module):
    """
    Lightweight Stage4 coarse saliency confidence head.

    It is not used as a prediction output and does not change the loss.
    It only provides a spatial control signal for selective IRAM.
    """

    def __init__(
        self,
        stage4_channels: int,
    ) -> None:
        super().__init__()

        self.head = nn.Conv2d(
            stage4_channels,
            1,
            kernel_size=1,
            bias=True,
        )

        nn.init.zeros_(
            self.head.weight
        )

        nn.init.zeros_(
            self.head.bias
        )

    def forward(
        self,
        stage4: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.head(
                stage4
            )
        )


class IRAMPrototypeGateNeck(nn.Module):
    """
    IRAM + foreground/background prototype discrimination confidence.
    """

    def __init__(
        self,
        stage2_channels: int,
        stage3_channels: int,
        stage4_channels: int,
        embedding_channels: int = 128,
        stage2_index: int = 1,
        stage3_index: int = 2,
        stage4_index: int = 3,
    ) -> None:
        super().__init__()

        self.stage2_index = stage2_index
        self.stage3_index = stage3_index
        self.stage4_index = stage4_index

        self.gate = PrototypeDiscriminationGate(
            stage2_channels=(
                stage2_channels
            ),
            stage3_channels=(
                stage3_channels
            ),
            stage4_channels=(
                stage4_channels
            ),
            embedding_channels=(
                embedding_channels
            ),
        )

        self.stage2_adapter = (
            SelectiveIRAMAdapter(
                stage2_channels
            )
        )

        self.stage3_adapter = (
            SelectiveIRAMAdapter(
                stage3_channels
            )
        )

        self.last_gates: dict[
            str,
            torch.Tensor,
        ] = {}

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

        stage2 = output[
            self.stage2_index
        ]

        stage3 = output[
            self.stage3_index
        ]

        stage4 = output[
            self.stage4_index
        ]

        gate2, gate3 = self.gate(
            stage2,
            stage3,
            stage4,
        )

        output[
            self.stage2_index
        ] = self.stage2_adapter(
            stage2,
            gate2,
        )

        output[
            self.stage3_index
        ] = self.stage3_adapter(
            stage3,
            gate3,
        )

        self.last_gates = {
            "stage2": gate2.detach(),
            "stage3": gate3.detach(),
        }

        return output


class IRAMUncertaintyGateNeck(nn.Module):
    """
    IRAM + Stage4 semantic uncertainty gate.

        p = sigmoid(h(F4))
        U = 4 p (1-p)

    The IRAM residual is concentrated on locations where the coarse
    Stage4 semantic evidence is uncertain, while confident regions are
    progressively protected from unnecessary frequency correction.
    """

    def __init__(
        self,
        stage2_channels: int,
        stage3_channels: int,
        stage4_channels: int,
        stage2_index: int = 1,
        stage3_index: int = 2,
        stage4_index: int = 3,
    ) -> None:
        super().__init__()

        self.stage2_index = stage2_index
        self.stage3_index = stage3_index
        self.stage4_index = stage4_index

        self.semantic_score = (
            CoarseSemanticScore(
                stage4_channels
            )
        )

        self.stage2_adapter = (
            SelectiveIRAMAdapter(
                stage2_channels
            )
        )

        self.stage3_adapter = (
            SelectiveIRAMAdapter(
                stage3_channels
            )
        )

        self.last_gates: dict[
            str,
            torch.Tensor,
        ] = {}

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

        probability = (
            self.semantic_score(
                output[
                    self.stage4_index
                ]
            )
        )

        uncertainty = (
            4.0
            * probability
            * (
                1.0
                - probability
            )
        )

        gate2 = F.interpolate(
            uncertainty,
            size=output[
                self.stage2_index
            ].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(
            min=0.0,
            max=1.0,
        )

        gate3 = F.interpolate(
            uncertainty,
            size=output[
                self.stage3_index
            ].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(
            min=0.0,
            max=1.0,
        )

        output[
            self.stage2_index
        ] = self.stage2_adapter(
            output[
                self.stage2_index
            ],
            gate2,
        )

        output[
            self.stage3_index
        ] = self.stage3_adapter(
            output[
                self.stage3_index
            ],
            gate3,
        )

        self.last_gates = {
            "stage2": gate2.detach(),
            "stage3": gate3.detach(),
        }

        return output


class IRAMRegionConsistencyGateNeck(
    nn.Module
):
    """
    IRAM + region-consistency gate.

    A Stage4 coarse semantic map defines regional confidence. A location
    receives a high gate when its score agrees with the local region:

        C = 1 - |p - AvgPool(p)|

    This suppresses isolated/locally inconsistent frequency corrections
    without introducing a new spatial enhancement stream.
    """

    def __init__(
        self,
        stage2_channels: int,
        stage3_channels: int,
        stage4_channels: int,
        stage2_index: int = 1,
        stage3_index: int = 2,
        stage4_index: int = 3,
        region_kernel: int = 3,
    ) -> None:
        super().__init__()

        self.stage2_index = stage2_index
        self.stage3_index = stage3_index
        self.stage4_index = stage4_index
        self.region_kernel = (
            region_kernel
        )

        self.semantic_score = (
            CoarseSemanticScore(
                stage4_channels
            )
        )

        self.stage2_adapter = (
            SelectiveIRAMAdapter(
                stage2_channels
            )
        )

        self.stage3_adapter = (
            SelectiveIRAMAdapter(
                stage3_channels
            )
        )

        self.last_gates: dict[
            str,
            torch.Tensor,
        ] = {}

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

        probability = (
            self.semantic_score(
                output[
                    self.stage4_index
                ]
            )
        )

        local_mean = F.avg_pool2d(
            probability,
            kernel_size=(
                self.region_kernel
            ),
            stride=1,
            padding=(
                self.region_kernel
                // 2
            ),
        )

        consistency = (
            1.0
            - (
                probability
                - local_mean
            ).abs()
        ).clamp(
            min=0.0,
            max=1.0,
        )

        gate2 = F.interpolate(
            consistency,
            size=output[
                self.stage2_index
            ].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(
            min=0.0,
            max=1.0,
        )

        gate3 = F.interpolate(
            consistency,
            size=output[
                self.stage3_index
            ].shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(
            min=0.0,
            max=1.0,
        )

        output[
            self.stage2_index
        ] = self.stage2_adapter(
            output[
                self.stage2_index
            ],
            gate2,
        )

        output[
            self.stage3_index
        ] = self.stage3_adapter(
            output[
                self.stage3_index
            ],
            gate3,
        )

        self.last_gates = {
            "stage2": gate2.detach(),
            "stage3": gate3.detach(),
        }

        return output
