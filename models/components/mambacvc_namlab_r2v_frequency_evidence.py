from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.mambacvc_blocks import ConvBNReLU
from models.components.mambacvc_namlab_local_cross import (
    RegionMeanPreservingEncoder,
)
from models.components.mambacvc_namlab_r2v_local import (
    LocalRegionToVisualInteraction,
)


class FrequencyEvidenceAdaptiveFusion(nn.Module):
    """Use fixed low/high-frequency descriptors to weight F, R, and R->F.

    Frequency is used only as evidence for source reliability. It does not
    reconstruct or replace any of the three source features.

    For each source X in {F, R, G}:
        X -> fixed FFT low/high decomposition
          -> mean absolute low/high spatial evidence (2 channels)

    The six evidence maps predict three spatial softmax weights:
        [w_F, w_R, w_G],  w_F + w_R + w_G = 1 at every pixel.
    """

    def __init__(
        self,
        channels: int = 32,
        hidden_channels: int = 16,
        low_frequency_radius: float = 0.25,
    ) -> None:
        super().__init__()

        if channels < 1:
            raise ValueError("channels must be positive")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if not 0.0 < low_frequency_radius < 0.5:
            raise ValueError("low_frequency_radius must be in (0, 0.5)")

        self.channels = int(channels)
        self.low_frequency_radius = float(low_frequency_radius)

        self.controller = nn.Sequential(
            ConvBNReLU(6, hidden_channels, 3),
            nn.Conv2d(
                hidden_channels,
                3,
                kernel_size=1,
                bias=True,
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
            grid_x.square() + grid_y.square()
        )
        return (
            radius <= self.low_frequency_radius
        ).float().view(1, 1, height, width)

    def frequency_descriptor(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        if feature.ndim != 4:
            raise ValueError("feature must be BCHW")
        if feature.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} channels, got {feature.shape[1]}"
            )

        input_dtype = feature.dtype
        height, width = feature.shape[-2:]

        with torch.autocast(
            device_type=feature.device.type,
            enabled=False,
        ):
            feature_float = feature.float()
            spectrum = torch.fft.fft2(
                feature_float,
                norm="ortho",
            )
            mask = self._frequency_mask(
                height=height,
                width=width,
                device=feature.device,
            )
            low = torch.fft.ifft2(
                spectrum * mask,
                norm="ortho",
            ).real
            high = torch.fft.ifft2(
                spectrum * (1.0 - mask),
                norm="ortho",
            ).real

            descriptor = torch.cat(
                [
                    low.abs().mean(dim=1, keepdim=True),
                    high.abs().mean(dim=1, keepdim=True),
                ],
                dim=1,
            )

        return descriptor.to(dtype=input_dtype)

    def forward(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
        selected_evidence: torch.Tensor,
    ) -> torch.Tensor:
        if not (
            visual_feature.shape
            == region_feature.shape
            == selected_evidence.shape
        ):
            raise ValueError("F, R, and selected R->F evidence must match")

        evidence = torch.cat(
            [
                self.frequency_descriptor(visual_feature),
                self.frequency_descriptor(region_feature),
                self.frequency_descriptor(selected_evidence),
            ],
            dim=1,
        )
        logits = self.controller(evidence)
        return F.softmax(logits, dim=1)


class FrequencyEvidenceLocalRegionToVisualInteraction(
    LocalRegionToVisualInteraction
):
    """Validated [F, R, R->F] fusion with frequency-evidence source weights."""

    def __init__(
        self,
        channels: int = 32,
        inner_channels: int = 16,
        num_heads: int = 4,
        window_size: int = 3,
        frequency_hidden_channels: int = 16,
        low_frequency_radius: float = 0.25,
    ) -> None:
        super().__init__(
            channels=channels,
            inner_channels=inner_channels,
            num_heads=num_heads,
            window_size=window_size,
        )
        self.frequency_evidence = FrequencyEvidenceAdaptiveFusion(
            channels=channels,
            hidden_channels=frequency_hidden_channels,
            low_frequency_radius=low_frequency_radius,
        )

    def forward(
        self,
        visual_feature: torch.Tensor,
        region_feature: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        selected_evidence = self.compute_region_to_visual(
            visual_feature=visual_feature,
            region_feature=region_feature,
            mean_60=mean_60,
        )
        weights = self.frequency_evidence(
            visual_feature=visual_feature,
            region_feature=region_feature,
            selected_evidence=selected_evidence,
        )

        fused = torch.cat(
            [
                weights[:, 0:1] * visual_feature,
                weights[:, 1:2] * region_feature,
                weights[:, 2:3] * selected_evidence,
            ],
            dim=1,
        )
        return self.reconstruction(fused)


class RegionMeanToVisualFrequencyEvidenceGuidance(nn.Module):
    """Encode M60, compute R->F evidence, then frequency-weight F/R/R->F."""

    def __init__(
        self,
        channels: int = 32,
        inner_channels: int = 16,
        num_heads: int = 4,
        window_size: int = 3,
        frequency_hidden_channels: int = 16,
        low_frequency_radius: float = 0.25,
    ) -> None:
        super().__init__()
        self.region_encoder = RegionMeanPreservingEncoder(
            channels=channels
        )
        self.interaction = FrequencyEvidenceLocalRegionToVisualInteraction(
            channels=channels,
            inner_channels=inner_channels,
            num_heads=num_heads,
            window_size=window_size,
            frequency_hidden_channels=frequency_hidden_channels,
            low_frequency_radius=low_frequency_radius,
        )

    def forward(
        self,
        feature: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> torch.Tensor:
        region = self.region_encoder(
            mean_60,
            target_size=feature.shape[-2:],
        )
        return self.interaction(
            visual_feature=feature,
            region_feature=region,
            mean_60=mean_60,
        )
