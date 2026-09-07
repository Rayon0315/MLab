from __future__ import annotations

import torch
import torch.nn as nn


class RegionSpectralPriorReconstruction(nn.Module):
    """
    Reconstruct a compact diffusion prior from four evidence streams:
      1) raw MambaVision feature
      2) region-enhanced feature
      3) Hybrid Region Mean feature
      4) progressive-decoder feature

    The spatial-spectral branch is inspired by IPDiff's information
    reconstruction idea, while remaining self-contained and lightweight
    enough to integrate into MLab without extra dependencies.
    """

    def __init__(
        self,
        input_channels: tuple[int, int, int, int],
        out_channels: int = 64,
        low_frequency_cutoff: float = 0.18,
    ) -> None:
        super().__init__()

        self.low_frequency_cutoff = low_frequency_cutoff

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                )
                for channels in input_channels
            ]
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                out_channels * 4,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

        self.low_gain = nn.Parameter(torch.ones(1))
        self.high_gain = nn.Parameter(torch.ones(1))

        self.reconstruction_gate = nn.Sequential(
            nn.Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        self.output = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def _spectral_reconstruct(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        height, width = feature.shape[-2:]

        spectrum = torch.fft.rfft2(
            feature.float(),
            norm="ortho",
        )

        fy = torch.fft.fftfreq(
            height,
            device=feature.device,
        ).abs()[:, None]
        fx = torch.fft.rfftfreq(
            width,
            device=feature.device,
        ).abs()[None, :]
        radius = torch.sqrt(fy.square() + fx.square())

        low_mask = torch.sigmoid(
            (self.low_frequency_cutoff - radius) * 40.0
        )[None, None]

        frequency_weight = (
            low_mask * self.low_gain
            + (1.0 - low_mask) * self.high_gain
        )

        reconstructed = torch.fft.irfft2(
            spectrum * frequency_weight,
            s=(height, width),
            norm="ortho",
        )

        return reconstructed.to(feature.dtype)

    def forward(
        self,
        raw_visual: torch.Tensor,
        enhanced_visual: torch.Tensor,
        region_feature: torch.Tensor,
        decoder_feature: torch.Tensor,
    ) -> torch.Tensor:
        streams = (
            raw_visual,
            enhanced_visual,
            region_feature,
            decoder_feature,
        )

        projected = [
            projection(stream)
            for projection, stream in zip(
                self.projections,
                streams,
            )
        ]

        fused = self.fusion(
            torch.cat(projected, dim=1)
        )
        reconstructed = self._spectral_reconstruct(fused)

        gate = self.reconstruction_gate(
            torch.cat(
                [fused, reconstructed],
                dim=1,
            )
        )

        return self.output(
            fused + gate * reconstructed
        )
