from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.blocks import (
    InformationPerturbation,
    PriorConditionBlock,
    SpatialSelfAttention,
    TimeResBlock,
    timestep_embedding,
)


class MultiPriorDenoiser(nn.Module):
    """
    IPDiff-style hierarchical conditional denoiser.

    Inputs at full 352x352 resolution:
      noisy mask x_t (1ch)
      Hybrid saliency prior (1ch)
      raw M60 region-mean image (3ch)

    Hierarchical conditions:
      prior1: 88x88
      prior2: 44x44
      prior3: 22x22
      prior4: 11x11
    """

    def __init__(
        self,
        prior_channels: int = 64,
        time_dim: int = 256,
        ipm_probability: float = 0.2,
    ) -> None:
        super().__init__()

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        self.stem = nn.Conv2d(
            5,
            32,
            kernel_size=3,
            padding=1,
        )

        self.enc352 = TimeResBlock(32, 32, time_dim)
        self.down176 = nn.Conv2d(
            32,
            48,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.enc176 = TimeResBlock(48, 48, time_dim)

        self.down88 = nn.Conv2d(
            48,
            64,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.enc88 = TimeResBlock(64, 64, time_dim)
        self.cond88 = PriorConditionBlock(prior_channels, 64)

        self.down44 = nn.Conv2d(
            64,
            128,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.enc44 = TimeResBlock(128, 128, time_dim)
        self.cond44 = PriorConditionBlock(prior_channels, 128)

        self.down22 = nn.Conv2d(
            128,
            256,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.enc22 = TimeResBlock(256, 256, time_dim)
        self.cond22 = PriorConditionBlock(prior_channels, 256)

        self.down11 = nn.Conv2d(
            256,
            512,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.enc11 = TimeResBlock(512, 512, time_dim)
        self.cond11 = PriorConditionBlock(prior_channels, 512)

        self.ipm88 = InformationPerturbation(ipm_probability)
        self.ipm44 = InformationPerturbation(ipm_probability)
        self.ipm22 = InformationPerturbation(ipm_probability)
        self.ipm11 = InformationPerturbation(ipm_probability)

        self.mid1 = TimeResBlock(512, 512, time_dim)
        self.mid_attention = SpatialSelfAttention(512, heads=8)
        self.mid2 = TimeResBlock(512, 512, time_dim)

        self.up22_projection = nn.Conv2d(512, 256, kernel_size=1)
        self.dec22 = TimeResBlock(512, 256, time_dim)

        self.up44_projection = nn.Conv2d(256, 128, kernel_size=1)
        self.dec44 = TimeResBlock(256, 128, time_dim)

        self.up88_projection = nn.Conv2d(128, 64, kernel_size=1)
        self.dec88 = TimeResBlock(128, 64, time_dim)

        self.up176_projection = nn.Conv2d(64, 48, kernel_size=1)
        self.dec176 = TimeResBlock(96, 48, time_dim)

        self.up352_projection = nn.Conv2d(48, 32, kernel_size=1)
        self.dec352 = TimeResBlock(64, 32, time_dim)

        self.output = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(
                32,
                1,
                kernel_size=3,
                padding=1,
            ),
        )

    @staticmethod
    def _upsample(
        feature: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        return F.interpolate(
            feature,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        noisy_mask: torch.Tensor,
        timesteps: torch.Tensor,
        saliency_prior: torch.Tensor,
        mean_60: torch.Tensor,
        priors: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        time = self.time_mlp(
            timestep_embedding(
                timesteps,
                self.time_dim,
            )
        )

        x = torch.cat(
            [
                noisy_mask,
                saliency_prior,
                mean_60,
            ],
            dim=1,
        )

        s352 = self.enc352(self.stem(x), time)
        s176 = self.enc176(self.down176(s352), time)

        s88 = self.enc88(self.down88(s176), time)
        s88 = self.cond88(s88, priors[0])
        s88 = self.ipm88(s88)

        s44 = self.enc44(self.down44(s88), time)
        s44 = self.cond44(s44, priors[1])
        s44 = self.ipm44(s44)

        s22 = self.enc22(self.down22(s44), time)
        s22 = self.cond22(s22, priors[2])
        s22 = self.ipm22(s22)

        s11 = self.enc11(self.down11(s22), time)
        s11 = self.cond11(s11, priors[3])
        s11 = self.ipm11(s11)

        x = self.mid1(s11, time)
        x = self.mid_attention(x)
        x = self.mid2(x, time)

        x = self.up22_projection(
            self._upsample(x, s22.shape[-2:])
        )
        x = self.dec22(
            torch.cat([x, s22], dim=1),
            time,
        )

        x = self.up44_projection(
            self._upsample(x, s44.shape[-2:])
        )
        x = self.dec44(
            torch.cat([x, s44], dim=1),
            time,
        )

        x = self.up88_projection(
            self._upsample(x, s88.shape[-2:])
        )
        x = self.dec88(
            torch.cat([x, s88], dim=1),
            time,
        )

        x = self.up176_projection(
            self._upsample(x, s176.shape[-2:])
        )
        x = self.dec176(
            torch.cat([x, s176], dim=1),
            time,
        )

        x = self.up352_projection(
            self._upsample(x, s352.shape[-2:])
        )
        x = self.dec352(
            torch.cat([x, s352], dim=1),
            time,
        )

        return self.output(x)
