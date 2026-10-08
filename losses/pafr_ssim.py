from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _gaussian(window_size: int, sigma: float) -> Tensor:
    values = [
        exp(-((x - window_size // 2) ** 2) / (2.0 * sigma * sigma))
        for x in range(window_size)
    ]
    kernel = torch.tensor(values, dtype=torch.float32)
    return kernel / kernel.sum()


def _create_window(window_size: int, channels: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    one_d = _gaussian(window_size, 1.5).unsqueeze(1)
    two_d = one_d.mm(one_d.t()).unsqueeze(0).unsqueeze(0)
    return two_d.expand(channels, 1, window_size, window_size).contiguous().to(
        device=device,
        dtype=dtype,
    )


class SSIM(nn.Module):
    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, image1: Tensor, image2: Tensor) -> Tensor:
        channels = image1.shape[1]
        window = _create_window(
            self.window_size,
            channels,
            image1.device,
            image1.dtype,
        )
        padding = self.window_size // 2

        mu1 = F.conv2d(image1, window, padding=padding, groups=channels)
        mu2 = F.conv2d(image2, window, padding=padding, groups=channels)

        mu1_sq = mu1.square()
        mu2_sq = mu2.square()
        mu12 = mu1 * mu2

        sigma1_sq = (
            F.conv2d(image1 * image1, window, padding=padding, groups=channels)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(image2 * image2, window, padding=padding, groups=channels)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(image1 * image2, window, padding=padding, groups=channels)
            - mu12
        )

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = (
            (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
        ) / (
            (mu1_sq + mu2_sq + c1)
            * (sigma1_sq + sigma2_sq + c2)
        )
        return ssim_map.mean()
