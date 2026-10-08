from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GatedCRF(nn.Module):
    """Gated CRF regularizer adapted from the official PAFR source."""

    def forward(
        self,
        prediction: Tensor,
        kernels_desc: list[dict[str, float]],
        kernels_radius: int,
        sample: dict[str, Tensor],
    ) -> Tensor:
        if prediction.dim() != 4:
            raise ValueError("prediction must be NCHW")

        batch, channels, height, width = prediction.shape
        kernels = self._create_kernels(
            kernels_desc=kernels_desc,
            radius=kernels_radius,
            sample=sample,
            batch=batch,
            height=height,
            width=width,
            device=prediction.device,
        )

        unfolded = self._unfold(prediction, kernels_radius)
        center = unfolded[
            :, :, kernels_radius, kernels_radius, :, :
        ].view(batch, channels, 1, 1, height, width)
        differences = (center - unfolded).abs()

        diameter = kernels_radius * 2 + 1
        weighted = (
            kernels * differences
        ).view(
            batch,
            channels,
            diameter * diameter,
            height,
            width,
        )
        return weighted.sum(dim=2).mean()

    @classmethod
    def _create_kernels(
        cls,
        kernels_desc: list[dict[str, float]],
        radius: int,
        sample: dict[str, Tensor],
        batch: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        kernels: Tensor | None = None

        for desc in kernels_desc:
            weight = float(desc["weight"])
            features: list[Tensor] = []

            for modality, sigma_value in desc.items():
                if modality == "weight":
                    continue
                sigma = float(sigma_value)
                if modality == "xy":
                    feature = cls._get_mesh(
                        batch,
                        height,
                        width,
                        device,
                    )
                else:
                    if modality not in sample:
                        raise KeyError(
                            f"GatedCRF modality '{modality}' is missing"
                        )
                    feature = sample[modality]
                    if feature.shape[-2:] != (height, width):
                        feature = F.interpolate(
                            feature,
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        )

                # Do not mutate sample tensors in-place. The official source uses
                # `feature /= sigma`, which would repeatedly rescale the same RGB
                # tensor when seven PAFR outputs are regularized in one iteration.
                features.append(feature / sigma)

            feature_stack = torch.cat(features, dim=1)
            kernel = weight * cls._create_kernels_from_features(
                feature_stack,
                radius,
            )
            kernels = kernel if kernels is None else kernels + kernel

        if kernels is None:
            raise ValueError("kernels_desc must contain at least one kernel")
        return kernels

    @classmethod
    def _create_kernels_from_features(
        cls,
        features: Tensor,
        radius: int,
    ) -> Tensor:
        batch, channels, height, width = features.shape
        unfolded = cls._unfold(features, radius)
        center = unfolded[:, :, radius, radius, :, :].view(
            batch,
            channels,
            1,
            1,
            height,
            width,
        )
        delta = unfolded - center
        return (-0.5 * delta.square()).sum(dim=1, keepdim=True).exp()

    @staticmethod
    def _get_mesh(
        batch: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        x = torch.arange(width, dtype=torch.float32, device=device)
        y = torch.arange(height, dtype=torch.float32, device=device)
        x = x.view(1, 1, 1, width).repeat(batch, 1, height, 1)
        y = y.view(1, 1, height, 1).repeat(batch, 1, 1, width)
        return torch.cat((x, y), dim=1)

    @staticmethod
    def _unfold(image: Tensor, radius: int) -> Tensor:
        batch, channels, height, width = image.shape
        diameter = 2 * radius + 1
        return F.unfold(
            image,
            kernel_size=diameter,
            padding=radius,
        ).view(
            batch,
            channels,
            diameter,
            diameter,
            height,
            width,
        )
