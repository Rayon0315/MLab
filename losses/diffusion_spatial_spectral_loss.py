from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionSpatialSpectralLoss(nn.Module):
    """
    Diffusion objective + SOD structure reconstruction + edge + spectral
    consistency + auxiliary preservation of the original Hybrid prior.
    """

    def __init__(
        self,
        diffusion_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        edge_weight: float = 0.2,
        spectral_weight: float = 0.05,
        prior_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.diffusion_weight = diffusion_weight
        self.reconstruction_weight = reconstruction_weight
        self.edge_weight = edge_weight
        self.spectral_weight = spectral_weight
        self.prior_weight = prior_weight

    @staticmethod
    def _structure_loss(
        probability: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        probability = probability.clamp(1e-5, 1.0 - 1e-5)

        weight = 1.0 + 5.0 * torch.abs(
            F.avg_pool2d(
                target,
                kernel_size=31,
                stride=1,
                padding=15,
            )
            - target
        )

        logits = torch.logit(probability)
        weighted_bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        weighted_bce = (
            (weight * weighted_bce).sum(dim=(2, 3))
            / weight.sum(dim=(2, 3))
        ).mean()

        intersection = (
            probability * target * weight
        ).sum(dim=(2, 3))
        union = (
            (probability + target) * weight
        ).sum(dim=(2, 3))

        weighted_iou = (
            1.0
            - (intersection + 1.0)
            / (union - intersection + 1.0)
        ).mean()

        return weighted_bce + weighted_iou

    @staticmethod
    def _edge_loss(
        probability: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_edge = torch.abs(
            probability
            - F.avg_pool2d(
                probability,
                kernel_size=5,
                stride=1,
                padding=2,
            )
        )
        target_edge = torch.abs(
            target
            - F.avg_pool2d(
                target,
                kernel_size=5,
                stride=1,
                padding=2,
            )
        )
        return F.l1_loss(pred_edge, target_edge)

    @staticmethod
    def _spectral_loss(
        probability: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred_freq = torch.fft.rfft2(
            probability.float(),
            norm="ortho",
        )
        target_freq = torch.fft.rfft2(
            target.float(),
            norm="ortho",
        )

        pred_magnitude = torch.log1p(torch.abs(pred_freq))
        target_magnitude = torch.log1p(torch.abs(target_freq))
        magnitude_loss = F.l1_loss(
            pred_magnitude,
            target_magnitude,
        )

        pred_unit = pred_freq / (torch.abs(pred_freq) + 1e-6)
        target_unit = target_freq / (torch.abs(target_freq) + 1e-6)
        phase_loss = torch.abs(
            pred_unit - target_unit
        ).mean()

        return magnitude_loss + 0.1 * phase_loss

    def forward(
        self,
        outputs: dict[str, object],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        v_prediction = outputs["v_prediction"]
        v_target = outputs["v_target"]
        x0_prediction = outputs["x0_prediction"]
        saliency_prior = outputs["saliency_prior"]

        diffusion_loss = F.mse_loss(
            v_prediction.float(),
            v_target.float(),
        )

        prediction = (
            (x0_prediction.float() + 1.0) * 0.5
        ).clamp(0.0, 1.0)
        target = target.float()

        reconstruction_loss = self._structure_loss(
            prediction,
            target,
        )
        edge_loss = self._edge_loss(
            prediction,
            target,
        )
        spectral_loss = self._spectral_loss(
            prediction,
            target,
        )
        prior_loss = self._structure_loss(
            saliency_prior.float(),
            target,
        )

        total = (
            self.diffusion_weight * diffusion_loss
            + self.reconstruction_weight * reconstruction_loss
            + self.edge_weight * edge_loss
            + self.spectral_weight * spectral_loss
            + self.prior_weight * prior_loss
        )

        return {
            "loss": total,
            "loss_diffusion": diffusion_loss,
            "loss_reconstruction": reconstruction_loss,
            "loss_edge": edge_loss,
            "loss_spectral": spectral_loss,
            "loss_prior": prior_loss,
        }
