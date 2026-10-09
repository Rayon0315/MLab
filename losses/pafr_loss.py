from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from losses.pafr_gated_crf import GatedCRF
from losses.pafr_ssim import SSIM


class PAFRWeakLoss(nn.Module):
    """MLab implementation of the official PAFR weak-supervision objective.

    Official source terms retained:
      - partial BCE on scribble-valid pixels for all seven predictions;
      - gated CRF on all seven predictions;
      - SRC scale/color consistency on the refined and first side output.
    """

    def __init__(
        self,
        crf_weight: float = 0.3,
        crf_radius: int = 5,
        crf_xy_sigma: float = 6.0,
        crf_rgb_sigma: float = 0.1,
        src_alpha: float = 0.85,
        crf_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.crf_weight = crf_weight
        self.crf_radius = crf_radius
        self.src_alpha = src_alpha
        self.crf_scale = crf_scale
        self.kernels_desc = [
            {
                "weight": 1.0,
                "xy": crf_xy_sigma,
                "rgb": crf_rgb_sigma,
            }
        ]
        self.gated_crf = GatedCRF()
        self.ssim = SSIM(window_size=11)

    def forward(
        self,
        outputs: dict[str, Any],
        scaled_outputs: dict[str, Any],
        image: Tensor,
        target: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        logits_list = outputs["pafr_logits"]
        scaled_logits_list = scaled_outputs["pafr_logits"]

        if len(logits_list) != 7:
            raise ValueError(f"PAFR expects 7 outputs, got {len(logits_list)}")
        if len(scaled_logits_list) < 2:
            raise ValueError("Scaled PAFR pass must provide at least 2 outputs")

        image_float = image.float()
        crf_image = F.interpolate(
            image_float,
            scale_factor=self.crf_scale,
            mode="bilinear",
            align_corners=True,
        )
        sample = {"rgb": crf_image}

        pce_losses: list[Tensor] = []
        crf_losses: list[Tensor] = []
        output_losses: list[Tensor] = []

        for logits in logits_list:
            logits_float = logits.float()
            pce = self.masked_bce_with_logits(
                logits_float,
                target,
                valid_mask,
            )
            probability = torch.sigmoid(logits_float)
            crf_probability = F.interpolate(
                probability,
                size=crf_image.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            crf = self.gated_crf(
                crf_probability,
                self.kernels_desc,
                self.crf_radius,
                sample,
            )
            pce_losses.append(pce)
            crf_losses.append(crf)
            output_losses.append(pce + self.crf_weight * crf)

        src_refined = self._src_loss(
            scaled_logits=scaled_logits_list[0],
            native_logits=logits_list[0],
        )
        src_side1 = self._src_loss(
            scaled_logits=scaled_logits_list[1],
            native_logits=logits_list[1],
        )
        src_loss = src_refined + src_side1

        # Official train_B.py adds SRC only to loss1, then sums all seven losses.
        total_loss = torch.stack(output_losses).sum() + src_loss

        valid = valid_mask.detach().float()
        target_float = target.detach().float()
        valid_count = valid.sum().clamp_min(1.0)

        return {
            "loss": total_loss,
            "loss_main": output_losses[0] + src_loss,
            "loss_pce": torch.stack(pce_losses).sum(),
            "loss_crf": torch.stack(crf_losses).sum(),
            "loss_src": src_loss,
            "loss_src_refined": src_refined,
            "loss_src_side1": src_side1,
            "stat_valid_ratio": valid.mean(),
            "stat_fg_ratio_on_valid": (target_float * valid).sum() / valid_count,
        }

    def _src_loss(
        self,
        scaled_logits: Tensor,
        native_logits: Tensor,
    ) -> Tensor:
        # SSIM computes local variances as E[x^2] - E[x]^2.
        # Under BF16 autocast this cancellation can become numerically
        # unstable and make SSIM exceed 1 by a large margin, turning the
        # SRC term negative.  PAFR's original training code evaluates this
        # path in FP32, so explicitly leave autocast for the complete SRC
        # calculation instead of only casting the input tensors to float.
        with torch.autocast(
            device_type=scaled_logits.device.type,
            enabled=False,
        ):
            scaled_probability = torch.sigmoid(
                scaled_logits.float()
            )
            native_probability = torch.sigmoid(
                native_logits.float()
            )
            native_probability = F.interpolate(
                native_probability,
                size=scaled_probability.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

            ssim_term = 1.0 - self.ssim(
                scaled_probability,
                native_probability,
            )
            l1_term = (
                scaled_probability
                - native_probability
            ).abs().mean()

            return (
                self.src_alpha * ssim_term
                + (1.0 - self.src_alpha) * l1_term
            )

    @staticmethod
    def masked_bce_with_logits(
        logits: Tensor,
        target: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        target = target.float()
        valid_mask = valid_mask.float()
        loss_map = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        valid_count = valid_mask.sum().clamp_min(1.0)
        return (loss_map * valid_mask).sum() / valid_count
