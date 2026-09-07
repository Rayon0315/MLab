from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.gaussian_diffusion import GaussianMaskDiffusion
from models.diffusion.multi_prior_denoiser import MultiPriorDenoiser
from models.diffusion.prior_reconstruction import (
    RegionSpectralPriorReconstruction,
)
from models.networks.mambavision_small_progressive_region_direct_hier60_region_hybrid_sod import (
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD,
)
from models.networks.mambavision_small_progressive_region_direct_sod import (
    PRETRAINED_PATH,
)


class HybridRegionDiffusionPriorExtractor(
    MambaVisionSmallProgressiveRegionDirectHier60RegionHybridSOD
):
    """
    The validated mv-region-hybrid network, with an additional method that
    exposes multi-scale evidence for diffusion. The original network file is
    untouched and its normal forward remains available.
    """

    def forward_with_priors(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[str, object]:
        input_size = image.shape[-2:]

        (
            raw_stage1,
            raw_stage2,
            raw_stage3,
            raw_stage4,
        ) = self.backbone(image)

        # Hier60 assignment: RGB-M60 for Stage1, M60 for Stage2/3/4.
        (
            detail_region,
            fine_region,
            middle_region,
            coarse_region,
        ) = self.region_hierarchy(
            image=image,
            mean_20=mean_60,
            mean_40=mean_60,
            mean_60=mean_60,
        )

        (
            region1,
            region2,
            region3,
            region4,
        ) = self.region_encoder(
            detail_region=detail_region,
            fine_region=fine_region,
            middle_region=middle_region,
            coarse_region=coarse_region,
            stage1_size=raw_stage1.shape[-2:],
            stage2_size=raw_stage2.shape[-2:],
            stage3_size=raw_stage3.shape[-2:],
            stage4_size=raw_stage4.shape[-2:],
        )

        stage4 = self.stage4_region_interaction(
            visual_feature=raw_stage4,
            region_feature=region4,
        )
        stage3 = self.stage3_region_interaction(
            visual_feature=raw_stage3,
            region_feature=region3,
        )
        stage2 = self.stage2_region_reconstruction(
            visual_feature=raw_stage2,
            region_feature=region2,
        )
        stage1 = self.stage1_detail_reconstruction(
            visual_feature=raw_stage1,
            detail_feature=region1,
        )

        decoded4 = self.context4(
            self.deep_projection(stage4)
        )
        prediction4 = self.pred4(decoded4)

        global3 = self.global3(
            stage4,
            target_size=stage3.shape[-2:],
        )
        decoded3 = self.fusion3(
            low_feature=stage3,
            high_feature=decoded4,
            global_feature=global3,
        )
        prediction3 = self.pred3(decoded3)
        decoded3_reduced = self.reduce3(decoded3)

        global2 = self.global2(
            stage4,
            target_size=stage2.shape[-2:],
        )
        decoded2 = self.fusion2(
            low_feature=stage2,
            high_feature=decoded3_reduced,
            global_feature=global2,
        )
        prediction2 = self.pred2(decoded2)
        decoded2_reduced = self.reduce2(decoded2)

        stage1_feature = self.stage1_adapter(stage1)
        global1 = self.global1(
            stage4,
            target_size=stage1.shape[-2:],
        )
        decoded1 = self.fusion1(
            low_feature=stage1_feature,
            high_feature=decoded2_reduced,
            global_feature=global1,
        )

        stage2_boundary = self.stage2_boundary_adapter(stage2)
        decoded1 = self.boundary_refinement(
            shallow_feature=stage1_feature,
            semantic_feature=stage2_boundary,
            saliency_feature=decoded1,
        )

        prediction1_native = self.pred1(decoded1)
        prediction1 = F.interpolate(
            prediction1_native,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return {
            "outputs": {
                "pred": prediction1,
                "aux": [
                    prediction2,
                    prediction3,
                    prediction4,
                ],
            },
            "prior_sources": (
                (
                    raw_stage1,
                    stage1,
                    region1,
                    decoded1,
                ),
                (
                    raw_stage2,
                    stage2,
                    region2,
                    decoded2,
                ),
                (
                    raw_stage3,
                    stage3,
                    region3,
                    decoded3,
                ),
                (
                    raw_stage4,
                    stage4,
                    region4,
                    decoded4,
                ),
            ),
        }


class MambaVisionSmallHybridRegionDiffusionSOD(nn.Module):
    """
    IPDiff-inspired diffusion extension of mv-region-hybrid.

    The Hybrid network generates:
      - coarse saliency prior
      - four hierarchical region-aware evidence groups

    Four spectral prior reconstruction blocks convert them into compact
    diffusion conditions. A full-resolution conditional denoising U-Net then
    predicts v over noisy masks and iteratively samples the final saliency map.
    """

    input_keys = ("image", "mean_60")

    def __init__(
        self,
        pretrained_path: str | Path | None = PRETRAINED_PATH,
        prior_channels: int = 64,
        train_timesteps: int = 1000,
        sample_steps: int = 10,
        ipm_probability: float = 0.2,
    ) -> None:
        super().__init__()

        self.prior_net = HybridRegionDiffusionPriorExtractor(
            pretrained_path=pretrained_path,
        )

        c1, c2, c3, c4 = self.prior_net.backbone.out_channels

        self.prior_reconstruction = nn.ModuleList(
            [
                RegionSpectralPriorReconstruction(
                    (c1, c1, c1, 128),
                    out_channels=prior_channels,
                ),
                RegionSpectralPriorReconstruction(
                    (c2, c2, c2, c2),
                    out_channels=prior_channels,
                ),
                RegionSpectralPriorReconstruction(
                    (c3, c3, c3, c3),
                    out_channels=prior_channels,
                ),
                RegionSpectralPriorReconstruction(
                    (c4, c4, c4, c3),
                    out_channels=prior_channels,
                ),
            ]
        )

        denoiser = MultiPriorDenoiser(
            prior_channels=prior_channels,
            time_dim=256,
            ipm_probability=ipm_probability,
        )

        self.diffusion = GaussianMaskDiffusion(
            denoiser=denoiser,
            train_timesteps=train_timesteps,
            sample_steps=sample_steps,
        )

    @property
    def denoiser(self) -> nn.Module:
        return self.diffusion.denoiser

    def set_prior_trainable(self, trainable: bool) -> None:
        self.prior_net.requires_grad_(trainable)
        if trainable:
            self.prior_net.train()
        else:
            self.prior_net.eval()

    def load_prior_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> dict:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.prior_net.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
        return checkpoint

    def extract_conditions(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        dict[str, torch.Tensor | list[torch.Tensor]],
    ]:
        extracted = self.prior_net.forward_with_priors(
            image=image,
            mean_60=mean_60,
        )
        outputs = extracted["outputs"]
        prior_sources = extracted["prior_sources"]

        priors = tuple(
            block(*sources)
            for block, sources in zip(
                self.prior_reconstruction,
                prior_sources,
            )
        )

        saliency_prior = torch.sigmoid(outputs["pred"])

        return saliency_prior, priors, outputs

    def forward_train(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, object]:
        saliency_prior, priors, prior_outputs = (
            self.extract_conditions(
                image=image,
                mean_60=mean_60,
            )
        )

        x_start = target * 2.0 - 1.0

        diffusion_outputs = self.diffusion.forward_train(
            x_start=x_start,
            saliency_prior=saliency_prior,
            mean_60=mean_60,
            priors=priors,
        )

        return {
            **diffusion_outputs,
            "saliency_prior": saliency_prior,
            "prior_outputs": prior_outputs,
        }

    @torch.inference_mode()
    def sample(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
        sample_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        saliency_prior, priors, prior_outputs = (
            self.extract_conditions(
                image=image,
                mean_60=mean_60,
            )
        )

        sampled_x0 = self.diffusion.sample(
            shape=(
                image.shape[0],
                1,
                image.shape[-2],
                image.shape[-1],
            ),
            saliency_prior=saliency_prior,
            mean_60=mean_60,
            priors=priors,
            sample_steps=sample_steps,
        )

        prediction = (sampled_x0 + 1.0) * 0.5

        return {
            "pred": prediction.clamp(0.0, 1.0),
            "prior_pred": saliency_prior,
            "prior_logits": prior_outputs["pred"],
        }

    def forward(
        self,
        image: torch.Tensor,
        mean_60: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Inference-oriented forward for convenience. Dedicated training uses
        # forward_train() because diffusion also needs the ground-truth mask.
        return self.sample(
            image=image,
            mean_60=mean_60,
        )


def build_model() -> MambaVisionSmallHybridRegionDiffusionSOD:
    return MambaVisionSmallHybridRegionDiffusionSOD()
