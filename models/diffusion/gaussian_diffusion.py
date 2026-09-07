from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMaskDiffusion(nn.Module):
    """
    Cosine-schedule mask diffusion with v-prediction and deterministic
    DDIM-style sampling. Training uses many noise levels; inference uses
    a short iterative chain (default: 10 steps), matching the role of
    diffusion in IPDiff without importing an external diffusion package.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        train_timesteps: int = 1000,
        sample_steps: int = 10,
        cosine_s: float = 0.008,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.train_timesteps = train_timesteps
        self.sample_steps = sample_steps

        steps = train_timesteps + 1
        x = torch.linspace(0, train_timesteps, steps)
        alpha_bar = torch.cos(
            ((x / train_timesteps) + cosine_s)
            / (1.0 + cosine_s)
            * math.pi
            * 0.5
        ).square()
        alpha_bar = alpha_bar / alpha_bar[0]

        betas = 1.0 - (
            alpha_bar[1:] / alpha_bar[:-1]
        )
        betas = betas.clamp(1e-5, 0.999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            "alphas_cumprod",
            alphas_cumprod.float(),
        )
        self.register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod).float(),
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).float(),
        )

    @staticmethod
    def _extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        selected = values.gather(0, timesteps)
        return selected.reshape(
            timesteps.shape[0],
            *((1,) * (len(shape) - 1)),
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._extract(
            self.sqrt_alphas_cumprod,
            timesteps,
            x_start.shape,
        )
        sigma = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            x_start.shape,
        )
        return alpha * x_start + sigma * noise

    def training_targets(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self._extract(
            self.sqrt_alphas_cumprod,
            timesteps,
            x_start.shape,
        )
        sigma = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            x_start.shape,
        )

        x_t = alpha * x_start + sigma * noise
        v_target = alpha * noise - sigma * x_start
        return x_t, v_target

    def predict_x0_from_v(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        v_prediction: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._extract(
            self.sqrt_alphas_cumprod,
            timesteps,
            x_t.shape,
        )
        sigma = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            timesteps,
            x_t.shape,
        )
        return alpha * x_t - sigma * v_prediction

    def forward_train(
        self,
        x_start: torch.Tensor,
        saliency_prior: torch.Tensor,
        mean_60: torch.Tensor,
        priors: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> dict[str, torch.Tensor]:
        batch = x_start.shape[0]
        timesteps = torch.randint(
            0,
            self.train_timesteps,
            (batch,),
            device=x_start.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x_start)

        x_t, v_target = self.training_targets(
            x_start,
            timesteps,
            noise,
        )

        v_prediction = self.denoiser(
            noisy_mask=x_t,
            timesteps=timesteps,
            saliency_prior=saliency_prior,
            mean_60=mean_60,
            priors=priors,
        )

        x0_prediction = self.predict_x0_from_v(
            x_t,
            timesteps,
            v_prediction,
        ).clamp(-1.0, 1.0)

        return {
            "v_prediction": v_prediction,
            "v_target": v_target,
            "x0_prediction": x0_prediction,
            "timesteps": timesteps,
        }

    @torch.inference_mode()
    def sample(
        self,
        shape: tuple[int, int, int, int],
        saliency_prior: torch.Tensor,
        mean_60: torch.Tensor,
        priors: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        sample_steps: int | None = None,
    ) -> torch.Tensor:
        sample_steps = sample_steps or self.sample_steps
        device = saliency_prior.device

        x = torch.randn(shape, device=device)

        times = torch.linspace(
            self.train_timesteps - 1,
            0,
            sample_steps,
            device=device,
        ).round().long()

        for index, timestep in enumerate(times):
            t = torch.full(
                (shape[0],),
                int(timestep.item()),
                device=device,
                dtype=torch.long,
            )

            v_prediction = self.denoiser(
                noisy_mask=x,
                timesteps=t,
                saliency_prior=saliency_prior,
                mean_60=mean_60,
                priors=priors,
            )

            alpha_t = self._extract(
                self.sqrt_alphas_cumprod,
                t,
                x.shape,
            )
            sigma_t = self._extract(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                x.shape,
            )

            x0 = (
                alpha_t * x
                - sigma_t * v_prediction
            ).clamp(-1.0, 1.0)
            eps = (
                sigma_t * x
                + alpha_t * v_prediction
            )

            if index == len(times) - 1:
                x = x0
                break

            t_next = torch.full(
                (shape[0],),
                int(times[index + 1].item()),
                device=device,
                dtype=torch.long,
            )
            alpha_next = self._extract(
                self.sqrt_alphas_cumprod,
                t_next,
                x.shape,
            )
            sigma_next = self._extract(
                self.sqrt_one_minus_alphas_cumprod,
                t_next,
                x.shape,
            )

            x = alpha_next * x0 + sigma_next * eps

        return x.clamp(-1.0, 1.0)
