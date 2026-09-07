from __future__ import annotations

import logging
import time

import torch
from torch import nn
from torch.utils.data import DataLoader


LOGGER = logging.getLogger(__name__)


def _amp_dtype(device: torch.device) -> torch.dtype:
    if (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16


def train_diffusion_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    use_amp: bool,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 5.0,
    log_interval: int = 50,
    prior_trainable: bool = True,
) -> tuple[dict[str, float], int]:
    model.train()
    if hasattr(model, "set_prior_trainable"):
        model.set_prior_trainable(prior_trainable)
    dtype = _amp_dtype(device)

    totals: dict[str, float] = {}
    sample_batches = 0
    start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(data_loader, start=1):
        image = batch["image"].to(device, non_blocking=True)
        mean_60 = batch["mean_60"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=use_amp,
        ):
            outputs = model.forward_train(
                image=image,
                mean_60=mean_60,
                target=target,
            )
            losses = criterion(outputs, target)
            loss = losses["loss"] / grad_accum_steps

        scaler.scale(loss).backward()

        should_step = (
            batch_index % grad_accum_steps == 0
            or batch_index == len(data_loader)
        )

        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(
                value.detach()
            )
        sample_batches += 1

        if batch_index % log_interval == 0:
            LOGGER.info(
                "Epoch %03d | Batch %04d/%04d | Loss %.6f | Diff %.6f | Recon %.6f | Edge %.6f | Spec %.6f | Prior %.6f",
                epoch,
                batch_index,
                len(data_loader),
                float(losses["loss"].detach()),
                float(losses["loss_diffusion"].detach()),
                float(losses["loss_reconstruction"].detach()),
                float(losses["loss_edge"].detach()),
                float(losses["loss_spectral"].detach()),
                float(losses["loss_prior"].detach()),
            )

    statistics = {
        key: value / sample_batches
        for key, value in totals.items()
    }
    statistics["lr_diffusion"] = optimizer.param_groups[0]["lr"]
    statistics["lr_prior"] = optimizer.param_groups[1]["lr"]
    statistics["time_seconds"] = time.perf_counter() - start

    return statistics, global_step
