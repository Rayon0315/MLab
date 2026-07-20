# engine/trainer.py

import logging
import time

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from engine.model_inputs import prepare_model_inputs


logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    use_amp: bool,
    log_interval: int,
) -> tuple[dict[str, float], int]:
    model.train()

    total_samples = 0
    loss_sums: dict[str, float] = {}

    start_time = time.perf_counter()

    for batch_index, batch in enumerate(
        data_loader,
        start=1,
    ):
        model_inputs = prepare_model_inputs(
            model=model,
            batch=batch,
            device=device,
        )

        mask = batch["mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(**model_inputs)
            loss_dict = criterion(outputs, mask)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = model_inputs["image"].shape[0]

        total_samples += batch_size
        global_step += 1

        for name, value in loss_dict.items():
            loss_sums[name] = (
                loss_sums.get(name, 0.0)
                + value.detach().item() * batch_size
            )

        if (
            batch_index % log_interval == 0
            or batch_index == len(data_loader)
        ):
            logger.info(
                "Epoch %03d | Batch %05d/%05d | "
                "Step %07d | Loss %.6f",
                epoch,
                batch_index,
                len(data_loader),
                global_step,
                loss.detach().item(),
            )

    elapsed_time = time.perf_counter() - start_time

    statistics = {
        name: value / total_samples
        for name, value in loss_sums.items()
    }

    statistics["lr"] = optimizer.param_groups[0]["lr"]
    statistics["time_seconds"] = elapsed_time

    return statistics, global_step