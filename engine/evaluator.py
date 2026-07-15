# engine/evaluator.py

import time

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()

    mae_sum = 0.0
    sample_count = 0
    start_time = time.perf_counter()

    for batch in data_loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(image)["pred"]

        prediction = torch.sigmoid(logits)

        batch_mae = (
            prediction.sub(mask)
            .abs()
            .flatten(start_dim=1)
            .mean(dim=1)
        )

        mae_sum += batch_mae.sum().item()
        sample_count += image.shape[0]

    return {
        "mae": mae_sum / sample_count,
        "time_seconds": time.perf_counter() - start_time,
    }