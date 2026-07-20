# engine/evaluator.py

import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from engine.model_inputs import prepare_model_inputs


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

    progress_bar = tqdm(
        data_loader,
        desc="Validation",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for batch in progress_bar:
        model_inputs = prepare_model_inputs(
            model=model,
            batch=batch,
            device=device,
        )

        mask = batch["mask"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(**model_inputs)["pred"]

        prediction = torch.sigmoid(logits)

        batch_mae = (
            prediction.sub(mask)
            .abs()
            .flatten(start_dim=1)
            .mean(dim=1)
        )

        mae_sum += batch_mae.sum().item()
        sample_count += model_inputs["image"].shape[0]

        progress_bar.set_postfix(
            mae=f"{mae_sum / sample_count:.6f}",
        )

    return {
        "mae": mae_sum / sample_count,
        "time_seconds": (
            time.perf_counter() - start_time
        ),
    }