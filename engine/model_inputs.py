# engine/model_inputs.py

import torch
from torch import nn


def prepare_model_inputs(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_keys = getattr(
        model,
        "input_keys",
        ("image",),
    )

    return {
        key: batch[key].to(
            device,
            non_blocking=True,
        )
        for key in input_keys
    }