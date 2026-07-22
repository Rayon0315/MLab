# engine/model_inputs.py

import torch
from torch import nn


def get_model_input_keys(
    model: nn.Module,
) -> tuple[str, ...]:
    return tuple(
        getattr(
            model,
            "input_keys",
            ("image",),
        )
    )


def model_uses_nam(
    model: nn.Module,
) -> bool:
    return any(
        key.startswith("nam_")
        for key in get_model_input_keys(model)
    )


def prepare_model_inputs(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(
            device,
            non_blocking=True,
        )
        for key in get_model_input_keys(model)
    }