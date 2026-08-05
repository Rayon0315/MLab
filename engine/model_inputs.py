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


def get_model_nam_hierarchies(
    model: nn.Module,
) -> tuple[int, ...]:
    hierarchies: list[int] = []

    for key in get_model_input_keys(model):
        if not key.startswith("nam_"):
            continue

        hierarchy_text = key.removeprefix(
            "nam_"
        )

        if not hierarchy_text.isdigit():
            raise ValueError(
                f"Invalid NAM input key: {key}"
            )

        hierarchies.append(
            int(hierarchy_text)
        )

    return tuple(hierarchies)


def model_uses_nam(
    model: nn.Module,
) -> bool:
    return bool(
        get_model_nam_hierarchies(model)
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