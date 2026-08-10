# engine/model_inputs.py

from __future__ import annotations

import torch
from torch import nn


def _unwrap_model(
    model: nn.Module,
) -> nn.Module:
    wrapped_model = getattr(
        model,
        "module",
        None,
    )

    if isinstance(
        wrapped_model,
        nn.Module,
    ):
        return wrapped_model

    return model


def get_model_input_keys(
    model: nn.Module,
) -> tuple[str, ...]:
    model = _unwrap_model(
        model
    )

    return tuple(
        getattr(
            model,
            "input_keys",
            ("image",),
        )
    )


def get_model_nam_keys(
    model: nn.Module,
) -> tuple[str, ...]:
    return tuple(
        key
        for key
        in get_model_input_keys(model)
        if key.startswith("nam_")
    )


def get_model_nam_hierarchies(
    model: nn.Module,
) -> tuple[int, ...]:
    hierarchies: list[int] = []

    for key in get_model_nam_keys(
        model
    ):
        hierarchy_text = (
            key.removeprefix(
                "nam_"
            )
        )

        if not hierarchy_text.isdigit():
            raise ValueError(
                f"Invalid NAM input key: {key}"
            )

        hierarchies.append(
            int(hierarchy_text)
        )

    return tuple(
        hierarchies
    )


def get_model_mean_keys(
    model: nn.Module,
) -> tuple[str, ...]:
    return tuple(
        key
        for key
        in get_model_input_keys(model)
        if key.startswith("mean_")
    )


def get_model_mean_hierarchies(
    model: nn.Module,
) -> tuple[int, ...]:
    hierarchies: list[int] = []

    for key in get_model_mean_keys(
        model
    ):
        hierarchy_text = (
            key.removeprefix(
                "mean_"
            )
        )

        if not hierarchy_text.isdigit():
            raise ValueError(
                "Invalid region-mean input key: "
                f"{key}"
            )

        hierarchies.append(
            int(hierarchy_text)
        )

    return tuple(
        hierarchies
    )


def get_edge_target_key(
    model: nn.Module,
) -> str | None:
    model = _unwrap_model(
        model
    )

    explicit_key = getattr(
        model,
        "edge_target_key",
        None,
    )

    if explicit_key is not None:
        if explicit_key not in (
            get_model_input_keys(model)
        ):
            raise ValueError(
                "edge_target_key is not listed "
                f"in input_keys: {explicit_key}"
            )

        return explicit_key

    nam_keys = get_model_nam_keys(
        model
    )

    if not nam_keys:
        return None

    return max(
        nam_keys,
        key=lambda key: int(
            key.removeprefix(
                "nam_"
            )
        ),
    )


def model_uses_nam(
    model: nn.Module,
) -> bool:
    return bool(
        get_model_nam_keys(
            model
        )
    )


def model_uses_mean(
    model: nn.Module,
) -> bool:
    return bool(
        get_model_mean_keys(
            model
        )
    )


def prepare_model_inputs(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model_inputs: dict[
        str,
        torch.Tensor,
    ] = {}

    for key in get_model_input_keys(
        model
    ):
        if key not in batch:
            raise KeyError(
                "Batch does not contain required "
                f"model input: {key}"
            )

        model_inputs[key] = batch[
            key
        ].to(
            device,
            non_blocking=True,
        )

    return model_inputs