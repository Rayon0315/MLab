from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def build_scaled_pafr_model_inputs(
    model_inputs: dict[str, Tensor],
    scaled_image: Tensor,
) -> dict[str, Tensor]:
    """Build the second PAFR/SRC forward inputs at the scaled image size."""
    scaled_inputs: dict[str, Tensor] = {"image": scaled_image}

    for key, value in model_inputs.items():
        if key == "image":
            continue
        if key.startswith("mean_"):
            scaled_inputs[key] = F.interpolate(
                value.float(),
                size=scaled_image.shape[-2:],
                mode="nearest",
            ).to(dtype=value.dtype)
            continue
        raise KeyError(
            "Unsupported non-image input for scaled PAFR forward: "
            f"{key}"
        )

    return scaled_inputs
