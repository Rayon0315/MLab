# tools/check_enf_nam_input.py
from __future__ import annotations

import sys 
sys.path.append('.')
sys.path.append('..')

import argparse
import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data.dataset import SODDataset
from engine.model_inputs import (
    get_model_input_keys,
    get_model_nam_hierarchies,
)


IMAGE_MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(3, 1, 1)

IMAGE_STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that hier_60 NAM maps are loaded "
            "and passed into the ENF NAM condition network."
        ),
    )

    parser.add_argument(
        "--network",
        default=(
            "models.networks."
            "mambavision_small_enf_nam_sod"
        ),
    )

    parser.add_argument(
        "--image-dir",
        default="datasets/EORSSD/train-images",
    )

    parser.add_argument(
        "--mask-dir",
        default="datasets/EORSSD/train-labels",
    )

    parser.add_argument(
        "--nam-dir",
        default="datasets/EORSSD/train-nam",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=352,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--output",
        default="tools/nam_input_check.png",
    )

    return parser.parse_args()


def build_model(
    network_path: str,
) -> torch.nn.Module:
    network_module = importlib.import_module(
        network_path
    )

    return network_module.build_model()


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "model",
        checkpoint,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )


def tensor_statistics(
    name: str,
    tensor: torch.Tensor,
) -> None:
    tensor_float = (
        tensor.detach()
        .float()
        .cpu()
    )

    unique_values = torch.unique(
        tensor_float
    )

    if unique_values.numel() <= 10:
        unique_text = str(
            unique_values.tolist()
        )
    else:
        unique_text = (
            f"{unique_values.numel()} values"
        )

    print(
        f"{name}: "
        f"shape={tuple(tensor_float.shape)} | "
        f"min={tensor_float.min().item():.6f} | "
        f"max={tensor_float.max().item():.6f} | "
        f"mean={tensor_float.mean().item():.6f} | "
        f"sum={tensor_float.sum().item():.1f} | "
        f"unique={unique_text}"
    )


def denormalize_image(
    image: torch.Tensor,
) -> torch.Tensor:
    image = (
        image.detach()
        .float()
        .cpu()
    )

    image = (
        image * IMAGE_STD
        + IMAGE_MEAN
    )

    return image.clamp(
        0.0,
        1.0,
    )


def prepare_single_sample(
    sample: dict,
    input_keys: tuple[str, ...],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model_inputs: dict[
        str,
        torch.Tensor,
    ] = {}

    for key in input_keys:
        if key not in sample:
            raise KeyError(
                f"Dataset sample does not contain '{key}'. "
                f"Available keys: {tuple(sample.keys())}"
            )

        value = sample[key]

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(
                f"Model input '{key}' is not a tensor."
            )

        model_inputs[key] = (
            value.unsqueeze(0)
            .to(device)
        )

    return model_inputs


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device
    )

    model = build_model(
        args.network
    )

    input_keys = get_model_input_keys(
        model
    )

    nam_hierarchies = (
        get_model_nam_hierarchies(
            model
        )
    )

    print(
        f"Network: {args.network}"
    )
    print(
        f"Model input keys: {input_keys}"
    )
    print(
        f"NAM hierarchies: {nam_hierarchies}"
    )

    expected_input_keys = (
        "image",
        "nam_60",
    )

    if input_keys != expected_input_keys:
        raise RuntimeError(
            "The ENF model input declaration is incorrect.\n"
            f"Current input_keys: {input_keys}\n"
            f"Expected input_keys: {expected_input_keys}\n"
            "Add the following declaration inside "
            "MambaVisionSmallENFNAMSOD:\n\n"
            "    input_keys = (\n"
            '        "image",\n'
            '        "nam_60",\n'
            "    )"
        )

    dataset = SODDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        nam_dir=args.nam_dir,
        nam_hierarchies=nam_hierarchies,
        image_size=(
            args.image_size,
            args.image_size,
        ),
        augment_8way=False,
    )

    if not (
        0 <= args.index < len(dataset)
    ):
        raise IndexError(
            f"Index {args.index} is outside "
            f"dataset size {len(dataset)}."
        )

    sample = dataset[
        args.index
    ]

    sample_name = sample[
        "name"
    ]

    nam_path = (
        dataset.nam_maps[
            60
        ][sample_name]
    )

    print(
        f"Dataset size: {len(dataset)}"
    )
    print(
        f"Sample index: {args.index}"
    )
    print(
        f"Sample name: {sample_name}"
    )
    print(
        f"NAM source path: {nam_path}"
    )

    if args.checkpoint is not None:
        load_checkpoint(
            model=model,
            checkpoint_path=(
                args.checkpoint
            ),
        )

        print(
            f"Checkpoint loaded: "
            f"{args.checkpoint}"
        )
    else:
        print(
            "Checkpoint: not loaded"
        )

    if not hasattr(
        model,
        "nam_condition",
    ):
        child_names = [
            name
            for name, _ in (
                model.named_children()
            )
        ]

        raise AttributeError(
            "The model does not contain "
            "'nam_condition'.\n"
            f"Top-level modules: {child_names}"
        )

    model = model.to(
        device
    )

    model.eval()

    model_inputs = (
        prepare_single_sample(
            sample=sample,
            input_keys=input_keys,
            device=device,
        )
    )

    captured: dict[
        str,
        torch.Tensor,
    ] = {}

    def model_input_hook(
        module: torch.nn.Module,
        hook_args: tuple,
        hook_kwargs: dict,
    ) -> None:
        del module
        del hook_args

        if "nam_60" not in hook_kwargs:
            raise RuntimeError(
                "The top-level model did not receive "
                "nam_60 as a keyword argument."
            )

        captured[
            "model_nam"
        ] = (
            hook_kwargs[
                "nam_60"
            ]
            .detach()
            .float()
            .cpu()
        )

    def condition_input_hook(
        module: torch.nn.Module,
        hook_args: tuple,
        hook_kwargs: dict,
    ) -> None:
        del module

        if "nam_60" in hook_kwargs:
            nam_tensor = hook_kwargs[
                "nam_60"
            ]
        elif hook_args:
            nam_tensor = hook_args[
                0
            ]
        else:
            raise RuntimeError(
                "NAM condition network received "
                "neither positional nor keyword input."
            )

        captured[
            "condition_input"
        ] = (
            nam_tensor.detach()
            .float()
            .cpu()
        )

    def condition_output_hook(
        module: torch.nn.Module,
        hook_args: tuple,
        output: torch.Tensor,
    ) -> None:
        del module
        del hook_args

        captured[
            "condition_output"
        ] = (
            output.detach()
            .float()
            .cpu()
        )

    model_pre_handle = (
        model.register_forward_pre_hook(
            model_input_hook,
            with_kwargs=True,
        )
    )

    condition_pre_handle = (
        model.nam_condition
        .register_forward_pre_hook(
            condition_input_hook,
            with_kwargs=True,
        )
    )

    condition_output_handle = (
        model.nam_condition
        .register_forward_hook(
            condition_output_hook,
        )
    )

    try:
        with torch.no_grad():
            outputs = model(
                **model_inputs
            )
    finally:
        model_pre_handle.remove()
        condition_pre_handle.remove()
        condition_output_handle.remove()

    required_captures = (
        "model_nam",
        "condition_input",
        "condition_output",
    )

    for capture_name in required_captures:
        if capture_name not in captured:
            raise RuntimeError(
                f"Hook did not capture "
                f"'{capture_name}'."
            )

    dataset_nam = (
        sample[
            "nam_60"
        ]
        .unsqueeze(0)
        .float()
        .cpu()
    )

    model_nam = captured[
        "model_nam"
    ]

    condition_input = captured[
        "condition_input"
    ]

    dataset_to_model_difference = (
        dataset_nam
        - model_nam
    ).abs().max().item()

    model_to_condition_difference = (
        model_nam
        - condition_input
    ).abs().max().item()

    tensor_statistics(
        "Dataset nam_60",
        dataset_nam,
    )

    tensor_statistics(
        "Top-level model nam_60",
        model_nam,
    )

    tensor_statistics(
        "NAM condition input",
        condition_input,
    )

    tensor_statistics(
        "NAM condition output",
        captured[
            "condition_output"
        ],
    )

    print(
        "Dataset -> model max difference: "
        f"{dataset_to_model_difference:.8f}"
    )

    print(
        "Model -> condition network max difference: "
        f"{model_to_condition_difference:.8f}"
    )

    nam_loaded_correctly = (
        dataset_to_model_difference
        == 0.0
        and model_to_condition_difference
        == 0.0
    )

    if not nam_loaded_correctly:
        raise RuntimeError(
            "NAM verification failed: the NAM tensor "
            "changed before entering the condition network."
        )

    if dataset_nam.sum().item() == 0:
        print(
            "Warning: this sample's NAM map is all zero."
        )

    if dataset_nam.sum().item() == (
        dataset_nam.numel()
    ):
        print(
            "Warning: this sample's NAM map is all one."
        )

    prediction = outputs[
        "pred"
    ]

    prediction = F.interpolate(
        prediction,
        size=sample[
            "nam_60"
        ].shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    prediction = (
        torch.sigmoid(
            prediction
        )[0, 0]
        .detach()
        .float()
        .cpu()
    )

    condition_visualization = (
        captured[
            "condition_output"
        ][0]
        .abs()
        .mean(dim=0)
    )

    condition_visualization = (
        condition_visualization
        - condition_visualization.min()
    ) / (
        condition_visualization.max()
        - condition_visualization.min()
        + 1e-8
    )

    rgb_image = denormalize_image(
        sample[
            "image"
        ]
    ).permute(
        1,
        2,
        0,
    )

    mask_image = (
        sample[
            "mask"
        ][0]
        .detach()
        .float()
        .cpu()
    )

    dataset_nam_image = (
        dataset_nam[
            0,
            0,
        ]
    )

    condition_nam_image = (
        condition_input[
            0,
            0,
        ]
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(15, 10),
    )

    axes[0, 0].imshow(
        rgb_image.numpy()
    )
    axes[0, 0].set_title(
        "RGB input"
    )

    axes[0, 1].imshow(
        mask_image.numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0, 1].set_title(
        "Ground-truth mask"
    )

    axes[0, 2].imshow(
        dataset_nam_image.numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0, 2].set_title(
        "Dataset nam_60"
    )

    axes[1, 0].imshow(
        condition_nam_image.numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1, 0].set_title(
        "NAM received by condition network"
    )

    axes[1, 1].imshow(
        condition_visualization.numpy(),
        cmap="gray",
    )
    axes[1, 1].set_title(
        "Condition feature mean magnitude"
    )

    axes[1, 2].imshow(
        prediction.numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1, 2].set_title(
        "Network prediction"
    )

    for axis in axes.flat:
        axis.axis(
            "off"
        )

    figure.suptitle(
        (
            f"{sample_name} | "
            f"NAM verification: PASS"
        ),
        fontsize=14,
    )

    figure.tight_layout()

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print()
    print(
        "NAM verification: PASS"
    )
    print(
        f"Visualization saved to: "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()