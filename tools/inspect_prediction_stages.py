# tools/inspect_prediction_stages.py

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from data.dataset import SODDataset
from engine.model_inputs import (
    get_model_input_keys,
    get_model_mean_hierarchies,
    get_model_nam_hierarchies,
)
from test import build_model, load_checkpoint


IMAGE_SIZE = 352


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect intermediate SOD predictions "
            "for one test image."
        ),
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the test image.",
    )

    parser.add_argument(
        "--network",
        required=True,
        help="Python module path of the network.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained checkpoint.",
    )

    return parser.parse_args()


def find_file_by_stem(
    directory: Path,
    stem: str,
) -> Path:
    suffixes = (
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    )

    for suffix in suffixes:
        path = directory / f"{stem}{suffix}"

        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Cannot find {stem} in {directory}"
    )


def load_image_tensor(
    path: Path,
) -> tuple[torch.Tensor, Image.Image]:
    with Image.open(path) as raw_image:
        image = ImageOps.exif_transpose(
            raw_image
        ).convert("RGB")

    original = image.copy()

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.Resampling.BILINEAR,
    )

    tensor = SODDataset._image_to_tensor(
        image
    ).unsqueeze(0)

    return tensor, original


def load_mean_tensor(
    path: Path,
) -> torch.Tensor:
    with Image.open(path) as raw_image:
        mean_map = ImageOps.exif_transpose(
            raw_image
        ).convert("RGB")

    mean_map = mean_map.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.Resampling.NEAREST,
    )

    return (
        SODDataset._rgb_to_tensor(
            mean_map
        )
        .unsqueeze(0)
    )


def load_nam_tensor(
    path: Path,
) -> torch.Tensor:
    with Image.open(path) as raw_image:
        nam_map = ImageOps.exif_transpose(
            raw_image
        ).convert("L")

    nam_map = nam_map.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        resample=Image.Resampling.NEAREST,
    )

    return (
        SODDataset._binary_to_tensor(
            nam_map
        )
        .unsqueeze(0)
    )


def load_gt(
    path: Path,
) -> Image.Image:
    with Image.open(path) as raw_mask:
        mask = ImageOps.exif_transpose(
            raw_mask
        ).convert("L")

    return mask


def prediction_to_image(
    logits: torch.Tensor,
    output_size: tuple[int, int],
) -> Image.Image:
    width, height = output_size

    logits = F.interpolate(
        logits.float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    probability = torch.sigmoid(
        logits
    )[0, 0]

    array = (
        probability
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    return Image.fromarray(
        array,
        mode="L",
    )


def make_panel(
    images: list[tuple[str, Image.Image]],
    output_path: Path,
) -> None:
    target_height = 352
    label_height = 32

    resized_images = []

    for name, image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")

        scale = (
            target_height
            / image.height
        )

        target_width = int(
            image.width * scale
        )

        image = image.resize(
            (
                target_width,
                target_height,
            ),
            resample=Image.Resampling.BILINEAR,
        )

        resized_images.append(
            (
                name,
                image,
            )
        )

    total_width = sum(
        image.width
        for _, image
        in resized_images
    )

    panel = Image.new(
        "RGB",
        (
            total_width,
            target_height + label_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        panel
    )

    x = 0

    for name, image in resized_images:
        panel.paste(
            image,
            (x, label_height),
        )

        draw.text(
            (
                x + 8,
                8,
            ),
            name,
            fill="black",
        )

        x += image.width

    panel.save(
        output_path
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    image_path = Path(
        args.image
    )

    sample_name = (
        image_path.stem
    )

    # datasets/EORSSD/test-images/1935.jpg
    #
    # dataset_root:
    # datasets/EORSSD
    dataset_root = (
        image_path.parent.parent
    )

    mask_dir = (
        dataset_root
        / "test-labels"
    )

    mean_dir = (
        dataset_root
        / "test-mean"
    )

    nam_dir = (
        dataset_root
        / "test-nam"
    )

    output_dir = (
        Path("outputs")
        / "stage_inspection"
        / sample_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Image:      {image_path}"
    )
    print(
        f"Network:    {args.network}"
    )
    print(
        f"Checkpoint: {args.checkpoint}"
    )
    print(
        f"Device:     {device}"
    )

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    model = build_model(
        args.network
    )

    load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        network_path=args.network,
    )

    model = model.to(
        device
    )

    model.eval()

    input_keys = (
        get_model_input_keys(
            model
        )
    )

    mean_hierarchies = (
        get_model_mean_hierarchies(
            model
        )
    )

    nam_hierarchies = (
        get_model_nam_hierarchies(
            model
        )
    )

    print(
        "Model inputs:",
        ", ".join(input_keys),
    )

    # -------------------------------------------------
    # Image
    # -------------------------------------------------

    image_tensor, original_image = (
        load_image_tensor(
            image_path
        )
    )

    model_inputs: dict[
        str,
        torch.Tensor,
    ] = {
        "image": (
            image_tensor.to(
                device
            )
        ),
    }

    # -------------------------------------------------
    # Region mean inputs
    # -------------------------------------------------

    for hierarchy in mean_hierarchies:
        mean_path = find_file_by_stem(
            mean_dir
            / f"region_mean_{hierarchy}",
            sample_name,
        )

        print(
            f"mean_{hierarchy}: "
            f"{mean_path}"
        )

        model_inputs[
            f"mean_{hierarchy}"
        ] = (
            load_mean_tensor(
                mean_path
            ).to(
                device
            )
        )

    # -------------------------------------------------
    # NAM inputs
    # -------------------------------------------------

    for hierarchy in nam_hierarchies:
        nam_path = find_file_by_stem(
            nam_dir
            / f"hier_{hierarchy}",
            sample_name,
        )

        print(
            f"nam_{hierarchy}: "
            f"{nam_path}"
        )

        model_inputs[
            f"nam_{hierarchy}"
        ] = (
            load_nam_tensor(
                nam_path
            ).to(
                device
            )
        )

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=(
            device.type == "cuda"
        ),
    ):
        outputs = model(
            **model_inputs
        )

    final_logits = outputs[
        "pred"
    ]

    aux_outputs = outputs.get(
        "aux",
        [],
    )

    if len(aux_outputs) != 3:
        raise RuntimeError(
            "This diagnostic expects "
            "three auxiliary predictions: "
            "[pred2, pred3, pred4]."
        )

    prediction2 = (
        aux_outputs[0]
    )

    prediction3 = (
        aux_outputs[1]
    )

    prediction4 = (
        aux_outputs[2]
    )

    print()
    print(
        "Raw output sizes:"
    )
    print(
        "pred4:",
        tuple(
            prediction4.shape
        ),
    )
    print(
        "pred3:",
        tuple(
            prediction3.shape
        ),
    )
    print(
        "pred2:",
        tuple(
            prediction2.shape
        ),
    )
    print(
        "final:",
        tuple(
            final_logits.shape
        ),
    )

    # -------------------------------------------------
    # Restore predictions to original resolution
    # -------------------------------------------------

    original_size = (
        original_image.size
    )

    pred4_image = prediction_to_image(
        prediction4,
        original_size,
    )

    pred3_image = prediction_to_image(
        prediction3,
        original_size,
    )

    pred2_image = prediction_to_image(
        prediction2,
        original_size,
    )

    final_image = prediction_to_image(
        final_logits,
        original_size,
    )

    # -------------------------------------------------
    # GT
    # -------------------------------------------------

    gt_path = find_file_by_stem(
        mask_dir,
        sample_name,
    )

    gt_image = load_gt(
        gt_path
    )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    original_image.save(
        output_dir
        / "00_image.png"
    )

    gt_image.save(
        output_dir
        / "01_gt.png"
    )

    pred4_image.save(
        output_dir
        / "02_pred4.png"
    )

    pred3_image.save(
        output_dir
        / "03_pred3.png"
    )

    pred2_image.save(
        output_dir
        / "04_pred2.png"
    )

    final_image.save(
        output_dir
        / "05_final.png"
    )

    make_panel(
        [
            (
                "Image",
                original_image,
            ),
            (
                "GT",
                gt_image,
            ),
            (
                "Pred4",
                pred4_image,
            ),
            (
                "Pred3",
                pred3_image,
            ),
            (
                "Pred2",
                pred2_image,
            ),
            (
                "Final",
                final_image,
            ),
        ],
        output_dir
        / "comparison.png",
    )

    print()
    print(
        f"Saved to: {output_dir}"
    )
    print(
        "  00_image.png"
    )
    print(
        "  01_gt.png"
    )
    print(
        "  02_pred4.png"
    )
    print(
        "  03_pred3.png"
    )
    print(
        "  04_pred2.png"
    )
    print(
        "  05_final.png"
    )
    print(
        "  comparison.png"
    )


if __name__ == "__main__":
    main()