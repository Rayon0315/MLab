# tools/build_salient_instance_maps.py

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


MASK_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build connected-component instance maps "
            "from binary salient-object masks."
        ),
    )

    parser.add_argument(
        "--mask-dir",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_paths = [
        path
        for path in sorted(
            mask_dir.iterdir()
        )
        if path.is_file()
        and path.suffix.lower()
        in MASK_SUFFIXES
    ]

    structure = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    total_objects = 0

    for index, mask_path in enumerate(
        mask_paths,
        start=1,
    ):
        with Image.open(
            mask_path
        ) as raw_mask:
            mask = ImageOps.exif_transpose(
                raw_mask
            ).convert(
                "L"
            )

        binary = (
            np.asarray(
                mask,
                dtype=np.uint8,
            )
            >= 128
        )

        labels, object_count = (
            ndimage.label(
                binary,
                structure=structure,
            )
        )

        total_objects += int(
            object_count
        )

        label_image = Image.fromarray(
            labels.astype(
                np.uint16
            ),
            mode="I;16",
        )

        label_image.save(
            output_dir
            / f"{mask_path.stem}.png"
        )

        if (
            index % 100 == 0
            or index == len(mask_paths)
        ):
            print(
                f"{index}/{len(mask_paths)} | "
                f"objects={total_objects}"
            )

    mean_objects = (
        total_objects
        / max(
            len(mask_paths),
            1,
        )
    )

    print(
        "Completed | "
        f"images={len(mask_paths)} | "
        f"objects={total_objects} | "
        f"mean_objects/image={mean_objects:.3f} | "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
