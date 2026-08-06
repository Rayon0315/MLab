# tools/check_scan.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.backbones.mambavision import window_partition
from models.backbones.mambavision_nam_scan import (
    build_hierarchical_region_permutation,
    label_region_components,
    reverse_valid_permutation,
    resize_boundary_map,
    validate_permutation,
)

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

STAGE_CONFIG = {
    3: {
        "stride": 16,
        "window_size": 14,
        "schedule": (
            (20, False),
            (40, False),
            (60, False),
            (60, True),
        ),
    },
    4: {
        "stride": 32,
        "window_size": 7,
        "schedule": (
            (20, False),
            (40, True),
            (60, False),
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the patch sequence produced by "
            "NAM hierarchical scanning."
        )
    )

    parser.add_argument(
        "--nam-root",
        default="datasets/EORSSD/test-nam",
    )

    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Sample stem or filename. "
            "The first hier_20 sample is used when omitted."
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=352,
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=(3, 4),
        default=3,
    )

    parser.add_argument(
        "--mamba-block",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--window-index",
        type=int,
        default=-1,
        help=(
            "-1 selects the window whose NAM sequence "
            "differs most from row-major."
        ),
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    return parser.parse_args()


def find_map(
    directory: Path,
    name: str | None,
) -> Path:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    if name is None:
        paths = sorted(
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_SUFFIXES
            )
        )

        if not paths:
            raise FileNotFoundError(
                f"No maps found in {directory}"
            )

        return paths[0]

    stem = Path(name).stem

    matches = sorted(
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.stem == stem
            and path.suffix.lower()
            in IMAGE_SUFFIXES
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"Map not found: {stem} in {directory}"
        )

    return matches[0]


def load_binary_map(
    path: Path,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("L")

        image = image.resize(
            (
                image_size,
                image_size,
            ),
            resample=Image.Resampling.NEAREST,
        )

        array = np.array(
            image,
            dtype=np.uint8,
            copy=True,
        )

    return (
        torch.from_numpy(array)
        .to(
            device=device,
            dtype=torch.float32,
        )
        .ge(128)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
    )


def ceil_div(
    value: int,
    divisor: int,
) -> int:
    return (
        value + divisor - 1
    ) // divisor


def compact_labels(
    labels: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    result = torch.full_like(
        labels,
        -1,
    )

    unique_labels = torch.unique(
        labels[valid],
        sorted=True,
    )

    for compact_id, label in enumerate(
        unique_labels.tolist()
    ):
        result[
            valid
            & labels.eq(label)
        ] = compact_id

    return result


def print_integer_grid(
    title: str,
    values: torch.Tensor,
    valid: torch.Tensor,
    window_size: int,
) -> None:
    values = (
        values
        .reshape(
            window_size,
            window_size,
        )
        .detach()
        .cpu()
    )

    valid = (
        valid
        .reshape(
            window_size,
            window_size,
        )
        .detach()
        .cpu()
    )

    if bool(valid.any()):
        maximum = int(
            values[valid]
            .max()
            .item()
        )

        width = max(
            2,
            len(str(maximum)),
        )
    else:
        width = 2

    print()
    print(title)

    for row in range(
        window_size
    ):
        cells = []

        for column in range(
            window_size
        ):
            if not bool(
                valid[
                    row,
                    column,
                ]
            ):
                cells.append(
                    "X".rjust(width)
                )
            else:
                cells.append(
                    str(
                        int(
                            values[
                                row,
                                column,
                            ].item()
                        )
                    ).rjust(width)
                )

        print(
            " ".join(cells)
        )


def print_boundary_grid(
    title: str,
    boundary: torch.Tensor,
    valid: torch.Tensor,
    window_size: int,
) -> None:
    boundary = (
        boundary
        .reshape(
            window_size,
            window_size,
        )
        .detach()
        .cpu()
    )

    valid = (
        valid
        .reshape(
            window_size,
            window_size,
        )
        .detach()
        .cpu()
    )

    print()
    print(title)
    print(
        "#=boundary, .=interior, X=padding"
    )

    for row in range(
        window_size
    ):
        cells = []

        for column in range(
            window_size
        ):
            if not bool(
                valid[
                    row,
                    column,
                ]
            ):
                cells.append("X")

            elif bool(
                boundary[
                    row,
                    column,
                ]
            ):
                cells.append("#")

            else:
                cells.append(".")

        print(
            " ".join(cells)
        )


def get_valid_sequence(
    permutation: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    ordered_valid = torch.gather(
        valid,
        dim=0,
        index=permutation,
    )

    return permutation[
        ordered_valid
    ]


def spatial_neighbor_ratio(
    permutation: torch.Tensor,
    valid: torch.Tensor,
    window_size: int,
) -> float:
    sequence = get_valid_sequence(
        permutation,
        valid,
    )

    if sequence.numel() <= 1:
        return 1.0

    rows = torch.div(
        sequence,
        window_size,
        rounding_mode="floor",
    )

    columns = (
        sequence
        % window_size
    )

    distance = (
        rows[1:]
        .sub(
            rows[:-1]
        )
        .abs()
        + columns[1:]
        .sub(
            columns[:-1]
        )
        .abs()
    )

    return float(
        distance.eq(1)
        .float()
        .mean()
        .item()
    )


def region_crossings(
    permutation: torch.Tensor,
    valid: torch.Tensor,
    labels: torch.Tensor,
) -> int:
    sequence = get_valid_sequence(
        permutation,
        valid,
    )

    ordered_labels = torch.gather(
        labels,
        dim=0,
        index=sequence,
    )

    return int(
        ordered_labels[
            1:
        ]
        .ne(
            ordered_labels[
                :-1
            ]
        )
        .sum()
        .item()
    )


def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device
    )

    config = STAGE_CONFIG[
        args.stage
    ]

    stride = int(
        config["stride"]
    )

    window_size = int(
        config["window_size"]
    )

    schedule = config[
        "schedule"
    ]

    if not (
        0
        <= args.mamba_block
        < len(schedule)
    ):
        raise ValueError(
            "Invalid Mamba block | "
            f"stage={args.stage} | "
            f"block={args.mamba_block} | "
            f"valid=0..{len(schedule) - 1}"
        )

    hierarchy, reverse_path = (
        schedule[
            args.mamba_block
        ]
    )

    nam_root = Path(
        args.nam_root
    )

    path_20 = find_map(
        nam_root / "hier_20",
        args.name,
    )

    sample_name = (
        path_20.stem
    )

    nam_maps = {
        20: load_binary_map(
            path_20,
            args.image_size,
            device,
        ),
        40: load_binary_map(
            find_map(
                nam_root / "hier_40",
                sample_name,
            ),
            args.image_size,
            device,
        ),
        60: load_binary_map(
            find_map(
                nam_root / "hier_60",
                sample_name,
            ),
            args.image_size,
            device,
        ),
    }

    feature_height = ceil_div(
        args.image_size,
        stride,
    )

    feature_width = ceil_div(
        args.image_size,
        stride,
    )

    pad_right = (
        window_size
        - feature_width % window_size
    ) % window_size

    pad_bottom = (
        window_size
        - feature_height % window_size
    ) % window_size

    valid = torch.ones(
        (
            1,
            1,
            feature_height,
            feature_width,
        ),
        device=device,
        dtype=torch.float32,
    )

    valid = F.pad(
        valid,
        (
            0,
            pad_right,
            0,
            pad_bottom,
        ),
        value=0.0,
    )

    valid_windows = (
        window_partition(
            valid,
            window_size,
        )
        .squeeze(-1)
        .bool()
    )

    raw_boundaries: dict[
        int,
        torch.Tensor,
    ] = {}

    for (
        level,
        nam_map,
    ) in nam_maps.items():
        boundary = resize_boundary_map(
            nam_map,
            (
                feature_height,
                feature_width,
            ),
        ).float()

        boundary = F.pad(
            boundary,
            (
                0,
                pad_right,
                0,
                pad_bottom,
            ),
            value=0.0,
        )

        raw_boundaries[
            level
        ] = (
            window_partition(
                boundary,
                window_size,
            )
            .squeeze(-1)
            .bool()
        )

    effective_boundaries: dict[
        int,
        torch.Tensor,
    ] = {}

    accumulated = (
        torch.zeros_like(
            raw_boundaries[20]
        )
    )

    for level in (
        20,
        40,
        60,
    ):
        accumulated = (
            accumulated
            | raw_boundaries[
                level
            ]
        )

        effective_boundaries[
            level
        ] = accumulated.clone()

    labels_by_hierarchy: dict[
        int,
        torch.Tensor,
    ] = {}

    assigned_boundaries: dict[
        int,
        torch.Tensor,
    ] = {}

    for level in (
        20,
        40,
        60,
    ):
        labels, assigned = (
            label_region_components(
                effective_boundaries[
                    level
                ],
                valid_windows,
                window_size,
            )
        )

        labels_by_hierarchy[
            level
        ] = labels

        assigned_boundaries[
            level
        ] = assigned

    hierarchy_order = {
        20: (
            20,
        ),
        40: (
            20,
            40,
        ),
        60: (
            20,
            40,
            60,
        ),
    }[
        hierarchy
    ]

    permutation = (
        build_hierarchical_region_permutation(
            labels_by_hierarchy,
            assigned_boundaries,
            valid_windows,
            hierarchy_order,
            window_size,
        )
    )

    if reverse_path:
        permutation = (
            reverse_valid_permutation(
                permutation,
                valid_windows,
            )
        )

    validate_permutation(
        permutation
    )

    sequence_length = (
        window_size
        * window_size
    )

    identity = torch.arange(
        sequence_length,
        device=device,
        dtype=torch.long,
    ).view(
        1,
        -1,
    )

    changed_per_window = (
        permutation
        .ne(identity)
        .sum(
            dim=1
        )
    )

    if args.window_index < 0:
        window_index = int(
            changed_per_window
            .argmax()
            .item()
        )
    else:
        window_index = (
            args.window_index
        )

    if not (
        0
        <= window_index
        < permutation.shape[0]
    ):
        raise ValueError(
            "Window index out of range | "
            f"requested={window_index} | "
            f"valid=0..{permutation.shape[0] - 1}"
        )

    num_window_columns = ceil_div(
        feature_width,
        window_size,
    )

    window_row = (
        window_index
        // num_window_columns
    )

    window_column = (
        window_index
        % num_window_columns
    )

    selected_valid = (
        valid_windows[
            window_index
        ]
    )

    selected_permutation = (
        permutation[
            window_index
        ]
    )

    selected_identity = (
        identity[0]
    )

    selected_sequence = (
        get_valid_sequence(
            selected_permutation,
            selected_valid,
        )
    )

    token_indices = torch.arange(
        sequence_length,
        device=device,
        dtype=torch.long,
    )

    scan_rank = torch.empty_like(
        selected_permutation
    )

    scan_rank.scatter_(
        dim=0,
        index=selected_permutation,
        src=torch.arange(
            sequence_length,
            device=device,
            dtype=torch.long,
        ),
    )

    compact_regions = {
        level: compact_labels(
            labels_by_hierarchy[
                level
            ][
                window_index
            ],
            selected_valid,
        )
        for level in (
            20,
            40,
            60,
        )
    }

    print(
        "NAM scan inspection"
    )

    print(
        f"Sample: {sample_name}"
    )

    print(
        f"Stage: {args.stage} | "
        f"Mamba block: {args.mamba_block} | "
        f"Hierarchy: nam_{hierarchy} | "
        f"Reverse: {reverse_path}"
    )

    print(
        f"Feature: "
        f"{feature_height}x{feature_width} | "
        f"Window: {window_size} | "
        f"Padding: right={pad_right}, "
        f"bottom={pad_bottom}"
    )

    print(
        f"Selected window: "
        f"{window_index} | "
        f"window grid="
        f"({window_row}, {window_column})"
    )

    print()
    print(
        "Changed positions in every window"
    )

    for (
        index,
        changed,
    ) in enumerate(
        changed_per_window.tolist()
    ):
        print(
            f"window={index:2d} | "
            f"changed={changed:3d}/"
            f"{sequence_length}"
        )

    print_integer_grid(
        "Original token-index grid",
        token_indices,
        selected_valid,
        window_size,
    )

    for level in (
        20,
        40,
        60,
    ):
        print_boundary_grid(
            (
                "Effective NAM boundary grid | "
                f"hierarchy={level}"
            ),
            effective_boundaries[
                level
            ][
                window_index
            ],
            selected_valid,
            window_size,
        )

        print_integer_grid(
            (
                "Connected-region grid | "
                f"hierarchy={level}"
            ),
            compact_regions[
                level
            ],
            selected_valid,
            window_size,
        )

    print_integer_grid(
        "NAM-guided scan-rank grid",
        scan_rank,
        selected_valid,
        window_size,
    )

    row_major_sequence = (
        get_valid_sequence(
            selected_identity,
            selected_valid,
        )
    )

    print()
    print(
        "Row-major coordinate sequence"
    )

    print(
        " -> ".join(
            (
                f"({index // window_size},"
                f"{index % window_size})"
            )
            for index
            in row_major_sequence.tolist()
        )
    )

    print()
    print(
        "NAM-guided coordinate sequence"
    )

    print(
        " -> ".join(
            (
                f"({index // window_size},"
                f"{index % window_size})"
            )
            for index
            in selected_sequence.tolist()
        )
    )

    print()
    print(
        "step token row col "
        "r20 r40 r60 b20 b40 b60"
    )

    for (
        step,
        token_index,
    ) in enumerate(
        selected_sequence.tolist()
    ):
        print(
            f"{step:4d} "
            f"{token_index:5d} "
            f"{token_index // window_size:3d} "
            f"{token_index % window_size:3d} "
            f"{int(compact_regions[20][token_index]):3d} "
            f"{int(compact_regions[40][token_index]):3d} "
            f"{int(compact_regions[60][token_index]):3d} "
            f"{int(effective_boundaries[20][window_index, token_index]):3d} "
            f"{int(effective_boundaries[40][window_index, token_index]):3d} "
            f"{int(effective_boundaries[60][window_index, token_index]):3d}"
        )

    changed_positions = int(
        selected_permutation
        .ne(
            selected_identity
        )
        .sum()
        .item()
    )

    print()
    print(
        "Sequence summary"
    )

    print(
        f"Changed positions: "
        f"{changed_positions}/"
        f"{sequence_length}"
    )

    print(
        "Spatial neighbor ratio | "
        "row-major="
        f"{spatial_neighbor_ratio(
            selected_identity,
            selected_valid,
            window_size,
        ):.4f} | "
        "NAM="
        f"{spatial_neighbor_ratio(
            selected_permutation,
            selected_valid,
            window_size,
        ):.4f}"
    )

    for level in (
        20,
        40,
        60,
    ):
        row_major_crossings = (
            region_crossings(
                selected_identity,
                selected_valid,
                labels_by_hierarchy[
                    level
                ][
                    window_index
                ],
            )
        )

        nam_crossings = (
            region_crossings(
                selected_permutation,
                selected_valid,
                labels_by_hierarchy[
                    level
                ][
                    window_index
                ],
            )
        )

        print(
            f"Region crossings | "
            f"hierarchy={level} | "
            f"row-major={row_major_crossings} | "
            f"NAM={nam_crossings}"
        )


if __name__ == "__main__":
    main()