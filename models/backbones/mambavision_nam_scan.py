# models/backbones/mambavision_nam_scan.py
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.mambavision import (
    MambaVisionBackbone,
    MambaVisionMixer,
    unwrap_state_dict,
    window_partition,
    window_reverse,
)


ScanMode = Literal[
    "identity",
    "serpentine",
    "nam_hierarchical",
    "nam_60_only",
    "shuffled_nam",
]

VALID_SCAN_MODES = (
    "identity",
    "serpentine",
    "nam_hierarchical",
    "nam_60_only",
    "shuffled_nam",
)

STAGE_MAMBA_SCHEDULES: dict[
    int,
    tuple[
        tuple[int, bool],
        ...,
    ],
] = {
    2: (
        (20, False),
        (40, False),
        (60, False),
        (60, True),
    ),
    3: (
        (20, False),
        (40, True),
        (60, False),
    ),
}


def check_scan_mode(
    scan_mode: str,
) -> None:
    if scan_mode not in VALID_SCAN_MODES:
        choices = ", ".join(
            VALID_SCAN_MODES
        )

        raise ValueError(
            f"Unsupported scan mode: {scan_mode}. "
            f"Expected one of: {choices}"
        )


def gather_tokens(
    tokens: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if (
        tokens.ndim != 3
        or indices.ndim != 2
    ):
        raise ValueError(
            "Expected tokens [N, L, C] "
            "and indices [N, L]."
        )

    if tokens.shape[:2] != indices.shape:
        raise ValueError(
            "Shape mismatch: "
            f"tokens={tuple(tokens.shape)}, "
            f"indices={tuple(indices.shape)}"
        )

    index = (
        indices
        .unsqueeze(-1)
        .expand(
            -1,
            -1,
            tokens.shape[-1],
        )
    )

    return torch.gather(
        tokens,
        dim=1,
        index=index,
    )


def build_inverse_permutation(
    permutation: torch.Tensor,
) -> torch.Tensor:
    if permutation.ndim != 2:
        raise ValueError(
            "Permutation must have shape [N, L]."
        )

    num_windows, sequence_length = (
        permutation.shape
    )

    source = torch.arange(
        sequence_length,
        device=permutation.device,
        dtype=permutation.dtype,
    ).view(
        1,
        -1,
    ).expand(
        num_windows,
        -1,
    )

    inverse = torch.empty_like(
        permutation
    )

    inverse.scatter_(
        dim=1,
        index=permutation,
        src=source,
    )

    return inverse


def validate_permutation(
    permutation: torch.Tensor,
) -> None:
    if permutation.ndim != 2:
        raise AssertionError(
            "Permutation must have shape [N, L]."
        )

    sequence_length = (
        permutation.shape[1]
    )

    expected = torch.arange(
        sequence_length,
        device=permutation.device,
        dtype=permutation.dtype,
    ).view(
        1,
        -1,
    ).expand_as(
        permutation
    )

    sorted_permutation = torch.sort(
        permutation,
        dim=1,
    ).values

    if not torch.equal(
        sorted_permutation,
        expected,
    ):
        raise AssertionError(
            "Permutation contains duplicate "
            "or missing indices."
        )


def make_permutation_pair(
    permutation: torch.Tensor,
    validate: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    if validate:
        validate_permutation(
            permutation
        )

    inverse = build_inverse_permutation(
        permutation
    )

    if validate:
        sequence_length = (
            permutation.shape[1]
        )

        expected = torch.arange(
            sequence_length,
            device=permutation.device,
            dtype=permutation.dtype,
        ).view(
            1,
            -1,
        ).expand_as(
            permutation
        )

        restored = torch.gather(
            permutation,
            dim=1,
            index=inverse,
        )

        if not torch.equal(
            restored,
            expected,
        ):
            raise AssertionError(
                "Inverse permutation is incorrect."
            )

    return (
        permutation,
        inverse,
    )


def build_serpentine_rank(
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows = torch.arange(
        window_size,
        device=device,
        dtype=torch.long,
    ).view(
        -1,
        1,
    )

    columns = torch.arange(
        window_size,
        device=device,
        dtype=torch.long,
    ).view(
        1,
        -1,
    ).expand(
        window_size,
        -1,
    )

    scan_columns = torch.where(
        rows.remainder(2).eq(0),
        columns,
        window_size - 1 - columns,
    )

    return (
        rows * window_size
        + scan_columns
    ).reshape(-1)


def build_serpentine_permutation(
    valid_windows: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    sequence_length = (
        window_size * window_size
    )

    if (
        valid_windows.ndim != 2
        or valid_windows.shape[1]
        != sequence_length
    ):
        raise ValueError(
            "Valid-window mask has "
            "an unexpected shape."
        )

    rank = build_serpentine_rank(
        window_size,
        valid_windows.device,
    ).view(
        1,
        -1,
    ).expand(
        valid_windows.shape[0],
        -1,
    )

    key = (
        (~valid_windows.bool()).long()
        * sequence_length
        + rank
    )

    return torch.argsort(
        key,
        dim=1,
        stable=True,
    )


def reverse_valid_permutation(
    forward_permutation: torch.Tensor,
    valid_windows: torch.Tensor,
) -> torch.Tensor:
    if (
        forward_permutation.shape
        != valid_windows.shape
    ):
        raise ValueError(
            "Permutation and valid-window "
            "masks must match."
        )

    ordered_valid = torch.gather(
        valid_windows,
        dim=1,
        index=forward_permutation,
    )

    valid_count = (
        ordered_valid
        .long()
        .sum(
            dim=1,
            keepdim=True,
        )
    )

    positions = torch.arange(
        forward_permutation.shape[1],
        device=forward_permutation.device,
        dtype=torch.long,
    ).view(
        1,
        -1,
    ).expand_as(
        forward_permutation
    )

    reverse_positions = torch.where(
        positions < valid_count,
        valid_count - 1 - positions,
        positions,
    )

    return torch.gather(
        forward_permutation,
        dim=1,
        index=reverse_positions,
    )


def minimum_neighbor_labels(
    labels: torch.Tensor,
    sentinel: int,
) -> torch.Tensor:
    up = torch.full_like(
        labels,
        sentinel,
    )

    down = torch.full_like(
        labels,
        sentinel,
    )

    left = torch.full_like(
        labels,
        sentinel,
    )

    right = torch.full_like(
        labels,
        sentinel,
    )

    up[:, 1:, :] = labels[:, :-1, :]
    down[:, :-1, :] = labels[:, 1:, :]
    left[:, :, 1:] = labels[:, :, :-1]
    right[:, :, :-1] = labels[:, :, 1:]

    result = torch.minimum(
        up,
        down,
    )

    result = torch.minimum(
        result,
        left,
    )

    return torch.minimum(
        result,
        right,
    )


@torch.no_grad()
def label_region_components(
    boundary_windows: torch.Tensor,
    valid_windows: torch.Tensor,
    window_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    if (
        boundary_windows.shape
        != valid_windows.shape
    ):
        raise ValueError(
            "Boundary and valid-window "
            "masks must match."
        )

    sequence_length = (
        window_size * window_size
    )

    if (
        boundary_windows.ndim != 2
        or boundary_windows.shape[1]
        != sequence_length
    ):
        raise ValueError(
            "Boundary-window mask has "
            "an unexpected shape."
        )

    num_windows = (
        boundary_windows.shape[0]
    )

    boundary = (
        boundary_windows
        .bool()
        .reshape(
            num_windows,
            window_size,
            window_size,
        )
    )

    valid = (
        valid_windows
        .bool()
        .reshape(
            num_windows,
            window_size,
            window_size,
        )
    )

    interior = (
        valid
        & ~boundary
    )

    token_ids = torch.arange(
        sequence_length,
        device=boundary.device,
        dtype=torch.long,
    ).reshape(
        1,
        window_size,
        window_size,
    ).expand(
        num_windows,
        -1,
        -1,
    )

    sentinel = (
        sequence_length + 1
    )

    labels = torch.where(
        interior,
        token_ids,
        torch.full_like(
            token_ids,
            sentinel,
        ),
    )

    propagation_steps = (
        2 * (window_size - 1)
    )

    for _ in range(
        propagation_steps
    ):
        neighbor_minimum = (
            minimum_neighbor_labels(
                labels,
                sentinel,
            )
        )

        labels = torch.where(
            interior,
            torch.minimum(
                labels,
                neighbor_minimum,
            ),
            labels,
        )

    has_interior = (
        interior
        .flatten(
            start_dim=1
        )
        .any(
            dim=1
        )
        .view(
            -1,
            1,
            1,
        )
    )

    for _ in range(
        propagation_steps
    ):
        neighbor_minimum = (
            minimum_neighbor_labels(
                labels,
                sentinel,
            )
        )

        fill_mask = (
            valid
            & labels.eq(sentinel)
            & neighbor_minimum.lt(
                sentinel
            )
        )

        labels = torch.where(
            fill_mask,
            neighbor_minimum,
            labels,
        )

    labels = torch.where(
        valid
        & ~has_interior,
        torch.zeros_like(
            labels
        ),
        labels,
    )

    labels = torch.where(
        valid
        & labels.eq(sentinel),
        torch.zeros_like(
            labels
        ),
        labels,
    )

    return (
        labels.reshape(
            num_windows,
            sequence_length,
        ),
        (
            boundary
            & valid
        ).reshape(
            num_windows,
            sequence_length,
        ),
    )


def build_component_anchor_keys(
    labels: torch.Tensor,
    valid_windows: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    sequence_length = (
        window_size * window_size
    )

    rank = build_serpentine_rank(
        window_size,
        labels.device,
    ).view(
        1,
        -1,
    ).expand_as(
        labels
    )

    sentinel = (
        sequence_length + 1
    )

    source = torch.where(
        valid_windows,
        rank,
        torch.full_like(
            rank,
            sentinel,
        ),
    )

    anchors = torch.full(
        (
            labels.shape[0],
            sequence_length + 2,
        ),
        sentinel,
        device=labels.device,
        dtype=torch.long,
    )

    anchors.scatter_reduce_(
        dim=1,
        index=labels.clamp(
            min=0,
            max=sequence_length + 1,
        ),
        src=source,
        reduce="amin",
        include_self=True,
    )

    safe_labels = labels.clamp(
        min=0,
        max=sequence_length + 1,
    )

    token_anchors = torch.gather(
        anchors,
        dim=1,
        index=safe_labels,
    )

    return torch.where(
        valid_windows,
        token_anchors,
        torch.full_like(
            token_anchors,
            sentinel,
        ),
    )


def build_hierarchical_region_permutation(
    labels_by_hierarchy: Mapping[
        int,
        torch.Tensor,
    ],
    boundary_by_hierarchy: Mapping[
        int,
        torch.Tensor,
    ],
    valid_windows: torch.Tensor,
    hierarchy_order: tuple[
        int,
        ...,
    ],
    window_size: int,
) -> torch.Tensor:
    if not hierarchy_order:
        raise ValueError(
            "Hierarchy order cannot be empty."
        )

    sequence_length = (
        window_size * window_size
    )

    if (
        valid_windows.ndim != 2
        or valid_windows.shape[1]
        != sequence_length
    ):
        raise ValueError(
            "Valid-window mask has "
            "an unexpected shape."
        )

    key = (
        ~valid_windows.bool()
    ).long()

    radix = (
        sequence_length + 2
    )

    for hierarchy in hierarchy_order:
        if (
            hierarchy
            not in labels_by_hierarchy
        ):
            raise KeyError(
                "Missing labels for "
                f"hierarchy {hierarchy}."
            )

        anchor_key = (
            build_component_anchor_keys(
                labels_by_hierarchy[
                    hierarchy
                ],
                valid_windows,
                window_size,
            )
        )

        key = (
            key * radix
            + anchor_key
        )

    target_hierarchy = (
        hierarchy_order[-1]
    )

    if (
        target_hierarchy
        not in boundary_by_hierarchy
    ):
        raise KeyError(
            "Missing boundary mask for "
            f"hierarchy {target_hierarchy}."
        )

    boundary_key = (
        boundary_by_hierarchy[
            target_hierarchy
        ]
        & valid_windows
    ).long()

    key = (
        key * 2
        + boundary_key
    )

    local_rank = (
        build_serpentine_rank(
            window_size,
            valid_windows.device,
        )
        .view(
            1,
            -1,
        )
        .expand(
            valid_windows.shape[0],
            -1,
        )
    )

    key = (
        key
        * (sequence_length + 1)
        + local_rank
    )

    return torch.argsort(
        key,
        dim=1,
        stable=True,
    )


@torch.no_grad()
def resize_boundary_map(
    boundary_map: torch.Tensor,
    output_size: tuple[
        int,
        int,
    ],
) -> torch.Tensor:
    if (
        boundary_map.ndim != 4
        or boundary_map.shape[1] != 1
    ):
        raise ValueError(
            "NAM boundary map must have "
            "shape [B, 1, H, W]."
        )

    input_height, input_width = (
        boundary_map.shape[-2:]
    )

    output_height, output_width = (
        output_size
    )

    boundary_map = (
        boundary_map.float()
    )

    if (
        output_height <= input_height
        and output_width <= input_width
    ):
        resized = F.adaptive_max_pool2d(
            boundary_map,
            output_size,
        )
    else:
        resized = F.interpolate(
            boundary_map,
            size=output_size,
            mode="nearest",
        )

    return resized.gt(0.5)


def forward_mamba_block(
    block: nn.Module,
    x: torch.Tensor,
    permutation: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> torch.Tensor:
    normalized = gather_tokens(
        block.norm1(x),
        permutation,
    )

    mixed = block.mixer(
        normalized
    )

    mixed = gather_tokens(
        mixed,
        inverse_permutation,
    )

    x = x + block.drop_path(
        block.gamma_1
        * mixed
    )

    x = x + block.drop_path(
        block.gamma_2
        * block.mlp(
            block.norm2(x)
        )
    )

    return x


class NAMScanMambaVisionBackbone(
    MambaVisionBackbone
):
    def __init__(
        self,
        *args: Any,
        scan_mode: ScanMode = (
            "nam_hierarchical"
        ),
        debug_validate_permutations: (
            bool
        ) = False,
        **kwargs: Any,
    ) -> None:
        check_scan_mode(
            scan_mode
        )

        super().__init__(
            *args,
            **kwargs,
        )

        self.scan_mode = (
            scan_mode
        )

        self.debug_validate_permutations = (
            debug_validate_permutations
        )

        self._validate_mamba_schedules()

    def _validate_mamba_schedules(
        self,
    ) -> None:
        for (
            stage_index,
            expected_schedule,
        ) in (
            STAGE_MAMBA_SCHEDULES.items()
        ):
            level = self.levels[
                stage_index
            ]

            mamba_count = sum(
                int(
                    isinstance(
                        block.mixer,
                        MambaVisionMixer,
                    )
                )
                for block in level.blocks
            )

            if (
                mamba_count
                != len(
                    expected_schedule
                )
            ):
                raise ValueError(
                    "Stage "
                    f"{stage_index + 1} "
                    "Mamba count mismatch: "
                    f"found={mamba_count}, "
                    "expected="
                    f"{len(expected_schedule)}"
                )

    def _get_mamba_schedule(
        self,
        stage_index: int,
    ) -> tuple[
        tuple[int, bool],
        ...,
    ]:
        schedule = (
            STAGE_MAMBA_SCHEDULES[
                stage_index
            ]
        )

        if (
            self.scan_mode
            == "nam_60_only"
        ):
            return tuple(
                (
                    60,
                    reverse_path,
                )
                for _, reverse_path
                in schedule
            )

        return schedule

    @staticmethod
    def _partition_valid_mask(
        batch_size: int,
        height: int,
        width: int,
        pad_right: int,
        pad_bottom: int,
        window_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        valid = torch.ones(
            (
                batch_size,
                1,
                height,
                width,
            ),
            device=device,
            dtype=torch.float32,
        )

        if (
            pad_right > 0
            or pad_bottom > 0
        ):
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

        return (
            window_partition(
                valid,
                window_size,
            )
            .squeeze(-1)
            .gt(0.5)
        )

    @staticmethod
    def _partition_boundary_maps(
        nam_maps: Mapping[
            int,
            torch.Tensor,
        ],
        required_hierarchies: tuple[
            int,
            ...,
        ],
        height: int,
        width: int,
        pad_right: int,
        pad_bottom: int,
        window_size: int,
    ) -> dict[
        int,
        torch.Tensor,
    ]:
        result: dict[
            int,
            torch.Tensor,
        ] = {}

        for hierarchy in (
            required_hierarchies
        ):
            if hierarchy not in nam_maps:
                raise KeyError(
                    "Missing NAM hierarchy: "
                    f"{hierarchy}"
                )

            boundary = (
                resize_boundary_map(
                    nam_maps[
                        hierarchy
                    ],
                    (
                        height,
                        width,
                    ),
                )
                .float()
            )

            if (
                pad_right > 0
                or pad_bottom > 0
            ):
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

            result[
                hierarchy
            ] = (
                window_partition(
                    boundary,
                    window_size,
                )
                .squeeze(-1)
                .gt(0.5)
            )

        return result

    @torch.no_grad()
    def _build_scan_paths(
        self,
        nam_maps: Mapping[
            int,
            torch.Tensor,
        ],
        stage_index: int,
        batch_size: int,
        height: int,
        width: int,
        pad_right: int,
        pad_bottom: int,
        window_size: int,
        device: torch.device,
    ) -> dict[
        tuple[
            int | str,
            bool,
        ],
        tuple[
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        valid_windows = (
            self._partition_valid_mask(
                batch_size,
                height,
                width,
                pad_right,
                pad_bottom,
                window_size,
                device,
            )
        )

        if (
            self.scan_mode
            == "serpentine"
        ):
            permutation = (
                build_serpentine_permutation(
                    valid_windows,
                    window_size,
                )
            )

            return {
                (
                    "serpentine",
                    False,
                ): make_permutation_pair(
                    permutation,
                    validate=(
                        self
                        .debug_validate_permutations
                    ),
                )
            }

        required_hierarchies = (
            (
                60,
            )
            if self.scan_mode
            == "nam_60_only"
            else (
                20,
                40,
                60,
            )
        )

        raw_boundaries = (
            self._partition_boundary_maps(
                nam_maps,
                required_hierarchies,
                height,
                width,
                pad_right,
                pad_bottom,
                window_size,
            )
        )

        effective_boundaries: dict[
            int,
            torch.Tensor,
        ] = {}

        accumulated_boundary = (
            torch.zeros_like(
                raw_boundaries[
                    required_hierarchies[
                        0
                    ]
                ]
            )
        )

        for hierarchy in (
            required_hierarchies
        ):
            accumulated_boundary = (
                accumulated_boundary
                | raw_boundaries[
                    hierarchy
                ]
            )

            effective_boundaries[
                hierarchy
            ] = (
                accumulated_boundary
            )

        labels_by_hierarchy: dict[
            int,
            torch.Tensor,
        ] = {}

        assigned_boundaries: dict[
            int,
            torch.Tensor,
        ] = {}

        for hierarchy in (
            required_hierarchies
        ):
            (
                labels,
                assigned_boundary,
            ) = label_region_components(
                effective_boundaries[
                    hierarchy
                ],
                valid_windows,
                window_size,
            )

            labels_by_hierarchy[
                hierarchy
            ] = labels

            assigned_boundaries[
                hierarchy
            ] = assigned_boundary

        hierarchy_orders = (
            {
                60: (
                    60,
                ),
            }
            if self.scan_mode
            == "nam_60_only"
            else {
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
            }
        )

        paths: dict[
            tuple[
                int | str,
                bool,
            ],
            tuple[
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}

        for (
            hierarchy,
            hierarchy_order,
        ) in hierarchy_orders.items():
            permutation = (
                build_hierarchical_region_permutation(
                    labels_by_hierarchy,
                    assigned_boundaries,
                    valid_windows,
                    hierarchy_order,
                    window_size,
                )
            )

            paths[
                (
                    hierarchy,
                    False,
                )
            ] = make_permutation_pair(
                permutation,
                validate=(
                    self
                    .debug_validate_permutations
                ),
            )

        schedule = (
            self._get_mamba_schedule(
                stage_index
            )
        )

        for (
            hierarchy,
            reverse_path,
        ) in schedule:
            if not reverse_path:
                continue

            reverse_permutation = (
                reverse_valid_permutation(
                    paths[
                        (
                            hierarchy,
                            False,
                        )
                    ][0],
                    valid_windows,
                )
            )

            paths[
                (
                    hierarchy,
                    True,
                )
            ] = make_permutation_pair(
                reverse_permutation,
                validate=(
                    self
                    .debug_validate_permutations
                ),
            )

        return paths

    def _forward_scan_level(
        self,
        level: nn.Module,
        x: torch.Tensor,
        nam_maps: Mapping[
            int,
            torch.Tensor,
        ],
        stage_index: int,
    ) -> torch.Tensor:
        (
            batch_size,
            _,
            height,
            width,
        ) = x.shape

        window_size = (
            level.window_size
        )

        pad_right = (
            window_size
            - width % window_size
        ) % window_size

        pad_bottom = (
            window_size
            - height % window_size
        ) % window_size

        if (
            pad_right > 0
            or pad_bottom > 0
        ):
            x = F.pad(
                x,
                (
                    0,
                    pad_right,
                    0,
                    pad_bottom,
                ),
            )

        padded_height, padded_width = (
            x.shape[-2:]
        )

        x = window_partition(
            x,
            window_size,
        )

        scan_paths = (
            self._build_scan_paths(
                nam_maps,
                stage_index,
                batch_size,
                height,
                width,
                pad_right,
                pad_bottom,
                window_size,
                x.device,
            )
        )

        schedule = (
            self._get_mamba_schedule(
                stage_index
            )
            if self.scan_mode
            in (
                "nam_hierarchical",
                "nam_60_only",
                "shuffled_nam",
            )
            else ()
        )

        mamba_index = 0

        for block in level.blocks:
            if not isinstance(
                block.mixer,
                MambaVisionMixer,
            ):
                x = block(x)
                continue

            if (
                self.scan_mode
                == "serpentine"
            ):
                (
                    permutation,
                    inverse_permutation,
                ) = scan_paths[
                    (
                        "serpentine",
                        False,
                    )
                ]
            else:
                (
                    hierarchy,
                    reverse_path,
                ) = schedule[
                    mamba_index
                ]

                (
                    permutation,
                    inverse_permutation,
                ) = scan_paths[
                    (
                        hierarchy,
                        reverse_path,
                    )
                ]

            x = forward_mamba_block(
                block,
                x,
                permutation,
                inverse_permutation,
            )

            mamba_index += 1

        x = window_reverse(
            x,
            window_size,
            padded_height,
            padded_width,
        )

        if (
            pad_right > 0
            or pad_bottom > 0
        ):
            x = x[
                :,
                :,
                :height,
                :width,
            ].contiguous()

        return x

    @staticmethod
    def _validate_nam_inputs(
        image: torch.Tensor,
        nam_maps: Mapping[
            int,
            torch.Tensor,
        ],
    ) -> None:
        for hierarchy in (
            20,
            40,
            60,
        ):
            if hierarchy not in nam_maps:
                raise KeyError(
                    "Missing NAM hierarchy: "
                    f"{hierarchy}"
                )

            nam_map = nam_maps[
                hierarchy
            ]

            if (
                nam_map.ndim != 4
                or nam_map.shape[1] != 1
            ):
                raise ValueError(
                    "NAM hierarchy "
                    f"{hierarchy} must have "
                    "shape [B, 1, H, W], "
                    "found "
                    f"{tuple(nam_map.shape)}"
                )

            if (
                nam_map.shape[0]
                != image.shape[0]
            ):
                raise ValueError(
                    "Image and NAM hierarchy "
                    f"{hierarchy} batch sizes "
                    "differ."
                )

    def _prepare_nam_maps(
        self,
        nam_20: torch.Tensor,
        nam_40: torch.Tensor,
        nam_60: torch.Tensor,
    ) -> dict[
        int,
        torch.Tensor,
    ]:
        nam_maps = {
            20: nam_20,
            40: nam_40,
            60: nam_60,
        }

        if (
            self.scan_mode
            == "shuffled_nam"
        ):
            nam_maps = {
                hierarchy: torch.roll(
                    nam_map,
                    shifts=1,
                    dims=0,
                )
                for hierarchy, nam_map
                in nam_maps.items()
            }

        return nam_maps

    def forward(
        self,
        image: torch.Tensor,
        nam_20: torch.Tensor,
        nam_40: torch.Tensor,
        nam_60: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        ...,
    ]:
        nam_maps = (
            self._prepare_nam_maps(
                nam_20,
                nam_40,
                nam_60,
            )
        )

        self._validate_nam_inputs(
            image,
            nam_maps,
        )

        x = self.patch_embed(
            image
        )

        features: list[
            torch.Tensor
        ] = []

        for (
            stage_index,
            level,
        ) in enumerate(
            self.levels
        ):
            if (
                stage_index < 2
                or self.scan_mode
                == "identity"
            ):
                stage_feature = (
                    level.forward_blocks(
                        x
                    )
                )
            else:
                stage_feature = (
                    self._forward_scan_level(
                        level,
                        x,
                        nam_maps,
                        stage_index,
                    )
                )

            if (
                stage_index
                == len(self.levels) - 1
            ):
                stage_feature = (
                    self.norm(
                        stage_feature
                    )
                )

            features.append(
                stage_feature
            )

            if (
                level.downsample
                is not None
            ):
                x = level.downsample(
                    stage_feature
                )
            else:
                x = stage_feature

        return tuple(
            features
        )

    def load_pretrained(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "MambaVision pretrained "
                "checkpoint not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = unwrap_state_dict(
            checkpoint
        )

        state_dict = {
            key: value
            for key, value
            in state_dict.items()
            if not key.startswith(
                "head."
            )
        }

        incompatible = (
            self.load_state_dict(
                state_dict,
                strict=True,
            )
        )

        print(
            "MambaVision NAM-scan "
            "pretrained load | "
            f"path={checkpoint_path} | "
            f"loaded={len(state_dict)} | "
            "missing="
            f"{len(incompatible.missing_keys)} | "
            "unexpected="
            f"{len(incompatible.unexpected_keys)}"
        )


def mamba_vision_small_nam_scan(
    pretrained_path: (
        str | Path | None
    ) = None,
    scan_mode: ScanMode = (
        "nam_hierarchical"
    ),
    debug_validate_permutations: (
        bool
    ) = False,
    **kwargs: Any,
) -> NAMScanMambaVisionBackbone:
    model = (
        NAMScanMambaVisionBackbone(
            depths=kwargs.pop(
                "depths",
                [
                    3,
                    3,
                    7,
                    5,
                ],
            ),
            num_heads=kwargs.pop(
                "num_heads",
                [
                    2,
                    4,
                    8,
                    16,
                ],
            ),
            window_size=kwargs.pop(
                "window_size",
                [
                    8,
                    8,
                    14,
                    7,
                ],
            ),
            dim=kwargs.pop(
                "dim",
                96,
            ),
            in_dim=kwargs.pop(
                "in_dim",
                64,
            ),
            mlp_ratio=kwargs.pop(
                "mlp_ratio",
                4.0,
            ),
            drop_path_rate=kwargs.pop(
                "drop_path_rate",
                0.2,
            ),
            scan_mode=scan_mode,
            debug_validate_permutations=(
                debug_validate_permutations
            ),
            **kwargs,
        )
    )

    if pretrained_path is not None:
        model.load_pretrained(
            pretrained_path
        )

    return model