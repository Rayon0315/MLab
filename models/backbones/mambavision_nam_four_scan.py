# models/backbones/mambavision_nam_four_scan.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.mambavision import (
    MambaVisionMixer,
    unwrap_state_dict,
    window_partition,
    window_reverse,
)
from models.backbones.mambavision_nam_scan import (
    NAMScanMambaVisionBackbone,
    ScanMode,
    build_component_anchor_keys,
    build_serpentine_permutation,
    build_serpentine_rank,
    gather_tokens,
    label_region_components,
    make_permutation_pair,
)


FOUR_PATH_VARIANTS = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)

FOUR_PATH_NAMES = {
    (False, False): "region_forward_token_forward",
    (False, True): "region_forward_token_reverse",
    (True, False): "region_reverse_token_forward",
    (True, True): "region_reverse_token_reverse",
}

STAGE_MAMBA_HIERARCHIES: dict[int, tuple[int, ...]] = {
    2: (20, 40, 60, 60),
    3: (20, 40, 60),
}


def build_hierarchical_region_keys(
    labels_by_hierarchy: Mapping[int, torch.Tensor],
    boundary_by_hierarchy: Mapping[int, torch.Tensor],
    valid_windows: torch.Tensor,
    hierarchy_order: tuple[int, ...],
    window_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    if not hierarchy_order:
        raise ValueError(
            "Hierarchy order cannot be empty."
        )

    sequence_length = window_size * window_size

    if (
        valid_windows.ndim != 2
        or valid_windows.shape[1] != sequence_length
    ):
        raise ValueError(
            "Valid-window mask has an unexpected shape."
        )

    region_radix = sequence_length + 2
    region_space = region_radix ** len(
        hierarchy_order
    )

    region_key = torch.zeros_like(
        valid_windows,
        dtype=torch.long,
    )

    for hierarchy in hierarchy_order:
        if hierarchy not in labels_by_hierarchy:
            raise KeyError(
                f"Missing labels for hierarchy {hierarchy}."
            )

        anchor_key = build_component_anchor_keys(
            labels_by_hierarchy[hierarchy],
            valid_windows,
            window_size,
        )

        region_key = (
            region_key * region_radix
            + anchor_key
        )

    target_hierarchy = hierarchy_order[-1]

    if target_hierarchy not in boundary_by_hierarchy:
        raise KeyError(
            "Missing boundary mask for "
            f"hierarchy {target_hierarchy}."
        )

    local_rank = build_serpentine_rank(
        window_size,
        valid_windows.device,
    ).view(
        1,
        -1,
    ).expand(
        valid_windows.shape[0],
        -1,
    )

    boundary_key = (
        boundary_by_hierarchy[target_hierarchy]
        & valid_windows
    ).long()

    within_radix = sequence_length + 1
    within_space = 2 * within_radix

    within_key = (
        boundary_key * within_radix
        + local_rank
    )

    region_key = torch.where(
        valid_windows,
        region_key,
        torch.zeros_like(region_key),
    )

    within_key = torch.where(
        valid_windows,
        within_key,
        torch.zeros_like(within_key),
    )

    return (
        region_key,
        within_key,
        region_space,
        within_space,
    )


def build_hierarchical_region_permutations(
    labels_by_hierarchy: Mapping[int, torch.Tensor],
    boundary_by_hierarchy: Mapping[int, torch.Tensor],
    valid_windows: torch.Tensor,
    hierarchy_order: tuple[int, ...],
    window_size: int,
) -> dict[
    tuple[bool, bool],
    torch.Tensor,
]:
    (
        region_key,
        within_key,
        region_space,
        within_space,
    ) = build_hierarchical_region_keys(
        labels_by_hierarchy,
        boundary_by_hierarchy,
        valid_windows,
        hierarchy_order,
        window_size,
    )

    padding_key = (
        ~valid_windows.bool()
    ).long()

    valid_key_space = (
        region_space * within_space
    )

    permutations: dict[
        tuple[bool, bool],
        torch.Tensor,
    ] = {}

    for (
        reverse_region,
        reverse_within_region,
    ) in FOUR_PATH_VARIANTS:
        ordered_region_key = (
            region_space - 1 - region_key
            if reverse_region
            else region_key
        )

        ordered_within_key = (
            within_space - 1 - within_key
            if reverse_within_region
            else within_key
        )

        sort_key = (
            padding_key * valid_key_space
            + ordered_region_key * within_space
            + ordered_within_key
        )

        permutations[
            (
                reverse_region,
                reverse_within_region,
            )
        ] = torch.argsort(
            sort_key,
            dim=1,
            stable=True,
        )

    return permutations


def forward_four_path_mamba_block(
    block: nn.Module,
    x: torch.Tensor,
    path_pairs: Sequence[
        tuple[
            torch.Tensor,
            torch.Tensor,
        ]
    ],
) -> torch.Tensor:
    if len(path_pairs) != 4:
        raise ValueError(
            "Four-path Mamba requires exactly four paths."
        )

    normalized = block.norm1(x)

    reordered_paths = [
        gather_tokens(
            normalized,
            permutation,
        )
        for permutation, _
        in path_pairs
    ]

    mixed_batch = block.mixer(
        torch.cat(
            reordered_paths,
            dim=0,
        )
    )

    mixed_paths = mixed_batch.chunk(
        len(path_pairs),
        dim=0,
    )

    restored_paths = [
        gather_tokens(
            mixed_path,
            inverse_permutation,
        )
        for (
            mixed_path,
            (
                _,
                inverse_permutation,
            ),
        ) in zip(
            mixed_paths,
            path_pairs,
        )
    ]

    mixed = torch.stack(
        restored_paths,
        dim=0,
    ).mean(
        dim=0
    )

    x = x + block.drop_path(
        block.gamma_1 * mixed
    )

    x = x + block.drop_path(
        block.gamma_2
        * block.mlp(
            block.norm2(x)
        )
    )

    return x


class NAMFourScanMambaVisionBackbone(
    NAMScanMambaVisionBackbone
):
    def _validate_mamba_schedules(
        self,
    ) -> None:
        for (
            stage_index,
            expected_hierarchies,
        ) in STAGE_MAMBA_HIERARCHIES.items():
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
                    expected_hierarchies
                )
            ):
                raise ValueError(
                    "Stage "
                    f"{stage_index + 1} "
                    "Mamba count mismatch: "
                    f"found={mamba_count}, "
                    "expected="
                    f"{len(expected_hierarchies)}"
                )

    def _get_mamba_hierarchies(
        self,
        stage_index: int,
    ) -> tuple[int, ...]:
        hierarchies = (
            STAGE_MAMBA_HIERARCHIES[
                stage_index
            ]
        )

        if (
            self.scan_mode
            == "nam_60_only"
        ):
            return tuple(
                60
                for _ in hierarchies
            )

        return hierarchies

    @torch.no_grad()
    def _build_four_scan_paths(
        self,
        nam_maps: Mapping[
            int,
            torch.Tensor,
        ],
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
            if (
                self.scan_mode
                == "nam_60_only"
            )
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
                .clone()
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
            if (
                self.scan_mode
                == "nam_60_only"
            )
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
            permutations = (
                build_hierarchical_region_permutations(
                    labels_by_hierarchy,
                    assigned_boundaries,
                    valid_windows,
                    hierarchy_order,
                    window_size,
                )
            )

            for (
                reverse_region,
                reverse_within_region,
            ), permutation in (
                permutations.items()
            ):
                paths[
                    (
                        hierarchy,
                        reverse_region,
                        reverse_within_region,
                    )
                ] = make_permutation_pair(
                    permutation,
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

        (
            padded_height,
            padded_width,
        ) = x.shape[-2:]

        x = window_partition(
            x,
            window_size,
        )

        scan_paths = (
            self._build_four_scan_paths(
                nam_maps,
                batch_size,
                height,
                width,
                pad_right,
                pad_bottom,
                window_size,
                x.device,
            )
        )

        hierarchies = (
            self._get_mamba_hierarchies(
                stage_index
            )
            if self.scan_mode in (
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
                        False,
                    )
                ]

                normalized = (
                    gather_tokens(
                        block.norm1(x),
                        permutation,
                    )
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
            else:
                hierarchy = hierarchies[
                    mamba_index
                ]

                path_pairs = [
                    scan_paths[
                        (
                            hierarchy,
                            reverse_region,
                            reverse_within_region,
                        )
                    ]
                    for (
                        reverse_region,
                        reverse_within_region,
                    ) in FOUR_PATH_VARIANTS
                ]

                x = (
                    forward_four_path_mamba_block(
                        block,
                        x,
                        path_pairs,
                    )
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
            "MambaVision NAM four-path "
            "pretrained load | "
            f"path={checkpoint_path} | "
            f"loaded={len(state_dict)} | "
            "missing="
            f"{len(incompatible.missing_keys)} | "
            "unexpected="
            f"{len(incompatible.unexpected_keys)}"
        )


def mamba_vision_small_nam_four_scan(
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
) -> NAMFourScanMambaVisionBackbone:
    model = (
        NAMFourScanMambaVisionBackbone(
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