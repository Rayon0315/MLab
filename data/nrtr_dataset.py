"""NRTR dataset shim loading official NAMLab partition NPZ files.

Expected default layout:
    datasets/EORSSD/train-images/
    datasets/EORSSD/train-mean/
    datasets/EORSSD/train-nam-npz/*.npz

    datasets/EORSSD/test-images/
    datasets/EORSSD/test-mean/
    datasets/EORSSD/test-nam-npz/*.npz

Each NPZ is produced from the official NAMLab MAT `segs` hierarchy and contains:
    partition: H x W integer region IDs, contiguous 0..K-1

The partition directory can be overridden per process with:
    MLAB_NRTR_PARTITION_DIR=/path/to/train-nam-npz
or:
    MLAB_NRTR_PARTITION_DIR=/path/to/test-nam-npz

This avoids changing the canonical train.py / test.py CLI.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.dataset import SODDataset


class NRTRSODDataset(SODDataset):
    NPZ_SUFFIXES = {".npz"}

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

        self.partition_dir = (
            self._resolve_partition_dir()
        )

        self.partition_map = (
            self._collect_file_map(
                directory=self.partition_dir,
                allowed_suffixes=self.NPZ_SUFFIXES,
            )
        )

        missing = sorted(
            set(self.names)
            - set(self.partition_map)
        )

        if missing:
            preview = ", ".join(
                missing[:20]
            )

            raise FileNotFoundError(
                "Missing NRTR NAM partition NPZ files | "
                f"count={len(missing)} | "
                f"samples={preview} | "
                f"directory={self.partition_dir}"
            )

    def _resolve_partition_dir(
        self,
    ) -> Path:
        override = os.environ.get(
            "MLAB_NRTR_PARTITION_DIR"
        )

        if override:
            return Path(
                override
            )

        # Preferred inference from train-images / test-images:
        #   train-images -> train-nam-npz
        #   test-images  -> test-nam-npz
        image_name = (
            self.image_dir.name
        )

        if image_name.endswith(
            "-images"
        ):
            split = image_name[
                :-len("-images")
            ]

            return (
                self.image_dir.parent
                / f"{split}-nam-npz"
            )

        # Fallback inference from train-mean / test-mean.
        if self.mean_dir is not None:
            mean_name = (
                self.mean_dir.name
            )

            if mean_name.endswith(
                "-mean"
            ):
                split = mean_name[
                    :-len("-mean")
                ]

                return (
                    self.mean_dir.parent
                    / f"{split}-nam-npz"
                )

        raise ValueError(
            "Cannot infer NRTR partition directory. "
            "Set MLAB_NRTR_PARTITION_DIR explicitly."
        )

    @staticmethod
    def _load_partition(
        path: Path,
    ) -> np.ndarray:
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            if "partition" not in data:
                raise KeyError(
                    f"{path} does not contain 'partition'."
                )

            partition = np.array(
                data["partition"],
                dtype=np.int32,
                copy=True,
            )

            expected_num_regions = (
                int(
                    data["num_regions"]
                )
                if "num_regions" in data
                else None
            )

        if partition.ndim != 2:
            raise ValueError(
                f"{path}: partition must be 2D, "
                f"got shape={partition.shape}"
            )

        if partition.size == 0:
            raise ValueError(
                f"{path}: empty partition"
            )

        if partition.min() < 0:
            raise ValueError(
                f"{path}: partition contains negative IDs"
            )

        if expected_num_regions is not None:
            minimum_id = int(
                partition.min()
            )
            maximum_id = int(
                partition.max()
            )

            if (
                minimum_id != 0
                or maximum_id
                != expected_num_regions - 1
            ):
                raise ValueError(
                    f"{path}: expected IDs in "
                    f"0..{expected_num_regions - 1}, "
                    f"got min={minimum_id}, "
                    f"max={maximum_id}. "
                    "Run check_nrtr_npz_dataset.py "
                    "for full contiguity validation."
                )

        return partition

    def __getitem__(
        self,
        index: int,
    ):
        sample = super().__getitem__(
            index
        )

        if self.augment_8way:
            base_index, transform_index = (
                divmod(
                    index,
                    8,
                )
            )
        else:
            base_index = index
            transform_index = 0

        name = self.names[
            base_index
        ]

        partition = (
            self._load_partition(
                self.partition_map[
                    name
                ]
            )
        )

        # PIL mode "I" keeps integer region IDs exactly while allowing the same
        # 8-way transform and nearest-neighbour resize as masks/mean maps.
        partition_image = Image.fromarray(
            partition,
            mode="I",
        )

        partition_image = (
            self._apply_geometric_transform(
                partition_image,
                transform_index,
            )
        )

        target_height, target_width = (
            self.image_size
        )

        partition_image = (
            partition_image.resize(
                (
                    target_width,
                    target_height,
                ),
                resample=(
                    Image.Resampling.NEAREST
                ),
            )
        )

        partition_array = np.array(
            partition_image,
            dtype=np.int64,
            copy=True,
        )

        sample[
            "region_partition"
        ] = torch.from_numpy(
            partition_array
        )

        return sample
