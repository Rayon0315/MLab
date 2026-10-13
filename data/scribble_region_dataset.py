from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.scribble_dataset import ScribbleSODDataset


class ScribbleRegionSODDataset(ScribbleSODDataset):
    """Scribble dataset with official NAMLab region-partition NPZ labels."""

    def __init__(
        self,
        *args,
        region_partition_dir: str | Path,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.region_partition_dir = Path(region_partition_dir)
        self.region_partition_map = self._collect_file_map(
            directory=self.region_partition_dir,
            allowed_suffixes={".npz"},
        )
        self._validate_required_map(
            file_map=self.region_partition_map,
            label="NAMLab region partition",
            directory=self.region_partition_dir,
        )

    @staticmethod
    def _load_partition(path: Path) -> np.ndarray:
        with np.load(path, allow_pickle=False) as data:
            if "partition" not in data:
                raise KeyError(
                    f"{path} does not contain 'partition'."
                )
            partition = np.array(
                data["partition"],
                dtype=np.int32,
                copy=True,
            )

        if partition.ndim != 2:
            raise ValueError(
                f"{path}: partition must be 2D, got {partition.shape}"
            )
        if partition.min() < 0:
            raise ValueError(
                f"{path}: partition contains negative region IDs"
            )
        return partition

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)

        if self.augment_8way:
            base_index, transform_index = divmod(index, 8)
        else:
            base_index = index
            transform_index = 0

        name = self.names[base_index]
        partition = self._load_partition(
            self.region_partition_map[name]
        )
        partition_image = Image.fromarray(partition, mode="I")
        partition_image = self._apply_geometric_transform(
            partition_image,
            transform_index,
        )

        target_height, target_width = self.image_size
        partition_image = partition_image.resize(
            (target_width, target_height),
            resample=Image.Resampling.NEAREST,
        )
        partition_array = np.array(
            partition_image,
            dtype=np.int64,
            copy=True,
        )
        sample["region_partition"] = torch.from_numpy(
            partition_array
        )
        return sample
