# data/object_scale_dataset.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from data.dataset import SODDataset


class ObjectScaleSODDataset(SODDataset):
    """
    Standard SODDataset + connected-component label map.

    Each positive integer in object_labels identifies one
    salient connected component. Background is 0.
    """

    def __init__(
        self,
        *args: Any,
        instance_dir: str | Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

        self.instance_dir = Path(
            instance_dir
        )

        self.instance_maps = (
            self._collect_file_map(
                directory=(
                    self.instance_dir
                ),
                allowed_suffixes={
                    ".png",
                },
            )
        )

        missing_names = sorted(
            set(self.names)
            - set(self.instance_maps)
        )

        if missing_names:
            preview = ", ".join(
                missing_names[:20]
            )

            raise FileNotFoundError(
                "Missing salient instance maps | "
                f"count={len(missing_names)} | "
                f"samples={preview} | "
                f"directory={self.instance_dir}"
            )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        sample = super().__getitem__(
            index
        )

        if self.augment_8way:
            (
                base_index,
                transform_index,
            ) = divmod(
                index,
                8,
            )
        else:
            base_index = index
            transform_index = 0

        name = self.names[
            base_index
        ]

        with Image.open(
            self.instance_maps[
                name
            ]
        ) as raw_labels:
            object_labels = (
                ImageOps.exif_transpose(
                    raw_labels
                ).copy()
            )

        object_labels = (
            self._apply_geometric_transform(
                object_labels,
                transform_index,
            )
        )

        target_height, target_width = (
            self.image_size
        )

        object_labels = (
            object_labels.resize(
                (
                    target_width,
                    target_height,
                ),
                resample=(
                    Image.Resampling.NEAREST
                ),
            )
        )

        label_array = np.asarray(
            object_labels,
            dtype=np.int64,
        ).copy()

        sample[
            "object_labels"
        ] = (
            torch.from_numpy(
                label_array
            )
            .unsqueeze(0)
            .long()
        )

        return sample
