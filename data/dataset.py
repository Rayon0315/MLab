# data/dataset.py

from __future__ import annotations

from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset


class SODSample(TypedDict):
    image: Tensor
    mask: Tensor

    nam_20: NotRequired[Tensor]
    nam_40: NotRequired[Tensor]
    nam_60: NotRequired[Tensor]

    mean_20: NotRequired[Tensor]
    mean_40: NotRequired[Tensor]
    mean_60: NotRequired[Tensor]

    name: str
    original_size: Tensor


class SODDataset(Dataset):
    """RGB SOD dataset with optional NAMLab maps and region-mean maps."""

    IMAGE_SUFFIXES = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    MASK_SUFFIXES = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    IMAGE_MEAN = torch.tensor(
        [0.485, 0.456, 0.406],
        dtype=torch.float32,
    ).view(3, 1, 1)

    IMAGE_STD = torch.tensor(
        [0.229, 0.224, 0.225],
        dtype=torch.float32,
    ).view(3, 1, 1)

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        nam_dir: str | Path | None = None,
        nam_hierarchies: tuple[int, ...] = (),
        mean_dir: str | Path | None = None,
        mean_hierarchies: tuple[int, ...] = (),
        image_size: tuple[int, int] = (352, 352),
        augment_8way: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.nam_dir = (
            Path(nam_dir)
            if nam_dir is not None
            else None
        )

        self.mean_dir = (
            Path(mean_dir)
            if mean_dir is not None
            else None
        )

        self.image_size = image_size
        self.augment_8way = augment_8way

        self.nam_hierarchies = tuple(
            sorted(
                set(nam_hierarchies)
            )
        )

        self.mean_hierarchies = tuple(
            sorted(
                set(mean_hierarchies)
            )
        )

        self.image_map = self._collect_file_map(
            directory=self.image_dir,
            allowed_suffixes=self.IMAGE_SUFFIXES,
        )

        self.mask_map = self._collect_file_map(
            directory=self.mask_dir,
            allowed_suffixes=self.MASK_SUFFIXES,
        )

        if not self.image_map:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

        missing_masks = sorted(
            set(self.image_map)
            - set(self.mask_map)
        )

        if missing_masks:
            preview = ", ".join(
                missing_masks[:20]
            )

            raise FileNotFoundError(
                "Missing masks | "
                f"count={len(missing_masks)} | "
                f"samples={preview} | "
                f"directory={self.mask_dir}"
            )

        self.names = sorted(
            self.image_map
        )

        self.nam_maps: dict[
            int,
            dict[str, Path],
        ] = {}

        if self.nam_hierarchies:
            if self.nam_dir is None:
                raise ValueError(
                    "NAM hierarchies were requested, "
                    "but nam_dir is None."
                )

            self.nam_maps = {
                hierarchy: self._collect_file_map(
                    directory=(
                        self.nam_dir
                        / f"hier_{hierarchy}"
                    ),
                    allowed_suffixes=self.MASK_SUFFIXES,
                )
                for hierarchy
                in self.nam_hierarchies
            }

            self._validate_nam_maps()

        self.mean_maps: dict[
            int,
            dict[str, Path],
        ] = {}

        if self.mean_hierarchies:
            if self.mean_dir is None:
                raise ValueError(
                    "Region-mean hierarchies were requested, "
                    "but mean_dir is None."
                )

            self.mean_maps = {
                hierarchy: self._collect_file_map(
                    directory=(
                        self.mean_dir
                        / f"region_mean_{hierarchy}"
                    ),
                    allowed_suffixes=self.IMAGE_SUFFIXES,
                )
                for hierarchy
                in self.mean_hierarchies
            }

            self._validate_mean_maps()

    def __len__(self) -> int:
        multiplier = (
            8
            if self.augment_8way
            else 1
        )

        return (
            len(self.names)
            * multiplier
        )

    def __getitem__(
        self,
        index: int,
    ) -> SODSample:
        if self.augment_8way:
            base_index, transform_index = divmod(
                index,
                8,
            )
        else:
            base_index = index
            transform_index = 0

        name = self.names[
            base_index
        ]

        image = self._read_rgb_image(
            self.image_map[name]
        )

        mask = self._read_binary_map(
            self.mask_map[name]
        )

        original_width, original_height = (
            image.size
        )

        original_size = torch.tensor(
            [
                original_height,
                original_width,
            ],
            dtype=torch.long,
        )

        image = self._apply_geometric_transform(
            image,
            transform_index,
        )

        mask = self._apply_geometric_transform(
            mask,
            transform_index,
        )

        target_height, target_width = (
            self.image_size
        )

        target_size = (
            target_width,
            target_height,
        )

        image = image.resize(
            target_size,
            resample=Image.Resampling.BILINEAR,
        )

        mask = mask.resize(
            target_size,
            resample=Image.Resampling.NEAREST,
        )

        sample: SODSample = {
            "image": self._image_to_tensor(
                image
            ),
            "mask": self._binary_to_tensor(
                mask
            ),
            "name": name,
            "original_size": original_size,
        }

        for hierarchy in self.nam_hierarchies:
            nam_map = self._read_binary_map(
                self.nam_maps[
                    hierarchy
                ][name]
            )

            nam_map = (
                self._apply_geometric_transform(
                    nam_map,
                    transform_index,
                )
            )

            nam_map = nam_map.resize(
                target_size,
                resample=Image.Resampling.NEAREST,
            )

            sample[
                f"nam_{hierarchy}"
            ] = self._binary_to_tensor(
                nam_map
            )

        for hierarchy in self.mean_hierarchies:
            mean_map = self._read_rgb_image(
                self.mean_maps[
                    hierarchy
                ][name]
            )

            mean_map = (
                self._apply_geometric_transform(
                    mean_map,
                    transform_index,
                )
            )

            mean_map = mean_map.resize(
                target_size,
                resample=Image.Resampling.NEAREST,
            )

            sample[
                f"mean_{hierarchy}"
            ] = self._rgb_to_tensor(
                mean_map
            )

        return sample

    def _validate_nam_maps(
        self,
    ) -> None:
        for hierarchy in self.nam_hierarchies:
            missing_names = sorted(
                set(self.names)
                - set(
                    self.nam_maps[
                        hierarchy
                    ]
                )
            )

            if not missing_names:
                continue

            preview = ", ".join(
                missing_names[:20]
            )

            raise FileNotFoundError(
                "Missing NAMLab maps | "
                f"hierarchy={hierarchy} | "
                f"count={len(missing_names)} | "
                f"samples={preview} | "
                f"directory="
                f"{self.nam_dir}/"
                f"hier_{hierarchy}"
            )

    def _validate_mean_maps(
        self,
    ) -> None:
        for hierarchy in self.mean_hierarchies:
            missing_names = sorted(
                set(self.names)
                - set(
                    self.mean_maps[
                        hierarchy
                    ]
                )
            )

            if not missing_names:
                continue

            preview = ", ".join(
                missing_names[:20]
            )

            raise FileNotFoundError(
                "Missing region-mean maps | "
                f"hierarchy={hierarchy} | "
                f"count={len(missing_names)} | "
                f"samples={preview} | "
                f"directory="
                f"{self.mean_dir}/"
                f"region_mean_{hierarchy}"
            )

    @staticmethod
    def _apply_geometric_transform(
        image: Image.Image,
        transform_index: int,
    ) -> Image.Image:
        if transform_index == 0:
            return image

        if transform_index == 1:
            return image.transpose(
                Image.Transpose.ROTATE_90
            )

        if transform_index == 2:
            return image.transpose(
                Image.Transpose.ROTATE_180
            )

        if transform_index == 3:
            return image.transpose(
                Image.Transpose.ROTATE_270
            )

        flipped = image.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        if transform_index == 4:
            return flipped

        if transform_index == 5:
            return flipped.transpose(
                Image.Transpose.ROTATE_90
            )

        if transform_index == 6:
            return flipped.transpose(
                Image.Transpose.ROTATE_180
            )

        if transform_index == 7:
            return flipped.transpose(
                Image.Transpose.ROTATE_270
            )

        raise ValueError(
            "Invalid transform index: "
            f"{transform_index}"
        )

    @staticmethod
    def _collect_file_map(
        directory: Path,
        allowed_suffixes: set[str],
    ) -> dict[str, Path]:
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        return {
            path.stem: path
            for path in sorted(
                directory.iterdir()
            )
            if path.is_file()
            and path.suffix.lower()
            in allowed_suffixes
        }

    @staticmethod
    def _read_rgb_image(
        path: Path,
    ) -> Image.Image:
        with Image.open(
            path
        ) as raw_image:
            image = ImageOps.exif_transpose(
                raw_image
            )

            return image.convert(
                "RGB"
            )

    @staticmethod
    def _read_binary_map(
        path: Path,
    ) -> Image.Image:
        with Image.open(
            path
        ) as raw_map:
            binary_map = (
                ImageOps.exif_transpose(
                    raw_map
                )
            )

            return binary_map.convert(
                "L"
            )

    @classmethod
    def _image_to_tensor(
        cls,
        image: Image.Image,
    ) -> Tensor:
        image_array = np.array(
            image,
            dtype=np.float32,
            copy=True,
        )

        image_tensor = torch.from_numpy(
            image_array
        )

        image_tensor = (
            image_tensor
            .permute(2, 0, 1)
            .contiguous()
        )

        image_tensor = (
            image_tensor / 255.0
        )

        return (
            image_tensor
            - cls.IMAGE_MEAN
        ) / cls.IMAGE_STD

    @staticmethod
    def _rgb_to_tensor(
        image: Image.Image,
    ) -> Tensor:
        image_array = np.array(
            image,
            dtype=np.float32,
            copy=True,
        )

        image_tensor = (
            torch.from_numpy(
                image_array
            )
            .permute(2, 0, 1)
            .contiguous()
        )

        return (
            image_tensor / 255.0
        )

    @staticmethod
    def _binary_to_tensor(
        binary_map: Image.Image,
    ) -> Tensor:
        map_array = np.array(
            binary_map,
            dtype=np.float32,
            copy=True,
        )

        map_tensor = (
            torch.from_numpy(
                map_array
            )
            .unsqueeze(0)
        )

        map_tensor = (
            map_tensor / 255.0
        )

        return (
            map_tensor >= 0.5
        ).float()