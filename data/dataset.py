# data/dataset.py

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

    name: str
    original_size: Tensor


class SODDataset(Dataset):
    """RGB SOD dataset with optional NAMLab edge maps."""

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
        image_size: tuple[int, int] = (352, 352),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.nam_dir = (
            Path(nam_dir)
            if nam_dir is not None
            else None
        )
        self.image_size = image_size

        self.image_map = self._collect_file_map(
            directory=self.image_dir,
            allowed_suffixes=self.IMAGE_SUFFIXES,
        )

        self.mask_map = self._collect_file_map(
            directory=self.mask_dir,
            allowed_suffixes=self.MASK_SUFFIXES,
        )

        self.names = sorted(self.image_map)

        self.nam_maps: dict[int, dict[str, Path]] | None = None

        if self.nam_dir is not None:
            self.nam_maps = {
                hierarchy: self._collect_file_map(
                    directory=self.nam_dir / f"hier_{hierarchy}",
                    allowed_suffixes=self.MASK_SUFFIXES,
                )
                for hierarchy in (20, 40, 60)
            }

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> SODSample:
        name = self.names[index]

        image_path = self.image_map[name]
        mask_path = self.mask_map[name]

        image = self._read_rgb_image(image_path)
        mask = self._read_binary_map(mask_path)

        original_width, original_height = image.size

        original_size = torch.tensor(
            [original_height, original_width],
            dtype=torch.long,
        )

        target_height, target_width = self.image_size
        target_size = (target_width, target_height)

        image = image.resize(
            target_size,
            resample=Image.Resampling.BILINEAR,
        )

        mask = mask.resize(
            target_size,
            resample=Image.Resampling.NEAREST,
        )

        sample: SODSample = {
            "image": self._image_to_tensor(image),
            "mask": self._binary_to_tensor(mask),
            "name": name,
            "original_size": original_size,
        }

        if self.nam_maps is not None:
            nam_20 = self._read_binary_map(
                self.nam_maps[20][name]
            )

            nam_40 = self._read_binary_map(
                self.nam_maps[40][name]
            )

            nam_60 = self._read_binary_map(
                self.nam_maps[60][name]
            )

            nam_20 = nam_20.resize(
                target_size,
                resample=Image.Resampling.NEAREST,
            )

            nam_40 = nam_40.resize(
                target_size,
                resample=Image.Resampling.NEAREST,
            )

            nam_60 = nam_60.resize(
                target_size,
                resample=Image.Resampling.NEAREST,
            )

            sample["nam_20"] = self._binary_to_tensor(
                nam_20
            )

            sample["nam_40"] = self._binary_to_tensor(
                nam_40
            )

            sample["nam_60"] = self._binary_to_tensor(
                nam_60
            )

        return sample

    @staticmethod
    def _collect_file_map(
        directory: Path,
        allowed_suffixes: set[str],
    ) -> dict[str, Path]:
        return {
            path.stem: path
            for path in sorted(directory.iterdir())
            if path.is_file()
            and path.suffix.lower() in allowed_suffixes
        }

    @staticmethod
    def _read_rgb_image(
        path: Path,
    ) -> Image.Image:
        with Image.open(path) as raw_image:
            image = ImageOps.exif_transpose(raw_image)
            return image.convert("RGB")

    @staticmethod
    def _read_binary_map(
        path: Path,
    ) -> Image.Image:
        with Image.open(path) as raw_map:
            binary_map = ImageOps.exif_transpose(raw_map)
            return binary_map.convert("L")

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

        image_tensor = image_tensor.permute(
            2,
            0,
            1,
        ).contiguous()

        image_tensor = image_tensor / 255.0

        image_tensor = (
            image_tensor - cls.IMAGE_MEAN
        ) / cls.IMAGE_STD

        return image_tensor

    @staticmethod
    def _binary_to_tensor(
        binary_map: Image.Image,
    ) -> Tensor:
        map_array = np.array(
            binary_map,
            dtype=np.float32,
            copy=True,
        )

        map_tensor = torch.from_numpy(
            map_array
        ).unsqueeze(0)

        map_tensor = map_tensor / 255.0

        return (
            map_tensor >= 0.5
        ).float()