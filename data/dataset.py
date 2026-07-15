# data/dataset.py

from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset


class SODSample(TypedDict):
    image: Tensor
    mask: Tensor
    name: str
    original_size: Tensor


class SODDataset(Dataset):
    """RGB salient object detection dataset."""

    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
    MASK_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

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
        image_size: tuple[int, int] = (352, 352),
    ) -> None:
        """
        Args:
            image_dir:
                RGB 图像目录。

            mask_dir:
                灰度 Mask 目录。

            image_size:
                Resize 后的尺寸，顺序为 (height, width)。
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self._check_directories()
        self._check_image_size()

        image_files = self._collect_files(
            self.image_dir,
            self.IMAGE_SUFFIXES,
        )
        mask_files = self._collect_files(
            self.mask_dir,
            self.MASK_SUFFIXES,
        )

        image_map = self._build_file_map(
            files=image_files,
            file_type="image",
        )
        mask_map = self._build_file_map(
            files=mask_files,
            file_type="mask",
        )

        self.samples = self._pair_files(
            image_map=image_map,
            mask_map=mask_map,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SODSample:
        image_path, mask_path = self.samples[index]

        image = self._read_rgb_image(image_path)
        mask = self._read_mask(mask_path)

        if image.size != mask.size:
            raise ValueError(
                "Image and mask have different original sizes:\n"
                f"image: {image_path} -> {image.size}\n"
                f"mask:  {mask_path} -> {mask.size}"
            )

        original_width, original_height = image.size

        original_size = torch.tensor(
            [original_height, original_width],
            dtype=torch.long,
        )

        target_height, target_width = self.image_size

        image = image.resize(
            (target_width, target_height),
            resample=Image.Resampling.BILINEAR,
        )

        mask = mask.resize(
            (target_width, target_height),
            resample=Image.Resampling.NEAREST,
        )

        image_tensor = self._image_to_tensor(image)
        mask_tensor = self._mask_to_tensor(mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "name": image_path.stem,
            "original_size": original_size,
        }

    def _check_directories(self) -> None:
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory does not exist: {self.image_dir}"
            )

        if not self.image_dir.is_dir():
            raise NotADirectoryError(
                f"Image path is not a directory: {self.image_dir}"
            )

        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Mask directory does not exist: {self.mask_dir}"
            )

        if not self.mask_dir.is_dir():
            raise NotADirectoryError(
                f"Mask path is not a directory: {self.mask_dir}"
            )

    def _check_image_size(self) -> None:
        if len(self.image_size) != 2:
            raise ValueError(
                "image_size must contain exactly two values: "
                "(height, width)."
            )

        height, width = self.image_size

        if height <= 0 or width <= 0:
            raise ValueError(
                "image_size values must be positive, "
                f"but got {self.image_size}."
            )

    @staticmethod
    def _collect_files(
        directory: Path,
        allowed_suffixes: set[str],
    ) -> list[Path]:
        files = sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in allowed_suffixes
        )

        if not files:
            raise RuntimeError(
                f"No supported files found in: {directory}"
            )

        return files

    @staticmethod
    def _build_file_map(
        files: list[Path],
        file_type: str,
    ) -> dict[str, Path]:
        file_map: dict[str, Path] = {}

        for path in files:
            name = path.stem

            if name in file_map:
                raise RuntimeError(
                    f"Duplicate {file_type} name found: {name}\n"
                    f"first:  {file_map[name]}\n"
                    f"second: {path}"
                )

            file_map[name] = path

        return file_map

    @staticmethod
    def _pair_files(
        image_map: dict[str, Path],
        mask_map: dict[str, Path],
    ) -> list[tuple[Path, Path]]:
        image_names = set(image_map)
        mask_names = set(mask_map)

        missing_masks = sorted(image_names - mask_names)
        missing_images = sorted(mask_names - image_names)

        if missing_masks or missing_images:
            messages = ["Image-mask pairing failed."]

            if missing_masks:
                messages.append(
                    "Images without masks: "
                    + ", ".join(missing_masks[:10])
                )

            if missing_images:
                messages.append(
                    "Masks without images: "
                    + ", ".join(missing_images[:10])
                )

            raise RuntimeError("\n".join(messages))

        return [
            (image_map[name], mask_map[name])
            for name in sorted(image_names)
        ]

    @staticmethod
    def _read_rgb_image(path: Path) -> Image.Image:
        with Image.open(path) as raw_image:
            image = ImageOps.exif_transpose(raw_image)
            return image.convert("RGB")

    @staticmethod
    def _read_mask(path: Path) -> Image.Image:
        with Image.open(path) as raw_mask:
            mask = ImageOps.exif_transpose(raw_mask)
            return mask.convert("L")

    @classmethod
    def _image_to_tensor(cls, image: Image.Image) -> Tensor:
        image_array = np.array(
            image,
            dtype=np.float32,
            copy=True,
        )

        image_tensor = torch.from_numpy(image_array)
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()
        image_tensor = image_tensor / 255.0

        image_tensor = (
            image_tensor - cls.IMAGE_MEAN
        ) / cls.IMAGE_STD

        return image_tensor

    @staticmethod
    def _mask_to_tensor(mask: Image.Image) -> Tensor:
        mask_array = np.array(
            mask,
            dtype=np.float32,
            copy=True,
        )

        mask_tensor = torch.from_numpy(mask_array)
        mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor / 255.0

        # 当前阶段将显著图作为二值 Mask 处理。
        mask_tensor = (mask_tensor >= 0.5).float()

        return mask_tensor