# tools/check_nam_dataset.py

import argparse

import torch
from torch.utils.data import DataLoader

from data.dataset import SODDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        default=(
            "datasets/DUTS/DUTS-TR/"
            "DUTS-TR-Image"
        ),
    )

    parser.add_argument(
        "--masks",
        default=(
            "datasets/DUTS/DUTS-TR/"
            "DUTS-TR-Mask"
        ),
    )

    parser.add_argument(
        "--nam",
        default=(
            "datasets/DUTS/DUTS-TR/nam"
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=352,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    return parser.parse_args()


def print_tensor(
    name: str,
    tensor: torch.Tensor,
) -> None:
    print(
        f"{name}: "
        f"shape={tuple(tensor.shape)} | "
        f"dtype={tensor.dtype} | "
        f"min={tensor.min().item():.4f} | "
        f"max={tensor.max().item():.4f} | "
        f"unique={torch.unique(tensor).tolist()}"
    )


def main() -> None:
    args = parse_args()

    dataset = SODDataset(
        image_dir=args.images,
        mask_dir=args.masks,
        nam_dir=args.nam,
        image_size=(
            args.image_size,
            args.image_size,
        ),
    )

    print(f"Dataset samples: {len(dataset)}")

    sample = dataset[0]

    print("\nSingle sample")
    print(f"Keys: {sample.keys()}")
    print(f"Name: {sample['name']}")
    print(
        "Original size: "
        f"{sample['original_size'].tolist()}"
    )

    print_tensor(
        "image",
        sample["image"],
    )

    print_tensor(
        "mask",
        sample["mask"],
    )

    print_tensor(
        "nam_20",
        sample["nam_20"],
    )

    print_tensor(
        "nam_40",
        sample["nam_40"],
    )

    print_tensor(
        "nam_60",
        sample["nam_60"],
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    batch = next(iter(data_loader))

    print("\nBatch")
    print(f"Keys: {batch.keys()}")
    print(f"Names: {batch['name']}")

    print_tensor(
        "image",
        batch["image"],
    )

    print_tensor(
        "mask",
        batch["mask"],
    )

    print_tensor(
        "nam_20",
        batch["nam_20"],
    )

    print_tensor(
        "nam_40",
        batch["nam_40"],
    )

    print_tensor(
        "nam_60",
        batch["nam_60"],
    )


if __name__ == "__main__":
    main()