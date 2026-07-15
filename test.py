# test.py

import argparse
import importlib
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset

from data.dataset import SODDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test an RGB SOD network."
    )

    parser.add_argument(
        "--network",
        default="models.networks.resnet18_baseline",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--test-images",
        default="datasets/DUTS/DUTS-TE/DUTS-TE-Image",
    )
    parser.add_argument(
        "--test-masks",
        default="datasets/DUTS/DUTS-TE/DUTS-TE-Mask",
    )

    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/resnet18_baseline/DUTS-TE",
    )
    parser.add_argument("--log-interval", type=int, default=50)

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit test samples for debugging.",
    )

    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                log_path,
                mode="w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def build_model(network_path: str) -> torch.nn.Module:
    network_module = importlib.import_module(network_path)
    return network_module.build_model()


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    network_path: str,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint["network"] != network_path:
        raise RuntimeError(
            "Checkpoint network does not match:\n"
            f'checkpoint: {checkpoint["network"]}\n'
            f"command:    {network_path}"
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    return checkpoint


@torch.inference_mode()
def test(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    prediction_dir: Path,
    log_interval: int,
) -> dict[str, float]:
    logger = logging.getLogger(__name__)

    model.eval()

    mae_sum = 0.0
    sample_count = 0
    start_time = time.perf_counter()

    for batch_index, batch in enumerate(data_loader, start=1):
        image = batch["image"].to(
            device,
            non_blocking=True,
        )
        mask = batch["mask"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(image)["pred"]

        prediction = torch.sigmoid(logits)

        batch_mae = (
            prediction.sub(mask)
            .abs()
            .flatten(start_dim=1)
            .mean(dim=1)
        )

        mae_sum += batch_mae.sum().item()
        sample_count += image.shape[0]

        for index, name in enumerate(batch["name"]):
            original_height = int(
                batch["original_size"][index, 0]
            )
            original_width = int(
                batch["original_size"][index, 1]
            )

            restored_prediction = F.interpolate(
                prediction[index:index + 1],
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )

            prediction_array = (
                restored_prediction[0, 0]
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

            Image.fromarray(
                prediction_array,
                mode="L",
            ).save(
                prediction_dir / f"{name}.png"
            )

        if (
            batch_index % log_interval == 0
            or batch_index == len(data_loader)
        ):
            logger.info(
                "Batch %05d/%05d | Saved %d predictions",
                batch_index,
                len(data_loader),
                sample_count,
            )

    elapsed_time = time.perf_counter() - start_time

    return {
        "mae": mae_sum / sample_count,
        "samples": sample_count,
        "time_seconds": elapsed_time,
        "milliseconds_per_image": elapsed_time * 1000 / sample_count,
    }


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir / "test.log")
    logger = logging.getLogger(__name__)

    logger.info("Network: %s", args.network)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Device: %s", device)
    logger.info("AMP: %s", use_amp)

    dataset = SODDataset(
        image_dir=args.test_images,
        mask_dir=args.test_masks,
        image_size=(args.image_size, args.image_size),
    )

    if args.max_samples is not None:
        dataset = Subset(
            dataset,
            range(min(args.max_samples, len(dataset))),
        )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.network)
    checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        network_path=args.network,
    )
    model = model.to(device)

    logger.info(
        "Loaded checkpoint | Epoch: %d | Best MAE: %s",
        checkpoint["epoch"],
        checkpoint.get("best_metric"),
    )
    logger.info("Test samples: %d", len(dataset))

    statistics = test(
        model=model,
        data_loader=data_loader,
        device=device,
        use_amp=use_amp,
        prediction_dir=prediction_dir,
        log_interval=args.log_interval,
    )

    results = {
        "network": args.network,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_best_metric": checkpoint.get("best_metric"),
        "test_dataset": "DUTS-TE",
        "metric_space": "resized_input",
        **statistics,
    }

    with (output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Test completed | Samples %d | MAE %.6f | "
        "Time %.1fs | %.2f ms/image",
        statistics["samples"],
        statistics["mae"],
        statistics["time_seconds"],
        statistics["milliseconds_per_image"],
    )
    logger.info(
        "Predictions saved to: %s",
        prediction_dir,
    )


if __name__ == "__main__":
    main()