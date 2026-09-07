from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SODDataset
from metrics.sod_metrics import (
    evaluate_prediction_directory,
    save_metric_curves,
)
from models.networks.mambavision_small_progressive_region_direct_hier60_region_hybrid_diffusion_sod import (
    MambaVisionSmallHybridRegionDiffusionSOD,
)
from train import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test mv-region-hybrid + IPDiff-style diffusion.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--test-images",
        default="datasets/EORSSD/test-images",
    )
    parser.add_argument(
        "--test-masks",
        default="datasets/EORSSD/test-labels",
    )
    parser.add_argument(
        "--test-mean",
        default="datasets/EORSSD/test-mean",
    )
    parser.add_argument("--dataset-name", default="EORSSD")
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
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
        default="runs/mv_h60_region_hybrid_diffusion_eorssd_aug8_e45/test/EORSSD",
    )
    return parser.parse_args()


def setup_logging(path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )


def build_from_checkpoint(
    checkpoint: dict,
) -> MambaVisionSmallHybridRegionDiffusionSOD:
    train_args = checkpoint.get("args", {})
    return MambaVisionSmallHybridRegionDiffusionSOD(
        prior_channels=int(train_args.get("prior_channels", 64)),
        train_timesteps=int(train_args.get("train_timesteps", 1000)),
        sample_steps=int(train_args.get("sample_steps", 10)),
        ipm_probability=float(train_args.get("ipm_probability", 0.2)),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    output_dir = Path(args.output_dir)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "test.log")
    logger = logging.getLogger(__name__)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = build_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device)
    model.eval()

    dataset = SODDataset(
        image_dir=args.test_images,
        mask_dir=args.test_masks,
        mean_dir=args.test_mean,
        mean_hierarchies=(60,),
        image_size=(args.image_size, args.image_size),
        augment_8way=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    logger.info("Dataset: %s", args.dataset_name)
    logger.info("Samples: %d", len(dataset))
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Diffusion sample steps: %d", args.sample_steps)
    logger.info("Sampling seed: %d", args.seed)

    for batch in tqdm(loader, desc="Diffusion inference", dynamic_ncols=True):
        image = batch["image"].to(device, non_blocking=True)
        mean_60 = batch["mean_60"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            outputs = model.sample(
                image=image,
                mean_60=mean_60,
                sample_steps=args.sample_steps,
            )
            probability = outputs["pred"].float()

        for index, name in enumerate(batch["name"]):
            original_height = int(batch["original_size"][index, 0])
            original_width = int(batch["original_size"][index, 1])

            restored = F.interpolate(
                probability[index:index + 1],
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            array = (
                restored.clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            Image.fromarray(array).save(
                prediction_dir / f"{name}.png"
            )

    metrics, curves, sample_count = evaluate_prediction_directory(
        prediction_dir=prediction_dir,
        ground_truth_dir=args.test_masks,
        show_progress=True,
    )
    save_metric_curves(
        output_dir / "metric_curves.npz",
        curves,
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    logger.info("Evaluated samples: %d", sample_count)
    for key, value in metrics.items():
        logger.info("%s: %.6f", key, value)


if __name__ == "__main__":
    main()
