# test.py

import argparse
import importlib
import json
import logging
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.dataset import SODDataset
from engine.model_inputs import (
    get_model_input_keys,
    model_uses_nam,
    prepare_model_inputs,
)
from metrics.sod_metrics import (
    evaluate_prediction_directory,
    get_metric_library_version,
    save_metric_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SOD inference and standard "
            "original-size evaluation."
        )
    )

    parser.add_argument(
        "--network",
        default="models.networks.resnet18_baseline",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--test-images",
        default=(
            "datasets/DUTS/DUTS-TE/"
            "DUTS-TE-Image"
        ),
    )

    parser.add_argument(
        "--test-masks",
        default=(
            "datasets/DUTS/DUTS-TE/"
            "DUTS-TE-Mask"
        ),
    )

    parser.add_argument(
        "--test-nam",
        default=(
            "datasets/DUTS/DUTS-TE/"
            "nam"
        ),
    )

    parser.add_argument(
        "--dataset-name",
        default="DUTS-TE",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=352,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/resnet18_baseline/"
            "DUTS-TE"
        ),
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help=(
            "Progress bar refresh interval "
            "in batches."
        ),
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Use only the first N test "
            "samples."
        ),
    )

    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Skip model inference and evaluate "
            "the existing predictions directory."
        ),
    )

    args = parser.parse_args()

    if (
        not args.evaluate_only
        and args.checkpoint is None
    ):
        parser.error(
            "--checkpoint is required unless "
            "--evaluate-only is used."
        )

    return args


def setup_logging(
    log_path: Path,
) -> None:
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


def build_model(
    network_path: str,
) -> nn.Module:
    network_module = importlib.import_module(
        network_path
    )

    return network_module.build_model()


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
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
            f"command: {network_path}"
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    return checkpoint


def synchronize(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def warm_up(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    warmup_steps: int,
) -> None:
    if warmup_steps <= 0:
        return

    batch = next(iter(data_loader))

    model_inputs = prepare_model_inputs(
        model=model,
        batch=batch,
        device=device,
    )

    for _ in range(warmup_steps):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            model(**model_inputs)

    synchronize(device)


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    prediction_dir: Path,
    log_interval: int,
) -> dict[str, float | int]:
    model.eval()

    sample_count = 0
    forward_seconds = 0.0
    prediction_save_seconds = 0.0

    inference_start = time.perf_counter()

    progress_bar = tqdm(
        data_loader,
        desc="Inference",
        unit="batch",
        dynamic_ncols=True,
        miniters=log_interval,
    )

    for batch in progress_bar:
        model_inputs = prepare_model_inputs(
            model=model,
            batch=batch,
            device=device,
        )

        synchronize(device)

        forward_start = time.perf_counter()

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            outputs = model(**model_inputs)
            logits = outputs["pred"]

        synchronize(device)

        forward_seconds += (
            time.perf_counter()
            - forward_start
        )

        save_start = time.perf_counter()

        for index, name in enumerate(
            batch["name"]
        ):
            original_height = int(
                batch["original_size"][index, 0]
            )

            original_width = int(
                batch["original_size"][index, 1]
            )

            restored_logits = F.interpolate(
                logits[
                    index:index + 1
                ].float(),
                size=(
                    original_height,
                    original_width,
                ),
                mode="bilinear",
                align_corners=False,
            )

            prediction = torch.sigmoid(
                restored_logits
            )

            prediction_array = (
                prediction[0, 0]
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

            Image.fromarray(
                prediction_array
            ).save(
                prediction_dir
                / f"{name}.png"
            )

        prediction_save_seconds += (
            time.perf_counter()
            - save_start
        )

        sample_count += len(batch["name"])

        progress_bar.set_postfix(
            saved=sample_count,
        )

    end_to_end_seconds = (
        time.perf_counter()
        - inference_start
    )

    return {
        "samples": sample_count,
        "forward_seconds": forward_seconds,
        "forward_ms_per_image": (
            forward_seconds
            * 1000
            / sample_count
        ),
        "prediction_save_seconds": (
            prediction_save_seconds
        ),
        "end_to_end_seconds": (
            end_to_end_seconds
        ),
        "end_to_end_ms_per_image": (
            end_to_end_seconds
            * 1000
            / sample_count
        ),
    }


def get_device_name(
    device: torch.device,
) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(
            device
        )

    return "CPU"


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    use_amp = (
        args.amp
        and device.type == "cuda"
    )

    output_dir = Path(args.output_dir)
    prediction_dir = (
        output_dir / "predictions"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    setup_logging(
        output_dir / "test.log"
    )

    logger = logging.getLogger(__name__)

    with (
        output_dir / "test_args.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vars(args),
            file,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Dataset: %s",
        args.dataset_name,
    )

    logger.info(
        "Device: %s",
        device,
    )

    logger.info(
        "Device name: %s",
        get_device_name(device),
    )

    logger.info(
        "AMP: %s",
        use_amp,
    )

    checkpoint_epoch = None
    checkpoint_best_metric = None
    inference_statistics = None

    if not args.evaluate_only:
        logger.info(
            "Network: %s",
            args.network,
        )

        logger.info(
            "Checkpoint: %s",
            args.checkpoint,
        )

        if prediction_dir.exists():
            shutil.rmtree(
                prediction_dir
            )

        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model = build_model(
            args.network
        )

        model_input_keys = (
            get_model_input_keys(model)
        )

        test_nam_dir = (
            args.test_nam
            if model_uses_nam(model)
            else None
        )

        logger.info(
            "Model inputs: %s",
            ", ".join(model_input_keys),
        )

        if test_nam_dir is not None:
            logger.info(
                "NAM directory: %s",
                test_nam_dir,
            )

        checkpoint = load_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            network_path=args.network,
        )

        checkpoint_epoch = checkpoint["epoch"]

        checkpoint_best_metric = (
            checkpoint.get("best_metric")
        )

        model = model.to(device)
        model.eval()

        dataset = SODDataset(
            image_dir=args.test_images,
            mask_dir=args.test_masks,
            nam_dir=test_nam_dir,
            image_size=(
                args.image_size,
                args.image_size,
            ),
        )

        if args.max_samples is not None:
            dataset = Subset(
                dataset,
                range(
                    min(
                        args.max_samples,
                        len(dataset),
                    )
                ),
            )

        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(
                device.type == "cuda"
            ),
            persistent_workers=(
                args.num_workers > 0
            ),
        )

        logger.info(
            "Loaded checkpoint | "
            "Epoch %d | Best metric %s",
            checkpoint_epoch,
            checkpoint_best_metric,
        )

        logger.info(
            "Test samples: %d",
            len(dataset),
        )

        warm_up(
            model=model,
            data_loader=data_loader,
            device=device,
            use_amp=use_amp,
            warmup_steps=args.warmup_steps,
        )

        inference_statistics = run_inference(
            model=model,
            data_loader=data_loader,
            device=device,
            use_amp=use_amp,
            prediction_dir=prediction_dir,
            log_interval=args.log_interval,
        )

        logger.info(
            "Inference completed | "
            "Forward %.3f ms/image | "
            "End-to-end %.3f ms/image",
            inference_statistics[
                "forward_ms_per_image"
            ],
            inference_statistics[
                "end_to_end_ms_per_image"
            ],
        )

    else:
        logger.info(
            "Evaluation-only mode"
        )

    logger.info(
        "Predictions: %s",
        prediction_dir,
    )

    evaluation_start = time.perf_counter()

    (
        metric_results,
        metric_curves,
        evaluated_samples,
    ) = evaluate_prediction_directory(
        prediction_dir=prediction_dir,
        ground_truth_dir=args.test_masks,
    )

    evaluation_seconds = (
        time.perf_counter()
        - evaluation_start
    )

    save_metric_curves(
        path=output_dir / "curves.npz",
        curves=metric_curves,
    )

    metric_library_version = (
        get_metric_library_version()
    )

    results = {
        "dataset": args.dataset_name,
        "network": args.network,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": (
            checkpoint_epoch
        ),
        "checkpoint_best_metric": (
            checkpoint_best_metric
        ),
        "samples": evaluated_samples,
        "metrics": metric_results,
        "inference": (
            {
                "input_size": (
                    args.image_size
                ),
                "batch_size": (
                    args.batch_size
                ),
                "amp": use_amp,
                "device": str(device),
                "device_name": (
                    get_device_name(device)
                ),
                "warmup_steps": (
                    args.warmup_steps
                ),
                **inference_statistics,
            }
            if inference_statistics
            else None
        ),
        "evaluation": {
            "seconds": evaluation_seconds,
            "prediction_space": (
                "original_size_uint8"
            ),
            "metric_library": (
                "pysodmetrics"
            ),
            "metric_library_version": (
                metric_library_version
            ),
            "fmeasure_beta": 0.3,
            "per_image_min_max_normalization": (
                False
            ),
        },
    }

    with (
        output_dir / "metrics.json"
    ).open(
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
        "Evaluation completed | "
        "Samples %d | Time %.1fs",
        evaluated_samples,
        evaluation_seconds,
    )

    logger.info(
        "MAE %.6f | "
        "S-measure %.6f | "
        "Weighted F-measure %.6f",
        metric_results["mae"],
        metric_results["smeasure"],
        metric_results[
            "weighted_fmeasure"
        ],
    )

    logger.info(
        "F-measure | "
        "Max %.6f | "
        "Mean %.6f | "
        "Adaptive %.6f",
        metric_results[
            "max_fmeasure"
        ],
        metric_results[
            "mean_fmeasure"
        ],
        metric_results[
            "adaptive_fmeasure"
        ],
    )

    logger.info(
        "E-measure | "
        "Max %.6f | "
        "Mean %.6f | "
        "Adaptive %.6f",
        metric_results[
            "max_emeasure"
        ],
        metric_results[
            "mean_emeasure"
        ],
        metric_results[
            "adaptive_emeasure"
        ],
    )

    logger.info(
        "Predictions: %s",
        prediction_dir,
    )

    logger.info(
        "Metrics: %s",
        output_dir / "metrics.json",
    )

    logger.info(
        "Curves: %s",
        output_dir / "curves.npz",
    )


if __name__ == "__main__":
    main()