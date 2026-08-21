# tools/train_tail.py

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from data.dataset import SODDataset
from engine.model_inputs import (
    get_model_input_keys,
    get_model_mean_hierarchies,
    get_model_nam_hierarchies,
    model_uses_mean,
    model_uses_nam,
)
from engine.trainer import train_one_epoch
from losses.sod_loss import SODLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue a trained SOD checkpoint "
            "with a fixed low learning rate."
        )
    )

    parser.add_argument(
        "--network",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--run-dir",
        required=True,
    )

    parser.add_argument(
        "--train-images",
        required=True,
    )

    parser.add_argument(
        "--train-masks",
        required=True,
    )

    parser.add_argument(
        "--train-nam",
        default=None,
    )

    parser.add_argument(
        "--train-mean",
        default=None,
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
        default=8,
    )

    parser.add_argument(
        "--tail-epochs",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--aux-weight",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--edge-weight",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--region-weight",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--region-hierarchy",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--augment-8way",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
    )

    return parser.parse_args()


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(
    log_path: Path,
) -> logging.Logger:
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

    return logging.getLogger(
        __name__
    )


def build_model(
    network_path: str,
) -> tuple[
    torch.nn.Module,
    object,
]:
    network_module = (
        importlib.import_module(
            network_path
        )
    )

    model = (
        network_module.build_model()
    )

    return (
        model,
        network_module,
    )


def load_source_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    network_path: str,
) -> tuple[int, int, dict]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_network = checkpoint.get(
        "network"
    )

    if checkpoint_network != network_path:
        raise RuntimeError(
            "Checkpoint network does not match:\n"
            f"checkpoint: {checkpoint_network}\n"
            f"command: {network_path}"
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    if (
        "scaler" in checkpoint
        and checkpoint["scaler"]
    ):
        scaler.load_state_dict(
            checkpoint["scaler"]
        )

    source_epoch = int(
        checkpoint.get(
            "epoch",
            0,
        )
    )

    global_step = int(
        checkpoint.get(
            "global_step",
            0,
        )
    )

    return (
        source_epoch,
        global_step,
        checkpoint,
    )


def force_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
) -> None:
    for parameter_group in (
        optimizer.param_groups
    ):
        parameter_group["lr"] = (
            learning_rate
        )

        parameter_group[
            "initial_lr"
        ] = learning_rate


def save_tail_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    source_checkpoint: str,
    source_epoch: int,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "network": args.network,
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": (
                optimizer.state_dict()
            ),
            "scaler": (
                scaler.state_dict()
            ),
            "scheduler": None,
            "args": vars(args),
            "tail_training": {
                "source_checkpoint": (
                    source_checkpoint
                ),
                "source_epoch": (
                    source_epoch
                ),
                "fixed_lr": args.lr,
            },
        },
        path,
    )


def prepare_metrics_file(
    path: Path,
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                "epoch",
                "global_step",
                "train_loss",
                "train_loss_main",
                "train_loss_aux",
                "train_loss_region",
                "train_loss_edge",
                "learning_rate",
                "train_time_seconds",
            ]
        )


def append_metrics(
    path: Path,
    epoch: int,
    global_step: int,
    statistics: dict[str, float],
) -> None:
    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            [
                epoch,
                global_step,
                statistics["loss"],
                statistics.get(
                    "loss_main",
                    "",
                ),
                statistics.get(
                    "loss_aux",
                    "",
                ),
                statistics.get(
                    "loss_region",
                    "",
                ),
                statistics.get(
                    "loss_edge",
                    "",
                ),
                statistics["lr"],
                statistics[
                    "time_seconds"
                ],
            ]
        )


def main() -> None:
    args = parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        args.device
    )

    use_amp = (
        args.amp
        and device.type == "cuda"
    )

    run_dir = Path(
        args.run_dir
    )

    checkpoint_dir = (
        run_dir
        / "checkpoints"
    )

    log_dir = (
        run_dir
        / "logs"
    )

    source_dir = (
        run_dir
        / "network_source"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = setup_logging(
        log_dir
        / "train_tail.log"
    )

    logger.info(
        "Run directory: %s",
        run_dir,
    )

    logger.info(
        "Source checkpoint: %s",
        args.checkpoint,
    )

    logger.info(
        "Network: %s",
        args.network,
    )

    logger.info(
        "Device: %s",
        device,
    )

    if device.type == "cuda":
        logger.info(
            "Device name: %s",
            torch.cuda.get_device_name(
                device
            ),
        )

    logger.info(
        "AMP: %s",
        use_amp,
    )

    logger.info(
        "Tail LR: %.8f",
        args.lr,
    )

    logger.info(
        "Tail epochs: %d",
        args.tail_epochs,
    )

    logger.info(
        "LR scheduler: disabled",
    )

    (
        model,
        network_module,
    ) = build_model(
        args.network
    )

    model_input_keys = (
        get_model_input_keys(
            model
        )
    )

    nam_hierarchies = (
        get_model_nam_hierarchies(
            model
        )
    )

    mean_hierarchies = (
        get_model_mean_hierarchies(
            model
        )
    )

    logger.info(
        "Model inputs: %s",
        ", ".join(
            model_input_keys
        ),
    )

    train_nam_dir = (
        args.train_nam
        if model_uses_nam(
            model
        )
        else None
    )

    train_mean_dir = (
        args.train_mean
        if model_uses_mean(
            model
        )
        else None
    )

    if (
        model_uses_mean(model)
        and train_mean_dir is None
    ):
        raise ValueError(
            "--train-mean is required "
            "for this network."
        )

    network_source_path = Path(
        network_module.__file__
    )

    shutil.copy2(
        network_source_path,
        source_dir
        / network_source_path.name,
    )

    model = model.to(
        device
    )

    criterion = SODLoss(
        aux_weight=args.aux_weight,
        edge_weight=args.edge_weight,
        region_weight=(
            args.region_weight
        ),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )
    )

    (
        source_epoch,
        global_step,
        source_checkpoint,
    ) = load_source_checkpoint(
        path=Path(
            args.checkpoint
        ),
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        network_path=args.network,
    )

    # Ignore the original cosine scheduler completely.
    # Keep AdamW state, but force every parameter group
    # onto the fixed tail learning rate.
    force_learning_rate(
        optimizer=optimizer,
        learning_rate=args.lr,
    )

    logger.info(
        "Loaded checkpoint | "
        "Epoch %d | Step %d",
        source_epoch,
        global_step,
    )

    logger.info(
        "Optimizer state restored",
    )

    logger.info(
        "Learning rate reset to %.8f",
        args.lr,
    )

    if (
        "scheduler"
        in source_checkpoint
    ):
        logger.info(
            "Original scheduler state ignored",
        )

    train_dataset = SODDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        nam_dir=train_nam_dir,
        nam_hierarchies=(
            nam_hierarchies
        ),
        mean_dir=train_mean_dir,
        mean_hierarchies=(
            mean_hierarchies
        ),
        image_size=(
            args.image_size,
            args.image_size,
        ),
        augment_8way=(
            args.augment_8way
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    region_target_key = None

    if args.region_weight > 0.0:
        region_target_key = (
            f"mean_{args.region_hierarchy}"
        )

    start_epoch = (
        source_epoch
        + 1
    )

    final_epoch = (
        source_epoch
        + args.tail_epochs
    )

    logger.info(
        "Training samples: %d",
        len(train_dataset),
    )

    logger.info(
        "Batches per epoch: %d",
        len(train_loader),
    )

    logger.info(
        "Tail range: epoch %d -> %d",
        start_epoch,
        final_epoch,
    )

    with (
        run_dir
        / "args.json"
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

    metrics_path = (
        log_dir
        / "metrics.csv"
    )

    prepare_metrics_file(
        metrics_path
    )

    for epoch in range(
        start_epoch,
        final_epoch + 1,
    ):
        # Enforce fixed LR every epoch.
        force_learning_rate(
            optimizer=optimizer,
            learning_rate=args.lr,
        )

        (
            train_statistics,
            global_step,
        ) = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            use_amp=use_amp,
            log_interval=(
                args.log_interval
            ),
            region_target_key=(
                region_target_key
            ),
        )

        append_metrics(
            path=metrics_path,
            epoch=epoch,
            global_step=global_step,
            statistics=(
                train_statistics
            ),
        )

        logger.info(
            "Epoch %03d completed | "
            "Train loss %.6f | "
            "Main %.6f | "
            "Aux %.6f | "
            "LR %.8f | "
            "Train %.1fs",
            epoch,
            train_statistics["loss"],
            train_statistics.get(
                "loss_main",
                0.0,
            ),
            train_statistics.get(
                "loss_aux",
                0.0,
            ),
            optimizer.param_groups[
                0
            ]["lr"],
            train_statistics[
                "time_seconds"
            ],
        )

        save_tail_checkpoint(
            path=(
                checkpoint_dir
                / "latest.pth"
            ),
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            epoch=epoch,
            global_step=global_step,
            source_checkpoint=(
                args.checkpoint
            ),
            source_epoch=(
                source_epoch
            ),
        )

        if (
            epoch
            % args.save_every
            == 0
        ):
            save_tail_checkpoint(
                path=(
                    checkpoint_dir
                    / (
                        f"epoch_"
                        f"{epoch:04d}.pth"
                    )
                ),
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=(
                    global_step
                ),
                source_checkpoint=(
                    args.checkpoint
                ),
                source_epoch=(
                    source_epoch
                ),
            )

    save_tail_checkpoint(
        path=(
            checkpoint_dir
            / "final.pth"
        ),
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        args=args,
        epoch=final_epoch,
        global_step=global_step,
        source_checkpoint=(
            args.checkpoint
        ),
        source_epoch=source_epoch,
    )

    logger.info(
        "Tail training completed | "
        "Final epoch %d | "
        "Checkpoint: %s",
        final_epoch,
        checkpoint_dir
        / "final.pth",
    )


if __name__ == "__main__":
    main()