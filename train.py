# train.py

import argparse
import csv
import importlib
import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from data.dataset import SODDataset
from engine.evaluator import evaluate
from engine.trainer import train_one_epoch
from losses.sod_loss import SODLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an RGB SOD network."
    )

    parser.add_argument(
        "--network",
        default="models.networks.resnet18_baseline",
    )

    parser.add_argument(
        "--train-images",
        default="datasets/DUTS/DUTS-TR/DUTS-TR-Image",
    )
    parser.add_argument(
        "--train-masks",
        default="datasets/DUTS/DUTS-TR/DUTS-TR-Mask",
    )

    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.4)

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
        "--run-dir",
        default="runs/resnet18_baseline",
    )
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--val-count",
        type=int,
        default=500,
        help="Number of DUTS-TR samples reserved for validation.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Limit training samples for debugging.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Limit validation samples for debugging.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(
    log_path: Path,
    resume: bool,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                log_path,
                mode="a" if resume else "w",
                encoding="utf-8",
            ),
        ],
        force=True,
    )


def build_model(
    network_path: str,
) -> tuple[torch.nn.Module, object]:
    network_module = importlib.import_module(network_path)
    model = network_module.build_model()

    return model, network_module


def split_dataset(
    dataset: SODDataset,
    val_count: int,
    seed: int,
    max_train_samples: int | None,
    max_val_samples: int | None,
) -> tuple[Subset, Subset, list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        len(dataset),
        generator=generator,
    ).tolist()

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    if max_train_samples is not None:
        train_indices = train_indices[:max_train_samples]

    if max_val_samples is not None:
        val_indices = val_indices[:max_val_samples]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return (
        train_dataset,
        val_dataset,
        train_indices,
        val_indices,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_metric: float,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "network": args.network,
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "scaler": scaler.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    network_path: str,
) -> tuple[int, int, float]:
    checkpoint = torch.load(
        path,
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
    optimizer.load_state_dict(checkpoint["optimizer"])

    if checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    best_metric = checkpoint.get("best_metric")

    if best_metric is None:
        best_metric = float("inf")

    return (
        checkpoint["epoch"] + 1,
        checkpoint["global_step"],
        best_metric,
    )


def prepare_metrics_file(
    path: Path,
    resume: bool,
) -> None:
    if resume and path.exists():
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "global_step",
                "train_loss",
                "train_loss_main",
                "train_loss_aux",
                "val_mae",
                "learning_rate",
                "train_time_seconds",
                "val_time_seconds",
            ]
        )


def append_metrics(
    path: Path,
    epoch: int,
    global_step: int,
    train_statistics: dict[str, float],
    val_statistics: dict[str, float],
) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                epoch,
                global_step,
                train_statistics["loss"],
                train_statistics.get("loss_main", ""),
                train_statistics.get("loss_aux", ""),
                val_statistics["mae"],
                train_statistics["lr"],
                train_statistics["time_seconds"],
                val_statistics["time_seconds"],
            ]
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    source_dir = run_dir / "network_source"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_path=log_dir / "train.log",
        resume=args.resume is not None,
    )

    logger = logging.getLogger(__name__)

    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s", device)
    logger.info("AMP: %s", use_amp)
    logger.info("Network: %s", args.network)

    with (run_dir / "args.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            vars(args),
            file,
            indent=2,
            ensure_ascii=False,
        )

    model, network_module = build_model(args.network)

    network_source_path = Path(network_module.__file__)
    shutil.copy2(
        network_source_path,
        source_dir / network_source_path.name,
    )

    model = model.to(device)

    full_dataset = SODDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        image_size=(args.image_size, args.image_size),
    )

    (
        train_dataset,
        val_dataset,
        train_indices,
        val_indices,
    ) = split_dataset(
        dataset=full_dataset,
        val_count=args.val_count,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    with (run_dir / "data_split.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "train_indices": train_indices,
                "val_indices": val_indices,
            },
            file,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    criterion = SODLoss(
        aux_weight=args.aux_weight,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    start_epoch = 1
    global_step = 0
    best_metric = float("inf")

    if args.resume is not None:
        (
            start_epoch,
            global_step,
            best_metric,
        ) = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            network_path=args.network,
        )

        logger.info(
            "Resumed from %s | Next epoch: %d | "
            "Step: %d | Best MAE: %.6f",
            args.resume,
            start_epoch,
            global_step,
            best_metric,
        )

    metrics_path = log_dir / "metrics.csv"

    prepare_metrics_file(
        path=metrics_path,
        resume=args.resume is not None,
    )

    logger.info("Training samples: %d", len(train_dataset))
    logger.info("Validation samples: %d", len(val_dataset))
    logger.info("Batches per epoch: %d", len(train_loader))
    logger.info("Total epochs: %d", args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        train_statistics, global_step = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            use_amp=use_amp,
            log_interval=args.log_interval,
        )

        val_statistics = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            use_amp=use_amp,
        )

        append_metrics(
            path=metrics_path,
            epoch=epoch,
            global_step=global_step,
            train_statistics=train_statistics,
            val_statistics=val_statistics,
        )

        logger.info(
            "Epoch %03d completed | "
            "Train loss %.6f | "
            "Val MAE %.6f | "
            "LR %.8f | "
            "Train %.1fs | Val %.1fs",
            epoch,
            train_statistics["loss"],
            val_statistics["mae"],
            train_statistics["lr"],
            train_statistics["time_seconds"],
            val_statistics["time_seconds"],
        )

        if val_statistics["mae"] < best_metric:
            best_metric = val_statistics["mae"]

            save_checkpoint(
                path=checkpoint_dir / "best.pth",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
            )

            logger.info(
                "New best checkpoint | Epoch %03d | MAE %.6f",
                epoch,
                best_metric,
            )

        save_checkpoint(
            path=checkpoint_dir / "latest.pth",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
        )

        if (
            epoch % args.save_every == 0
            or epoch == args.epochs
        ):
            save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch:04d}.pth",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
            )

    logger.info(
        "Training completed | Best validation MAE: %.6f",
        best_metric,
    )


if __name__ == "__main__":
    main()