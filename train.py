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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from data.dataset import SODDataset
from engine.model_inputs import get_model_input_keys
from engine.trainer import train_one_epoch
from losses.sod_loss import SODLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SOD network.",
    )
    parser.add_argument(
        "--network",
        default="models.networks.resnet18_baseline",
    )
    parser.add_argument(
        "--train-images",
        default=(
            "datasets/DUTS/DUTS-TR/"
            "DUTS-TR-Image"
        ),
    )
    parser.add_argument(
        "--train-masks",
        default=(
            "datasets/DUTS/DUTS-TR/"
            "DUTS-TR-Mask"
        ),
    )
    parser.add_argument(
        "--train-nam",
        default=(
            "datasets/DUTS/DUTS-TR/"
            "nam"
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
        default=8,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--min-lr",
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
        default=0.2,
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
        "--run-dir",
        default="runs/resnet18_baseline",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--resume",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help=(
            "Use only the first N training "
            "samples for debugging."
        ),
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
    network_module = importlib.import_module(
        network_path
    )
    model = network_module.build_model()
    return model, network_module


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "network": args.network,
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    network_path: str,
) -> tuple[int, int]:
    checkpoint = torch.load(
        path,
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
    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )
    scheduler.load_state_dict(
        checkpoint["scheduler"]
    )
    scaler.load_state_dict(
        checkpoint["scaler"]
    )

    return (
        checkpoint["epoch"] + 1,
        checkpoint["global_step"],
    )


def prepare_metrics_file(
    path: Path,
    resume: bool,
) -> None:
    if resume and path.exists():
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "global_step",
                "train_loss",
                "train_loss_main",
                "train_loss_aux",
                "train_loss_edge",
                "learning_rate",
                "train_time_seconds",
            ]
        )


def append_metrics(
    path: Path,
    epoch: int,
    global_step: int,
    train_statistics: dict[str, float],
) -> None:
    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                epoch,
                global_step,
                train_statistics["loss"],
                train_statistics.get(
                    "loss_main",
                    "",
                ),
                train_statistics.get(
                    "loss_aux",
                    "",
                ),
                train_statistics.get(
                    "loss_edge",
                    "",
                ),
                train_statistics["lr"],
                train_statistics["time_seconds"],
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

    setup_logging(
        log_path=log_dir / "train.log",
        resume=args.resume is not None,
    )
    logger = logging.getLogger(__name__)

    logger.info(
        "Run directory: %s",
        run_dir,
    )
    logger.info(
        "Device: %s",
        device,
    )
    logger.info(
        "AMP: %s",
        use_amp,
    )
    logger.info(
        "Network: %s",
        args.network,
    )
    logger.info(
        "LR schedule: cosine | "
        "Initial LR: %.8f | "
        "Minimum LR: %.8f",
        args.lr,
        args.min_lr,
    )
    logger.info(
        "Loss weights | Aux: %.3f | Edge: %.3f",
        args.aux_weight,
        args.edge_weight,
    )

    with (
        run_dir / "args.json"
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

    model, network_module = build_model(
        args.network
    )
    model_input_keys = get_model_input_keys(model)
    train_nam_dir = (
        args.train_nam
        if "nam_20" in model_input_keys
        else None
    )

    logger.info(
        "Model inputs: %s",
        ", ".join(model_input_keys),
    )
    if train_nam_dir is not None:
        logger.info(
            "NAM directory: %s",
            train_nam_dir,
        )

    network_source_path = Path(
        network_module.__file__
    )
    shutil.copy2(
        network_source_path,
        source_dir / network_source_path.name,
    )

    model = model.to(device)

    train_dataset = SODDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        nam_dir=train_nam_dir,
        image_size=(
            args.image_size,
            args.image_size,
        ),
    )

    if args.max_train_samples is not None:
        train_dataset = Subset(
            train_dataset,
            range(
                min(
                    args.max_train_samples,
                    len(train_dataset),
                )
            ),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    criterion = SODLoss(
        aux_weight=args.aux_weight,
        edge_weight=args.edge_weight,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    start_epoch = 1
    global_step = 0

    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            network_path=args.network,
        )
        logger.info(
            "Resumed from %s | "
            "Next epoch: %d | "
            "Step: %d | "
            "LR: %.8f",
            args.resume,
            start_epoch,
            global_step,
            optimizer.param_groups[0]["lr"],
        )

    metrics_path = log_dir / "metrics.csv"
    prepare_metrics_file(
        path=metrics_path,
        resume=args.resume is not None,
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
        "Total epochs: %d",
        args.epochs,
    )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
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

        append_metrics(
            path=metrics_path,
            epoch=epoch,
            global_step=global_step,
            train_statistics=train_statistics,
        )

        logger.info(
            "Epoch %03d completed | "
            "Train loss %.6f | "
            "Main %.6f | "
            "Aux %.6f | "
            "Edge %.6f | "
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
            train_statistics.get(
                "loss_edge",
                0.0,
            ),
            train_statistics["lr"],
            train_statistics["time_seconds"],
        )

        scheduler.step()

        save_checkpoint(
            path=checkpoint_dir / "latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            epoch=epoch,
            global_step=global_step,
        )

        if epoch % args.save_every == 0:
            save_checkpoint(
                path=(
                    checkpoint_dir
                    / f"epoch_{epoch:04d}.pth"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
            )

        if epoch == args.epochs:
            save_checkpoint(
                path=checkpoint_dir / "final.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
            )

    logger.info(
        "Training completed | "
        "Final checkpoint: %s",
        checkpoint_dir / "final.pth",
    )


if __name__ == "__main__":
    main()
