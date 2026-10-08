from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from data.scribble_dataset import ScribbleSODDataset
from engine.model_inputs import (
    get_model_input_keys,
    get_model_mean_hierarchies,
    get_model_nam_hierarchies,
    model_uses_mean,
    model_uses_nam,
)
from engine.weak_trainer import train_weak_one_epoch
from losses.scribble_loss import ScribbleSODLoss
from train import (
    build_model,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an SOD network with scribble supervision.",
    )

    parser.add_argument(
        "--network",
        default="models.networks.vmamba_small_fpn_sod",
    )
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-scribble-gt", required=True)
    parser.add_argument("--train-scribble-mask", required=True)

    parser.add_argument(
        "--train-nam",
        default=None,
    )
    parser.add_argument(
        "--train-mean",
        default=None,
    )

    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.4)

    parser.add_argument(
        "--augment-8way",
        action=argparse.BooleanOptionalAction,
        default=False,
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
        default="runs/vmamba_small_fpn_weak_eorssd",
    )
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)

    return parser.parse_args()


def prepare_metrics_file(path: Path, resume: bool) -> None:
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
                "scribble_valid_ratio",
                "scribble_fg_ratio_on_valid",
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
                train_statistics.get("loss_main", ""),
                train_statistics.get("loss_aux", ""),
                train_statistics.get("stat_valid_ratio", ""),
                train_statistics.get("stat_fg_ratio_on_valid", ""),
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
    logger.info("Supervision: scribble partial BCE")
    logger.info("8-way augmentation: %s", args.augment_8way)
    logger.info(
        "LR schedule: cosine | Initial LR: %.8f | Minimum LR: %.8f",
        args.lr,
        args.min_lr,
    )
    logger.info("Aux weight: %.3f", args.aux_weight)

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

    model_input_keys = get_model_input_keys(model)
    nam_hierarchies = get_model_nam_hierarchies(model)
    mean_hierarchies = get_model_mean_hierarchies(model)

    train_nam_dir = (
        args.train_nam
        if model_uses_nam(model)
        else None
    )
    train_mean_dir = (
        args.train_mean
        if model_uses_mean(model)
        else None
    )

    if model_uses_nam(model) and train_nam_dir is None:
        raise ValueError(
            "--train-nam is required because the selected model uses NAM inputs."
        )

    if model_uses_mean(model) and train_mean_dir is None:
        raise ValueError(
            "--train-mean is required because the selected model uses region-mean inputs."
        )

    logger.info(
        "Model inputs: %s",
        ", ".join(model_input_keys),
    )
    logger.info("Scribble GT: %s", args.train_scribble_gt)
    logger.info("Scribble mask: %s", args.train_scribble_mask)

    if train_nam_dir is not None:
        logger.info("NAM directory: %s", train_nam_dir)
    if train_mean_dir is not None:
        logger.info("Region-mean directory: %s", train_mean_dir)

    network_source_path = Path(network_module.__file__)
    shutil.copy2(
        network_source_path,
        source_dir / network_source_path.name,
    )

    for source_path in (
        Path(__file__),
        Path(__file__).parent / "data" / "scribble_dataset.py",
        Path(__file__).parent / "losses" / "scribble_loss.py",
        Path(__file__).parent / "engine" / "weak_trainer.py",
    ):
        shutil.copy2(
            source_path,
            source_dir / source_path.name,
        )

    model = model.to(device)

    train_dataset = ScribbleSODDataset(
        image_dir=args.train_images,
        scribble_gt_dir=args.train_scribble_gt,
        scribble_mask_dir=args.train_scribble_mask,
        nam_dir=train_nam_dir,
        nam_hierarchies=nam_hierarchies,
        mean_dir=train_mean_dir,
        mean_hierarchies=mean_hierarchies,
        image_size=(args.image_size, args.image_size),
        augment_8way=args.augment_8way,
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
        persistent_workers=args.num_workers > 0,
    )

    criterion = ScribbleSODLoss(
        aux_weight=args.aux_weight,
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
            "Resumed from %s | Next epoch: %d | Step: %d | LR: %.8f",
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

    logger.info("Training samples: %d", len(train_dataset))
    logger.info("Batches per epoch: %d", len(train_loader))
    logger.info("Total epochs: %d", args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        train_statistics, global_step = train_weak_one_epoch(
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
            "Epoch %03d completed | Loss %.6f | Main %.6f | Aux %.6f | "
            "Valid %.4f | FG(valid) %.4f | LR %.8f | Train %.1fs",
            epoch,
            train_statistics["loss"],
            train_statistics.get("loss_main", 0.0),
            train_statistics.get("loss_aux", 0.0),
            train_statistics.get("stat_valid_ratio", 0.0),
            train_statistics.get("stat_fg_ratio_on_valid", 0.0),
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
                path=checkpoint_dir / f"epoch_{epoch:04d}.pth",
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
        "Training completed | Final checkpoint: %s",
        checkpoint_dir / "final.pth",
    )


if __name__ == "__main__":
    main()
