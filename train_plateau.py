from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

import train as original_train


_ORIGINAL_PARSE_ARGS = original_train.parse_args

# Keep these names patchable so train_asymmetric_evidence_plateau.py can
# replace only the loss / metric functions exactly like the current
# train_asymmetric_evidence.py does.
SODLoss = original_train.SODLoss
prepare_metrics_file = original_train.prepare_metrics_file
append_metrics = original_train.append_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        add_help=False,
    )

    parser.add_argument(
        "--min-epochs",
        type=int,
        default=40,
        help=(
            "Do not early-stop before this epoch. "
            "--epochs remains the hard maximum epoch."
        ),
    )

    parser.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help=(
            "Number of plateau epochs tolerated before "
            "ReduceLROnPlateau lowers the learning rate."
        ),
    )

    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Multiplicative LR reduction factor on plateau.",
    )

    parser.add_argument(
        "--stop-patience",
        type=int,
        default=15,
        help=(
            "Stop after this many consecutive epochs without "
            "a meaningful train-loss improvement."
        ),
    )

    parser.add_argument(
        "--plateau-min-delta",
        type=float,
        default=1e-4,
        help=(
            "Minimum absolute train-loss decrease counted as "
            "an improvement."
        ),
    )

    custom_args, remaining = parser.parse_known_args()

    original_argv = sys.argv

    try:
        sys.argv = [
            original_argv[0],
            *remaining,
        ]
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv

    args.min_epochs = custom_args.min_epochs
    args.lr_patience = custom_args.lr_patience
    args.lr_factor = custom_args.lr_factor
    args.stop_patience = custom_args.stop_patience
    args.plateau_min_delta = custom_args.plateau_min_delta
    args.plateau_training = True

    return args


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_loss: float,
    bad_epochs: int,
) -> None:
    torch.save(
        {
            "format_version": 2,
            "network": args.network,
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "plateau_state": {
                "best_loss": best_loss,
                "bad_epochs": bad_epochs,
            },
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    network_path: str,
) -> tuple[int, int, float, int]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint["network"]
        != network_path
    ):
        raise RuntimeError(
            "Checkpoint network does not match:\n"
            f'checkpoint: {checkpoint["network"]}\n'
            f"command: {network_path}"
        )

    checkpoint_args = checkpoint.get(
        "args",
        {},
    )

    if not checkpoint_args.get(
        "plateau_training",
        False,
    ):
        raise RuntimeError(
            "This checkpoint was not created by the "
            "plateau training protocol. Resume from a "
            "plateau checkpoint instead."
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

    plateau_state = checkpoint.get(
        "plateau_state",
        {},
    )

    return (
        checkpoint["epoch"] + 1,
        checkpoint["global_step"],
        float(
            plateau_state.get(
                "best_loss",
                float("inf"),
            )
        ),
        int(
            plateau_state.get(
                "bad_epochs",
                0,
            )
        ),
    )


def main() -> None:
    args = parse_args()

    if args.min_epochs > args.epochs:
        raise ValueError(
            "--min-epochs cannot be larger than --epochs."
        )

    if args.lr_patience < 0:
        raise ValueError(
            "--lr-patience must be >= 0."
        )

    if args.stop_patience <= 0:
        raise ValueError(
            "--stop-patience must be > 0."
        )

    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError(
            "--lr-factor must be between 0 and 1."
        )

    if args.plateau_min_delta < 0.0:
        raise ValueError(
            "--plateau-min-delta must be >= 0."
        )

    original_train.set_seed(
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

    original_train.setup_logging(
        log_path=(
            log_dir
            / "train.log"
        ),
        resume=(
            args.resume
            is not None
        ),
    )

    logger = logging.getLogger(
        __name__
    )

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
        "8-way augmentation: %s",
        args.augment_8way,
    )

    logger.info(
        "LR schedule: ReduceLROnPlateau(train loss) | "
        "Initial LR: %.8f | "
        "Minimum LR: %.8f | "
        "Factor: %.3f | "
        "LR patience: %d",
        args.lr,
        args.min_lr,
        args.lr_factor,
        args.lr_patience,
    )

    logger.info(
        "Plateau stop | "
        "Min epochs: %d | "
        "Stop patience: %d | "
        "Min delta: %.8f | "
        "Max epochs: %d",
        args.min_epochs,
        args.stop_patience,
        args.plateau_min_delta,
        args.epochs,
    )

    logger.info(
        "Loss weights | "
        "Aux: %.3f | "
        "Region: %.3f | "
        "Edge: %.3f",
        args.aux_weight,
        args.region_weight,
        args.edge_weight,
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

    (
        model,
        network_module,
    ) = original_train.build_model(
        args.network
    )

    model_input_keys = (
        original_train
        .get_model_input_keys(
            model
        )
    )

    nam_hierarchies = (
        original_train
        .get_model_nam_hierarchies(
            model
        )
    )

    model_mean_hierarchies = (
        original_train
        .get_model_mean_hierarchies(
            model
        )
    )

    mean_hierarchy_set = set(
        model_mean_hierarchies
    )

    region_target_key = None

    if args.region_weight > 0.0:
        region_target_key = (
            f"mean_"
            f"{args.region_hierarchy}"
        )

        mean_hierarchy_set.add(
            args.region_hierarchy
        )

    mean_hierarchies = tuple(
        sorted(
            mean_hierarchy_set
        )
    )

    train_nam_dir = (
        args.train_nam
        if original_train.model_uses_nam(
            model
        )
        else None
    )

    mean_required = (
        original_train.model_uses_mean(
            model
        )
        or region_target_key
        is not None
    )

    train_mean_dir = (
        args.train_mean
        if mean_required
        else None
    )

    if (
        mean_required
        and train_mean_dir is None
    ):
        raise ValueError(
            "--train-mean is required when "
            "the model uses mean maps or "
            "region loss is enabled."
        )

    logger.info(
        "Model inputs: %s",
        ", ".join(
            model_input_keys
        ),
    )

    if train_nam_dir is not None:
        logger.info(
            "NAM directory: %s",
            train_nam_dir,
        )

    if nam_hierarchies:
        logger.info(
            "NAM hierarchies: %s",
            ", ".join(
                str(hierarchy)
                for hierarchy
                in nam_hierarchies
            ),
        )

    if train_mean_dir is not None:
        logger.info(
            "Region-mean directory: %s",
            train_mean_dir,
        )

    if mean_hierarchies:
        logger.info(
            "Region-mean hierarchies: %s",
            ", ".join(
                str(hierarchy)
                for hierarchy
                in mean_hierarchies
            ),
        )

    if region_target_key is not None:
        logger.info(
            "Region loss target: %s",
            region_target_key,
        )

    network_source_path = Path(
        network_module.__file__
    )

    shutil.copy2(
        network_source_path,
        source_dir
        / network_source_path.name,
    )

    # Save the exact plateau trainer used by this run as well.
    trainer_source_path = Path(
        __file__
    )

    shutil.copy2(
        trainer_source_path,
        source_dir
        / trainer_source_path.name,
    )

    model = model.to(
        device
    )

    train_dataset = original_train.SODDataset(
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

    if (
        args.max_train_samples
        is not None
    ):
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
        pin_memory=(
            device.type
            == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
    )

    criterion = SODLoss(
        aux_weight=(
            args.aux_weight
        ),
        edge_weight=(
            args.edge_weight
        ),
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

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        threshold=args.plateau_min_delta,
        threshold_mode="abs",
        min_lr=args.min_lr,
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )
    )

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    bad_epochs = 0

    if args.resume is not None:
        (
            start_epoch,
            global_step,
            best_loss,
            bad_epochs,
        ) = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            network_path=(
                args.network
            ),
        )

        logger.info(
            "Resumed from %s | "
            "Next epoch: %d | "
            "Step: %d | "
            "LR: %.8f | "
            "Best loss: %.6f | "
            "Bad epochs: %d",
            args.resume,
            start_epoch,
            global_step,
            optimizer
            .param_groups[0][
                "lr"
            ],
            best_loss,
            bad_epochs,
        )

    metrics_path = (
        log_dir
        / "metrics.csv"
    )

    prepare_metrics_file(
        path=metrics_path,
        resume=(
            args.resume
            is not None
        ),
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
        "Maximum epochs: %d",
        args.epochs,
    )

    final_epoch = start_epoch - 1
    stopped_early = False

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        (
            train_statistics,
            global_step,
        ) = original_train.train_one_epoch(
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

        current_loss = float(
            train_statistics["loss"]
        )

        improved = (
            current_loss
            < (
                best_loss
                - args.plateau_min_delta
            )
        )

        if improved:
            best_loss = current_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        append_metrics(
            path=metrics_path,
            epoch=epoch,
            global_step=global_step,
            train_statistics=(
                train_statistics
            ),
        )

        logger.info(
            "Epoch %03d completed | "
            "Train loss %.6f | "
            "Main %.6f | "
            "Aux %.6f | "
            "Region %.6f | "
            "Edge %.6f | "
            "LR %.8f | "
            "Best %.6f | "
            "Bad %d/%d | "
            "Train %.1fs",
            epoch,
            current_loss,
            train_statistics.get(
                "loss_main",
                0.0,
            ),
            train_statistics.get(
                "loss_aux",
                0.0,
            ),
            train_statistics.get(
                "loss_region",
                0.0,
            ),
            train_statistics.get(
                "loss_edge",
                0.0,
            ),
            train_statistics[
                "lr"
            ],
            best_loss,
            bad_epochs,
            args.stop_patience,
            train_statistics[
                "time_seconds"
            ],
        )

        old_lr = optimizer.param_groups[0][
            "lr"
        ]

        scheduler.step(
            current_loss
        )

        new_lr = optimizer.param_groups[0][
            "lr"
        ]

        if new_lr < old_lr:
            logger.info(
                "Plateau LR reduction | "
                "%.8f -> %.8f",
                old_lr,
                new_lr,
            )

        save_checkpoint(
            path=(
                checkpoint_dir
                / "latest.pth"
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            epoch=epoch,
            global_step=global_step,
            best_loss=best_loss,
            bad_epochs=bad_epochs,
        )

        if improved:
            save_checkpoint(
                path=(
                    checkpoint_dir
                    / "best_train_loss.pth"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                bad_epochs=bad_epochs,
            )

        if (
            epoch
            % args.save_every
            == 0
        ):
            save_checkpoint(
                path=(
                    checkpoint_dir
                    / (
                        f"epoch_"
                        f"{epoch:04d}.pth"
                    )
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                bad_epochs=bad_epochs,
            )

        final_epoch = epoch

        should_stop = (
            epoch >= args.min_epochs
            and bad_epochs
            >= args.stop_patience
        )

        if should_stop:
            stopped_early = True

            logger.info(
                "Training plateau reached | "
                "Epoch: %d | "
                "Best train loss: %.6f | "
                "No meaningful improvement "
                "for %d epochs",
                epoch,
                best_loss,
                bad_epochs,
            )

            save_checkpoint(
                path=(
                    checkpoint_dir
                    / "final.pth"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                bad_epochs=bad_epochs,
            )

            break

        if epoch == args.epochs:
            save_checkpoint(
                path=(
                    checkpoint_dir
                    / "final.pth"
                ),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                bad_epochs=bad_epochs,
            )

    logger.info(
        "Training completed | "
        "Final epoch: %d | "
        "Stopped on plateau: %s | "
        "Best train loss: %.6f | "
        "Final checkpoint: %s",
        final_epoch,
        stopped_early,
        best_loss,
        checkpoint_dir
        / "final.pth",
    )


if __name__ == "__main__":
    main()
