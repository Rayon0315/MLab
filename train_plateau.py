from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

import train as original_train


_ORIGINAL_PARSE_ARGS = original_train.parse_args

# Keep these names patchable so train_asymmetric_evidence_plateau_v2.py can
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
            "Training may reduce LR before this point, "
            "but it will not stop before this epoch."
        ),
    )

    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=2,
        help=(
            "Number of consecutive epochs whose relative loss "
            "improvement is below the threshold before reducing LR. "
            "At min LR, the same condition stops training."
        ),
    )

    parser.add_argument(
        "--relative-threshold",
        type=float,
        default=0.001,
        help=(
            "Minimum relative epoch-to-epoch train-loss improvement. "
            "0.001 means 0.1 percent."
        ),
    )

    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Multiplicative LR reduction factor on plateau.",
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
    args.plateau_patience = custom_args.plateau_patience
    args.relative_threshold = custom_args.relative_threshold
    args.lr_factor = custom_args.lr_factor
    args.plateau_training = True
    args.plateau_protocol = "relative_epoch_loss_v2"

    return args


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_loss: float,
    previous_loss: float | None,
    plateau_epochs: int,
) -> None:
    torch.save(
        {
            "format_version": 2,
            "network": args.network,
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "plateau_state": {
                "best_loss": best_loss,
                "previous_loss": previous_loss,
                "plateau_epochs": plateau_epochs,
            },
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    network_path: str,
) -> tuple[int, int, float, float | None, int]:
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

    checkpoint_args = checkpoint.get("args", {})

    if checkpoint_args.get("plateau_protocol") != "relative_epoch_loss_v2":
        raise RuntimeError(
            "This checkpoint was not created by the current "
            "relative-loss plateau protocol. Resume from a checkpoint "
            "produced by train_plateau_v2.py."
        )

    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])

    plateau_state = checkpoint.get("plateau_state", {})
    previous_loss = plateau_state.get("previous_loss")
    if previous_loss is not None:
        previous_loss = float(previous_loss)

    return (
        checkpoint["epoch"] + 1,
        checkpoint["global_step"],
        float(plateau_state.get("best_loss", float("inf"))),
        previous_loss,
        int(plateau_state.get("plateau_epochs", 0)),
    )


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def reduce_learning_rate(
    optimizer: torch.optim.Optimizer,
    factor: float,
    min_lr: float,
) -> tuple[float, float]:
    old_lr = get_current_lr(optimizer)
    for group in optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * factor, min_lr)
    return old_lr, get_current_lr(optimizer)


def main() -> None:
    args = parse_args()

    if args.min_epochs > args.epochs:
        raise ValueError(
            "--min-epochs cannot be larger than --epochs."
        )

    if args.plateau_patience <= 0:
        raise ValueError(
            "--plateau-patience must be > 0."
        )

    if args.relative_threshold < 0.0:
        raise ValueError(
            "--relative-threshold must be >= 0."
        )

    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError(
            "--lr-factor must be between 0 and 1."
        )

    if args.min_lr <= 0.0:
        raise ValueError(
            "--min-lr must be > 0."
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
        "LR schedule: relative-loss plateau | "
        "Initial LR: %.8f | "
        "Minimum LR: %.8f | "
        "Factor: %.3f",
        args.lr,
        args.min_lr,
        args.lr_factor,
    )

    logger.info(
        "Plateau rule | "
        "Relative threshold: %.6f (%.4f%%) | "
        "Patience: %d | "
        "Min epochs: %d | "
        "Max epochs: %d",
        args.relative_threshold,
        args.relative_threshold * 100.0,
        args.plateau_patience,
        args.min_epochs,
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

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=use_amp,
        )
    )

    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    previous_loss = None
    plateau_epochs = 0

    if args.resume is not None:
        (
            start_epoch,
            global_step,
            best_loss,
            previous_loss,
            plateau_epochs,
        ) = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            network_path=args.network,
        )

        logger.info(
            "Resumed from %s | "
            "Next epoch: %d | "
            "Step: %d | "
            "LR: %.8f | "
            "Best loss: %.6f | "
            "Previous loss: %s | "
            "Plateau: %d/%d",
            args.resume,
            start_epoch,
            global_step,
            get_current_lr(optimizer),
            best_loss,
            (
                f"{previous_loss:.6f}"
                if previous_loss is not None
                else "None"
            ),
            plateau_epochs,
            args.plateau_patience,
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

        absolute_best_improved = current_loss < best_loss
        if absolute_best_improved:
            best_loss = current_loss

        relative_improvement = None
        if previous_loss is not None:
            denominator = max(abs(previous_loss), 1e-12)
            relative_improvement = (previous_loss - current_loss) / denominator

            if relative_improvement < args.relative_threshold:
                plateau_epochs += 1
            else:
                plateau_epochs = 0

        plateau_before_action = plateau_epochs
        current_lr = get_current_lr(optimizer)
        at_min_lr = current_lr <= args.min_lr * (1.0 + 1e-12)
        plateau_triggered = (
            previous_loss is not None
            and plateau_epochs >= args.plateau_patience
        )

        action = "KEEP"
        should_stop = False

        if plateau_triggered:
            if not at_min_lr:
                _, new_lr = reduce_learning_rate(
                    optimizer=optimizer,
                    factor=args.lr_factor,
                    min_lr=args.min_lr,
                )
                action = f"LR_DOWN->{new_lr:.8f}"
                plateau_epochs = 0
            elif epoch >= args.min_epochs:
                action = "STOP"
                should_stop = True
            else:
                action = "MIN_LR_HOLD"
                plateau_epochs = min(
                    plateau_epochs,
                    args.plateau_patience,
                )

        next_lr = get_current_lr(optimizer)

        append_metrics(
            path=metrics_path,
            epoch=epoch,
            global_step=global_step,
            train_statistics=train_statistics,
        )

        relative_text = (
            "N/A"
            if relative_improvement is None
            else f"{relative_improvement * 100.0:+.4f}%"
        )

        logger.info(
            "Epoch %03d completed | "
            "Train loss %.6f | "
            "Main %.6f | "
            "Aux %.6f | "
            "Region %.6f | "
            "Edge %.6f | "
            "DeltaRel %s | "
            "Plateau %d/%d | "
            "LR used %.8f | "
            "Next LR %.8f | "
            "Action %s | "
            "Train %.1fs",
            epoch,
            current_loss,
            train_statistics.get("loss_main", 0.0),
            train_statistics.get("loss_aux", 0.0),
            train_statistics.get("loss_region", 0.0),
            train_statistics.get("loss_edge", 0.0),
            relative_text,
            plateau_before_action,
            args.plateau_patience,
            train_statistics["lr"],
            next_lr,
            action,
            train_statistics["time_seconds"],
        )

        previous_loss = current_loss
        final_epoch = epoch

        save_checkpoint(
            path=checkpoint_dir / "latest.pth",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            epoch=epoch,
            global_step=global_step,
            best_loss=best_loss,
            previous_loss=previous_loss,
            plateau_epochs=plateau_epochs,
        )

        if absolute_best_improved:
            save_checkpoint(
                path=checkpoint_dir / "best_train_loss.pth",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                previous_loss=previous_loss,
                plateau_epochs=plateau_epochs,
            )

        if epoch % args.save_every == 0:
            save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch:04d}.pth",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                previous_loss=previous_loss,
                plateau_epochs=plateau_epochs,
            )

        if should_stop:
            stopped_early = True

            logger.info(
                "Minimum-LR plateau reached | "
                "Epoch: %d | "
                "LR: %.8f | "
                "Loss: %.6f | "
                "Relative improvement: %s",
                epoch,
                next_lr,
                current_loss,
                relative_text,
            )

            save_checkpoint(
                path=checkpoint_dir / "final.pth",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                previous_loss=previous_loss,
                plateau_epochs=plateau_epochs,
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
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
                best_loss=best_loss,
                previous_loss=previous_loss,
                plateau_epochs=plateau_epochs,
            )

    logger.info(
        "Training completed | "
        "Final epoch: %d | "
        "Stopped on plateau: %s | "
        "Best train loss: %.6f | "
        "Final LR: %.8f | "
        "Final checkpoint: %s",
        final_epoch,
        stopped_early,
        best_loss,
        get_current_lr(optimizer),
        checkpoint_dir
        / "final.pth",
    )


if __name__ == "__main__":
    main()
