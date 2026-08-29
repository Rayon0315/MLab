# train_refinement.py
from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from data.dataset import SODDataset
from engine.model_inputs import (
    get_model_input_keys,
    get_model_mean_hierarchies,
    prepare_model_inputs,
)
from losses.sod_loss import SODLoss


MAX_GRAD_NORM = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen Stage-1 + trainable Stage-2 refinement.",
    )

    parser.add_argument(
        "--network",
        default=(
            "models.networks."
            "mambavision_small_progressive_region_direct_"
            "hier60_region_hybrid_refinement_sod"
        ),
    )
    parser.add_argument(
        "--stage1-checkpoint",
        default=(
            "runs/"
            "mv_progressive_region_direct_hier60_region_hybrid_"
            "eorssd_aug8_e45/checkpoints/final.pth"
        ),
    )
    parser.add_argument(
        "--train-images",
        default="datasets/EORSSD/train-images",
    )
    parser.add_argument(
        "--train-masks",
        default="datasets/EORSSD/train-labels",
    )
    parser.add_argument(
        "--train-mean",
        default="datasets/EORSSD/train-mean",
    )
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--augment-8way",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
        default=(
            "runs/"
            "mv_progressive_region_direct_hier60_region_hybrid_"
            "refinement_stage2_eorssd_aug8_e15"
        ),
    )
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_path: Path, resume: bool) -> None:
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


def get_amp_dtype(
    device: torch.device,
    use_amp: bool,
) -> torch.dtype | None:
    if not use_amp:
        return None

    if (
        device.type == "cuda"
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16

    return torch.float16


def build_model(network_path: str) -> tuple[nn.Module, object]:
    network_module = importlib.import_module(network_path)
    model = network_module.build_model()

    if not hasattr(model, "refinement_stage"):
        raise AttributeError(
            "Refinement network must expose model.refinement_stage."
        )

    return model, network_module


def load_stage1_weights(
    model: nn.Module,
    checkpoint_path: str,
    logger: logging.Logger,
) -> None:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    incompatible = model.load_state_dict(
        checkpoint["model"],
        strict=False,
    )

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    invalid_missing = [
        key
        for key in missing
        if not key.startswith("refinement_stage.")
    ]

    if invalid_missing:
        raise RuntimeError(
            "Stage-1 checkpoint is missing non-refinement keys:\n"
            + "\n".join(invalid_missing)
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected Stage-1 checkpoint keys:\n"
            + "\n".join(unexpected)
        )

    logger.info(
        "Loaded Stage-1 checkpoint: %s",
        checkpoint_path,
    )
    logger.info(
        "Stage-1 source network: %s",
        checkpoint.get("network", "unknown"),
    )
    logger.info(
        "Stage-1 source epoch: %s",
        checkpoint.get("epoch", "unknown"),
    )
    logger.info(
        "Fresh refinement parameters: %d tensors",
        len(missing),
    )


def freeze_stage1(model: nn.Module) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for parameter in model.refinement_stage.parameters():
        parameter.requires_grad_(True)

    return [
        parameter
        for parameter in model.refinement_stage.parameters()
        if parameter.requires_grad
    ]


def set_refinement_train_mode(model: nn.Module) -> None:
    # Keep the pretrained Stage 1 exactly in inference mode.
    model.eval()

    # Only the new Stage 2 uses train mode.
    model.refinement_stage.train()


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable


def save_checkpoint(
    path: Path,
    model: nn.Module,
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
            "training_mode": "frozen_stage1_refinement_only",
            "network": args.network,
            "stage1_checkpoint": args.stage1_checkpoint,
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


def load_refinement_checkpoint(
    path: str,
    model: nn.Module,
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

    if checkpoint.get("network") != network_path:
        raise RuntimeError(
            "Checkpoint network does not match:\n"
            f'checkpoint: {checkpoint.get("network")}\n'
            f"command: {network_path}"
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])

    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("global_step", 0)),
    )


def prepare_metrics_file(path: Path, resume: bool) -> None:
    if resume and path.exists():
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        csv.writer(file).writerow(
            [
                "epoch",
                "global_step",
                "train_loss",
                "learning_rate",
                "gradient_norm",
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
        csv.writer(file).writerow(
            [
                epoch,
                global_step,
                statistics["loss"],
                statistics["lr"],
                statistics["grad_norm"],
                statistics["time_seconds"],
            ]
        )


def train_one_refinement_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    trainable_parameters: list[nn.Parameter],
    device: torch.device,
    epoch: int,
    global_step: int,
    use_amp: bool,
    log_interval: int,
) -> tuple[dict[str, float], int]:
    logger = logging.getLogger(__name__)

    set_refinement_train_mode(model)

    criterion = criterion.to(device)
    criterion.train()

    amp_dtype = get_amp_dtype(
        device=device,
        use_amp=use_amp,
    )

    use_grad_scaler = (
        amp_dtype == torch.float16
        and scaler.is_enabled()
    )

    logger.info(
        "Precision: %s | GradScaler: %s | Max grad norm: %.1f",
        (
            str(amp_dtype).removeprefix("torch.")
            if amp_dtype is not None
            else "float32"
        ),
        use_grad_scaler,
        MAX_GRAD_NORM,
    )

    total_samples = 0
    loss_sum = 0.0
    grad_sum = 0.0
    start_time = time.perf_counter()

    for batch_index, batch in enumerate(
        data_loader,
        start=1,
    ):
        model_inputs = prepare_model_inputs(
            model=model,
            batch=batch,
            device=device,
        )

        mask = batch["mask"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=(
                amp_dtype
                if amp_dtype is not None
                else torch.float32
            ),
            enabled=amp_dtype is not None,
        ):
            outputs = model(**model_inputs)

            # Refinement-only objective:
            # do not supervise the frozen coarse/aux outputs again.
            loss_dict = criterion(
                {"pred": outputs["pred"]},
                mask,
            )
            loss = loss_dict["loss"]

        if not torch.isfinite(loss).all():
            raise FloatingPointError(
                "Non-finite refinement loss | "
                f"epoch={epoch} | batch={batch_index} | "
                f"step={global_step + 1}"
            )

        if use_grad_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=MAX_GRAD_NORM,
            error_if_nonfinite=False,
        )

        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                "Non-finite refinement gradient norm | "
                f"epoch={epoch} | batch={batch_index} | "
                f"step={global_step + 1}"
            )

        if use_grad_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        batch_size = model_inputs["image"].shape[0]
        total_samples += batch_size
        global_step += 1

        loss_sum += (
            loss.detach().float().item()
            * batch_size
        )
        grad_sum += (
            gradient_norm.detach().float().item()
            * batch_size
        )

        if (
            batch_index % log_interval == 0
            or batch_index == len(data_loader)
        ):
            logger.info(
                "Epoch %03d | Batch %05d/%05d | "
                "Step %07d | Refine loss %.6f | Grad %.4f",
                epoch,
                batch_index,
                len(data_loader),
                global_step,
                loss.detach().float().item(),
                gradient_norm.detach().float().item(),
            )

    elapsed = time.perf_counter() - start_time

    return (
        {
            "loss": loss_sum / total_samples,
            "grad_norm": grad_sum / total_samples,
            "lr": optimizer.param_groups[0]["lr"],
            "time_seconds": elapsed,
        },
        global_step,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    use_amp = (
        args.amp
        and device.type == "cuda"
    )

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
    logger.info(
        "Training mode: frozen Stage 1 + refinement-only Stage 2"
    )
    logger.info(
        "Stage-1 checkpoint: %s",
        args.stage1_checkpoint,
    )
    logger.info("Stage-2 epochs: %d", args.epochs)

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

    model, network_module = build_model(
        args.network
    )

    if args.resume is None:
        load_stage1_weights(
            model=model,
            checkpoint_path=args.stage1_checkpoint,
            logger=logger,
        )

    trainable_parameters = freeze_stage1(model)

    total_params, trainable_params = parameter_counts(model)
    logger.info(
        "Parameters | Total %.3f M | Trainable refinement %.3f M",
        total_params / 1e6,
        trainable_params / 1e6,
    )
    logger.info(
        "Trainable ratio: %.3f%%",
        100.0 * trainable_params / total_params,
    )

    input_keys = get_model_input_keys(model)
    mean_hierarchies = get_model_mean_hierarchies(
        model
    )

    logger.info(
        "Model inputs: %s",
        ", ".join(input_keys),
    )
    logger.info(
        "Region-mean hierarchies: %s",
        ", ".join(
            str(h)
            for h in mean_hierarchies
        ),
    )

    network_source_path = Path(
        network_module.__file__
    )
    shutil.copy2(
        network_source_path,
        source_dir / network_source_path.name,
    )
    shutil.copy2(
        Path(__file__),
        source_dir / Path(__file__).name,
    )

    model = model.to(device)

    train_dataset = SODDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        nam_dir=None,
        nam_hierarchies=(),
        mean_dir=args.train_mean,
        mean_hierarchies=mean_hierarchies,
        image_size=(
            args.image_size,
            args.image_size,
        ),
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

    # Exactly the original main BCE + soft IoU.
    criterion = SODLoss(
        aux_weight=0.0,
        edge_weight=0.0,
        region_weight=0.0,
    )

    optimizer = AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    amp_dtype = get_amp_dtype(
        device=device,
        use_amp=use_amp,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            amp_dtype == torch.float16
        ),
    )

    start_epoch = 1
    global_step = 0

    if args.resume is not None:
        start_epoch, global_step = (
            load_refinement_checkpoint(
                path=args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                network_path=args.network,
            )
        )

        trainable_parameters = freeze_stage1(model)

        logger.info(
            "Resumed from %s | Next epoch %d | "
            "Step %d | LR %.8f",
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

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        statistics, global_step = (
            train_one_refinement_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                trainable_parameters=trainable_parameters,
                device=device,
                epoch=epoch,
                global_step=global_step,
                use_amp=use_amp,
                log_interval=args.log_interval,
            )
        )

        append_metrics(
            path=metrics_path,
            epoch=epoch,
            global_step=global_step,
            statistics=statistics,
        )

        logger.info(
            "Epoch %03d completed | Refine loss %.6f | "
            "Grad %.4f | LR %.8f | Train %.1fs",
            epoch,
            statistics["loss"],
            statistics["grad_norm"],
            statistics["lr"],
            statistics["time_seconds"],
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
        "Refinement training completed | Final checkpoint: %s",
        checkpoint_dir / "final.pth",
    )


if __name__ == "__main__":
    main()
