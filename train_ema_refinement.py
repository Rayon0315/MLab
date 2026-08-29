# train_ema_refinement.py
from __future__ import annotations

import argparse
import copy
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
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from data.dataset import SODDataset
from engine.model_inputs import (
    get_model_input_keys,
    get_model_mean_hierarchies,
    get_model_nam_hierarchies,
    model_uses_mean,
    model_uses_nam,
    prepare_model_inputs,
)
from losses.sod_loss import SODLoss


MAX_GRAD_NORM = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NESS-inspired EMA refinement training for a pretrained SOD model."
        ),
    )

    parser.add_argument(
        "--network",
        default=(
            "models.networks."
            "mambavision_small_progressive_region_direct_"
            "hier60_region_hybrid_sod"
        ),
    )

    parser.add_argument(
        "--stage1-checkpoint",
        default=(
            "runs/"
            "mv_progressive_region_direct_hier60_region_hybrid_"
            "eorssd_aug8_e45/checkpoints/final.pth"
        ),
        help=(
            "Stage-1 checkpoint used to initialize both Student and Teacher."
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
        "--train-nam",
        default=None,
    )

    parser.add_argument(
        "--train-mean",
        default="datasets/EORSSD/train-mean",
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
        default=15,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
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
        help=(
            "Keep the Stage-1 auxiliary supervision unchanged during refinement."
        ),
    )

    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--augment-8way",
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
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--run-dir",
        default=(
            "runs/"
            "mv_progressive_region_direct_hier60_region_hybrid_"
            "ema_refinement_eorssd_aug8_e15"
        ),
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Resume an EMA-refinement checkpoint produced by this script."
        ),
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
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


def build_model(
    network_path: str,
) -> tuple[nn.Module, object]:
    network_module = importlib.import_module(
        network_path
    )

    model = network_module.build_model()

    return model, network_module


def load_stage1_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    network_path: str,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_network = checkpoint.get(
        "network"
    )

    if (
        checkpoint_network is not None
        and checkpoint_network != network_path
    ):
        raise RuntimeError(
            "Stage-1 checkpoint network does not match:\n"
            f"checkpoint: {checkpoint_network}\n"
            f"command: {network_path}"
        )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    return checkpoint


def make_teacher(
    student: nn.Module,
) -> nn.Module:
    teacher = copy.deepcopy(
        student
    )

    teacher.requires_grad_(False)
    teacher.eval()

    return teacher


@torch.no_grad()
def update_ema_teacher(
    teacher: nn.Module,
    student: nn.Module,
    decay: float,
) -> None:
    teacher_state = (
        teacher.state_dict()
    )
    student_state = (
        student.state_dict()
    )

    for key, teacher_value in teacher_state.items():
        student_value = student_state[key].detach()

        if torch.is_floating_point(
            teacher_value
        ):
            teacher_value.mul_(
                decay
            ).add_(
                student_value,
                alpha=(1.0 - decay),
            )
        else:
            teacher_value.copy_(
                student_value
            )


def save_checkpoint(
    path: Path,
    teacher: nn.Module,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> None:
    # Important:
    # "model" is deliberately the EMA Teacher state so the existing
    # test.py can load this checkpoint directly without modification.
    torch.save(
        {
            "format_version": 1,
            "training_mode": (
                "ness_inspired_ema_refinement"
            ),
            "network": args.network,
            "stage1_checkpoint": (
                args.stage1_checkpoint
            ),
            "ema_decay": args.ema_decay,
            "epoch": epoch,
            "global_step": global_step,
            "model": teacher.state_dict(),
            "student_model": (
                student.state_dict()
            ),
            "optimizer": (
                optimizer.state_dict()
            ),
            "scheduler": (
                scheduler.state_dict()
            ),
            "scaler": scaler.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_refinement_checkpoint(
    path: str,
    teacher: nn.Module,
    student: nn.Module,
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

    if (
        checkpoint.get("network")
        != network_path
    ):
        raise RuntimeError(
            "Refinement checkpoint network does not match:\n"
            f'checkpoint: {checkpoint.get("network")}\n'
            f"command: {network_path}"
        )

    teacher.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    student.load_state_dict(
        checkpoint["student_model"],
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

    teacher.requires_grad_(False)
    teacher.eval()

    return (
        int(checkpoint["epoch"]) + 1,
        int(
            checkpoint.get(
                "global_step",
                0,
            )
        ),
    )


def prepare_metrics_file(
    path: Path,
    resume: bool,
) -> None:
    if (
        resume
        and path.exists()
    ):
        return

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
                statistics["lr"],
                statistics["grad_norm"],
                statistics["time_seconds"],
            ]
        )


def train_one_ema_epoch(
    student: nn.Module,
    teacher: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    global_step: int,
    use_amp: bool,
    ema_decay: float,
    log_interval: int,
) -> tuple[
    dict[str, float],
    int,
]:
    logger = logging.getLogger(
        __name__
    )

    student.train()
    teacher.eval()

    criterion = criterion.to(
        device
    )
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
        "Training precision: %s | "
        "GradScaler: %s | "
        "Max grad norm: %.1f | "
        "EMA decay: %.6f",
        (
            str(amp_dtype)
            .removeprefix("torch.")
            if amp_dtype is not None
            else "float32"
        ),
        use_grad_scaler,
        MAX_GRAD_NORM,
        ema_decay,
    )

    total_samples = 0

    loss_sums: dict[
        str,
        float,
    ] = {}

    gradient_norm_sum = 0.0

    start_time = (
        time.perf_counter()
    )

    for batch_index, batch in enumerate(
        data_loader,
        start=1,
    ):
        model_inputs = (
            prepare_model_inputs(
                model=student,
                batch=batch,
                device=device,
            )
        )

        mask = batch[
            "mask"
        ].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=(
                amp_dtype
                if amp_dtype is not None
                else torch.float32
            ),
            enabled=(
                amp_dtype is not None
            ),
        ):
            outputs = student(
                **model_inputs
            )

            # First EMA-refinement experiment:
            # keep the original dense-GT SOD objective unchanged.
            # No teacher consistency and no pseudo labels yet.
            loss_dict = criterion(
                outputs,
                mask,
            )

            loss = loss_dict[
                "loss"
            ]

        if not torch.isfinite(
            loss
        ).all():
            raise FloatingPointError(
                "Non-finite EMA-refinement loss | "
                f"epoch={epoch} | "
                f"batch={batch_index} | "
                f"step={global_step + 1}"
            )

        if use_grad_scaler:
            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )
        else:
            loss.backward()

        gradient_norm = (
            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                max_norm=MAX_GRAD_NORM,
                error_if_nonfinite=False,
            )
        )

        if not torch.isfinite(
            gradient_norm
        ):
            raise FloatingPointError(
                "Non-finite EMA-refinement gradient norm | "
                f"epoch={epoch} | "
                f"batch={batch_index} | "
                f"step={global_step + 1}"
            )

        if use_grad_scaler:
            scaler.step(
                optimizer
            )
            scaler.update()
        else:
            optimizer.step()

        update_ema_teacher(
            teacher=teacher,
            student=student,
            decay=ema_decay,
        )

        batch_size = (
            model_inputs["image"]
            .shape[0]
        )

        total_samples += (
            batch_size
        )

        global_step += 1

        for name, value in (
            loss_dict.items()
        ):
            loss_sums[name] = (
                loss_sums.get(
                    name,
                    0.0,
                )
                + value
                .detach()
                .float()
                .item()
                * batch_size
            )

        gradient_norm_sum += (
            gradient_norm
            .detach()
            .float()
            .item()
            * batch_size
        )

        if (
            batch_index
            % log_interval
            == 0
            or batch_index
            == len(data_loader)
        ):
            zero = (
                loss.detach()
                .new_zeros(())
            )

            logger.info(
                "Epoch %03d | "
                "Batch %05d/%05d | "
                "Step %07d | "
                "Loss %.6f | "
                "Main %.6f | "
                "Aux %.6f | "
                "Grad %.4f",
                epoch,
                batch_index,
                len(data_loader),
                global_step,
                loss.detach()
                .float()
                .item(),
                loss_dict.get(
                    "loss_main",
                    loss,
                ).detach()
                .float()
                .item(),
                loss_dict.get(
                    "loss_aux",
                    zero,
                ).detach()
                .float()
                .item(),
                gradient_norm.detach()
                .float()
                .item(),
            )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    statistics = {
        name: (
            value
            / total_samples
        )
        for name, value
        in loss_sums.items()
    }

    statistics[
        "grad_norm"
    ] = (
        gradient_norm_sum
        / total_samples
    )

    statistics[
        "lr"
    ] = (
        optimizer
        .param_groups[0]["lr"]
    )

    statistics[
        "time_seconds"
    ] = elapsed_time

    return (
        statistics,
        global_step,
    )


def main() -> None:
    args = parse_args()

    if not (
        0.0
        < args.ema_decay
        < 1.0
    ):
        raise ValueError(
            "--ema-decay must be between 0 and 1."
        )

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

    setup_logging(
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
        "Refinement mode: "
        "Stage-1 checkpoint -> Student fine-tuning + EMA Teacher"
    )

    logger.info(
        "Stage-1 checkpoint: %s",
        args.stage1_checkpoint,
    )

    logger.info(
        "Refinement epochs: %d",
        args.epochs,
    )

    logger.info(
        "EMA decay: %.6f",
        args.ema_decay,
    )

    logger.info(
        "LR schedule: cosine | "
        "Initial LR %.8f | "
        "Minimum LR %.8f",
        args.lr,
        args.min_lr,
    )

    logger.info(
        "Loss: original BCE + IoU | "
        "Aux weight %.3f | "
        "No pseudo labels | "
        "No teacher consistency",
        args.aux_weight,
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
        student,
        network_module,
    ) = build_model(
        args.network
    )

    stage1_checkpoint = (
        load_stage1_checkpoint(
            checkpoint_path=(
                args.stage1_checkpoint
            ),
            model=student,
            network_path=args.network,
        )
    )

    logger.info(
        "Loaded Stage-1 checkpoint | "
        "Epoch %s | "
        "Global step %s",
        stage1_checkpoint.get(
            "epoch",
            "unknown",
        ),
        stage1_checkpoint.get(
            "global_step",
            "unknown",
        ),
    )

    teacher = make_teacher(
        student
    )

    model_input_keys = (
        get_model_input_keys(
            student
        )
    )

    nam_hierarchies = (
        get_model_nam_hierarchies(
            student
        )
    )

    mean_hierarchies = (
        get_model_mean_hierarchies(
            student
        )
    )

    train_nam_dir = (
        args.train_nam
        if model_uses_nam(
            student
        )
        else None
    )

    train_mean_dir = (
        args.train_mean
        if model_uses_mean(
            student
        )
        else None
    )

    logger.info(
        "Model inputs: %s",
        ", ".join(
            model_input_keys
        ),
    )

    if nam_hierarchies:
        logger.info(
            "NAM hierarchies: %s",
            ", ".join(
                str(h)
                for h
                in nam_hierarchies
            ),
        )

    if mean_hierarchies:
        logger.info(
            "Region-mean hierarchies: %s",
            ", ".join(
                str(h)
                for h
                in mean_hierarchies
            ),
        )

    network_source_path = Path(
        network_module.__file__
    )

    shutil.copy2(
        network_source_path,
        source_dir
        / network_source_path.name,
    )

    shutil.copy2(
        Path(__file__),
        source_dir
        / Path(__file__).name,
    )

    student = student.to(
        device
    )

    teacher = teacher.to(
        device
    )

    teacher.requires_grad_(False)
    teacher.eval()

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
        edge_weight=0.0,
        region_weight=0.0,
    )

    optimizer = AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=(
            args.weight_decay
        ),
    )

    scheduler = (
        CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    )

    amp_dtype = get_amp_dtype(
        device=device,
        use_amp=use_amp,
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=(
                amp_dtype
                == torch.float16
            ),
        )
    )

    start_epoch = 1
    global_step = 0

    if args.resume is not None:
        (
            start_epoch,
            global_step,
        ) = (
            load_refinement_checkpoint(
                path=args.resume,
                teacher=teacher,
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                network_path=(
                    args.network
                ),
            )
        )

        logger.info(
            "Resumed EMA refinement from %s | "
            "Next epoch %d | "
            "Step %d | "
            "LR %.8f",
            args.resume,
            start_epoch,
            global_step,
            optimizer
            .param_groups[0]["lr"],
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

    total_parameters = sum(
        parameter.numel()
        for parameter
        in student.parameters()
    )

    logger.info(
        "Student parameters: %.3f M",
        total_parameters / 1e6,
    )

    logger.info(
        "Teacher parameters: %.3f M",
        total_parameters / 1e6,
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
        (
            statistics,
            global_step,
        ) = train_one_ema_epoch(
            student=student,
            teacher=teacher,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            use_amp=use_amp,
            ema_decay=(
                args.ema_decay
            ),
            log_interval=(
                args.log_interval
            ),
        )

        append_metrics(
            path=metrics_path,
            epoch=epoch,
            global_step=global_step,
            statistics=statistics,
        )

        logger.info(
            "Epoch %03d completed | "
            "Loss %.6f | "
            "Main %.6f | "
            "Aux %.6f | "
            "Grad %.4f | "
            "LR %.8f | "
            "Train %.1fs",
            epoch,
            statistics["loss"],
            statistics.get(
                "loss_main",
                0.0,
            ),
            statistics.get(
                "loss_aux",
                0.0,
            ),
            statistics["grad_norm"],
            statistics["lr"],
            statistics["time_seconds"],
        )

        scheduler.step()

        save_checkpoint(
            path=(
                checkpoint_dir
                / "latest.pth"
            ),
            teacher=teacher,
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            args=args,
            epoch=epoch,
            global_step=global_step,
        )

        if (
            epoch
            % args.save_every
            == 0
        ):
            save_checkpoint(
                path=(
                    checkpoint_dir
                    / f"epoch_{epoch:04d}.pth"
                ),
                teacher=teacher,
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
            )

        if epoch == args.epochs:
            save_checkpoint(
                path=(
                    checkpoint_dir
                    / "final.pth"
                ),
                teacher=teacher,
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
                epoch=epoch,
                global_step=global_step,
            )

    logger.info(
        "EMA refinement completed | "
        "Final Teacher checkpoint: %s",
        checkpoint_dir
        / "final.pth",
    )


if __name__ == "__main__":
    main()
