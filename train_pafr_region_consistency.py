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

from data.scribble_region_dataset import ScribbleRegionSODDataset
from engine.model_inputs import (
    get_model_input_keys,
    get_model_mean_hierarchies,
    model_uses_mean,
)
from engine.pafr_region_trainer import train_pafr_region_one_epoch
from losses.pafr_region_consistency_loss import PAFRRegionConsistencyLoss
from train import (
    build_model,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PAFR with soft NAMLab region-consistency regularization.",
    )
    parser.add_argument("--network", required=True)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-scribble-gt", required=True)
    parser.add_argument("--train-scribble-mask", required=True)
    parser.add_argument("--train-region-partitions", required=True)
    parser.add_argument("--train-mean", default=None)
    parser.add_argument("--region-consistency-weight", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--crf-weight", type=float, default=0.3)
    parser.add_argument("--crf-radius", type=int, default=5)
    parser.add_argument("--crf-xy-sigma", type=float, default=6.0)
    parser.add_argument("--crf-rgb-sigma", type=float, default=0.1)
    parser.add_argument("--src-alpha", type=float, default=0.85)
    parser.add_argument("--src-scale", type=float, default=0.3)
    parser.add_argument("--crf-scale", type=float, default=0.25)
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
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    return parser.parse_args()


def prepare_metrics_file(path: Path, resume: bool) -> None:
    if resume and path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "epoch",
                "global_step",
                "train_loss",
                "train_loss_main",
                "train_loss_pce",
                "train_loss_crf",
                "train_loss_src",
                "train_loss_region_consistency",
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
    stats: dict[str, float],
) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                epoch,
                global_step,
                stats["loss"],
                stats.get("loss_main", ""),
                stats.get("loss_pce", ""),
                stats.get("loss_crf", ""),
                stats.get("loss_src", ""),
                stats.get("loss_region_consistency", ""),
                stats.get("stat_valid_ratio", ""),
                stats.get("stat_fg_ratio_on_valid", ""),
                stats["lr"],
                stats["time_seconds"],
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
    for directory in (checkpoint_dir, log_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_path=log_dir / "train.log",
        resume=args.resume is not None,
    )
    logger = logging.getLogger(__name__)

    logger.info("Run directory: %s", run_dir)
    logger.info("Device: %s | AMP: %s", device, use_amp)
    logger.info("Network: %s", args.network)
    logger.info(
        "PAFR region consistency | weight %.4f | partitions %s",
        args.region_consistency_weight,
        args.train_region_partitions,
    )
    logger.info(
        "PAFR loss | CRF weight %.3f | radius %d | xy %.3f | rgb %.3f | "
        "SRC alpha %.3f | SRC scale %.3f | CRF scale %.3f",
        args.crf_weight,
        args.crf_radius,
        args.crf_xy_sigma,
        args.crf_rgb_sigma,
        args.src_alpha,
        args.src_scale,
        args.crf_scale,
    )

    with (run_dir / "args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)

    model, network_module = build_model(args.network)
    network_source_path = Path(network_module.__file__)

    model_input_keys = get_model_input_keys(model)
    mean_hierarchies = get_model_mean_hierarchies(model)
    train_mean_dir = args.train_mean if model_uses_mean(model) else None
    if model_uses_mean(model) and train_mean_dir is None:
        raise ValueError(
            "--train-mean is required when the model uses region-mean maps."
        )

    logger.info("Model inputs: %s", ", ".join(model_input_keys))

    source_paths = [
        network_source_path,
        Path(__file__),
        Path(__file__).parent / "data" / "scribble_region_dataset.py",
        Path(__file__).parent / "engine" / "pafr_region_trainer.py",
        Path(__file__).parent / "losses" / "pafr_region_consistency_loss.py",
        Path(__file__).parent / "losses" / "pafr_loss.py",
    ]
    for source_path in source_paths:
        shutil.copy2(source_path, source_dir / source_path.name)

    model = model.to(device)

    train_dataset = ScribbleRegionSODDataset(
        image_dir=args.train_images,
        scribble_gt_dir=args.train_scribble_gt,
        scribble_mask_dir=args.train_scribble_mask,
        region_partition_dir=args.train_region_partitions,
        mean_dir=train_mean_dir,
        mean_hierarchies=mean_hierarchies,
        image_size=(args.image_size, args.image_size),
        augment_8way=args.augment_8way,
    )
    if args.max_train_samples is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(args.max_train_samples, len(train_dataset))),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    criterion = PAFRRegionConsistencyLoss(
        region_consistency_weight=args.region_consistency_weight,
        crf_weight=args.crf_weight,
        crf_radius=args.crf_radius,
        crf_xy_sigma=args.crf_xy_sigma,
        crf_rgb_sigma=args.crf_rgb_sigma,
        src_alpha=args.src_alpha,
        crf_scale=args.crf_scale,
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
            "Resumed from %s | Next epoch %d | Step %d",
            args.resume,
            start_epoch,
            global_step,
        )

    metrics_path = log_dir / "metrics.csv"
    prepare_metrics_file(metrics_path, resume=args.resume is not None)

    logger.info(
        "Training samples: %d | Batches/epoch: %d",
        len(train_dataset),
        len(train_loader),
    )

    for epoch in range(start_epoch, args.epochs + 1):
        stats, global_step = train_pafr_region_one_epoch(
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
            src_scale=args.src_scale,
        )

        append_metrics(metrics_path, epoch, global_step, stats)
        logger.info(
            "Epoch %03d completed | Loss %.6f | Main %.6f | PCE %.6f | "
            "CRF %.6f | SRC %.6f | Region %.6f | Valid %.4f | "
            "FG(valid) %.4f | LR %.8f | %.1fs",
            epoch,
            stats["loss"],
            stats.get("loss_main", 0.0),
            stats.get("loss_pce", 0.0),
            stats.get("loss_crf", 0.0),
            stats.get("loss_src", 0.0),
            stats.get("loss_region_consistency", 0.0),
            stats.get("stat_valid_ratio", 0.0),
            stats.get("stat_fg_ratio_on_valid", 0.0),
            stats["lr"],
            stats["time_seconds"],
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
