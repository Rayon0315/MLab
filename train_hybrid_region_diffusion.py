from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import SODDataset
from engine.diffusion_trainer import train_diffusion_one_epoch
from losses.diffusion_spatial_spectral_loss import (
    DiffusionSpatialSpectralLoss,
)
from models.networks.mambavision_small_progressive_region_direct_hier60_region_hybrid_diffusion_sod import (
    MambaVisionSmallHybridRegionDiffusionSOD,
)
from train import set_seed, setup_logging


NETWORK_PATH = (
    "models.networks."
    "mambavision_small_progressive_region_direct_hier60_"
    "region_hybrid_diffusion_sod"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train IPDiff-style diffusion on mv-region-hybrid.",
    )

    parser.add_argument("--prior-checkpoint", required=True)
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prior-lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-prior-epochs", type=int, default=5)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)

    parser.add_argument("--train-timesteps", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--prior-channels", type=int, default=64)
    parser.add_argument("--ipm-probability", type=float, default=0.2)

    parser.add_argument("--diffusion-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=0.2)
    parser.add_argument("--spectral-weight", type=float, default=0.05)
    parser.add_argument("--prior-weight", type=float, default=0.2)

    parser.add_argument(
        "--augment-8way",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--run-dir",
        default="runs/mv_h60_region_hybrid_diffusion_eorssd_aug8_e45",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume", default=None)

    return parser.parse_args()


def build_model(args: argparse.Namespace) -> MambaVisionSmallHybridRegionDiffusionSOD:
    return MambaVisionSmallHybridRegionDiffusionSOD(
        prior_channels=args.prior_channels,
        train_timesteps=args.train_timesteps,
        sample_steps=args.sample_steps,
        ipm_probability=args.ipm_probability,
    )


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
            "network": NETWORK_PATH,
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    run_dir = Path(args.run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        log_path=log_dir / "train.log",
        resume=args.resume is not None,
    )
    logger = logging.getLogger(__name__)

    with (run_dir / "args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)

    model = build_model(args)

    if args.resume is None:
        prior_checkpoint = model.load_prior_checkpoint(
            args.prior_checkpoint
        )
        logger.info(
            "Loaded Hybrid prior checkpoint | epoch=%s | network=%s",
            prior_checkpoint.get("epoch"),
            prior_checkpoint.get("network"),
        )

    model = model.to(device)

    dataset = SODDataset(
        image_dir=args.train_images,
        mask_dir=args.train_masks,
        mean_dir=args.train_mean,
        mean_hierarchies=(60,),
        image_size=(args.image_size, args.image_size),
        augment_8way=args.augment_8way,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    diffusion_parameters = [
        *model.prior_reconstruction.parameters(),
        *model.denoiser.parameters(),
    ]
    prior_parameters = list(model.prior_net.parameters())

    optimizer = AdamW(
        [
            {
                "params": diffusion_parameters,
                "lr": args.lr,
            },
            {
                "params": prior_parameters,
                "lr": args.prior_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp and amp_dtype == torch.float16,
    )

    criterion = DiffusionSpatialSpectralLoss(
        diffusion_weight=args.diffusion_weight,
        reconstruction_weight=args.reconstruction_weight,
        edge_weight=args.edge_weight,
        spectral_weight=args.spectral_weight,
        prior_weight=args.prior_weight,
    )

    start_epoch = 1
    global_step = 0

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        logger.info("Resumed from %s", args.resume)

    metrics_path = log_dir / "metrics.csv"
    if start_epoch == 1:
        with metrics_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "epoch",
                    "global_step",
                    "loss",
                    "loss_diffusion",
                    "loss_reconstruction",
                    "loss_edge",
                    "loss_spectral",
                    "loss_prior",
                    "lr_diffusion",
                    "lr_prior",
                    "prior_trainable",
                    "time_seconds",
                ]
            )

    logger.info("Samples: %d", len(dataset))
    logger.info(
        "Diffusion: train_timesteps=%d sample_steps=%d | freeze prior=%d epochs",
        args.train_timesteps,
        args.sample_steps,
        args.freeze_prior_epochs,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        prior_trainable = epoch > args.freeze_prior_epochs

        statistics, global_step = train_diffusion_one_epoch(
            model=model,
            data_loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            global_step=global_step,
            use_amp=use_amp,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
            prior_trainable=prior_trainable,
        )

        scheduler.step()

        with metrics_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    epoch,
                    global_step,
                    statistics["loss"],
                    statistics["loss_diffusion"],
                    statistics["loss_reconstruction"],
                    statistics["loss_edge"],
                    statistics["loss_spectral"],
                    statistics["loss_prior"],
                    statistics["lr_diffusion"],
                    statistics["lr_prior"],
                    int(prior_trainable),
                    statistics["time_seconds"],
                ]
            )

        logger.info(
            "Epoch %03d | Loss %.6f | Diff %.6f | Recon %.6f | Prior trainable=%s | %.1fs",
            epoch,
            statistics["loss"],
            statistics["loss_diffusion"],
            statistics["loss_reconstruction"],
            prior_trainable,
            statistics["time_seconds"],
        )

        save_checkpoint(
            checkpoint_dir / "latest.pth",
            model,
            optimizer,
            scheduler,
            scaler,
            args,
            epoch,
            global_step,
        )

        if epoch % args.save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch:04d}.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                epoch,
                global_step,
            )

    save_checkpoint(
        checkpoint_dir / "final.pth",
        model,
        optimizer,
        scheduler,
        scaler,
        args,
        args.epochs,
        global_step,
    )


if __name__ == "__main__":
    main()
