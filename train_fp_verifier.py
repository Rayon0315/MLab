from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import train as base_train

from losses.fp_verifier_loss import (
    FalsePositiveAwareSODLoss,
)


_ORIGINAL_PARSE_ARGS = base_train.parse_args

_FP_CONFIG = {
    "weight": 0.2,
    "gamma": 2.0,
    "easy_background_weight": 0.05,
}


def parse_args() -> argparse.Namespace:
    global _FP_CONFIG

    parser = argparse.ArgumentParser(
        add_help=False,
    )

    parser.add_argument(
        "--fp-risk-weight",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--fp-risk-gamma",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--fp-easy-background-weight",
        type=float,
        default=0.05,
    )

    custom_args, remaining = (
        parser.parse_known_args()
    )

    original_argv = sys.argv

    try:
        sys.argv = [
            original_argv[0],
            *remaining,
        ]

        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = original_argv

    args.fp_risk_weight = (
        custom_args.fp_risk_weight
    )
    args.fp_risk_gamma = (
        custom_args.fp_risk_gamma
    )
    args.fp_easy_background_weight = (
        custom_args.fp_easy_background_weight
    )

    _FP_CONFIG = {
        "weight": (
            custom_args.fp_risk_weight
        ),
        "gamma": (
            custom_args.fp_risk_gamma
        ),
        "easy_background_weight": (
            custom_args
            .fp_easy_background_weight
        ),
    }

    return args


def build_fp_loss(
    aux_weight: float = 0.4,
    edge_weight: float = 0.0,
    region_weight: float = 0.0,
) -> FalsePositiveAwareSODLoss:
    logger = logging.getLogger(
        __name__
    )

    logger.info(
        "FP-risk loss | "
        "Weight: %.3f | "
        "Gamma: %.3f | "
        "Easy-BG weight: %.3f",
        _FP_CONFIG["weight"],
        _FP_CONFIG["gamma"],
        _FP_CONFIG[
            "easy_background_weight"
        ],
    )

    return FalsePositiveAwareSODLoss(
        aux_weight=aux_weight,
        edge_weight=edge_weight,
        region_weight=region_weight,
        fp_risk_weight=(
            _FP_CONFIG["weight"]
        ),
        fp_risk_gamma=(
            _FP_CONFIG["gamma"]
        ),
        easy_background_weight=(
            _FP_CONFIG[
                "easy_background_weight"
            ]
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
                "train_loss_fp_risk",
                "train_loss_fp_positive",
                "train_loss_fp_negative",
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
    train_statistics: dict[
        str,
        float,
    ],
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
                    "loss_fp_risk",
                    "",
                ),
                train_statistics.get(
                    "loss_fp_risk_positive",
                    "",
                ),
                train_statistics.get(
                    "loss_fp_risk_negative",
                    "",
                ),
                train_statistics.get(
                    "loss_region",
                    "",
                ),
                train_statistics.get(
                    "loss_edge",
                    "",
                ),
                train_statistics["lr"],
                train_statistics[
                    "time_seconds"
                ],
            ]
        )


base_train.parse_args = parse_args
base_train.SODLoss = build_fp_loss
base_train.prepare_metrics_file = (
    prepare_metrics_file
)
base_train.append_metrics = (
    append_metrics
)


if __name__ == "__main__":
    base_train.main()
