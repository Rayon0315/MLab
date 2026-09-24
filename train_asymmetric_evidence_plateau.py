from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import train_plateau_v2 as base_train

from losses.asymmetric_evidence_loss import (
    AsymmetricEvidenceSODLoss,
)


_ORIGINAL_PARSE_ARGS = (
    base_train.parse_args
)

_CONFIG = {
    "fg_support_weight": 0.1,
    "fp_risk_weight": 0.2,
    "fp_risk_gamma": 2.0,
    "easy_background_weight": 0.05,
}


def parse_args() -> argparse.Namespace:
    global _CONFIG

    parser = argparse.ArgumentParser(
        add_help=False,
    )

    parser.add_argument(
        "--fg-support-weight",
        type=float,
        default=0.1,
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

    args.fg_support_weight = (
        custom_args.fg_support_weight
    )
    args.fp_risk_weight = (
        custom_args.fp_risk_weight
    )
    args.fp_risk_gamma = (
        custom_args.fp_risk_gamma
    )
    args.fp_easy_background_weight = (
        custom_args
        .fp_easy_background_weight
    )

    _CONFIG = {
        "fg_support_weight": (
            custom_args.fg_support_weight
        ),
        "fp_risk_weight": (
            custom_args.fp_risk_weight
        ),
        "fp_risk_gamma": (
            custom_args.fp_risk_gamma
        ),
        "easy_background_weight": (
            custom_args
            .fp_easy_background_weight
        ),
    }

    return args


def build_asymmetric_loss(
    aux_weight: float = 0.4,
    edge_weight: float = 0.0,
    region_weight: float = 0.0,
) -> AsymmetricEvidenceSODLoss:
    logger = logging.getLogger(
        __name__
    )

    logger.info(
        "Asymmetric evidence loss | "
        "FG support weight: %.3f | "
        "FP-risk weight: %.3f | "
        "FP gamma: %.3f | "
        "Easy-BG weight: %.3f",
        _CONFIG["fg_support_weight"],
        _CONFIG["fp_risk_weight"],
        _CONFIG["fp_risk_gamma"],
        _CONFIG[
            "easy_background_weight"
        ],
    )

    return AsymmetricEvidenceSODLoss(
        aux_weight=aux_weight,
        edge_weight=edge_weight,
        region_weight=region_weight,
        fg_support_weight=(
            _CONFIG[
                "fg_support_weight"
            ]
        ),
        fp_risk_weight=(
            _CONFIG[
                "fp_risk_weight"
            ]
        ),
        fp_risk_gamma=(
            _CONFIG[
                "fp_risk_gamma"
            ]
        ),
        easy_background_weight=(
            _CONFIG[
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
        writer = csv.writer(file)

        writer.writerow(
            [
                "epoch",
                "global_step",
                "train_loss",
                "train_loss_main",
                "train_loss_aux",
                "train_loss_fg_support",
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
                    "loss_fg_support",
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
base_train.SODLoss = (
    build_asymmetric_loss
)
base_train.prepare_metrics_file = (
    prepare_metrics_file
)
base_train.append_metrics = (
    append_metrics
)


if __name__ == "__main__":
    base_train.main()
