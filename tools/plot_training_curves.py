# tools/plot_training_curves.py

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training curves from metrics.csv."
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to logs/metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for output figures. "
            "Defaults to the metrics file directory."
        ),
    )
    return parser.parse_args()


def read_metrics(
    path: Path,
) -> dict[str, list[float]]:
    curves: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_loss_main": [],
        "train_loss_aux": [],
        "learning_rate": [],
        "train_time_seconds": [],
    }

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            for name in curves:
                value = row.get(name, "")

                if value == "":
                    curves[name].append(float("nan"))
                else:
                    curves[name].append(float(value))

    return curves


def plot_losses(
    curves: dict[str, list[float]],
    output_path: Path,
) -> None:
    epochs = curves["epoch"]

    plt.figure(figsize=(9, 6))
    plt.plot(
        epochs,
        curves["train_loss"],
        marker="o",
        label="Total loss",
    )
    plt.plot(
        epochs,
        curves["train_loss_main"],
        marker="o",
        label="Main loss",
    )
    plt.plot(
        epochs,
        curves["train_loss_aux"],
        marker="o",
        label="Auxiliary loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def plot_learning_rate(
    curves: dict[str, list[float]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))
    plt.plot(
        curves["epoch"],
        curves["learning_rate"],
        marker="o",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning Rate Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def plot_training_time(
    curves: dict[str, list[float]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))
    plt.plot(
        curves["epoch"],
        curves["train_time_seconds"],
        marker="o",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Time (seconds)")
    plt.title("Training Time per Epoch")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def main() -> None:
    args = parse_args()

    metrics_path = Path(args.metrics)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else metrics_path.parent
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    curves = read_metrics(metrics_path)

    plot_losses(
        curves,
        output_dir / "loss_curves.png",
    )
    plot_learning_rate(
        curves,
        output_dir / "learning_rate_curve.png",
    )
    plot_training_time(
        curves,
        output_dir / "training_time_curve.png",
    )

    print(f"Saved training curves to: {output_dir}")


if __name__ == "__main__":
    main()