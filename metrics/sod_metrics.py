# metrics/sod_metrics.py

from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
from py_sod_metrics import (
    Emeasure,
    FmeasureHandler,
    FmeasureV2,
    MAE,
    PrecisionHandler,
    RecallHandler,
    Smeasure,
    WeightedFmeasure,
)
from tqdm import tqdm


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}


class SODMetricRecorder:
    """
    Standard grayscale SOD metric recorder.

    Input:
        prediction: uint8 ndarray [H, W], range [0, 255]
        ground_truth: uint8 ndarray [H, W], range [0, 255]
    """

    def __init__(self) -> None:
        self.mae = MAE()
        self.smeasure = Smeasure()
        self.weighted_fmeasure = WeightedFmeasure()
        self.emeasure = Emeasure()

        self.fmeasure = FmeasureV2(
            metric_handlers={
                "fmeasure": FmeasureHandler(
                    beta=0.3,
                    with_adaptive=True,
                    with_dynamic=True,
                ),
                "precision": PrecisionHandler(
                    with_adaptive=True,
                    with_dynamic=True,
                ),
                "recall": RecallHandler(
                    with_adaptive=True,
                    with_dynamic=True,
                ),
            }
        )

    def step(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
    ) -> None:
        self.mae.step(prediction, ground_truth)
        self.smeasure.step(prediction, ground_truth)
        self.weighted_fmeasure.step(
            prediction,
            ground_truth,
        )
        self.emeasure.step(prediction, ground_truth)
        self.fmeasure.step(prediction, ground_truth)

    def get_results(
        self,
    ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        mae = self.mae.get_results()["mae"]
        smeasure = self.smeasure.get_results()["sm"]
        weighted_fmeasure = self.weighted_fmeasure.get_results()["wfm"]

        emeasure_results = self.emeasure.get_results()["em"]
        fmeasure_results = self.fmeasure.get_results()

        fmeasure = fmeasure_results["fmeasure"]
        precision = fmeasure_results["precision"]
        recall = fmeasure_results["recall"]

        fmeasure_curve = np.asarray(
            fmeasure["dynamic"],
            dtype=np.float64,
        )
        precision_curve = np.asarray(
            precision["dynamic"],
            dtype=np.float64,
        )
        recall_curve = np.asarray(
            recall["dynamic"],
            dtype=np.float64,
        )
        emeasure_curve = np.asarray(
            emeasure_results["curve"],
            dtype=np.float64,
        )

        metrics = {
            "mae": float(mae),
            "smeasure": float(smeasure),
            "weighted_fmeasure": float(weighted_fmeasure),
            "max_fmeasure": float(fmeasure_curve.max()),
            "mean_fmeasure": float(fmeasure_curve.mean()),
            "adaptive_fmeasure": float(
                fmeasure["adaptive"]
            ),
            "max_emeasure": float(emeasure_curve.max()),
            "mean_emeasure": float(emeasure_curve.mean()),
            "adaptive_emeasure": float(
                emeasure_results["adp"]
            ),
        }

        # PySODMetrics 的动态结果按阈值从高到低排列。
        # 保存时翻转为 0 -> 1，便于后续直接绘图。
        curves = {
            "thresholds": np.linspace(
                0.0,
                1.0,
                256,
                dtype=np.float64,
            ),
            "precision": np.flip(
                precision_curve
            ).copy(),
            "recall": np.flip(
                recall_curve
            ).copy(),
            "fmeasure": np.flip(
                fmeasure_curve
            ).copy(),
            "emeasure": np.flip(
                emeasure_curve
            ).copy(),
        }

        return metrics, curves

def evaluate_prediction_directory(
    prediction_dir: str | Path,
    ground_truth_dir: str | Path,
    show_progress: bool = True,
) -> tuple[
    dict[str, float],
    dict[str, np.ndarray],
    int,
]:
    """
    Evaluate saved prediction images against original ground truth masks.
    Predictions are paired with ground truths by file stem.
    """
    prediction_dir = Path(prediction_dir)
    ground_truth_dir = Path(ground_truth_dir)

    prediction_map = _collect_file_map(prediction_dir)
    ground_truth_map = _collect_file_map(ground_truth_dir)

    missing_ground_truths = sorted(
        set(prediction_map) - set(ground_truth_map)
    )
    if missing_ground_truths:
        raise RuntimeError(
            "Predictions without matching ground truths: "
            + ", ".join(missing_ground_truths[:10])
        )

    prediction_names = sorted(prediction_map)
    recorder = SODMetricRecorder()

    progress_bar = tqdm(
        prediction_names,
        desc="Evaluation",
        unit="image",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    for name in progress_bar:
        prediction = _read_grayscale(
            prediction_map[name]
        )
        ground_truth = _read_grayscale(
            ground_truth_map[name]
        )

        if prediction.shape != ground_truth.shape:
            prediction = _resize_prediction(
                prediction,
                target_shape=ground_truth.shape,
            )

        recorder.step(
            prediction=prediction,
            ground_truth=ground_truth,
        )

    metrics, curves = recorder.get_results()

    return metrics, curves, len(prediction_names)


def save_metric_curves(
    path: str | Path,
    curves: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(
        path,
        **curves,
    )


def get_metric_library_version() -> str:
    return version("pysodmetrics")


def _collect_file_map(
    directory: Path,
) -> dict[str, Path]:
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not files:
        raise RuntimeError(
            f"No prediction or mask images found in: {directory}"
        )

    file_map: dict[str, Path] = {}

    for path in files:
        if path.stem in file_map:
            raise RuntimeError(
                f"Duplicate file stem found: {path.stem}"
            )

        file_map[path.stem] = path

    return file_map


def _read_grayscale(
    path: Path,
) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(
            image.convert("L"),
            dtype=np.uint8,
            copy=True,
        )


def _resize_prediction(
    prediction: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    target_height, target_width = target_shape

    resized = Image.fromarray(prediction).resize(
        (target_width, target_height),
        resample=Image.Resampling.BILINEAR,
    )

    return np.array(
        resized,
        dtype=np.uint8,
        copy=True,
    )