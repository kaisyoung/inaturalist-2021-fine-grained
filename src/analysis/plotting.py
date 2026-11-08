"""Matplotlib-only plotting helpers for Stage 2 and Stage 3 artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Keep matplotlib's font/config cache inside a writable, disposable location.
_MPL_CACHE = Path(tempfile.gettempdir()) / "comp9517-matplotlib"
_XDG_CACHE = Path(tempfile.gettempdir()) / "comp9517-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save_figure_atomic(figure: plt.Figure, output_path: Path, *, dpi: int = 180) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=output_path.suffix, dir=output_path.parent
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, dpi=dpi, bbox_inches="tight")
        os.replace(temporary_path, output_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)


def plot_confusion_matrix_overview(
    matrix: np.ndarray,
    output_path: Path,
    *,
    title: str,
    normalize: bool = True,
) -> None:
    """Plot a legible overview of a potentially 500-by-500 confusion matrix."""

    values = np.asarray(matrix)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("confusion matrix must be square")
    display = values.astype(np.float64)
    colorbar_label = "Count"
    if normalize:
        row_totals = display.sum(axis=1, keepdims=True)
        display = np.divide(
            display,
            row_totals,
            out=np.zeros_like(display),
            where=row_totals != 0,
        )
        colorbar_label = "Fraction of true class"

    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(display, interpolation="nearest", aspect="auto", cmap="viridis")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=colorbar_label)
    tick_step = max(1, values.shape[0] // 10)
    ticks = np.arange(0, values.shape[0], tick_step)
    axis.set_xticks(ticks, labels=[str(int(value)) for value in ticks], rotation=45)
    axis.set_yticks(ticks, labels=[str(int(value)) for value in ticks])
    axis.set_xlabel("Predicted new_label")
    axis.set_ylabel("True new_label")
    axis.set_title(title)
    _save_figure_atomic(figure, output_path)


def plot_top_confused_pairs(
    pairs: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    """Plot total errors for ranked unordered class pairs."""

    required = {
        "class_a_new_label",
        "class_b_new_label",
        "total_confusions",
    }
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"top-confusion table is missing columns: {missing}")
    figure_height = max(4.0, 0.35 * max(1, len(pairs)))
    figure, axis = plt.subplots(figsize=(10, figure_height))
    if pairs.empty:
        axis.text(0.5, 0.5, "No cross-class errors", ha="center", va="center")
        axis.set_axis_off()
    else:
        ordered = pairs.iloc[::-1]
        labels = [
            f"{int(first)} ↔ {int(second)}"
            for first, second in zip(
                ordered["class_a_new_label"], ordered["class_b_new_label"]
            )
        ]
        axis.barh(labels, ordered["total_confusions"], color="#4472C4")
        axis.set_xlabel("Misclassifications in both directions")
        axis.set_ylabel("Class pair (new_label)")
    axis.set_title(title)
    _save_figure_atomic(figure, output_path)


def plot_model_comparison(
    results: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Stage 2 traditional model comparison",
) -> None:
    """Plot validation and test top-1/top-5 accuracy for completed models."""

    required = {"model_name", "val_top1", "val_top5", "test_top1", "test_top5"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"model comparison table is missing columns: {missing}")
    if results.empty:
        raise ValueError("cannot plot an empty model comparison table")

    positions = np.arange(len(results), dtype=float)
    width = 0.2
    figure, axis = plt.subplots(figsize=(max(8, len(results) * 2.5), 5.5))
    for offset, column, label in (
        (-1.5, "val_top1", "Validation top-1"),
        (-0.5, "val_top5", "Validation top-5"),
        (0.5, "test_top1", "Test top-1"),
        (1.5, "test_top5", "Test top-5"),
    ):
        axis.bar(positions + offset * width, results[column], width, label=label)
    axis.set_xticks(positions, labels=results["model_name"], rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Accuracy")
    axis.set_title(title)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_figure_atomic(figure, output_path)


def plot_validation_ablation(
    results: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Validation-only handcrafted feature ablation",
) -> None:
    """Plot validation top-1 and top-5 accuracy for feature ablations."""

    required = {"experiment_name", "val_top1", "val_top5"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"ablation table is missing columns: {missing}")
    if results.empty:
        raise ValueError("cannot plot an empty ablation table")
    positions = np.arange(len(results), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(max(8, len(results) * 2.2), 5.5))
    axis.bar(positions - width / 2, results["val_top1"], width, label="Top-1")
    axis.bar(positions + width / 2, results["val_top5"], width, label="Top-5")
    axis.set_xticks(
        positions, labels=results["experiment_name"], rotation=20, ha="right"
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Validation accuracy")
    axis.set_title(title)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_figure_atomic(figure, output_path)


def plot_deep_training_curves(
    history: list[dict[str, object]] | pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Deep-learning training curves",
) -> None:
    """Plot the six required epoch curves from a Stage 3 JSONL history."""

    frame = history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
    required = {
        "epoch",
        "train_loss",
        "val_loss",
        "train_top1",
        "val_top1",
        "val_macro_f1",
        "learning_rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"deep training history is missing columns: {missing}")
    if frame.empty:
        raise ValueError("cannot plot an empty deep training history")
    frame = frame.sort_values("epoch", kind="stable")
    epochs = frame["epoch"].to_numpy(dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    panels = (
        ("train_loss", "Train loss", "Loss", "#4472C4"),
        ("val_loss", "Validation loss", "Loss", "#ED7D31"),
        ("train_top1", "Train top-1", "Accuracy", "#70AD47"),
        ("val_top1", "Validation top-1", "Accuracy", "#A5A5A5"),
        ("val_macro_f1", "Validation macro F1", "Macro F1", "#FFC000"),
        ("learning_rate", "Learning rate", "Learning rate", "#5B9BD5"),
    )
    for axis, (column, panel_title, ylabel, color) in zip(axes.flat, panels):
        axis.plot(epochs, frame[column].to_numpy(dtype=float), marker="o", color=color)
        axis.set_title(panel_title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        if column in {"train_top1", "val_top1", "val_macro_f1"}:
            axis.set_ylim(0.0, 1.0)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure_atomic(figure, output_path)


def plot_stage3_validation_comparison(
    results: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Stage 3 validation comparison",
) -> None:
    """Compare top-1, top-5, and macro F1 for completed Stage 3 runs."""

    required = {"model_name", "val_top1", "val_top5", "val_macro_f1"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Stage 3 comparison table is missing columns: {missing}")
    if results.empty:
        raise ValueError("cannot plot an empty Stage 3 comparison table")
    positions = np.arange(len(results), dtype=float)
    width = 0.25
    figure, axis = plt.subplots(figsize=(max(9, 2.2 * len(results)), 5.8))
    axis.bar(positions - width, results["val_top1"], width, label="Top-1")
    axis.bar(positions, results["val_top5"], width, label="Top-5")
    axis.bar(positions + width, results["val_macro_f1"], width, label="Macro F1")
    axis.set_xticks(positions, labels=results["model_name"], rotation=22, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Validation score")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_figure_atomic(figure, output_path)
