"""Build compact report figures strictly from persisted Stage 1--4 artifacts.

This module deliberately contains no model imports and performs no inference.
It crops, composes, and compresses already-generated evidence so that the final
CVPR-format PDF remains readable and comfortably below the upload-size limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.analysis.stage5_contract import MAIN_REPORT_MODELS, PROJECT_ROOT, require_frozen_artifacts
from src.data.common import sha256_file


GENERATOR_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "final_report_figures"
SELECTION_TABLE = PROJECT_ROOT / "results" / "tables" / "gradcam_selected_examples.csv"


@dataclass(frozen=True)
class GradCamCase:
    image_id: int
    predicted_figure: Path
    true_figure: Path | None
    interpretation: str


GRADCAM_CASES: tuple[GradCamCase, ...] = (
    GradCamCase(
        2713957,
        PROJECT_ROOT / "results/figures/gradcam/correct_high_confidence/2713957__row-4863__target-predicted-label-486_gradcam.png",
        None,
        "correct",
    ),
    GradCamCase(
        2759395,
        PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence/2759395__row-4855__target-predicted-label-486_gradcam.png",
        PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence/2759395__row-4855__target-true-label-485_gradcam.png",
        "similar fruit",
    ),
    GradCamCase(
        2730603,
        PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence/2730603__row-1774__target-predicted-label-456_gradcam.png",
        PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence/2730603__row-1774__target-true-label-177_gradcam.png",
        "texture shortcut",
    ),
    GradCamCase(
        2758641,
        PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence/2758641__row-0255__target-predicted-label-004_gradcam.png",
        PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence/2758641__row-0255__target-true-label-025_gradcam.png",
        "arthropod confusion",
    ),
    GradCamCase(
        2782727,
        PROJECT_ROOT / "results/figures/gradcam/confused_pair/2782727__row-2177__target-predicted-label-218_gradcam.png",
        PROJECT_ROOT / "results/figures/gradcam/confused_pair/2782727__row-2177__target-true-label-217_gradcam.png",
        "top confused pair",
    ),
)


CATEGORY_GRADCAM: dict[str, tuple[tuple[int, Path], ...]] = {
    "correct": (
        (2713957, PROJECT_ROOT / "results/figures/gradcam/correct_high_confidence/2713957__row-4863__target-predicted-label-486_gradcam.png"),
        (2700732, PROJECT_ROOT / "results/figures/gradcam/correct_high_confidence/2700732__row-4301__target-predicted-label-430_gradcam.png"),
        (2783324, PROJECT_ROOT / "results/figures/gradcam/correct_high_confidence/2783324__row-1209__target-predicted-label-120_gradcam.png"),
        (2723623, PROJECT_ROOT / "results/figures/gradcam/correct_high_confidence/2723623__row-1761__target-predicted-label-176_gradcam.png"),
    ),
    "incorrect": (
        (2759395, PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence/2759395__row-4855__target-predicted-label-486_gradcam.png"),
        (2730603, PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence/2730603__row-1774__target-predicted-label-456_gradcam.png"),
        (2758641, PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence/2758641__row-0255__target-predicted-label-004_gradcam.png"),
        (2690086, PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence/2690086__row-2372__target-predicted-label-238_gradcam.png"),
    ),
    "confused": (
        (2782727, PROJECT_ROOT / "results/figures/gradcam/confused_pair/2782727__row-2177__target-predicted-label-218_gradcam.png"),
        (2723041, PROJECT_ROOT / "results/figures/gradcam/confused_pair/2723041__row-1152__target-predicted-label-116_gradcam.png"),
        (2687681, PROJECT_ROOT / "results/figures/gradcam/confused_pair/2687681__row-1850__target-predicted-label-186_gradcam.png"),
        (2745117, PROJECT_ROOT / "results/figures/gradcam/confused_pair/2745117__row-1917__target-predicted-label-192_gradcam.png"),
    ),
}


STATIC_SOURCES: tuple[Path, ...] = (
    PROJECT_ROOT / "results/figures/dataset_sample_grid.png",
    PROJECT_ROOT / "results/figures/deep_learning/stage3_validation_comparison.png",
    PROJECT_ROOT / "results/figures/deep_learning/resnet18_scratch_basic_training_curves.png",
    PROJECT_ROOT / "results/figures/deep_learning/efficientnet_b0_pretrained_strong_aug_training_curves.png",
    PROJECT_ROOT / "results/figures/deep_learning/efficientnet_b0_pretrained_strong_aug_test_confusion_matrix.png",
    PROJECT_ROOT / "results/figures/robustness/robustness_top1_curves.png",
    PROJECT_ROOT / "results/figures/robustness/robustness_macro_f1_curves.png",
    PROJECT_ROOT / "results/figures/robustness/corruption_examples.png",
    SELECTION_TABLE,
    PROJECT_ROOT / "results/figures/stage2_traditional_model_comparison.png",
    PROJECT_ROOT / "results/tables/final_model_comparison.csv",
    PROJECT_ROOT / "results/tables/stage4_confused_species_pairs.csv",
    PROJECT_ROOT / "results/figures/gradcam/correct_high_confidence_grid.png",
    PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence_grid.png",
    PROJECT_ROOT / "results/figures/gradcam/confused_pairs_grid.png",
    PROJECT_ROOT / "results/logs/deep_learning/resnet18_scratch_basic_train_log.jsonl",
    PROJECT_ROOT / "results/logs/deep_learning/efficientnet_b0_pretrained_strong_aug_train_log.jsonl",
    PROJECT_ROOT / "results/tables/stage2_traditional_results.csv",
    PROJECT_ROOT / "results/tables/robustness_results.csv",
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Required persisted figure is missing: {path}")
    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image)
        if oriented.mode == "RGBA":
            background = Image.new("RGBA", oriented.size, "white")
            background.alpha_composite(oriented)
            return background.convert("RGB")
        return oriented.convert("RGB")


def _largest_run(mask: np.ndarray) -> tuple[int, int]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return 0, int(mask.size)
    starts = np.r_[indices[0], indices[1:][np.diff(indices) > 1]]
    ends = np.r_[indices[:-1][np.diff(indices) > 1] + 1, indices[-1] + 1]
    lengths = ends - starts
    index = int(np.argmax(lengths))
    return int(starts[index]), int(ends[index])


def extract_dense_panel(source: Image.Image, panel_index: int, panel_count: int = 4) -> Image.Image:
    """Extract the dense raster panel from a persisted multi-panel Grad-CAM figure.

    Titles and annotations are sparse dark pixels on white.  The actual image
    panel is the largest contiguous run of rows/columns with substantial
    non-white coverage, which makes this robust to portrait and landscape
    source photographs without relying on a single pixel coordinate.
    """

    if not 0 <= panel_index < panel_count:
        raise ValueError(f"panel_index must be in [0, {panel_count}): {panel_index}")
    width, height = source.size
    left = round(panel_index * width / panel_count)
    right = round((panel_index + 1) * width / panel_count)
    segment = source.crop((left, 0, right, height)).convert("RGB")
    pixels = np.asarray(segment)
    nonwhite = np.any(pixels < 246, axis=2)
    row_start, row_end = _largest_run(nonwhite.mean(axis=1) > 0.12)
    dense_rows = nonwhite[row_start:row_end]
    col_start, col_end = _largest_run(dense_rows.mean(axis=0) > 0.12)
    padding = max(2, round(min(segment.size) * 0.006))
    return segment.crop(
        (
            max(0, col_start - padding),
            max(0, row_start - padding),
            min(segment.width, col_end + padding),
            min(segment.height, row_end + padding),
        )
    )


def _contain(image: Image.Image, size: tuple[int, int], background: str = "white") -> Image.Image:
    canvas = Image.new("RGB", size, background)
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - copy.width) // 2
    y = (size[1] - copy.height) // 2
    canvas.paste(copy, (x, y))
    return canvas


def _trim_white(image: Image.Image, threshold: int = 250, padding: int = 8) -> Image.Image:
    pixels = np.asarray(image.convert("RGB"))
    mask = np.any(pixels < threshold, axis=2)
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return image.copy()
    left = max(0, int(columns.min()) - padding)
    top = max(0, int(rows.min()) - padding)
    right = min(image.width, int(columns.max()) + padding + 1)
    bottom = min(image.height, int(rows.max()) + padding + 1)
    return image.crop((left, top, right, bottom))


def _save_jpeg(image: Image.Image, path: Path, *, quality: int = 84, max_width: int | None = None) -> None:
    output = image.convert("RGB")
    if max_width is not None and output.width > max_width:
        new_height = round(output.height * max_width / output.width)
        output = output.resize((max_width, new_height), Image.Resampling.LANCZOS)
    output.save(path, format="JPEG", quality=quality, optimize=True, progressive=True, subsampling=2)


def _save_png(image: Image.Image, path: Path, *, max_width: int | None = None) -> None:
    output = image.convert("RGB")
    if max_width is not None and output.width > max_width:
        new_height = round(output.height * max_width / output.width)
        output = output.resize((max_width, new_height), Image.Resampling.LANCZOS)
    output.save(path, format="PNG", optimize=True)


def _draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont, fill: str = "#152536") -> None:
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=4)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2),
        text,
        font=font,
        fill=fill,
        align="center",
        spacing=4,
    )


def build_method_overview(path: Path) -> None:
    canvas = Image.new("RGB", (1800, 720), "white")
    draw = ImageDraw.Draw(canvas)
    title = _font(38, bold=True)
    heading = _font(27, bold=True)
    body = _font(23)
    draw.text((900, 32), "Split-controlled species-recognition workflow", font=title, fill="#102A43", anchor="ma")

    boxes = {
        "data": (50, 245, 340, 500),
        "traditional": (445, 120, 790, 330),
        "deep": (445, 400, 790, 610),
        "validation": (900, 245, 1190, 500),
        "test": (1300, 245, 1745, 500),
    }
    colours = {
        "data": "#D9EAF7",
        "traditional": "#FDE3C2",
        "deep": "#DDEFD8",
        "validation": "#E8DDF4",
        "test": "#F6D6D6",
    }
    for key, box in boxes.items():
        draw.rounded_rectangle(box, radius=24, fill=colours[key], outline="#334E68", width=4)

    _draw_centered(draw, boxes["data"], "Frozen Stage 1 split\n500 species\n20k / 5k / 5k", heading)
    _draw_centered(draw, boxes["traditional"], "Traditional CV\n3,704-D LinearSVC\n2,424-D RF for robustness", body)
    _draw_centered(draw, boxes["deep"], "Deep learning\nResNet18 / EfficientNet-B0\nscratch + ImageNet transfer", body)
    _draw_centered(draw, boxes["validation"], "Validation only\nmodel selection\nablations + early stopping", body)
    _draw_centered(draw, boxes["test"], "Held-out test evidence\naccuracy + macro metrics\nrobustness + Grad-CAM", body)

    def arrow(start: tuple[int, int], end: tuple[int, int]) -> None:
        draw.line((start, end), fill="#486581", width=8)
        x, y = end
        draw.polygon(((x, y), (x - 24, y - 15), (x - 24, y + 15)), fill="#486581")

    arrow((340, 315), (445, 225))
    arrow((340, 430), (445, 505))
    arrow((790, 225), (900, 320))
    arrow((790, 505), (900, 430))
    arrow((1190, 373), (1300, 373))
    draw.text((900, 675), "All fitting uses train; current Stage 2 state is hash-bound, while historical pre-test locking is not independently established.", font=body, fill="#334E68", anchor="ms")
    _save_png(canvas, path)


def _load_history(path: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if record.get("event") == "epoch":
                records.append(record)
    if not records:
        raise ValueError(f"Training log contains no epoch events: {path}")
    frame = pd.DataFrame.from_records(records).sort_values("epoch", kind="stable")
    required = {"epoch", "train_loss", "val_loss", "train_top1", "val_top1", "val_top5", "val_macro_f1"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Training log {path} is missing fields: {missing}")
    return frame


def _pyplot():
    cache_dir = Path(tempfile.gettempdir()) / "comp9517-stage5-mpl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return plt


def _save_matplotlib(fig: object, path: Path) -> None:
    options: dict[str, object] = {
        "dpi": 180,
        "bbox_inches": "tight",
        "facecolor": "white",
    }
    if path.suffix.lower() == ".png":
        options["metadata"] = {"Software": "COMP9517 Stage 5 persisted-artifact compositor"}
    elif path.suffix.lower() == ".pdf":
        options["metadata"] = {
            "Creator": "COMP9517 Stage 5 persisted-artifact compositor",
            "Producer": "Matplotlib PDF backend",
            "CreationDate": None,
            "ModDate": None,
        }
    fig.savefig(path, **options)
    _pyplot().close(fig)


def build_method_overview_vector(path: Path) -> None:
    """Create the pipeline overview as true vector PDF drawing primitives."""

    plt = _pyplot()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, axis = plt.subplots(figsize=(12, 4.8), layout="constrained")
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5)
    axis.axis("off")
    axis.set_title("Split-controlled species-recognition workflow", fontsize=18, fontweight="bold", pad=8)
    boxes = {
        "data": ((0.2, 1.65), 2.1, 1.75, "#D9EAF7", "Frozen Stage 1 split\n500 species\n20k / 5k / 5k"),
        "traditional": ((3.0, 3.0), 2.5, 1.35, "#FDE3C2", "Traditional CV\n3,704-D LinearSVC\n2,424-D RF for robustness"),
        "deep": ((3.0, 0.65), 2.5, 1.35, "#DDEFD8", "Deep learning\nResNet18 / EfficientNet-B0\nscratch + ImageNet transfer"),
        "validation": ((6.25, 1.65), 2.1, 1.75, "#E8DDF4", "Validation only\nmodel selection\nablations + early stopping"),
        "test": ((9.25, 1.65), 2.55, 1.75, "#F6D6D6", "Held-out test evidence\naccuracy + macro metrics\nrobustness + Grad-CAM"),
    }
    for position, width, height, colour, label in boxes.values():
        patch = FancyBboxPatch(
            position,
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.16",
            linewidth=1.5,
            edgecolor="#334E68",
            facecolor=colour,
        )
        axis.add_patch(patch)
        axis.text(
            position[0] + width / 2,
            position[1] + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#152536",
            fontweight="bold" if label.startswith("Frozen Stage") else "normal",
        )

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=2.0, color="#486581")
        )

    arrow((2.3, 2.65), (3.0, 3.65))
    arrow((2.3, 2.35), (3.0, 1.3))
    arrow((5.5, 3.65), (6.25, 2.8))
    arrow((5.5, 1.3), (6.25, 2.2))
    arrow((8.35, 2.52), (9.25, 2.52))
    axis.text(
        6.0,
        0.18,
        "All fitting uses train; current Stage 2 is hash-bound, but historical pre-test locking is not independently established.",
        ha="center",
        va="center",
        fontsize=9.6,
        color="#334E68",
    )
    _save_matplotlib(fig, path)


def build_training_convergence(path: Path) -> None:
    histories = (
        ("ResNet18 scratch", _load_history(STATIC_SOURCES[15]), "#4472C4"),
        ("EfficientNet-B0 pretrained", _load_history(STATIC_SOURCES[16]), "#E07A1F"),
    )
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), sharex="row", layout="constrained")
    for row_index, (model_name, frame, colour) in enumerate(histories):
        epochs = frame["epoch"].to_numpy()
        axes[row_index, 0].plot(epochs, frame["train_loss"], marker="o", markersize=3.5, color=colour)
        axes[row_index, 1].plot(epochs, frame["val_macro_f1"], marker="o", markersize=3.5, color=colour)
        axes[row_index, 0].set_ylabel(f"{model_name}\nloss")
        axes[row_index, 1].set_ylabel("macro F1")
        axes[row_index, 0].set_xlabel("epoch")
        axes[row_index, 1].set_xlabel("epoch")
        axes[row_index, 1].set_ylim(0, 0.8)
        for axis in axes[row_index]:
            axis.grid(True, alpha=0.25)
            axis.set_xticks(epochs[:: max(1, len(epochs) // 6)])
    axes[0, 0].set_title("Training loss")
    axes[0, 1].set_title("Validation macro F1")
    fig.suptitle("Observed convergence on the frozen Stage 1 split", fontweight="bold")
    _save_matplotlib(fig, path)


def build_model_training_curves(history_path: Path, title: str, path: Path) -> None:
    frame = _load_history(history_path)
    epochs = frame["epoch"].to_numpy()
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.3), layout="constrained")
    axes[0, 0].plot(epochs, frame["train_loss"], label="train", marker="o", markersize=3)
    axes[0, 0].plot(epochs, frame["val_loss"], label="validation", marker="o", markersize=3)
    axes[0, 0].set_title("Cross-entropy loss")
    axes[0, 0].legend()
    axes[0, 1].plot(epochs, frame["train_top1"], label="train top-1", marker="o", markersize=3)
    axes[0, 1].plot(epochs, frame["val_top1"], label="validation top-1", marker="o", markersize=3)
    axes[0, 1].set_title("Top-1 accuracy")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[1, 0].plot(epochs, frame["val_top5"], color="#2A9D8F", marker="o", markersize=3)
    axes[1, 0].set_title("Validation top-5 accuracy")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 1].plot(epochs, frame["val_macro_f1"], color="#E9A800", marker="o", markersize=3)
    axes[1, 1].set_title("Validation macro F1")
    axes[1, 1].set_ylim(0, 1)
    for axis in axes.flat:
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.25)
    fig.suptitle(title, fontweight="bold")
    _save_matplotlib(fig, path)


def build_robustness_composite(path: Path, top1_source: Path, macro_f1_source: Path) -> None:
    top1 = _contain(_trim_white(_rgb(top1_source)), (1120, 720))
    macro_f1 = _contain(_trim_white(_rgb(macro_f1_source)), (1120, 720))
    canvas = Image.new("RGB", (2264, 780), "white")
    canvas.paste(top1, (8, 54))
    canvas.paste(macro_f1, (1136, 54))
    draw = ImageDraw.Draw(canvas)
    font = _font(31, bold=True)
    draw.text((568, 12), "Top-1 accuracy", font=font, fill="#102A43", anchor="ma")
    draw.text((1696, 12), "Macro F1", font=font, fill="#102A43", anchor="ma")
    _save_png(canvas, path, max_width=1900)


def build_robustness_metric(metric: str, path: Path) -> None:
    if metric not in {"top1", "macro_f1"}:
        raise ValueError(f"Unsupported robustness metric: {metric}")
    frame = pd.read_csv(STATIC_SOURCES[18])
    model_order = (
        "traditional_random_forest",
        "resnet18_scratch_basic",
        "resnet18_pretrained_strong_aug",
        "efficientnet_b0_pretrained_strong_aug",
    )
    display = {
        "traditional_random_forest": "RF (2,424-D handcrafted)",
        "resnet18_scratch_basic": "ResNet18 scratch",
        "resnet18_pretrained_strong_aug": "ResNet18 pretrained",
        "efficientnet_b0_pretrained_strong_aug": "EfficientNet-B0 pretrained",
    }
    colours = ("#4C78A8", "#F58518", "#54A24B", "#E45756")
    corruption_order = ("gaussian_blur", "gaussian_noise", "jpeg_compression", "motion_blur")
    titles = {
        "gaussian_blur": "Gaussian blur",
        "gaussian_noise": "Gaussian noise",
        "jpeg_compression": "JPEG compression",
        "motion_blur": "Motion blur",
    }
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharey=True, layout="constrained")
    for axis, corruption in zip(axes.flat, corruption_order, strict=True):
        for model, colour in zip(model_order, colours, strict=True):
            clean_rows = frame[(frame["model_name"] == model) & (frame["corruption_type"] == "clean")]
            corrupted = frame[(frame["model_name"] == model) & (frame["corruption_type"] == corruption)].sort_values("severity")
            if len(clean_rows) != 1 or len(corrupted) != 4:
                raise ValueError(f"Incomplete robustness trajectory: {model}/{corruption}")
            values = np.r_[float(clean_rows.iloc[0][metric]), corrupted[metric].astype(float).to_numpy()]
            axis.plot(np.arange(5), values, marker="o", linewidth=2, label=display[model], color=colour)
        parameters = corrupted["severity_value"].tolist()
        labels = ["clean"] + [f"{value:g}" for value in parameters]
        axis.set_xticks(np.arange(5), labels)
        axis.set_title(titles[corruption])
        axis.set_xlabel("severity parameter")
        axis.set_ylim(0, 1)
        axis.grid(True, alpha=0.25)
    axes[0, 0].set_ylabel("Top-1 accuracy" if metric == "top1" else "Macro F1")
    axes[1, 0].set_ylabel("Top-1 accuracy" if metric == "top1" else "Macro F1")
    fig.suptitle("Robustness to image degradation", fontweight="bold")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    _save_matplotlib(fig, path)


def build_final_model_comparison(path: Path) -> None:
    frame = pd.read_csv(STATIC_SOURCES[10])
    frame = frame[frame["model_name"].isin(MAIN_REPORT_MODELS)].copy()
    order = {name: index for index, name in enumerate(MAIN_REPORT_MODELS)}
    frame["_order"] = frame["model_name"].map(order)
    frame = frame.sort_values("_order", kind="stable")
    if tuple(frame["model_name"]) != MAIN_REPORT_MODELS:
        raise ValueError("Final model comparison does not contain the six frozen report models")
    if frame[["test_top1", "test_top5", "test_macro_f1"]].isna().any().any():
        raise ValueError("A frozen report model has missing test metrics")
    display_names = {
        "linear_svm": "Handcrafted + BoVW (LinearSVC)",
        "random_forest": "Random forest (2,424-D)",
        "sift_bovw_svm": "SIFT-BoVW (LinearSVC)",
        "resnet18_scratch_basic": "ResNet18 scratch",
        "resnet18_pretrained_strong_aug": "ResNet18 pretrained",
        "efficientnet_b0_pretrained_strong_aug": "EfficientNet-B0 pretrained",
    }
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(10.4, 5.3))
    y = np.arange(len(frame), dtype=float)
    height = 0.23
    for offset, column, label, colour in (
        (-height, "test_top1", "Top-1", "#3B82B8"),
        (0.0, "test_top5", "Top-5", "#69A84F"),
        (height, "test_macro_f1", "Macro F1", "#E69F3A"),
    ):
        values = frame[column].astype(float).to_numpy() * 100.0
        bars = axis.barh(y + offset, values, height=height * 0.9, label=label, color=colour)
        axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    axis.set_yticks(y, [display_names[name] for name in frame["model_name"]])
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Held-out test performance (%)")
    axis.set_title("Traditional and deep-learning model comparison")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(ncols=3, loc="upper right")
    fig.tight_layout()
    _save_matplotlib(fig, path)


def build_stage2_comparison(path: Path) -> None:
    frame = pd.read_csv(STATIC_SOURCES[17])
    expected = ("linear_svm", "random_forest", "sift_bovw_svm")
    frame = frame.set_index("model_name").loc[list(expected)].reset_index()
    labels = (
        "LinearSVC\n3,704-D",
        "Random forest\n2,424-D",
        "BoVW LinearSVC\n1,280-D",
    )
    metrics = (
        ("val_top1", "Val top-1", "#4C78A8"),
        ("val_top5", "Val top-5", "#F58518"),
        ("test_top1", "Test top-1", "#54A24B"),
        ("test_top5", "Test top-5", "#E45756"),
    )
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(9.4, 5.2), layout="constrained")
    x = np.arange(len(frame), dtype=float)
    width = 0.19
    for index, (column, label, colour) in enumerate(metrics):
        values = frame[column].astype(float).to_numpy()
        bars = axis.bar(x + (index - 1.5) * width, values, width=width, label=label, color=colour)
        axis.bar_label(bars, labels=[f"{value * 100:.1f}" for value in values], padding=2, fontsize=8)
    axis.set_xticks(x, labels)
    maximum = float(frame[[column for column, _, _ in metrics]].max().max())
    axis.set_ylim(0, min(1.0, max(0.16, maximum * 1.22)))
    axis.set_ylabel("Accuracy (fraction; labels show %)")
    axis.set_title("Traditional baselines (zoomed scale)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=2, loc="upper left")
    _save_matplotlib(fig, path)


def build_top_confused_pairs(path: Path) -> None:
    frame = pd.read_csv(STATIC_SOURCES[11]).head(10).copy()
    required = {"class_a_common_name", "class_b_common_name", "a_as_b", "b_as_a", "total_confusions"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Confused-pair table is missing columns: {missing}")
    labels = [
        f"{_short_name(row.class_a_common_name)} / {_short_name(row.class_b_common_name)}"
        for row in frame.itertuples(index=False)
    ]
    plt = _pyplot()
    from matplotlib.ticker import MaxNLocator

    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    y = np.arange(len(frame))
    first = frame["a_as_b"].astype(int).to_numpy()
    second = frame["b_as_a"].astype(int).to_numpy()
    axis.barh(y, first, color="#4C78A8", label="A predicted as B")
    axis.barh(y, second, left=first, color="#F28E2B", label="B predicted as A")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Number of held-out test errors")
    axis.set_title("Most frequently confused species pairs (EfficientNet-B0)")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    _save_matplotlib(fig, path)


def _short_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text if text else "unnamed class"


def build_gradcam_composite(path: Path) -> None:
    frame = pd.read_csv(SELECTION_TABLE)
    if frame["image_id"].duplicated().any():
        # Confused-pair sampling can contain directional rows, but every chosen
        # report image must still resolve unambiguously.
        chosen = frame[frame["image_id"].isin([case.image_id for case in GRADCAM_CASES])]
        duplicates = chosen[chosen["image_id"].duplicated()]["image_id"].tolist()
        if duplicates:
            raise ValueError(f"Ambiguous Grad-CAM selection rows: {duplicates}")
    by_id = frame.set_index("image_id", drop=False)
    panel_size = (500, 300)
    margin, gap, label_height, column_header = 24, 18, 64, 58
    width = margin * 2 + panel_size[0] * 3 + gap * 2
    row_height = label_height + panel_size[1]
    height = column_header + len(GRADCAM_CASES) * row_height + margin
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = _font(27, bold=True)
    row_font = _font(22, bold=True)
    small_font = _font(21)
    headers = ("Original image", "CAM for predicted class", "CAM for true class")
    for column, header in enumerate(headers):
        x = margin + column * (panel_size[0] + gap) + panel_size[0] / 2
        draw.text((x, 16), header, font=header_font, fill="#102A43", anchor="ma")

    for index, case in enumerate(GRADCAM_CASES):
        if case.image_id not in by_id.index:
            raise ValueError(f"Grad-CAM image {case.image_id} is absent from {SELECTION_TABLE}")
        row = by_id.loc[case.image_id]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"Grad-CAM image {case.image_id} has multiple selection rows")
        source_predicted = _rgb(case.predicted_figure)
        original = extract_dense_panel(source_predicted, 0)
        predicted_overlay = extract_dense_panel(source_predicted, 3)
        true_name = _short_name(row.get("common_name"))
        predicted_name = _short_name(row.get("predicted_common_name"))
        if true_name == "unnamed class":
            true_name = _short_name(row.get("scientific_name"))
        if predicted_name == "unnamed class":
            predicted_name = _short_name(row.get("predicted_scientific_name"))
        prefix = chr(ord("a") + index)
        caption = (
            f"({prefix}) {case.interpretation}: {true_name} → {predicted_name} "
            f"(p={float(row['top1_score']):.3f})"
        )
        top = column_header + index * row_height
        draw.text((margin, top + 5), caption, font=row_font, fill="#243B53")
        image_top = top + label_height
        panels: list[Image.Image | None] = [original, predicted_overlay]
        if case.true_figure is None:
            panels.append(None)
        else:
            panels.append(extract_dense_panel(_rgb(case.true_figure), 3))
        for column, panel in enumerate(panels):
            left = margin + column * (panel_size[0] + gap)
            if panel is None:
                empty = Image.new("RGB", panel_size, "#F3F6F8")
                empty_draw = ImageDraw.Draw(empty)
                empty_draw.text(
                    (panel_size[0] / 2, panel_size[1] / 2),
                    "prediction = truth\n(same CAM target)",
                    font=small_font,
                    fill="#627D98",
                    anchor="mm",
                    align="center",
                )
                fitted = empty
            else:
                fitted = _contain(panel, panel_size, background="#F8FAFC")
            canvas.paste(fitted, (left, image_top))
        if index < len(GRADCAM_CASES) - 1:
            separator_y = image_top + panel_size[1] + 5
            draw.line((margin, separator_y, width - margin, separator_y), fill="#D9E2EC", width=2)
    _save_jpeg(canvas, path, quality=85, max_width=1600)


def build_gradcam_category_composite(category: str, path: Path) -> None:
    """Build a compact 2x2 overlay figure from individual persisted CAM files."""

    if category not in CATEGORY_GRADCAM:
        raise ValueError(f"Unknown Grad-CAM category: {category}")
    frame = pd.read_csv(SELECTION_TABLE).set_index("image_id", drop=False)
    entries = CATEGORY_GRADCAM[category]
    cell_size = (700, 480)
    caption_height, margin, gap = 58, 20, 18
    canvas = Image.new(
        "RGB",
        (margin * 2 + cell_size[0] * 2 + gap, margin * 2 + (cell_size[1] + caption_height) * 2 + gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(22, bold=True)
    for index, (image_id, source_path) in enumerate(entries):
        if image_id not in frame.index:
            raise ValueError(f"Grad-CAM image {image_id} is absent from the selection manifest")
        row = frame.loc[image_id]
        if isinstance(row, pd.DataFrame):
            raise ValueError(f"Grad-CAM image {image_id} has multiple selection rows")
        overlay = extract_dense_panel(_rgb(source_path), 3)
        true_name = _short_name(row.get("common_name"))
        predicted_name = _short_name(row.get("predicted_common_name"))
        if true_name == "unnamed class":
            true_name = _short_name(row.get("scientific_name"))
        if predicted_name == "unnamed class":
            predicted_name = _short_name(row.get("predicted_scientific_name"))
        caption = f"{true_name} → {predicted_name}  (p={float(row['top1_score']):.3f})"
        column, grid_row = index % 2, index // 2
        left = margin + column * (cell_size[0] + gap)
        top = margin + grid_row * (cell_size[1] + caption_height + gap)
        canvas.paste(_contain(overlay, cell_size, background="#F8FAFC"), (left, top))
        draw.text((left + 4, top + cell_size[1] + 10), caption, font=font, fill="#243B53")
    _save_jpeg(canvas, path, quality=85, max_width=1500)


def _source_paths() -> tuple[Path, ...]:
    gradcam = tuple(
        path
        for case in GRADCAM_CASES
        for path in (case.predicted_figure, case.true_figure)
        if path is not None
    )
    category_gradcam = tuple(path for entries in CATEGORY_GRADCAM.values() for _, path in entries)
    # Bind the cache to the compositor implementation as well as evidence.
    # A source-code change therefore cannot silently reuse stale derivatives.
    candidates = (Path(__file__).resolve(),) + STATIC_SOURCES + gradcam + category_gradcam
    return tuple(dict.fromkeys(candidates))


def _source_manifest(paths: Iterable[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required persisted source is missing: {path}")
        rows.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return rows


def _existing_cache_valid(output_dir: Path, sources: list[dict[str, object]]) -> bool:
    manifest_path = output_dir / "figure_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != GENERATOR_SCHEMA_VERSION:
            return False
        if manifest.get("sources") != sources:
            return False
        outputs = manifest.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        for entry in outputs:
            path = output_dir / str(entry["file"])
            if not path.is_file() or sha256_file(path) != entry["sha256"]:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _write_manifest(staging: Path, sources: list[dict[str, object]], output_names: Sequence[str]) -> None:
    outputs: list[dict[str, object]] = []
    for name in output_names:
        path = staging / name
        if path.suffix.lower() == ".pdf":
            width = height = None
        else:
            with Image.open(path) as image:
                width, height = image.size
        outputs.append(
            {
                "file": name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "width": width,
                "height": height,
            }
        )
    manifest = {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "operation": "persisted-artifact composition only; no model inference",
        "sources": sources,
        "outputs": outputs,
    }
    (staging / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate(output_dir: Path, *, force: bool = False, verify_frozen: bool = True) -> str:
    if verify_frozen:
        require_frozen_artifacts()
    sources = _source_manifest(_source_paths())
    if _existing_cache_valid(output_dir, sources):
        return "cache_hit"
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Report-figure cache is stale or partial; pass --force to replace generated files: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".final-report-figures-", dir=output_dir.parent))
    output_names = (
        "method_overview.png",
        "dataset_examples.jpg",
        "deep_validation_comparison.png",
        "training_convergence.png",
        "efficientnet_confusion.png",
        "robustness_curves.png",
        "corruption_examples.jpg",
        "gradcam_examples.jpg",
        "stage2_traditional_results.png",
        "scratch_training_curves.png",
        "best_pretrained_training_curves.png",
        "final_model_comparison.png",
        "top_confused_pairs.png",
        "robustness_top1_curves.png",
        "robustness_macro_f1_curves.png",
        "gradcam_correct_examples.jpg",
        "gradcam_incorrect_examples.jpg",
        "gradcam_confused_pairs.jpg",
        "method_overview.pdf",
    )
    try:
        build_method_overview(staging / output_names[0])
        _save_jpeg(_trim_white(_rgb(STATIC_SOURCES[0])), staging / output_names[1], quality=85, max_width=1600)
        _save_png(_trim_white(_rgb(STATIC_SOURCES[1])), staging / output_names[2], max_width=1500)
        build_training_convergence(staging / output_names[3])
        _save_png(_trim_white(_rgb(STATIC_SOURCES[4])), staging / output_names[4], max_width=1400)
        _save_jpeg(_trim_white(_rgb(STATIC_SOURCES[7])), staging / output_names[6], quality=85, max_width=1500)
        build_gradcam_composite(staging / output_names[7])
        build_stage2_comparison(staging / output_names[8])
        build_model_training_curves(
            STATIC_SOURCES[15], "ResNet18 trained from scratch", staging / output_names[9]
        )
        build_model_training_curves(
            STATIC_SOURCES[16], "ImageNet-pretrained EfficientNet-B0", staging / output_names[10]
        )
        build_final_model_comparison(staging / output_names[11])
        build_top_confused_pairs(staging / output_names[12])
        build_robustness_metric("top1", staging / output_names[13])
        build_robustness_metric("macro_f1", staging / output_names[14])
        build_gradcam_category_composite("correct", staging / output_names[15])
        build_gradcam_category_composite("incorrect", staging / output_names[16])
        build_gradcam_category_composite("confused", staging / output_names[17])
        build_method_overview_vector(staging / output_names[18])
        build_robustness_composite(
            staging / output_names[5], staging / output_names[13], staging / output_names[14]
        )
        _write_manifest(staging, sources, output_names)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in (*output_names, "figure_manifest.json"):
            os.replace(staging / name, output_dir / name)
        # This was a managed Stage 5 derivative before convergence became a
        # lossless chart.  Remove only that obsolete generated filename.
        (output_dir / "training_convergence.jpg").unlink(missing_ok=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return "generated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compose compact final-report figures from persisted Stage 1-4 evidence; performs no inference."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Replace a stale or partial generated-figure cache.")
    parser.add_argument(
        "--skip-frozen-verification",
        action="store_true",
        help="Testing/development only: do not verify the accepted Stage 1-4 evidence hashes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        status = generate(
            args.output_dir.resolve(),
            force=args.force,
            verify_frozen=not args.skip_frozen_verification,
        )
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}")
        return 1
    if status == "cache_hit":
        print(f"[PASS] Valid report-figure cache hit: {args.output_dir.resolve()}")
    else:
        print(f"[PASS] Generated report figures without inference: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
