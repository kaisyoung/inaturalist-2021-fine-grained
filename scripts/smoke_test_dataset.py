#!/usr/bin/env python3
"""Validate and visually sample the frozen Stage 1 materialised dataset."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Direct script execution puts ``scripts/`` rather than the project root on
# sys.path.  Establish both imports and writable plotting caches before
# importing matplotlib or project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PLOTTING_CACHE = PROJECT_ROOT / "results" / "logs" / "plotting-cache"
PLOTTING_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOTTING_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOTTING_CACHE / "xdg"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from src.data.common import (
    CSV_COLUMNS,
    LOCAL_IMAGE_PATH_COLUMN,
    NEW_LABEL_COLUMN,
    NUM_CLASSES,
    TEST_IMAGES_PER_CLASS,
    TRAIN_IMAGES_PER_CLASS,
    VAL_IMAGES_PER_CLASS,
    is_safe_relative_posix_path,
)


SPLITS = ("train", "val", "test")


class SmokeTestError(RuntimeError):
    """Raised when the frozen dataset fails a smoke-test invariant."""


@dataclass(frozen=True)
class SmokeContract:
    """Dataset-size contract used by production and injectable unit tests.

    The command-line entry point always uses :data:`PRODUCTION_CONTRACT`.
    Tests may inject a smaller contract through :func:`run_smoke_test`, which
    exercises the same validation and decoding path without weakening the
    30,000-image production checks.
    """

    num_classes: int = NUM_CLASSES
    train_images_per_class: int = TRAIN_IMAGES_PER_CLASS
    val_images_per_class: int = VAL_IMAGES_PER_CLASS
    test_images_per_class: int = TEST_IMAGES_PER_CLASS

    def __post_init__(self) -> None:
        values = (
            self.num_classes,
            self.train_images_per_class,
            self.val_images_per_class,
            self.test_images_per_class,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        ):
            raise ValueError(
                "Smoke-contract class and per-class counts must be positive integers"
            )

    @property
    def expected_rows(self) -> dict[str, int]:
        """Return exact row counts for train, validation, and test."""

        return {
            "train": self.num_classes * self.train_images_per_class,
            "val": self.num_classes * self.val_images_per_class,
            "test": self.num_classes * self.test_images_per_class,
        }

    @property
    def expected_labels(self) -> set[int]:
        """Return the exact contiguous label set."""

        return set(range(self.num_classes))


PRODUCTION_CONTRACT = SmokeContract()


@dataclass(frozen=True)
class SampledImage:
    """Decoded image and evidence displayed in the smoke-test report."""

    split: str
    image_id: int
    new_label: int
    scientific_name: str
    local_image_path: str
    original_mode: str
    width: int
    height: int
    rgb: np.ndarray


def _load_manifest(
    path: Path,
    expected_split: str,
    contract: SmokeContract = PRODUCTION_CONTRACT,
) -> pd.DataFrame:
    """Read and validate one exact Stage 1 CSV contract."""

    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != CSV_COLUMNS:
        raise SmokeTestError(
            f"{path} schema mismatch: expected {CSV_COLUMNS}, got {list(frame.columns)}"
        )
    expected_rows = contract.expected_rows[expected_split]
    if len(frame) != expected_rows:
        raise SmokeTestError(
            f"{path} has {len(frame):,} rows; expected {expected_rows:,}"
        )
    if (frame == "").any().any():
        raise SmokeTestError(f"{path} contains empty values")
    if not frame["split"].eq(expected_split).all():
        raise SmokeTestError(f"{path} contains an incorrect split value")
    try:
        labels = pd.to_numeric(frame[NEW_LABEL_COLUMN], errors="raise")
        image_ids = pd.to_numeric(frame["image_id"], errors="raise")
    except (TypeError, ValueError) as error:
        raise SmokeTestError(f"{path} has non-numeric identifiers") from error
    if not np.equal(labels, np.floor(labels)).all():
        raise SmokeTestError(f"{path} contains a non-integer new_label")
    if not np.equal(image_ids, np.floor(image_ids)).all():
        raise SmokeTestError(f"{path} contains a non-integer image_id")
    frame[NEW_LABEL_COLUMN] = labels.astype(np.int64)
    frame["image_id"] = image_ids.astype(np.int64)
    if set(frame[NEW_LABEL_COLUMN].tolist()) != contract.expected_labels:
        raise SmokeTestError(
            f"{path} must contain every new_label from 0 to {contract.num_classes - 1}"
        )
    if frame[LOCAL_IMAGE_PATH_COLUMN].duplicated().any():
        raise SmokeTestError(f"{path} contains duplicate local_image_path values")
    return frame


def _check_materialised_paths(
    frames: dict[str, pd.DataFrame], image_root: Path
) -> dict[str, list[Path]]:
    """Check all 30,000 portable paths and materialised files."""

    if not image_root.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {image_root}")
    root = image_root.resolve()
    resolved_by_split: dict[str, list[Path]] = {}
    seen: set[str] = set()
    for split in SPLITS:
        paths: list[Path] = []
        values = frames[split][LOCAL_IMAGE_PATH_COLUMN].astype(str).tolist()
        for row_number, value in enumerate(
            tqdm(values, desc=f"checking {split} paths", unit="image"), start=2
        ):
            location = f"{split}.csv:{row_number}"
            if not is_safe_relative_posix_path(value):
                raise SmokeTestError(
                    f"{location}: unsafe local_image_path {value!r}"
                )
            relative = PurePosixPath(value)
            if relative.parts[:1] != (split,):
                raise SmokeTestError(
                    f"{location}: local_image_path must start with {split!r}"
                )
            if value in seen:
                raise SmokeTestError(
                    f"{location}: cross-split duplicate local_image_path {value!r}"
                )
            candidate = image_root.joinpath(*relative.parts)
            resolved = candidate.resolve(strict=False)
            if root not in resolved.parents:
                raise SmokeTestError(f"{location}: path escapes image root")
            if not resolved.is_file():
                raise FileNotFoundError(f"{location}: image not found: {candidate}")
            seen.add(value)
            paths.append(resolved)
        resolved_by_split[split] = paths
    return resolved_by_split


def _decode_rgb_image(
    path: Path, *, retain_pixels: bool
) -> tuple[str, int, int, np.ndarray | None]:
    """Fully decode one image through the RGB preprocessing safety path.

    Calling ``load`` on the converted image is deliberate: merely opening an
    image reads mostly header data and can miss truncated or corrupt pixel
    streams.  Alpha-bearing images are composited over white consistently with
    the handcrafted feature loader.
    """

    try:
        with Image.open(path) as opened:
            original_mode = opened.mode
            oriented = ImageOps.exif_transpose(opened)
            width, height = oriented.size
            if width <= 0 or height <= 0:
                raise SmokeTestError(f"Invalid decoded image size for {path}")
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                display_image = Image.alpha_composite(white, rgba).convert("RGB")
            else:
                display_image = oriented.convert("RGB")
            display_image.load()
            rgb = (
                np.asarray(display_image, dtype=np.uint8).copy()
                if retain_pixels
                else None
            )
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise SmokeTestError(f"Could not fully decode image {path}: {error}") from error
    return original_mode, width, height, rgb


def _verify_all_images_decodable(
    resolved_paths: dict[str, list[Path]],
) -> None:
    """Fully decode every materialised image, failing on the first bad file."""

    for split in SPLITS:
        for path in tqdm(
            resolved_paths[split],
            desc=f"decoding {split} images",
            unit="image",
        ):
            _decode_rgb_image(path, retain_pixels=False)


def _decode_sample(
    frames: dict[str, pd.DataFrame],
    resolved_paths: dict[str, list[Path]],
    *,
    seed: int,
    samples_per_split: int,
) -> dict[str, list[SampledImage]]:
    """Deterministically select and fully decode sample images."""

    if samples_per_split <= 0:
        raise ValueError("samples_per_split must be positive")
    generator = np.random.Generator(np.random.PCG64(seed))
    samples: dict[str, list[SampledImage]] = {}
    for split in SPLITS:
        if samples_per_split > len(frames[split]):
            raise ValueError(f"Cannot sample {samples_per_split} rows from {split}")
        positions = generator.choice(
            len(frames[split]), size=samples_per_split, replace=False
        )
        split_samples: list[SampledImage] = []
        for position_value in positions.tolist():
            position = int(position_value)
            row = frames[split].iloc[position]
            path = resolved_paths[split][position]
            original_mode, width, height, rgb = _decode_rgb_image(
                path, retain_pixels=True
            )
            if rgb is None:  # Defensive guard for the dataclass's strict array contract.
                raise AssertionError("Sample decoding did not retain RGB pixels")
            split_samples.append(
                SampledImage(
                    split=split,
                    image_id=int(row["image_id"]),
                    new_label=int(row[NEW_LABEL_COLUMN]),
                    scientific_name=str(row["scientific_name"]),
                    local_image_path=str(row[LOCAL_IMAGE_PATH_COLUMN]),
                    original_mode=original_mode,
                    width=width,
                    height=height,
                    rgb=rgb,
                )
            )
        samples[split] = split_samples
    return samples


def _atomic_save_figure(
    samples: dict[str, list[SampledImage]], output_path: Path, *, seed: int
) -> None:
    """Render a three-row sample grid and atomically install the PNG."""

    columns = len(samples[SPLITS[0]])
    figure, axes = plt.subplots(
        len(SPLITS), columns, figsize=(3.2 * columns, 8.5), squeeze=False
    )
    for row_index, split in enumerate(SPLITS):
        for column_index, sample in enumerate(samples[split]):
            axis = axes[row_index, column_index]
            axis.imshow(sample.rgb)
            axis.set_title(
                f"{split} · label {sample.new_label}\n"
                f"{sample.scientific_name}\n"
                f"{sample.original_mode}, {sample.width}×{sample.height}",
                fontsize=8,
            )
            axis.axis("off")
    figure.suptitle(
        f"Deterministic Stage 1 dataset sample (seed {seed})", fontsize=14
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".png"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(temporary_path, dpi=160, bbox_inches="tight")
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output_path)
    finally:
        plt.close(figure)
        temporary_path.unlink(missing_ok=True)


def _format_sample_statistics(
    split_samples: list[SampledImage],
) -> tuple[str, str, str]:
    """Return report-friendly mode, width, and height summaries."""

    modes = Counter(sample.original_mode for sample in split_samples)
    mode_text = ", ".join(f"{mode}: {count}" for mode, count in sorted(modes.items()))
    widths = np.asarray([sample.width for sample in split_samples], dtype=np.int64)
    heights = np.asarray([sample.height for sample in split_samples], dtype=np.int64)
    width_text = f"{widths.min()}/{np.median(widths):.1f}/{widths.max()}"
    height_text = f"{heights.min()}/{np.median(heights):.1f}/{heights.max()}"
    return mode_text, width_text, height_text


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 report text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _build_report(
    frames: dict[str, pd.DataFrame],
    samples: dict[str, list[SampledImage]],
    *,
    seed: int,
    image_root: Path,
    figure_path: Path,
) -> str:
    """Build a report containing only measurements made by this invocation."""

    lines = [
        "# Dataset Smoke Test Report",
        "",
        "**Status: PASS**",
        "",
        "The frozen Stage 1 CSV schema, counts, labels, paths, materialised file "
        "existence, full-image decoding, and deterministic decode sample all passed.",
        "",
        "## Contract checks",
        "",
        "| Split | Rows | Classes | Label range | Existing files | Fully decoded |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        labels = frames[split][NEW_LABEL_COLUMN]
        lines.append(
            f"| {split} | {len(frames[split]):,} | {labels.nunique():,} | "
            f"{labels.min()}–{labels.max()} | {len(frames[split]):,} | "
            f"{len(frames[split]):,} |"
        )
    lines.extend(
        [
            "",
            f"All paths were resolved safely beneath `{image_root.as_posix()}`.",
            f"All {sum(len(frame) for frame in frames.values()):,} images were fully "
            "decoded through EXIF correction and RGB conversion. Any unreadable image "
            "is fatal. The statistics below describe only the deterministic grid sample.",
            "",
            "## Decoded sample statistics",
            "",
            f"Seed: `{seed}`. Values below are based on "
            f"{len(samples['train'])} images per split. Size fields are min/median/max.",
            "",
            "| Split | Original modes | Width (px) | Height (px) |",
            "|---|---|---:|---:|",
        ]
    )
    for split in SPLITS:
        modes, widths, heights = _format_sample_statistics(samples[split])
        lines.append(f"| {split} | {modes} | {widths} | {heights} |")
    lines.extend(
        [
            "",
            "## Sampled records",
            "",
            "| Split | Image ID | New label | Scientific name | Local image path |",
            "|---|---:|---:|---|---|",
        ]
    )
    for split in SPLITS:
        for sample in samples[split]:
            name = sample.scientific_name.replace("|", "\\|")
            local_path = sample.local_image_path.replace("|", "\\|")
            lines.append(
                f"| {split} | {sample.image_id} | {sample.new_label} | "
                f"{name} | `{local_path}` |"
            )
    lines.extend(
        [
            "",
            "## Figure",
            "",
            f"The visualization grid was written to `{figure_path.as_posix()}`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_smoke_test(
    *,
    split_dir: Path,
    image_root: Path,
    figure_path: Path,
    report_path: Path,
    seed: int = 42,
    samples_per_split: int = 4,
    contract: SmokeContract = PRODUCTION_CONTRACT,
) -> None:
    """Run all smoke checks and write the grid/report only after they pass."""

    frames = {
        split: _load_manifest(split_dir / f"{split}.csv", split, contract)
        for split in SPLITS
    }
    resolved = _check_materialised_paths(frames, image_root)
    _verify_all_images_decodable(resolved)
    samples = _decode_sample(
        frames,
        resolved,
        seed=seed,
        samples_per_split=samples_per_split,
    )
    _atomic_save_figure(samples, figure_path, seed=seed)
    report = _build_report(
        frames,
        samples,
        seed=seed,
        image_root=image_root,
        figure_path=figure_path,
    )
    _atomic_write_text(report_path, report)

    print("PASS: frozen CSV schema and row counts")
    print(f"PASS: labels are exactly 0..{contract.num_classes - 1} in every split")
    print(f"PASS: all {sum(len(frame) for frame in frames.values()):,} image files exist")
    print(f"PASS: all {sum(len(frame) for frame in frames.values()):,} images fully decode")
    for split in SPLITS:
        modes, widths, heights = _format_sample_statistics(samples[split])
        print(
            f"PASS: {split} sample modes [{modes}], "
            f"width min/median/max [{widths}], height [{heights}]"
        )
    print(f"WROTE: {figure_path}")
    print(f"WROTE: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Smoke-test frozen Stage 1 manifests and materialised images."
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--image-root", type=Path, default=Path("data/processed/images")
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path("results/figures/dataset_sample_grid.png"),
    )
    parser.add_argument(
        "--report-path", type=Path, default=Path("DATASET_SMOKE_TEST_REPORT.md")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-split", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    args = build_parser().parse_args(argv)
    try:
        run_smoke_test(
            split_dir=args.split_dir,
            image_root=args.image_root,
            figure_path=args.figure_path,
            report_path=args.report_path,
            seed=args.seed,
            samples_per_split=args.samples_per_split,
        )
    except (OSError, ValueError, SmokeTestError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
