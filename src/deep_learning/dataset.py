"""Strict manifest-backed image dataset for the Stage 3 baselines.

The canonical manifests use the Stage 1 schema-v2 contract.  Validation is
performed when the dataset is constructed so malformed rows and missing files
cannot be silently skipped by data-loader workers.  Image decoding remains in
``__getitem__`` and raises a contextual :class:`DatasetImageError` on failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset

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


_EXPECTED_SOURCE_SPLIT = {
    "train": "train_mini",
    "val": "train_mini",
    "test": "val",
}


class DatasetManifestError(ValueError):
    """Raised when a Stage 3 manifest violates the schema-v2 contract."""


class DatasetImageError(RuntimeError):
    """Raised when an individual manifest image cannot be loaded or transformed."""


@dataclass(frozen=True)
class DatasetContract:
    """Expected split cardinalities.

    The default is the immutable project contract.  Supplying a smaller
    contract is useful for focused tests and disposable smoke tests; production
    callers should always use the default.
    """

    num_classes: int = NUM_CLASSES
    train_images_per_class: int = TRAIN_IMAGES_PER_CLASS
    val_images_per_class: int = VAL_IMAGES_PER_CLASS
    test_images_per_class: int = TEST_IMAGES_PER_CLASS

    def __post_init__(self) -> None:
        values = {
            "num_classes": self.num_classes,
            "train_images_per_class": self.train_images_per_class,
            "val_images_per_class": self.val_images_per_class,
            "test_images_per_class": self.test_images_per_class,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def images_per_class(self, split: str) -> int:
        """Return the exact expected count for one supported split."""

        try:
            return {
                "train": self.train_images_per_class,
                "val": self.val_images_per_class,
                "test": self.test_images_per_class,
            }[split]
        except KeyError as error:  # defensive; expected_split is checked earlier
            raise ValueError(f"Unsupported split {split!r}") from error


@dataclass(frozen=True)
class _ImageRecord:
    split: str
    source_split: str
    image_path: str
    local_image_path: str
    resolved_image_path: Path
    image_id: int
    new_label: int
    original_category_id: int
    scientific_name: str
    common_name: str

    def metadata(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "source_split": self.source_split,
            "image_path": self.image_path,
            "local_image_path": self.local_image_path,
            "resolved_image_path": str(self.resolved_image_path),
            "image_id": self.image_id,
            "original_category_id": self.original_category_id,
            "scientific_name": self.scientific_name,
            "common_name": self.common_name,
        }


def _load_manifest(csv_path: str | Path | pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if isinstance(csv_path, pd.DataFrame):
        return csv_path.copy(deep=True), "<dataframe>"

    manifest_path = Path(csv_path)
    try:
        frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    except (
        OSError,
        UnicodeError,
        ValueError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        raise DatasetManifestError(
            f"Could not read manifest {manifest_path}: {error}"
        ) from error
    return frame, str(manifest_path)


def _require_exact_integer_column(
    frame: pd.DataFrame,
    column: str,
    *,
    manifest_name: str,
) -> pd.Series:
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    limits = np.iinfo(np.int64)
    valid = (
        numeric.notna()
        & np.isfinite(numeric)
        & (numeric % 1 == 0)
        & (numeric >= limits.min)
        & (numeric <= limits.max)
    )
    if not bool(valid.all()):
        rows = [int(index) + 2 for index in frame.index[~valid][:5].tolist()]
        raise DatasetManifestError(
            f"Manifest {manifest_name} column {column!r} contains non-integer "
            f"values at CSV rows {rows}"
        )
    return numeric.astype("int64")


def _raise_bad_rows(
    condition: pd.Series,
    *,
    manifest_name: str,
    detail: str,
) -> None:
    if bool(condition.all()):
        return
    rows = [int(index) + 2 for index in condition.index[~condition][:5].tolist()]
    raise DatasetManifestError(
        f"Manifest {manifest_name} {detail}; offending CSV rows {rows}"
    )


def _resolved_local_path(image_root: Path, relative_path: str) -> Path:
    """Resolve a safe local path and reject symlink escapes from image_root."""

    candidate = image_root.joinpath(*PurePosixPath(relative_path).parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(image_root)
    except ValueError as error:
        raise DatasetManifestError(
            f"Local image path {relative_path!r} resolves outside image root "
            f"{image_root}"
        ) from error
    return resolved


class ManifestImageDataset(Dataset[tuple[Tensor, int, Mapping[str, Any]]]):
    """Load tensors from a validated Stage 1 schema-v2 split manifest.

    Parameters
    ----------
    csv_path:
        Manifest CSV path.  A DataFrame is accepted for focused tests.
    image_root:
        Root containing the materialized ``train/``, ``val/``, and ``test/``
        paths referenced by ``local_image_path``.
    transform:
        Callable converting an EXIF-oriented RGB PIL image into a tensor.
    expected_split:
        Exactly one of ``train``, ``val``, or ``test``.
    strict:
        When true (the canonical mode), enforce exact row and per-class counts.
        All schema, path, provenance, label-range, and existence checks remain
        active even when false.
    contract:
        Optional cardinality contract for miniature tests.  Omit in production.
    """

    def __init__(
        self,
        csv_path: str | Path | pd.DataFrame,
        image_root: str | Path,
        transform: Callable[[Image.Image], Tensor],
        expected_split: str,
        *,
        strict: bool = True,
        contract: DatasetContract | None = None,
    ) -> None:
        super().__init__()
        if expected_split not in _EXPECTED_SOURCE_SPLIT:
            raise ValueError(
                "expected_split must be one of 'train', 'val', or 'test'; "
                f"got {expected_split!r}"
            )
        if not callable(transform):
            raise TypeError("transform must be callable and return a torch.Tensor")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a bool")

        self.expected_split = expected_split
        self.transform = transform
        self.contract = contract or DatasetContract()
        self.strict = strict
        self.image_root = Path(image_root).expanduser().resolve(strict=False)
        if not self.image_root.is_dir():
            raise DatasetManifestError(
                f"Image root does not exist or is not a directory: {self.image_root}"
            )

        frame, manifest_name = _load_manifest(csv_path)
        self.manifest_name = manifest_name
        self._records = self._validate_and_build_records(frame)

    def _validate_and_build_records(self, frame: pd.DataFrame) -> list[_ImageRecord]:
        manifest_name = self.manifest_name
        if list(frame.columns) != CSV_COLUMNS:
            raise DatasetManifestError(
                f"Manifest {manifest_name} has invalid schema-v2 columns: "
                f"expected {CSV_COLUMNS}, got {list(frame.columns)}"
            )
        if frame.empty:
            raise DatasetManifestError(f"Manifest {manifest_name} is empty")

        for column in ("image_id", NEW_LABEL_COLUMN, "original_category_id"):
            frame[column] = _require_exact_integer_column(
                frame, column, manifest_name=manifest_name
            )

        required_nonempty = (
            "split",
            "source_split",
            "image_path",
            LOCAL_IMAGE_PATH_COLUMN,
            "scientific_name",
        )
        for column in required_nonempty:
            nonempty = frame[column].map(
                lambda value: isinstance(value, str) and bool(value.strip())
            )
            _raise_bad_rows(
                nonempty,
                manifest_name=manifest_name,
                detail=f"has empty required {column!r} values",
            )

        expected_source = _EXPECTED_SOURCE_SPLIT[self.expected_split]
        _raise_bad_rows(
            frame["split"] == self.expected_split,
            manifest_name=manifest_name,
            detail=f"contains rows outside expected split {self.expected_split!r}",
        )
        _raise_bad_rows(
            frame["source_split"] == expected_source,
            manifest_name=manifest_name,
            detail=(
                f"contains rows whose source_split is not {expected_source!r} "
                f"for {self.expected_split!r}"
            ),
        )

        safe_source_paths = frame["image_path"].map(is_safe_relative_posix_path)
        _raise_bad_rows(
            safe_source_paths,
            manifest_name=manifest_name,
            detail="contains unsafe image_path values",
        )
        safe_local_paths = frame[LOCAL_IMAGE_PATH_COLUMN].map(
            is_safe_relative_posix_path
        )
        _raise_bad_rows(
            safe_local_paths,
            manifest_name=manifest_name,
            detail="contains unsafe local_image_path values",
        )

        source_prefix_matches = frame["image_path"].map(
            lambda value: PurePosixPath(value).parts[0] == expected_source
        )
        _raise_bad_rows(
            source_prefix_matches,
            manifest_name=manifest_name,
            detail=f"has image_path values outside {expected_source!r}",
        )
        expected_local_paths = frame["image_path"].map(
            lambda value: PurePosixPath(
                self.expected_split, PurePosixPath(value).name
            ).as_posix()
        )
        _raise_bad_rows(
            frame[LOCAL_IMAGE_PATH_COLUMN] == expected_local_paths,
            manifest_name=manifest_name,
            detail="has local_image_path values that do not match split/filename",
        )

        labels = frame[NEW_LABEL_COLUMN]
        label_in_range = (labels >= 0) & (labels < self.contract.num_classes)
        _raise_bad_rows(
            label_in_range,
            manifest_name=manifest_name,
            detail=(
                "contains new_label values outside "
                f"0..{self.contract.num_classes - 1}"
            ),
        )

        duplicate_checks = {
            "image_id": frame["image_id"].duplicated(keep=False),
            "image_path": frame["image_path"].duplicated(keep=False),
            LOCAL_IMAGE_PATH_COLUMN: frame[LOCAL_IMAGE_PATH_COLUMN].duplicated(
                keep=False
            ),
        }
        for name, duplicate in duplicate_checks.items():
            if bool(duplicate.any()):
                rows = [int(index) + 2 for index in frame.index[duplicate][:5].tolist()]
                raise DatasetManifestError(
                    f"Manifest {manifest_name} contains duplicate {name} values "
                    f"at CSV rows {rows}"
                )

        # A label must carry one stable category and pair of display names.
        mapping_columns = [
            "original_category_id",
            "scientific_name",
            "common_name",
        ]
        mapping_sizes = frame.groupby(NEW_LABEL_COLUMN, sort=False)[
            mapping_columns
        ].nunique(dropna=False)
        inconsistent = (mapping_sizes > 1).any(axis=1)
        if bool(inconsistent.any()):
            bad_labels = [int(value) for value in inconsistent[inconsistent].index[:5]]
            raise DatasetManifestError(
                f"Manifest {manifest_name} has inconsistent category/name mapping "
                f"for labels {bad_labels}"
            )
        category_to_label = frame.groupby("original_category_id")[
            NEW_LABEL_COLUMN
        ].nunique()
        if bool((category_to_label > 1).any()):
            bad_categories = [
                int(value) for value in category_to_label[category_to_label > 1].index[:5]
            ]
            raise DatasetManifestError(
                f"Manifest {manifest_name} maps original categories "
                f"{bad_categories} to more than one new_label"
            )

        if self.strict:
            expected_labels = set(range(self.contract.num_classes))
            actual_labels = set(int(value) for value in labels.tolist())
            if actual_labels != expected_labels:
                missing = sorted(expected_labels - actual_labels)[:5]
                unexpected = sorted(actual_labels - expected_labels)[:5]
                raise DatasetManifestError(
                    f"Manifest {manifest_name} label set is not contiguous "
                    f"0..{self.contract.num_classes - 1}; missing={missing}, "
                    f"unexpected={unexpected}"
                )
            expected_per_class = self.contract.images_per_class(self.expected_split)
            expected_rows = self.contract.num_classes * expected_per_class
            if len(frame) != expected_rows:
                raise DatasetManifestError(
                    f"Manifest {manifest_name} has {len(frame)} rows; expected "
                    f"{expected_rows} for split {self.expected_split!r}"
                )
            counts = frame.groupby(NEW_LABEL_COLUMN).size().reindex(
                range(self.contract.num_classes), fill_value=0
            )
            bad_counts = counts[counts != expected_per_class]
            if not bad_counts.empty:
                examples = {
                    int(label): int(count)
                    for label, count in bad_counts.iloc[:5].items()
                }
                raise DatasetManifestError(
                    f"Manifest {manifest_name} must have exactly "
                    f"{expected_per_class} images per class; bad counts {examples}"
                )

        records: list[_ImageRecord] = []
        missing: list[tuple[int, str]] = []
        for row_index, row in frame.iterrows():
            relative_path = str(row[LOCAL_IMAGE_PATH_COLUMN])
            try:
                resolved = _resolved_local_path(self.image_root, relative_path)
            except DatasetManifestError as error:
                raise DatasetManifestError(
                    f"Manifest {manifest_name} CSV row {int(row_index) + 2}: {error}"
                ) from error
            if not resolved.is_file():
                missing.append((int(row_index) + 2, relative_path))
                if len(missing) >= 5:
                    break
            records.append(
                _ImageRecord(
                    split=str(row["split"]),
                    source_split=str(row["source_split"]),
                    image_path=str(row["image_path"]),
                    local_image_path=relative_path,
                    resolved_image_path=resolved,
                    image_id=int(row["image_id"]),
                    new_label=int(row[NEW_LABEL_COLUMN]),
                    original_category_id=int(row["original_category_id"]),
                    scientific_name=str(row["scientific_name"]),
                    common_name=str(row["common_name"]),
                )
            )
        if missing:
            detail = ", ".join(f"row {row}: {path}" for row, path in missing)
            raise DatasetManifestError(
                f"Manifest {manifest_name} references missing image files under "
                f"{self.image_root}: {detail}"
            )
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Tensor, int, Mapping[str, Any]]:
        record = self._records[index]
        context = (
            f"manifest={self.manifest_name}, row_index={index}, "
            f"image_id={record.image_id}, path={record.local_image_path!r}"
        )
        try:
            with Image.open(record.resolved_image_path) as opened:
                oriented = ImageOps.exif_transpose(opened)
                rgb = oriented.convert("RGB")
                # Detach from Pillow's lazy file decoder before the file closes.
                image = rgb.copy()
        except (OSError, ValueError, UnidentifiedImageError) as error:
            raise DatasetImageError(
                f"Could not decode Stage 3 image ({context}): {error}"
            ) from error

        try:
            tensor = self.transform(image)
        except Exception as error:
            raise DatasetImageError(
                f"Could not transform Stage 3 image ({context}): {error}"
            ) from error
        if not isinstance(tensor, torch.Tensor):
            raise DatasetImageError(
                f"Transform did not return a torch.Tensor ({context}); "
                f"got {type(tensor).__name__}"
            )
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise DatasetImageError(
                f"Transform returned tensor shape {tuple(tensor.shape)} ({context}); "
                "expected (3, height, width)"
            )
        return tensor, record.new_label, record.metadata()


# A descriptive compatibility alias for callers that prefer the stage name.
DeepLearningDataset = ManifestImageDataset


__all__ = [
    "DatasetContract",
    "DatasetImageError",
    "DatasetManifestError",
    "DeepLearningDataset",
    "ManifestImageDataset",
]
