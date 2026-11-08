"""Extract deterministic handcrafted caches without premature test access.

Train and validation are the default pre-selection operation.  The held-out
test manifest and images are not even opened until an immutable Stage 2
selection manifest has been validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd
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
    sha256_file,
)
from src.traditional.features import (
    DEFAULT_IMAGE_SIZE,
    ImageFeatureError,
    extract_features,
    feature_configuration,
)


SPLITS = ("train", "val", "test")
PRE_SELECTION_SPLITS = ("train", "val")
EXPECTED_ROWS = {
    "train": NUM_CLASSES * TRAIN_IMAGES_PER_CLASS,
    "val": NUM_CLASSES * VAL_IMAGES_PER_CLASS,
    "test": NUM_CLASSES * TEST_IMAGES_PER_CLASS,
}
EXPECTED_SOURCE = {"train": "train_mini", "val": "train_mini", "test": "val"}
OUTPUT_NAMES = {
    split: f"traditional_{split}_features.npz" for split in SPLITS
}
CONFIG_NAME = "traditional_feature_config.json"
TEST_CONFIG_NAME = "traditional_test_feature_config.json"
CACHE_SCHEMA_VERSION = 3
DEFAULT_SELECTION_MANIFEST = Path("results/metrics/stage2_selection_manifest.json")
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)
IMPLEMENTATION_SOURCE_PATHS = {
    "src/traditional/extract_features.py": Path(__file__).resolve(),
    "src/traditional/features.py": Path(__file__).with_name("features.py").resolve(),
}

NPZ_FIELDS = (
    "features",
    "new_labels",
    "image_ids",
    "splits",
    "source_splits",
    "image_paths",
    "local_image_paths",
    "original_category_ids",
    "scientific_names",
    "common_names",
    "source_file_sizes",
    "source_file_mtime_ns",
    "source_file_sha256",
    "manifest_sha256",
    "manifest_file_name",
)


class FeatureCacheError(RuntimeError):
    """Raised when a manifest or feature cache violates its contract."""


def _read_manifest(manifest_path: Path, expected_split: str) -> pd.DataFrame:
    """Load one Stage 1 manifest and enforce the frozen schema and counts."""

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    try:
        frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as error:
        raise FeatureCacheError(f"Could not read {manifest_path}: {error}") from error

    if list(frame.columns) != CSV_COLUMNS:
        raise FeatureCacheError(
            f"Unexpected schema in {manifest_path}: expected {CSV_COLUMNS}, "
            f"got {list(frame.columns)}"
        )
    expected_rows = EXPECTED_ROWS[expected_split]
    if len(frame) != expected_rows:
        raise FeatureCacheError(
            f"{manifest_path} has {len(frame):,} rows; expected {expected_rows:,}"
        )
    if (frame == "").any().any():
        columns = frame.columns[(frame == "").any()].tolist()
        raise FeatureCacheError(f"Empty manifest values in columns: {columns}")
    if not frame["split"].eq(expected_split).all():
        raise FeatureCacheError(f"{manifest_path} contains an incorrect split value")
    expected_source = EXPECTED_SOURCE[expected_split]
    if not frame["source_split"].eq(expected_source).all():
        raise FeatureCacheError(
            f"{manifest_path} must contain only source_split={expected_source!r}"
        )

    for column in ("image_id", NEW_LABEL_COLUMN, "original_category_id"):
        try:
            numeric = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise FeatureCacheError(
                f"{manifest_path}: {column} must contain integers"
            ) from error
        if not np.equal(numeric, np.floor(numeric)).all():
            raise FeatureCacheError(f"{manifest_path}: {column} contains non-integers")
        frame[column] = numeric.astype(np.int64)

    expected_labels = set(range(NUM_CLASSES))
    labels = set(frame[NEW_LABEL_COLUMN].astype(int).tolist())
    if labels != expected_labels:
        raise FeatureCacheError(
            f"{manifest_path}: new_label set must be exactly 0..{NUM_CLASSES - 1}"
        )
    expected_per_class = {
        "train": TRAIN_IMAGES_PER_CLASS,
        "val": VAL_IMAGES_PER_CLASS,
        "test": TEST_IMAGES_PER_CLASS,
    }[expected_split]
    per_class = frame[NEW_LABEL_COLUMN].value_counts().sort_index()
    if not per_class.eq(expected_per_class).all():
        raise FeatureCacheError(
            f"{manifest_path}: every class must have {expected_per_class} rows"
        )
    if frame["image_id"].duplicated().any():
        raise FeatureCacheError(f"{manifest_path}: duplicate image_id values")
    if frame[LOCAL_IMAGE_PATH_COLUMN].duplicated().any():
        raise FeatureCacheError(f"{manifest_path}: duplicate local_image_path values")
    if frame["image_path"].duplicated().any():
        raise FeatureCacheError(f"{manifest_path}: duplicate image_path values")
    return frame


def _resolve_image_paths(
    frame: pd.DataFrame,
    image_root: Path,
    expected_split: str,
) -> list[Path]:
    """Resolve and validate manifest paths without allowing root traversal."""

    if not image_root.is_dir():
        raise NotADirectoryError(f"Image root is not a directory: {image_root}")
    root = image_root.resolve()
    paths: list[Path] = []
    for row_number, local_value in enumerate(
        frame[LOCAL_IMAGE_PATH_COLUMN].astype(str), start=2
    ):
        location = f"{expected_split}.csv:{row_number}"
        if not is_safe_relative_posix_path(local_value):
            raise FeatureCacheError(
                f"{location}: unsafe local_image_path {local_value!r}"
            )
        relative = PurePosixPath(local_value)
        if relative.parts[:1] != (expected_split,):
            raise FeatureCacheError(
                f"{location}: local_image_path must start with {expected_split!r}"
            )
        candidate = image_root.joinpath(*relative.parts)
        resolved = candidate.resolve(strict=False)
        if root not in resolved.parents:
            raise FeatureCacheError(f"{location}: path escapes image root")
        if not resolved.is_file():
            raise FileNotFoundError(f"{location}: image file not found: {candidate}")
        paths.append(resolved)
    return paths


def _feature_worker(task: tuple[str, int]) -> tuple[np.ndarray, int, int, str]:
    """Process-pool entry point with path-aware failure messages."""

    path_value, image_size = task
    try:
        path = Path(path_value)
        before = path.stat()
        source_digest = sha256_file(path)
        feature = extract_features(path, image_size)
        after = path.stat()
        before_identity = (before.st_size, before.st_mtime_ns, before.st_ino)
        after_identity = (after.st_size, after.st_mtime_ns, after.st_ino)
        if before_identity != after_identity:
            raise ImageFeatureError(f"Source image changed while being read: {path}")
        return feature, after.st_size, after.st_mtime_ns, source_digest
    except Exception as error:
        raise ImageFeatureError(
            f"Feature extraction failed for {path_value}: {error}"
        ) from error


def _extract_matrix(
    image_paths: Iterable[Path],
    *,
    image_size: int,
    workers: int,
    description: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract an order-preserving feature matrix, serially or in processes."""

    paths = list(image_paths)
    if workers <= 0:
        raise ValueError("workers must be at least 1")
    dimension = int(feature_configuration(image_size)["feature_dim"])
    matrix = np.empty((len(paths), dimension), dtype=np.float32)
    source_sizes = np.empty(len(paths), dtype=np.int64)
    source_mtime_ns = np.empty(len(paths), dtype=np.int64)
    source_sha256 = np.empty(len(paths), dtype="<U64")
    tasks = [(str(path), image_size) for path in paths]

    if workers == 1:
        iterator: Iterable[tuple[np.ndarray, int, int, str]] = (
            _feature_worker(task)
            for task in tqdm(tasks, desc=description, unit="image")
        )
        for index, result in enumerate(iterator):
            _store_feature(
                matrix,
                source_sizes,
                source_mtime_ns,
                source_sha256,
                index,
                result,
                paths[index],
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_feature_worker, tasks, chunksize=16)
            for index, result in enumerate(
                tqdm(results, total=len(tasks), desc=description, unit="image")
            ):
                _store_feature(
                    matrix,
                    source_sizes,
                    source_mtime_ns,
                    source_sha256,
                    index,
                    result,
                    paths[index],
                )

    if not np.isfinite(matrix).all():
        raise FeatureCacheError(f"{description}: feature matrix contains non-finite values")
    return matrix, source_sizes, source_mtime_ns, source_sha256


def _store_feature(
    matrix: np.ndarray,
    source_sizes: np.ndarray,
    source_mtime_ns: np.ndarray,
    source_sha256: np.ndarray,
    index: int,
    result: tuple[np.ndarray, int, int, str],
    image_path: Path,
) -> None:
    """Validate one worker result before inserting it into the matrix."""

    feature, source_size, source_mtime, source_digest = result
    if feature.shape != (matrix.shape[1],):
        raise FeatureCacheError(
            f"Unexpected feature shape for {image_path}: {feature.shape}; "
            f"expected {(matrix.shape[1],)}"
        )
    if feature.dtype != np.float32:
        feature = feature.astype(np.float32)
    if source_size < 0 or source_mtime < 0:
        raise FeatureCacheError(f"Invalid source stat values for {image_path}")
    if len(source_digest) != 64 or any(
        character not in "0123456789abcdef" for character in source_digest
    ):
        raise FeatureCacheError(f"Invalid source SHA-256 for {image_path}")
    matrix[index] = feature
    source_sizes[index] = source_size
    source_mtime_ns[index] = source_mtime
    source_sha256[index] = source_digest


def _unicode_array(values: pd.Series) -> np.ndarray:
    """Return a non-object Unicode array that can load with allow_pickle=False."""

    return np.asarray(values.astype(str).tolist(), dtype=np.str_)


def _npz_payload(
    frame: pd.DataFrame,
    features: np.ndarray,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    source_sizes: np.ndarray,
    source_mtime_ns: np.ndarray,
    source_sha256: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the complete portable cache payload in manifest row order."""

    payload = {
        "features": np.asarray(features, dtype=np.float32),
        "new_labels": frame[NEW_LABEL_COLUMN].to_numpy(dtype=np.int64),
        "image_ids": frame["image_id"].to_numpy(dtype=np.int64),
        "splits": _unicode_array(frame["split"]),
        "source_splits": _unicode_array(frame["source_split"]),
        "image_paths": _unicode_array(frame["image_path"]),
        "local_image_paths": _unicode_array(frame[LOCAL_IMAGE_PATH_COLUMN]),
        "original_category_ids": frame["original_category_id"].to_numpy(
            dtype=np.int64
        ),
        "scientific_names": _unicode_array(frame["scientific_name"]),
        "common_names": _unicode_array(frame["common_name"]),
        "source_file_sizes": np.asarray(source_sizes, dtype=np.int64),
        "source_file_mtime_ns": np.asarray(source_mtime_ns, dtype=np.int64),
        "source_file_sha256": np.asarray(source_sha256, dtype="<U64"),
        "manifest_sha256": np.asarray(manifest_sha256, dtype=np.str_),
        "manifest_file_name": np.asarray(manifest_path.name, dtype=np.str_),
    }
    if tuple(payload) != NPZ_FIELDS:
        raise AssertionError("NPZ payload field order does not match its schema")
    row_count = len(frame)
    for field_name, value in payload.items():
        if field_name in {"manifest_sha256", "manifest_file_name"}:
            if value.ndim != 0:
                raise AssertionError(f"{field_name} must be a scalar")
        elif value.shape[0] != row_count:
            raise AssertionError(f"{field_name} does not preserve manifest row count")
        if value.dtype == object:
            raise AssertionError(f"{field_name} must not require pickled objects")
    return payload


def _atomic_save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Write a compressed NPZ beside its destination and atomically replace it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_save_json(path: Path, value: dict[str, Any]) -> None:
    """Write deterministic JSON and atomically replace its destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _package_versions() -> dict[str, str]:
    """Return versions that materially affect feature extraction."""

    packages = ("numpy", "pandas", "Pillow", "scikit-image")
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _implementation_fingerprint() -> dict[str, Any]:
    """Fingerprint source files that define extraction and feature semantics.

    Package/configuration fingerprints alone cannot detect a code-only change.
    Schema v2 therefore treats any relevant source-byte change as a stale cache
    that must be rebuilt explicitly with ``--force``.
    """

    file_hashes = {
        name: sha256_file(path)
        for name, path in sorted(IMPLEMENTATION_SOURCE_PATHS.items())
    }
    canonical = "".join(
        f"{name}:{digest}\n" for name, digest in sorted(file_hashes.items())
    ).encode("utf-8")
    return {
        "algorithm": "SHA-256",
        "combined_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": file_hashes,
    }


def _load_stage1_provenance(
    split_dir: Path,
    manifest_hashes: dict[str, str],
) -> dict[str, Any]:
    """Check the Stage 1 summary and return portable provenance fields."""

    summary_path = split_dir / "split_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Stage 1 split summary not found: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FeatureCacheError(f"Could not read {summary_path}: {error}") from error
    if summary.get("schema_version") != 2:
        raise FeatureCacheError("Stage 1 split_summary.json must use schema version 2")
    if summary.get("csv_columns") != CSV_COLUMNS:
        raise FeatureCacheError("Stage 1 summary CSV schema does not match src.data.common")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FeatureCacheError("Stage 1 summary has no artifacts mapping")
    for split, digest in manifest_hashes.items():
        recorded = artifacts.get(f"{split}.csv", {}).get("sha256")
        if recorded != digest:
            raise FeatureCacheError(
                f"Stage 1 summary hash mismatch for {split}.csv: "
                f"recorded {recorded!r}, actual {digest!r}"
            )
    return {
        "schema_version": int(summary["schema_version"]),
        "seed": int(summary["seed"]),
        "split_summary_file_name": summary_path.name,
        "split_summary_sha256": sha256_file(summary_path),
        "frozen_artifact_hashes": {
            name: details["sha256"] for name, details in sorted(artifacts.items())
        },
    }


def _validate_npz_contract(
    path: Path,
    *,
    expected_split: str,
    expected_rows: int,
    expected_dimension: int,
    expected_manifest_hash: str,
    source_paths: list[Path] | None = None,
) -> tuple[bool, str]:
    """Load an NPZ without pickle and validate its complete schema and shapes."""

    try:
        with np.load(path, allow_pickle=False) as archive:
            if tuple(archive.files) != NPZ_FIELDS:
                return False, f"{path.name} has unexpected NPZ fields"
            features = archive["features"]
            labels = archive["new_labels"]
            if features.dtype != np.float32:
                return False, f"{path.name} features are not float32"
            if features.shape != (expected_rows, expected_dimension):
                return False, f"{path.name} has feature shape {features.shape}"
            if not np.isfinite(features).all():
                return False, f"{path.name} contains non-finite features"
            if labels.dtype != np.int64 or labels.shape != (expected_rows,):
                return False, f"{path.name} has invalid new_labels"
            if set(labels.tolist()) != set(range(NUM_CLASSES)):
                return False, f"{path.name} labels are not exactly 0..{NUM_CLASSES - 1}"

            integer_fields = (
                "image_ids",
                "original_category_ids",
                "source_file_sizes",
                "source_file_mtime_ns",
            )
            text_fields = (
                "splits",
                "source_splits",
                "image_paths",
                "local_image_paths",
                "scientific_names",
                "common_names",
                "source_file_sha256",
            )
            for field_name in integer_fields:
                value = archive[field_name]
                if value.dtype != np.int64 or value.shape != (expected_rows,):
                    return False, f"{path.name} has invalid {field_name}"
            for field_name in text_fields:
                value = archive[field_name]
                if value.dtype.kind != "U" or value.shape != (expected_rows,):
                    return False, f"{path.name} has invalid {field_name}"
            if not np.all(archive["splits"] == expected_split):
                return False, f"{path.name} contains an incorrect split marker"
            if len(np.unique(archive["image_ids"])) != expected_rows:
                return False, f"{path.name} contains duplicate image IDs"
            if len(np.unique(archive["local_image_paths"])) != expected_rows:
                return False, f"{path.name} contains duplicate local image paths"
            if np.any(archive["source_file_sizes"] <= 0):
                return False, f"{path.name} contains non-positive source sizes"
            if np.any(archive["source_file_mtime_ns"] <= 0):
                return False, f"{path.name} contains invalid source mtimes"
            source_hashes = archive["source_file_sha256"].tolist()
            for digest in source_hashes:
                if len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    return False, f"{path.name} contains an invalid source SHA-256"
            if source_paths is not None:
                if len(source_paths) != expected_rows:
                    return False, f"{path.name} source path count is inconsistent"
                recorded_sizes = archive["source_file_sizes"]
                recorded_mtimes = archive["source_file_mtime_ns"]
                for index, source_path in enumerate(source_paths):
                    try:
                        current = source_path.stat()
                    except OSError as error:
                        return False, f"could not stat source image {source_path}: {error}"
                    if (
                        current.st_size != int(recorded_sizes[index])
                        or current.st_mtime_ns != int(recorded_mtimes[index])
                    ):
                        try:
                            current_digest = sha256_file(source_path)
                        except OSError as error:
                            return False, f"could not hash source image {source_path}: {error}"
                        if current_digest != source_hashes[index]:
                            return False, f"source image changed: {source_path}"
            manifest_digest = archive["manifest_sha256"]
            manifest_name = archive["manifest_file_name"]
            if manifest_digest.ndim != 0 or str(manifest_digest.item()) != expected_manifest_hash:
                return False, f"{path.name} has an incorrect manifest SHA-256"
            if manifest_name.ndim != 0 or str(manifest_name.item()) != f"{expected_split}.csv":
                return False, f"{path.name} has an incorrect manifest filename"
    except Exception as error:
        return False, f"could not safely load {path.name}: {error}"
    return True, "NPZ schema, arrays, labels, and provenance are valid"


def _cache_is_valid(
    config_path: Path,
    output_paths: dict[str, Path],
    *,
    image_size: int,
    manifest_hashes: dict[str, str],
    manifest_rows: dict[str, int],
    stage1_summary_hash: str,
    source_paths: dict[str, list[Path]],
) -> tuple[bool, str]:
    """Validate cache provenance and byte hashes without loading feature matrices."""

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return False, f"could not read feature config: {error}"
    expected_feature = feature_configuration(image_size)
    for key, expected_value in expected_feature.items():
        if config.get(key) != expected_value:
            return False, f"feature setting {key!r} changed"
    if config.get("schema_version") != CACHE_SCHEMA_VERSION:
        return False, "cache schema version changed"
    if config.get("stage1", {}).get("split_summary_sha256") != stage1_summary_hash:
        return False, "Stage 1 split summary changed"
    if config.get("package_versions") != _package_versions():
        return False, "feature extraction package versions changed"
    if config.get("implementation_fingerprint") != _implementation_fingerprint():
        return False, "feature extraction implementation changed"
    split_artifacts = config.get("split_artifacts", {})
    for split in output_paths:
        output_path = output_paths[split]
        if not output_path.is_file():
            return False, f"missing {output_path.name}"
        entry = split_artifacts.get(split, {})
        manifest = entry.get("manifest", {})
        feature_file = entry.get("feature_file", {})
        if manifest.get("sha256") != manifest_hashes[split]:
            return False, f"{split} manifest hash changed"
        if manifest.get("rows") != manifest_rows[split]:
            return False, f"{split} manifest row count changed"
        if feature_file.get("file_name") != output_path.name:
            return False, f"{split} cache filename changed"
        if feature_file.get("bytes") != output_path.stat().st_size:
            return False, f"{split} cache size changed"
        if feature_file.get("sha256") != sha256_file(output_path):
            return False, f"{split} cache hash changed"
        npz_valid, npz_reason = _validate_npz_contract(
            output_path,
            expected_split=split,
            expected_rows=manifest_rows[split],
            expected_dimension=int(expected_feature["feature_dim"]),
            expected_manifest_hash=manifest_hashes[split],
            source_paths=source_paths[split],
        )
        if not npz_valid:
            return False, npz_reason
    return True, "all settings and hashes match"


def extract_all_splits(
    *,
    split_dir: Path,
    image_root: Path,
    output_dir: Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    splits: Iterable[str] = PRE_SELECTION_SPLITS,
    selection_manifest: Path = DEFAULT_SELECTION_MANIFEST,
) -> dict[str, Any]:
    """Extract requested caches, gating the test split before any test read."""

    feature_contract = feature_configuration(image_size)
    implementation_fingerprint = _implementation_fingerprint()
    requested = tuple(str(split).strip().lower() for split in splits)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("--splits must contain unique split names")
    unknown = sorted(set(requested) - set(SPLITS))
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}")
    if "test" in requested and requested != ("test",):
        raise ValueError("test extraction must run separately with --splits test")

    selection_binding: dict[str, str] | None = None
    method_binding: dict[str, str] | None = None
    if requested == ("test",):
        # This block must remain before creation/resolution of any test manifest,
        # image path, cache path, or source-image stat.
        manifest_path = selection_manifest.resolve()
        try:
            from src.traditional.evaluate_traditional import _prevalidate_selection

            selection, selection_hash, _ = _prevalidate_selection(
                manifest_path,
                model_path=None,
                split_dir=split_dir,
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise FeatureCacheError(
                "A valid immutable Stage 2 selection manifest is required before "
                f"test feature extraction: {error}"
            ) from error
        base_config = output_dir / CONFIG_NAME
        if not base_config.is_file():
            raise FeatureCacheError(
                f"Pre-selection handcrafted configuration is missing: {base_config}"
            )
        base_hash = sha256_file(base_config)
        models = selection.get("models")
        if not isinstance(models, dict) or not any(
            isinstance(entry, dict)
            and entry.get("feature_config_sha256") == base_hash
            for entry in models.values()
        ):
            raise FeatureCacheError(
                "Selection manifest does not bind the current handcrafted config"
            )
        base_document = json.loads(base_config.read_text(encoding="utf-8"))
        for key, expected in feature_contract.items():
            if base_document.get(key) != expected:
                raise FeatureCacheError(
                    f"Pre-selection handcrafted setting {key!r} changed"
                )
        selection_binding = {
            "path": manifest_path.as_posix(),
            "sha256": selection_hash,
        }
        method_binding = {
            "path": base_config.resolve().as_posix(),
            "sha256": base_hash,
        }
    if workers <= 0:
        raise ValueError("--workers must be at least 1")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

    frames: dict[str, pd.DataFrame] = {}
    manifest_paths = {split: split_dir / f"{split}.csv" for split in requested}
    manifest_hashes: dict[str, str] = {}
    for split in requested:
        frames[split] = _read_manifest(manifest_paths[split], split)
        manifest_hashes[split] = sha256_file(manifest_paths[split])
    manifest_rows = {split: len(frame) for split, frame in frames.items()}
    stage1 = _load_stage1_provenance(split_dir, manifest_hashes)

    resolved_paths: dict[str, list[Path]] = {}
    all_local_paths: set[str] = set()
    for split in requested:
        local_values = set(frames[split][LOCAL_IMAGE_PATH_COLUMN].astype(str))
        overlap = all_local_paths.intersection(local_values)
        if overlap:
            example = sorted(overlap)[0]
            raise FeatureCacheError(f"Cross-split local_image_path duplicate: {example}")
        all_local_paths.update(local_values)
        resolved_paths[split] = _resolve_image_paths(frames[split], image_root, split)

    output_paths = {split: output_dir / OUTPUT_NAMES[split] for split in requested}
    config_path = output_dir / (
        TEST_CONFIG_NAME if requested == ("test",) else CONFIG_NAME
    )
    expected_paths = [*output_paths.values(), config_path]
    existing = [path for path in expected_paths if path.exists()]
    if existing and not force:
        if len(existing) != len(expected_paths):
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"Partial feature cache exists ({names}); use --force to rebuild it"
            )
        valid, reason = _cache_is_valid(
            config_path,
            output_paths,
            image_size=image_size,
            manifest_hashes=manifest_hashes,
            manifest_rows=manifest_rows,
            stage1_summary_hash=stage1["split_summary_sha256"],
            source_paths=resolved_paths,
        )
        if not valid:
            raise FileExistsError(
                f"Existing feature cache is stale ({reason}); use --force to rebuild it"
            )
        print(f"CACHE HIT: {config_path} ({reason})")
        return json.loads(config_path.read_text(encoding="utf-8"))

    for path in existing:
        if path.is_dir():
            raise IsADirectoryError(f"Expected an artifact file, found directory: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_artifacts: dict[str, dict[str, Any]] = {}
    artifacts_alias: dict[str, dict[str, Any]] = {}
    for split in requested:
        print(
            f"Extracting {split}: {len(frames[split]):,} images, "
            f"{feature_contract['feature_dim']:,} features, {workers} worker(s)"
        )
        started = time.perf_counter()
        matrix, source_sizes, source_mtime_ns, source_sha256 = _extract_matrix(
            resolved_paths[split],
            image_size=image_size,
            workers=workers,
            description=f"{split} features",
        )
        extraction_seconds = time.perf_counter() - started
        if sha256_file(manifest_paths[split]) != manifest_hashes[split]:
            raise FeatureCacheError(
                f"{manifest_paths[split]} changed during extraction; cache not written"
            )
        payload = _npz_payload(
            frames[split],
            matrix,
            manifest_path=manifest_paths[split],
            manifest_sha256=manifest_hashes[split],
            source_sizes=source_sizes,
            source_mtime_ns=source_mtime_ns,
            source_sha256=source_sha256,
        )
        output_path = output_paths[split]
        _atomic_save_npz(output_path, payload)
        feature_digest = sha256_file(output_path)
        feature_bytes = output_path.stat().st_size
        split_artifacts[split] = {
            "manifest": {
                "file_name": manifest_paths[split].name,
                "sha256": manifest_hashes[split],
                "rows": len(frames[split]),
            },
            "feature_file": {
                "file_name": output_path.name,
                "sha256": feature_digest,
                "bytes": feature_bytes,
            },
            "extraction_seconds": round(extraction_seconds, 6),
        }
        artifacts_alias[split] = {
            "path": output_path.name,
            "sha256": feature_digest,
            "bytes": feature_bytes,
            "rows": len(frames[split]),
        }
        print(
            f"WROTE: {output_path} ({feature_bytes:,} bytes, "
            f"{extraction_seconds:.2f} s extraction)"
        )

    current_summary_hash = sha256_file(split_dir / "split_summary.json")
    if current_summary_hash != stage1["split_summary_sha256"]:
        raise FeatureCacheError(
            "Stage 1 split_summary.json changed during extraction; rerun with --force"
        )
    for split in requested:
        if sha256_file(manifest_paths[split]) != manifest_hashes[split]:
            raise FeatureCacheError(
                f"{manifest_paths[split]} changed during extraction; rerun with --force"
            )
    if _implementation_fingerprint() != implementation_fingerprint:
        raise FeatureCacheError(
            "Feature extraction implementation changed during extraction; "
            "rerun with --force"
        )

    config: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        **feature_contract,
        "npz_fields": list(NPZ_FIELDS),
        "npz_storage": {
            "compression": "numpy savez_compressed",
            "features_dtype": "float32",
            "integer_metadata_dtype": "int64",
            "text_metadata_dtype": "numpy unicode",
            "source_integrity": "size, mtime_ns, and SHA-256 captured while reading",
            "allow_pickle_required": False,
            "row_order": "unchanged from each Stage 1 CSV manifest",
        },
        "workers": workers,
        "package_versions": _package_versions(),
        "implementation_fingerprint": implementation_fingerprint,
        "stage1": stage1,
        "split_artifacts": split_artifacts,
        "artifacts": artifacts_alias,
    }
    if selection_binding is not None and method_binding is not None:
        config["selection_manifest"] = selection_binding
        config["method_config"] = method_binding
    _atomic_save_json(config_path, config)
    print(f"WROTE: {config_path}")
    return config


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Extract cached colour, multi-scale texture, and HOG features "
            "without accessing test before Stage 2 selection."
        )
    )
    parser.add_argument(
        "--splits",
        default=",".join(PRE_SELECTION_SPLITS),
        help="comma-separated splits; test must be requested alone after selection",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
        help="immutable Stage 2 selection manifest required for --splits test",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits"),
        help="directory containing train.csv, val.csv, test.csv, and split_summary.json",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/processed/images"),
        help="root against which local_image_path is resolved",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/features"),
        help="feature-cache output directory",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"square resize dimension (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"process workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace a complete, partial, or stale existing feature cache",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""

    args = build_parser().parse_args(argv)
    try:
        extract_all_splits(
            split_dir=args.split_dir,
            image_root=args.image_root,
            output_dir=args.output_dir,
            image_size=args.image_size,
            workers=args.workers,
            force=args.force,
            splits=tuple(part.strip() for part in args.splits.split(",") if part.strip()),
            selection_manifest=args.selection_manifest,
        )
    except (OSError, ValueError, FeatureCacheError, ImageFeatureError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
