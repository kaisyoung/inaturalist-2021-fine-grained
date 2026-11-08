"""Train the Stage 2 traditional classifiers from a frozen feature cache.

Only the training split is read by this module.  Validation and test features are
deliberately left to :mod:`src.traditional.evaluate_traditional`, which makes it
harder to accidentally tune a model on the held-out test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.common import deterministic_json_bytes, sha256_file
from src.traditional.features import BLOCK_ORDER
from src.traditional.normalize import PowerNormalizer, blocks_from_config


SCHEMA_VERSION = 2
EXPECTED_CLASSES = np.arange(500, dtype=np.int64)
EXPECTED_TRAIN_ROWS = 20_000
EXPECTED_TRAIN_IMAGES_PER_CLASS = 40
DEFAULT_FEATURE_CONFIG = Path("results/features/traditional_feature_config.json")
DEFAULT_BOVW_FEATURE_CONFIG = Path("results/features/bovw/bovw_feature_config.json")
DEFAULT_OUTPUT_DIR = Path("results/models/traditional")
DEFAULT_SPLIT_DIR = Path("data/splits")
EXPERIMENT_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")
CANONICAL_LINEARSVC_PARAMETERS = {
    "classifier__C": 1.0,
    "classifier__class_weight": None,
    "classifier__dual": "auto",
    "classifier__fit_intercept": True,
    "classifier__intercept_scaling": 1,
    "classifier__loss": "squared_hinge",
    "classifier__max_iter": 5000,
    "classifier__multi_class": "ovr",
    "classifier__penalty": "l2",
    "classifier__random_state": 42,
    "classifier__tol": 1e-4,
    "classifier__verbose": 0,
}
CANONICAL_RANDOM_FOREST_PARAMETERS = {
    "classifier__bootstrap": True,
    "classifier__ccp_alpha": 0.0,
    "classifier__class_weight": None,
    "classifier__criterion": "gini",
    "classifier__max_depth": None,
    "classifier__max_features": "sqrt",
    "classifier__max_leaf_nodes": None,
    "classifier__max_samples": None,
    "classifier__min_impurity_decrease": 0.0,
    "classifier__min_samples_leaf": 2,
    "classifier__min_samples_split": 2,
    "classifier__min_weight_fraction_leaf": 0.0,
    "classifier__monotonic_cst": None,
    "classifier__n_estimators": 300,
    "classifier__n_jobs": -1,
    "classifier__oob_score": False,
    "classifier__random_state": 42,
    "classifier__verbose": 0,
    "classifier__warm_start": False,
}
CANONICAL_MODEL_SPECS = {
    "linear_svm": {
        "model": "svm",
        "feature_set": "combined",
        "components": ("color", "texture", "hog", "bovw"),
        "feature_dim": 3704,
        "estimator_parameters": CANONICAL_LINEARSVC_PARAMETERS,
    },
    "random_forest": {
        "model": "random_forest",
        "feature_set": "handcrafted",
        "components": ("color", "texture", "hog"),
        "feature_dim": 2424,
        "estimator_parameters": CANONICAL_RANDOM_FOREST_PARAMETERS,
    },
    "sift_bovw_svm": {
        "model": "svm",
        "feature_set": "bovw",
        "components": ("bovw",),
        "feature_dim": 1280,
        "estimator_parameters": CANONICAL_LINEARSVC_PARAMETERS,
    },
}


@dataclass(frozen=True)
class FeatureSelection:
    """Resolved columns for one requested feature-component selection."""

    names: tuple[str, ...]
    indices: np.ndarray


def combined_configuration(
    handcrafted: Mapping[str, Any], bovw: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive the exact color+texture+HOG+BoVW feature contract."""

    expected_handcrafted = tuple(BLOCK_ORDER)
    if tuple(handcrafted.get("concatenation_order", ())) != expected_handcrafted:
        raise ValueError("Handcrafted config has a non-canonical component order")
    if tuple(bovw.get("concatenation_order", ())) != ("bovw",):
        raise ValueError("BoVW config has a non-canonical component order")
    components: dict[str, dict[str, Any]] = {}
    offsets: dict[str, list[int]] = {}
    cursor = 0
    for source, names in (
        (handcrafted, expected_handcrafted),
        (bovw, ("bovw",)),
    ):
        for name in names:
            entry = source.get("components", {}).get(name)
            if not isinstance(entry, Mapping):
                raise ValueError(f"Feature config is missing component {name!r}")
            start, stop = _component_bounds(entry, name)
            width = stop - start
            components[name] = {
                **dict(entry),
                "start": cursor,
                "stop": cursor + width,
                "dimension": width,
            }
            offsets[name] = [cursor, cursor + width]
            cursor += width
    if cursor != 3704:
        raise ValueError(f"Combined feature dimension must be 3704, got {cursor}")
    return {
        "schema_version": 1,
        "feature_type": "color_texture_hog_sift_bovw",
        "feature_dim": cursor,
        "image_size": handcrafted.get("image_size"),
        "concatenation_order": [*expected_handcrafted, "bovw"],
        "components": components,
        "component_offsets": offsets,
        "normalisation": "applied by the model pipeline, not at extraction time",
    }


def _cache_row_identity(path: Path) -> dict[str, np.ndarray | str]:
    """Load only row-order fields needed to prove safe cache concatenation."""

    try:
        with np.load(path, allow_pickle=False) as cache:
            identity: dict[str, np.ndarray | str] = {
                "new_labels": np.asarray(cache["new_labels"], dtype=np.int64),
                "image_ids": np.asarray(cache["image_ids"], dtype=np.int64),
                "image_paths": np.asarray(cache["image_paths"]).astype(str),
                "local_image_paths": np.asarray(cache["local_image_paths"]).astype(str),
                "manifest_sha256": str(np.asarray(cache["manifest_sha256"]).item()),
                "manifest_file_name": str(
                    np.asarray(cache["manifest_file_name"]).item()
                ),
            }
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"Could not validate cache row order for {path}: {error}") from error
    return identity


def _require_same_cache_rows(left: Path, right: Path) -> None:
    left_identity = _cache_row_identity(left)
    right_identity = _cache_row_identity(right)
    for key in left_identity:
        first = left_identity[key]
        second = right_identity[key]
        equal = first == second if isinstance(first, str) else np.array_equal(first, second)
        if not bool(equal):
            raise ValueError(
                f"Cannot combine feature caches: row identity field {key!r} differs"
            )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Feature config does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in feature config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Feature config must contain a JSON object: {path}")
    return value


def _path_candidates(value: str | Path, config_path: Path) -> list[Path]:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return [candidate]
    # Stage 2 configs normally store a basename.  Project-relative paths are also
    # accepted so caches can be moved with the repository.
    return [config_path.parent / candidate, Path.cwd() / candidate]


def _artifact_path_from_entry(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        for key in ("output_file", "output_path", "path", "file", "filename"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                return value
        nested = entry.get("feature_file")
        if isinstance(nested, Mapping):
            for key in ("file_name", "filename", "path"):
                value = nested.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def resolve_feature_artifact(
    config: Mapping[str, Any], config_path: Path, split: str
) -> Path:
    """Resolve a feature NPZ while tolerating documented config-key variants."""

    raw: str | None = None
    for container_name in ("split_artifacts", "artifacts", "output_files", "files"):
        container = config.get(container_name)
        if isinstance(container, Mapping) and split in container:
            raw = _artifact_path_from_entry(container[split])
            if raw:
                break
    if raw is None:
        for key in (f"{split}_features", f"{split}_feature_file"):
            raw = _artifact_path_from_entry(config.get(key))
            if raw:
                break
    if raw is None:
        feature_type = str(config.get("feature_type", ""))
        basename = (
            f"bovw_{split}_features.npz"
            if "bovw" in feature_type.lower()
            else f"traditional_{split}_features.npz"
        )
        raw = basename

    candidates = _path_candidates(raw, config_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise ValueError(f"Could not find {split!r} feature artifact; tried: {rendered}")


def _component_bounds(entry: Any, name: str) -> tuple[int, int]:
    if isinstance(entry, Mapping):
        start = entry.get("start", entry.get("offset"))
        stop = entry.get("stop", entry.get("end"))
        if stop is None and start is not None and entry.get("dimension") is not None:
            stop = int(start) + int(entry["dimension"])
        if start is not None and stop is not None:
            return int(start), int(stop)
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        if len(entry) == 2:
            return int(entry[0]), int(entry[1])
    raise ValueError(f"Feature config has invalid bounds for component {name!r}")


def parse_components(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    cleaned = tuple(part.strip().lower() for part in parts if part.strip())
    if not cleaned:
        raise ValueError("At least one feature component is required")
    return cleaned


def resolve_feature_selection(
    config: Mapping[str, Any], requested: str | Iterable[str], feature_dim: int
) -> FeatureSelection:
    """Resolve requested component names to stable feature-column indices."""

    components = config.get("components")
    if not isinstance(components, Mapping) or not components:
        if str(config.get("feature_type", "")).lower() in {"sift_bovw", "bovw"}:
            components = {"bovw": {"start": 0, "stop": feature_dim}}
        else:
            raise ValueError("Feature config is missing a non-empty 'components' object")

    aliases = {
        "rgb": "rgb_histogram",
        "rgb_hist": "rgb_histogram",
        "hsv": "hsv_histogram",
        "hsv_hist": "hsv_histogram",
        "color": "color",
        "colour": "color",
    }
    names = parse_components(requested)
    if names == ("full",) or names == ("all",):
        order = config.get("concatenation_order")
        if isinstance(order, list) and order:
            names = tuple(str(item) for item in order)
        else:
            names = tuple(str(item) for item in components)

    expanded: list[str] = []
    for raw_name in names:
        canonical = aliases.get(raw_name, raw_name)
        if canonical == "color" and "color" not in components:
            expanded.extend(("rgb_histogram", "hsv_histogram"))
        else:
            expanded.append(canonical)
    # Preserve user/config order but reject duplicates, which usually indicate a
    # misspelled ablation specification.
    if len(expanded) != len(set(expanded)):
        raise ValueError(f"Duplicate feature component in request: {expanded}")

    indices: list[np.ndarray] = []
    for name in expanded:
        if name not in components:
            available = ", ".join(str(key) for key in components)
            raise ValueError(f"Unknown feature component {name!r}; available: {available}")
        start, stop = _component_bounds(components[name], name)
        if not (0 <= start < stop <= feature_dim):
            raise ValueError(
                f"Component {name!r} has out-of-range bounds [{start}, {stop}) "
                f"for feature_dim={feature_dim}"
            )
        indices.append(np.arange(start, stop, dtype=np.int64))
    combined = np.concatenate(indices)
    if len(np.unique(combined)) != len(combined):
        raise ValueError("Selected feature components overlap")
    return FeatureSelection(tuple(expanded), combined)


def load_training_features(
    feature_config_path: Path,
    components: str | Iterable[str],
    *,
    split_dir: Path = DEFAULT_SPLIT_DIR,
    require_production_contract: bool = True,
) -> tuple[np.ndarray, np.ndarray, FeatureSelection, dict[str, Any], Path, dict[str, str]]:
    """Load and validate only the frozen training feature artifact."""

    config_path = feature_config_path.resolve()
    config = _load_json(config_path)
    train_path = resolve_feature_artifact(config, config_path, "train")
    try:
        with np.load(train_path, allow_pickle=False) as cache:
            if "features" not in cache or "new_labels" not in cache:
                raise ValueError(
                    f"{train_path} must contain 'features' and 'new_labels' arrays"
                )
            raw_features = cache["features"]
            raw_labels = cache["new_labels"]
            if raw_features.dtype != np.float32:
                raise ValueError(
                    f"Training features must use float32, got {raw_features.dtype}"
                )
            if raw_labels.dtype != np.int64:
                raise ValueError(
                    f"Training new_labels must use int64, got {raw_labels.dtype}"
                )
            features = np.asarray(raw_features)
            labels = np.asarray(raw_labels)
            provenance = {}
            for key in ("manifest_sha256", "manifest_file_name"):
                if key in cache:
                    raw = np.asarray(cache[key])
                    if raw.ndim != 0:
                        raise ValueError(f"Training feature {key} must be a scalar")
                    provenance[key] = str(raw.item())
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and "must contain" in str(exc):
            raise
        raise ValueError(f"Could not load feature artifact {train_path}: {exc}") from exc

    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError(
            f"Invalid training array shapes: features={features.shape}, labels={labels.shape}"
        )
    if not np.isfinite(features).all():
        raise ValueError("Training features contain NaN or infinite values")
    config_dim = int(config.get("feature_dim", features.shape[1]))
    if config_dim != features.shape[1]:
        raise ValueError(
            f"Config feature_dim={config_dim} does not match NPZ dimension={features.shape[1]}"
        )
    expected_hash = None
    expected_manifest_hash = None
    expected_manifest_name = None
    expected_manifest_rows = None
    split_artifacts = config.get("split_artifacts")
    if isinstance(split_artifacts, Mapping):
        train_entry = split_artifacts.get("train")
        if isinstance(train_entry, Mapping):
            expected_hash = train_entry.get("output_sha256", train_entry.get("sha256"))
            feature_file = train_entry.get("feature_file")
            if expected_hash is None and isinstance(feature_file, Mapping):
                expected_hash = feature_file.get("sha256")
            manifest = train_entry.get("manifest")
            if isinstance(manifest, Mapping):
                expected_manifest_hash = manifest.get("sha256")
                expected_manifest_name = manifest.get("file_name")
                expected_manifest_rows = manifest.get("rows")
            else:
                expected_manifest_hash = train_entry.get("manifest_sha256")
                expected_manifest_name = train_entry.get("manifest_file")
                expected_manifest_rows = train_entry.get("rows")
    artifacts = config.get("artifacts")
    if expected_hash is None and isinstance(artifacts, Mapping):
        train_entry = artifacts.get("train")
        if isinstance(train_entry, Mapping):
            expected_hash = train_entry.get("sha256")
    actual_hash = sha256_file(train_path)
    if expected_hash is not None and str(expected_hash) != actual_hash:
        raise ValueError(
            f"Training feature hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    if require_production_contract:
        if expected_hash is None:
            raise ValueError("Feature config is missing the train feature SHA-256")
        if not expected_manifest_hash or not expected_manifest_name:
            raise ValueError(
                "Feature config is missing split_artifacts.train.manifest provenance"
            )
        if str(expected_manifest_name) != "train.csv":
            raise ValueError("Training feature config manifest must be named train.csv")
        if expected_manifest_rows is None or int(expected_manifest_rows) != EXPECTED_TRAIN_ROWS:
            raise ValueError(
                "Feature config train manifest must record exactly "
                f"{EXPECTED_TRAIN_ROWS} rows"
            )
        if provenance.get("manifest_file_name") != str(expected_manifest_name):
            raise ValueError(
                "Training feature manifest_file_name does not match feature config"
            )
        if provenance.get("manifest_sha256") != str(expected_manifest_hash):
            raise ValueError(
                "Training feature manifest_sha256 does not match feature config"
            )
        manifest_path = Path(split_dir) / str(expected_manifest_name)
        if not manifest_path.is_file():
            raise ValueError(f"Current frozen training manifest does not exist: {manifest_path}")
        current_manifest_hash = sha256_file(manifest_path)
        if current_manifest_hash != str(expected_manifest_hash):
            raise ValueError(
                "Current frozen training manifest hash does not match feature config: "
                f"expected {expected_manifest_hash}, got {current_manifest_hash}"
            )
        if len(labels) != EXPECTED_TRAIN_ROWS:
            raise ValueError(
                f"Training feature cache has {len(labels)} rows; "
                f"expected {EXPECTED_TRAIN_ROWS}"
            )
        counts = np.bincount(labels, minlength=len(EXPECTED_CLASSES))
        if counts.shape != (len(EXPECTED_CLASSES),) or not np.all(
            counts == EXPECTED_TRAIN_IMAGES_PER_CLASS
        ):
            raise ValueError(
                "Training feature cache must contain exactly "
                f"{EXPECTED_TRAIN_IMAGES_PER_CLASS} records for every class 0..499"
            )

    selection = resolve_feature_selection(config, components, features.shape[1])
    selected = np.ascontiguousarray(features[:, selection.indices], dtype=np.float32)
    return selected, labels, selection, config, train_path, {
        **provenance,
        "feature_npz_sha256": actual_hash,
    }


def _feature_binding(
    *,
    role: str,
    config_path: Path,
    config: Mapping[str, Any],
    train_path: Path,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "role": role,
        "feature_config_path": _portable_path(config_path.resolve()),
        "feature_config_sha256": sha256_file(config_path),
        "train_feature_path": _portable_path(train_path.resolve()),
        "train_feature_sha256": provenance["feature_npz_sha256"],
        "train_manifest_file_name": provenance.get("manifest_file_name"),
        "train_manifest_sha256": provenance.get("manifest_sha256"),
        "feature_dim": int(config.get("feature_dim", 0)),
    }
    vocabulary = config.get("vocabulary")
    if isinstance(vocabulary, Mapping):
        raw_path = vocabulary.get("output_file")
        expected_hash = vocabulary.get("output_sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            raise ValueError("BoVW config has an incomplete vocabulary binding")
        vocabulary_path = config_path.parent / raw_path
        if not vocabulary_path.is_file():
            raise ValueError(f"BoVW vocabulary does not exist: {vocabulary_path}")
        actual_hash = sha256_file(vocabulary_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"BoVW vocabulary hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        binding["vocabulary_path"] = _portable_path(vocabulary_path.resolve())
        binding["vocabulary_sha256"] = actual_hash
        binding["vocabulary_training_manifest_sha256"] = vocabulary.get(
            "training_manifest_sha256"
        )
    return binding


def balanced_pilot_indices(
    labels: np.ndarray, images_per_class: int = 5
) -> np.ndarray:
    """Select the first manifest-order records per class for the SVM gate."""

    if images_per_class <= 0:
        raise ValueError("images_per_class must be positive")
    selected: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < images_per_class:
            raise ValueError(
                f"Class {int(label)} has only {len(indices)} training records; "
                f"{images_per_class} required for the SVM pilot"
            )
        selected.extend(int(index) for index in indices[:images_per_class])
    return np.asarray(selected, dtype=np.int64)


def projected_linearsvc_seconds(
    pilot_seconds: float, pilot_samples: int, full_samples: int
) -> float:
    if pilot_seconds < 0 or pilot_samples <= 0 or full_samples <= 0:
        raise ValueError("Invalid values for LinearSVC runtime projection")
    return float(2.0 * pilot_seconds * full_samples / pilot_samples)


def make_linearsvc_pipeline(
    seed: int,
    *,
    blocks: Sequence[tuple[int, int]] | None = None,
    alpha: float = 0.3,
    blockwise: bool = True,
    C: float = 1.0,
) -> Pipeline:
    return Pipeline(
        [
            (
                "normalise",
                PowerNormalizer(blocks=blocks, alpha=alpha, blockwise=blockwise),
            ),
            (
                "classifier",
                LinearSVC(C=C, dual="auto", max_iter=5000, random_state=seed),
            ),
        ]
    )


def make_sgd_svm_pipeline(
    seed: int,
    *,
    blocks: Sequence[tuple[int, int]] | None = None,
    alpha: float = 0.3,
    blockwise: bool = True,
) -> Pipeline:
    return Pipeline(
        [
            (
                "normalise",
                PowerNormalizer(blocks=blocks, alpha=alpha, blockwise=blockwise),
            ),
            (
                "classifier",
                SGDClassifier(
                    loss="hinge",
                    alpha=1e-4,
                    max_iter=2000,
                    tol=1e-3,
                    average=True,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def make_random_forest(
    seed: int,
    *,
    blocks: Sequence[tuple[int, int]] | None = None,
    alpha: float = 0.3,
    blockwise: bool = True,
    n_estimators: int = 300,
) -> Pipeline:
    return Pipeline(
        [
            (
                "normalise",
                PowerNormalizer(blocks=blocks, alpha=alpha, blockwise=blockwise),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def _fit_and_time(estimator: Any, features: np.ndarray, labels: np.ndarray) -> float:
    start = time.perf_counter()
    estimator.fit(features, labels)
    return float(time.perf_counter() - start)


def train_linear_svm(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    backend: str,
    pilot_images_per_class: int = 5,
    max_projected_seconds: float = 90.0 * 60.0,
    blocks: Sequence[tuple[int, int]] | None = None,
    alpha: float = 0.3,
    blockwise: bool = True,
    C: float = 1.0,
) -> tuple[Any, float, dict[str, Any], str]:
    """Fit a linear SVM, applying the documented gate and explicit fallback."""

    if backend not in {"auto", "linearsvc", "sgd"}:
        raise ValueError(f"Unsupported SVM backend: {backend}")
    gate: dict[str, Any] = {
        "requested_backend": backend,
        "pilot_images_per_class": pilot_images_per_class,
        "threshold_seconds": float(max_projected_seconds),
        "fallback_used": False,
    }
    if backend == "sgd":
        estimator = make_sgd_svm_pipeline(
            seed, blocks=blocks, alpha=alpha, blockwise=blockwise
        )
        elapsed = _fit_and_time(estimator, features, labels)
        gate.update({"status": "BYPASSED_EXPLICIT_SGD", "selected_backend": "sgd"})
        return estimator, elapsed, gate, "sgd"

    pilot_indices = balanced_pilot_indices(labels, pilot_images_per_class)
    pilot = make_linearsvc_pipeline(
        seed, blocks=blocks, alpha=alpha, blockwise=blockwise, C=C
    )
    caught: list[warnings.WarningMessage]
    pilot_error: str | None = None
    start = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pilot.fit(features[pilot_indices], labels[pilot_indices])
    except Exception as exc:  # estimator failures need an auditable fallback
        caught = []
        pilot_error = f"{type(exc).__name__}: {exc}"
    pilot_seconds = float(time.perf_counter() - start)
    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    projection = projected_linearsvc_seconds(
        pilot_seconds, len(pilot_indices), len(labels)
    )
    gate.update(
        {
            "pilot_samples": int(len(pilot_indices)),
            "full_samples": int(len(labels)),
            "pilot_seconds": pilot_seconds,
            "projected_seconds": projection,
            "pilot_converged": not convergence_warning and pilot_error is None,
            "pilot_error": pilot_error,
            "pilot_warnings": [
                {"category": item.category.__name__, "message": str(item.message)}
                for item in caught
            ],
        }
    )
    gate_passed = (
        pilot_error is None
        and not convergence_warning
        and projection <= max_projected_seconds
    )

    if not gate_passed:
        reason = (
            pilot_error
            or ("LinearSVC pilot emitted ConvergenceWarning" if convergence_warning else None)
            or (
                f"projected {projection:.3f}s exceeds "
                f"threshold {max_projected_seconds:.3f}s"
            )
        )
        if backend == "linearsvc":
            raise RuntimeError(f"LinearSVC gate failed with fallback disabled: {reason}")
        estimator = make_sgd_svm_pipeline(
            seed, blocks=blocks, alpha=alpha, blockwise=blockwise
        )
        elapsed = _fit_and_time(estimator, features, labels)
        gate.update(
            {
                "status": "FALLBACK_AFTER_GATE",
                "fallback_used": True,
                "fallback_reason": reason,
                "selected_backend": "sgd",
            }
        )
        return estimator, elapsed, gate, "sgd"

    estimator = make_linearsvc_pipeline(
        seed, blocks=blocks, alpha=alpha, blockwise=blockwise, C=C
    )
    full_error: str | None = None
    start = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(features, labels)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            full_error = "LinearSVC full fit emitted ConvergenceWarning"
    except Exception as exc:
        full_error = f"{type(exc).__name__}: {exc}"
    elapsed = float(time.perf_counter() - start)
    gate["full_fit_warnings"] = [
        {"category": item.category.__name__, "message": str(item.message)}
        for item in caught
    ]
    if full_error is None:
        gate.update({"status": "PASSED", "selected_backend": "linearsvc"})
        return estimator, elapsed, gate, "linearsvc"
    if backend == "linearsvc":
        raise RuntimeError(f"LinearSVC full fit failed with fallback disabled: {full_error}")

    fallback = make_sgd_svm_pipeline(
        seed, blocks=blocks, alpha=alpha, blockwise=blockwise
    )
    fallback_elapsed = _fit_and_time(fallback, features, labels)
    gate.update(
        {
            "status": "FALLBACK_AFTER_FULL_FIT",
            "fallback_used": True,
            "fallback_reason": full_error,
            "failed_linearsvc_seconds": elapsed,
            "selected_backend": "sgd",
        }
    )
    return fallback, fallback_elapsed, gate, "sgd"


def _atomic_joblib_dump(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_dump(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(deterministic_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_staged_group(staged: Mapping[Path, Path]) -> None:
    """Publish a complete model/config/metadata set with rollback."""

    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for destination, candidate in staged.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                descriptor, name = tempfile.mkstemp(
                    prefix=f".{destination.name}.backup.", dir=destination.parent
                )
                os.close(descriptor)
                backup = Path(name)
                backup.unlink()
                os.replace(destination, backup)
                backups[destination] = backup
            os.replace(candidate, destination)
            installed.append(destination)
    except OSError:
        for destination in reversed(installed):
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        for candidate in staged.values():
            candidate.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return {
        "class": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def estimator_parameters(estimator: Any) -> dict[str, Any]:
    """Return every recursively exposed estimator parameter in JSON-safe form."""

    return {
        str(key): _json_safe(value)
        for key, value in estimator.get_params(deep=True).items()
    }


def _parameter_value_matches(observed: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if expected is None:
        return observed is None
    if isinstance(expected, (int, float)):
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            return False
        return math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
        )
    return type(observed) is type(expected) and observed == expected


def validate_canonical_estimator_parameters(
    parameters: Mapping[str, Any], expected: Mapping[str, Any], *, context: str
) -> None:
    """Require the complete, exact classifier parameter surface for a canonical name."""

    expected_keys = set(expected)
    observed_keys = {
        str(key) for key in parameters if str(key).startswith("classifier__")
    }
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"{context} has an incomplete canonical classifier parameter set; "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatches = [
        key
        for key, expected_value in expected.items()
        if not _parameter_value_matches(parameters.get(key), expected_value)
    ]
    if mismatches:
        raise ValueError(
            f"{context} violates canonical estimator parameter(s): {mismatches}"
        )


def validate_canonical_training_request(
    args: argparse.Namespace,
    *,
    experiment_name: str,
    feature_set: str,
    alpha: float,
    blockwise: bool,
) -> Mapping[str, Any] | None:
    """Reject canonical artifact names unless every requested knob is canonical."""

    canonical = CANONICAL_MODEL_SPECS.get(experiment_name)
    if canonical is None:
        return None
    svm_backend = str(
        getattr(args, "svm_backend", getattr(args, "solver", "linearsvc"))
    )
    requested_c = getattr(args, "C", 1.0)
    requested_trees = getattr(args, "n_estimators", 300)
    canonical_request = (
        args.model == canonical["model"]
        and feature_set == canonical["feature_set"]
        and np.isclose(alpha, 0.3, rtol=0.0, atol=1e-12)
        and blockwise
        and _parameter_value_matches(getattr(args, "seed", None), 42)
        and (
            args.model != "svm"
            or (
                svm_backend in {"linearsvc", "liblinear"}
                and _parameter_value_matches(requested_c, 1.0)
            )
        )
        and (
            args.model != "random_forest"
            or _parameter_value_matches(requested_trees, 300)
        )
    )
    if not canonical_request:
        raise ValueError(
            f"Canonical model {experiment_name!r} must use its frozen feature set, "
            "alpha=0.3 blockwise normalisation, seed=42, LinearSVC C=1, or the "
            "300-tree canonical random forest as applicable"
        )
    return canonical


def estimator_diagnostics(estimator: Any) -> dict[str, Any]:
    """Record fitted iteration/convergence evidence for the final classifier."""

    classifier = (
        estimator.named_steps["classifier"]
        if isinstance(estimator, Pipeline)
        else estimator
    )
    n_iter = getattr(classifier, "n_iter_", None)
    max_iter = getattr(classifier, "max_iter", None)
    converged: bool | None = None
    if n_iter is not None and max_iter is not None:
        values = np.asarray(n_iter)
        converged = bool(np.all(values < int(max_iter)))
    elif isinstance(classifier, RandomForestClassifier):
        converged = True
    return {
        "classifier": type(classifier).__name__,
        "n_iter": _json_safe(n_iter),
        "max_iter": _json_safe(max_iter),
        "converged": converged,
    }


def train_and_save(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    default_name = "linear_svm" if args.model == "svm" else "random_forest"
    experiment_name = args.experiment_name or default_name
    if not EXPERIMENT_NAME_PATTERN.fullmatch(experiment_name):
        raise ValueError(
            "experiment-name must start with a lowercase letter or digit and contain "
            "only lowercase letters, digits, underscores, and hyphens"
        )
    output_dir = Path(args.output_dir)
    model_path = output_dir / f"{experiment_name}.joblib"
    metadata_path = output_dir / f"{experiment_name}.metadata.json"
    derived_contract_path = output_dir / f"{experiment_name}.feature_contract.json"
    feature_set = str(getattr(args, "feature_set", "handcrafted"))
    if feature_set not in {"handcrafted", "bovw", "combined"}:
        raise ValueError(f"Unsupported feature_set: {feature_set}")
    artifacts = [model_path, metadata_path]
    if feature_set == "combined":
        artifacts.append(derived_contract_path)
    conflicts = [path for path in artifacts if path.exists()]
    dangerous_overwrite = bool(getattr(args, "dangerously_overwrite", False))
    if conflicts and not dangerous_overwrite:
        raise FileExistsError(
            "Refusing to replace existing model artifact(s) without "
            "--dangerously-overwrite: "
            + ", ".join(str(path) for path in conflicts)
        )

    split_dir = Path(getattr(args, "split_dir", DEFAULT_SPLIT_DIR))
    requested_components = getattr(args, "components", "full")
    feature_bindings: list[dict[str, Any]] = []
    derived_contract: dict[str, Any] | None = None
    if feature_set == "combined":
        handcrafted_path = Path(args.feature_config)
        bovw_path = Path(
            getattr(args, "bovw_feature_config", DEFAULT_BOVW_FEATURE_CONFIG)
        )
        (
            handcrafted_features,
            handcrafted_labels,
            _handcrafted_selection,
            handcrafted_config,
            handcrafted_train_path,
            handcrafted_provenance,
        ) = load_training_features(
            handcrafted_path,
            "full",
            split_dir=split_dir,
        )
        (
            bovw_features,
            bovw_labels,
            _bovw_selection,
            bovw_config,
            bovw_train_path,
            bovw_provenance,
        ) = load_training_features(
            bovw_path,
            "bovw",
            split_dir=split_dir,
        )
        _require_same_cache_rows(handcrafted_train_path, bovw_train_path)
        if not np.array_equal(handcrafted_labels, bovw_labels):
            raise ValueError("Handcrafted and BoVW training labels differ")
        feature_config = combined_configuration(handcrafted_config, bovw_config)
        combined = np.hstack((handcrafted_features, bovw_features)).astype(
            np.float32, copy=False
        )
        selection = resolve_feature_selection(
            feature_config, requested_components, combined.shape[1]
        )
        features = np.ascontiguousarray(combined[:, selection.indices])
        labels = handcrafted_labels
        feature_bindings = [
            _feature_binding(
                role="handcrafted",
                config_path=handcrafted_path,
                config=handcrafted_config,
                train_path=handcrafted_train_path,
                provenance=handcrafted_provenance,
            ),
            _feature_binding(
                role="bovw",
                config_path=bovw_path,
                config=bovw_config,
                train_path=bovw_train_path,
                provenance=bovw_provenance,
            ),
        ]
        derived_contract = {
            **feature_config,
            "source_feature_bindings": feature_bindings,
            "train_row_order": "exact-match across underlying caches",
        }
        feature_config_path = derived_contract_path
        train_path = handcrafted_train_path
        provenance = handcrafted_provenance
    else:
        feature_config_path = Path(
            getattr(args, "bovw_feature_config", DEFAULT_BOVW_FEATURE_CONFIG)
            if feature_set == "bovw"
            else args.feature_config
        )
        component_request = "bovw" if feature_set == "bovw" else requested_components
        features, labels, selection, feature_config, train_path, provenance = (
            load_training_features(
                feature_config_path,
                component_request,
                split_dir=split_dir,
            )
        )
        feature_bindings = [
            _feature_binding(
                role=feature_set,
                config_path=feature_config_path,
                config=feature_config,
                train_path=train_path,
                provenance=provenance,
            )
        ]

    blocks = blocks_from_config(feature_config, selection.names)
    alpha = float(getattr(args, "alpha", 0.3))
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("--alpha must be finite and positive")
    blockwise = not bool(getattr(args, "global_norm", False))

    canonical = validate_canonical_training_request(
        args,
        experiment_name=experiment_name,
        feature_set=feature_set,
        alpha=alpha,
        blockwise=blockwise,
    )
    if canonical is not None:
        if (
            tuple(selection.names) != canonical["components"]
            or features.shape[1] != canonical["feature_dim"]
        ):
            raise ValueError(
                f"Canonical model {experiment_name!r} must use its frozen feature "
                "components and dimension"
            )
    unique_labels = np.unique(labels)
    if not np.array_equal(unique_labels, EXPECTED_CLASSES):
        raise ValueError(
            "Training labels must contain every class 0..499 exactly; got "
            f"{len(unique_labels)} classes spanning "
            f"{int(unique_labels.min()) if len(unique_labels) else 'N/A'}.."
            f"{int(unique_labels.max()) if len(unique_labels) else 'N/A'}"
        )

    if args.model == "svm":
        requested_backend = str(
            getattr(args, "svm_backend", getattr(args, "solver", "linearsvc"))
        )
        if requested_backend == "liblinear":
            requested_backend = "linearsvc"
        estimator, train_seconds, gate, backend = train_linear_svm(
            features,
            labels,
            seed=args.seed,
            backend=requested_backend,
            blocks=blocks,
            alpha=alpha,
            blockwise=blockwise,
            C=float(getattr(args, "C", 1.0)),
        )
        model_type = "linear_svm"
        training_details: dict[str, Any] = {"svm_gate": gate, "svm_backend": backend}
    elif args.model == "random_forest":
        estimator = make_random_forest(
            args.seed,
            blocks=blocks,
            alpha=alpha,
            blockwise=blockwise,
            n_estimators=int(getattr(args, "n_estimators", 300)),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train_seconds = _fit_and_time(estimator, features, labels)
        model_type = "random_forest"
        training_details = {
            "training_warnings": [
                {
                    "category": item.category.__name__,
                    "message": str(item.message),
                }
                for item in caught
            ]
        }
    else:  # argparse prevents this; retain a programmatic guard
        raise ValueError(f"Unsupported model: {args.model}")

    final_estimator_parameters = estimator_parameters(estimator)
    if canonical is not None:
        validate_canonical_estimator_parameters(
            final_estimator_parameters,
            canonical["estimator_parameters"],
            context=f"Canonical model {experiment_name!r}",
        )

    config_path_resolved = feature_config_path.resolve()
    config_hash = (
        hashlib.sha256(deterministic_json_bytes(derived_contract)).hexdigest()
        if derived_contract is not None
        else sha256_file(config_path_resolved)
    )
    image_size = feature_config.get("image_size")
    scalar_metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": experiment_name,
        "model_type": model_type,
        "classifier": type(
            estimator.named_steps["classifier"]
            if isinstance(estimator, Pipeline)
            else estimator
        ).__name__,
        "feature_type": str(feature_config.get("feature_type", "unknown")),
        "feature_set": feature_set,
        "feature_components": list(selection.names),
        "feature_dim": int(features.shape[1]),
        "image_size": image_size,
        "training_samples": int(len(labels)),
        "classes": [int(value) for value in estimator.classes_],
        "training_time_seconds": float(train_seconds),
        "seed": int(args.seed),
        "normalisation": {
            "alpha": alpha,
            "blockwise": blockwise,
            "blocks": [list(span) for span in blocks],
        },
        "feature_config_path": _portable_path(config_path_resolved),
        "feature_config_sha256": config_hash,
        "train_feature_path": _portable_path(train_path),
        "train_feature_sha256": provenance["feature_npz_sha256"],
        "train_manifest_file_name": provenance.get("manifest_file_name"),
        "train_manifest_sha256": provenance.get("manifest_sha256"),
        "input_split_hashes": {
            "train": provenance.get("manifest_sha256"),
        },
        "feature_bindings": feature_bindings,
        "estimator_parameters": final_estimator_parameters,
        "estimator_diagnostics": estimator_diagnostics(estimator),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        **training_details,
    }
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": experiment_name,
        "model_type": model_type,
        "estimator": estimator,
        "classes": scalar_metadata["classes"],
        "feature_config_path": scalar_metadata["feature_config_path"],
        "feature_config_sha256": scalar_metadata["feature_config_sha256"],
        "feature_components": list(selection.names),
        "feature_set": feature_set,
        "feature_bindings": feature_bindings,
        "feature_indices": selection.indices,
        "training_time_seconds": float(train_seconds),
        "seed": int(args.seed),
        "metadata": scalar_metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[Path, Path] = {}
    try:
        if derived_contract is not None:
            descriptor, name = tempfile.mkstemp(
                prefix=f".{derived_contract_path.name}.", suffix=".candidate", dir=output_dir
            )
            contract_candidate = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(deterministic_json_bytes(derived_contract))
                stream.flush()
                os.fsync(stream.fileno())
            if sha256_file(contract_candidate) != config_hash:
                raise RuntimeError("Staged combined feature contract hash changed")
            staged[derived_contract_path] = contract_candidate

        descriptor, name = tempfile.mkstemp(
            prefix=f".{model_path.name}.", suffix=".candidate", dir=output_dir
        )
        os.close(descriptor)
        model_candidate = Path(name)
        joblib.dump(bundle, model_candidate, compress=3)
        staged[model_path] = model_candidate
        model_hash = sha256_file(model_candidate)
        metadata = {
            **scalar_metadata,
            "model_file": model_path.name,
            "model_sha256": model_hash,
        }
        descriptor, name = tempfile.mkstemp(
            prefix=f".{metadata_path.name}.", suffix=".candidate", dir=output_dir
        )
        metadata_candidate = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(deterministic_json_bytes(metadata))
            stream.flush()
            os.fsync(stream.fileno())
        staged[metadata_path] = metadata_candidate
        _publish_staged_group(staged)
    except Exception:
        for candidate in staged.values():
            candidate.unlink(missing_ok=True)
        raise
    return model_path, metadata_path, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Stage 2 traditional classifier from cached train features."
    )
    parser.add_argument(
        "--model",
        choices=("svm", "random_forest"),
        required=True,
        help="Classifier family to train.",
    )
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=DEFAULT_FEATURE_CONFIG,
        help=f"Feature-cache config (default: {DEFAULT_FEATURE_CONFIG}).",
    )
    parser.add_argument(
        "--bovw-feature-config",
        type=Path,
        default=DEFAULT_BOVW_FEATURE_CONFIG,
        help="BoVW config used for --feature-set bovw or combined",
    )
    parser.add_argument(
        "--feature-set",
        choices=("handcrafted", "bovw", "combined"),
        default="handcrafted",
    )
    parser.add_argument(
        "--components",
        default="full",
        help="Comma-separated feature components, 'color', or 'full' (default: full).",
    )
    parser.add_argument(
        "--experiment-name",
        help="Artifact stem; defaults to linear_svm or random_forest.",
    )
    parser.add_argument(
        "--svm-backend",
        choices=("auto", "linearsvc", "sgd"),
        default="linearsvc",
        help="Linear-SVM implementation; auto applies the documented runtime gate.",
    )
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--global-norm", action="store_true")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help=(
            "directory containing the current frozen train.csv used to verify "
            "feature-cache provenance"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--dangerously-overwrite",
        action="store_true",
        help="explicitly replace an existing named model/config/metadata set",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        model_path, metadata_path, metadata = train_and_save(args)
    except (ValueError, FileExistsError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"PASS model: {model_path}")
    print(f"PASS metadata: {metadata_path}")
    print(f"classifier: {metadata['classifier']}")
    print(f"feature_dim: {metadata['feature_dim']}")
    print(f"training_time_seconds: {metadata['training_time_seconds']:.6f}")
    if "svm_gate" in metadata:
        print(f"svm_gate_status: {metadata['svm_gate']['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
