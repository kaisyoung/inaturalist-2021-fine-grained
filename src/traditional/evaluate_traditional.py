"""Evaluate a frozen traditional classifier on validation or held-out test data."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from src.analysis.metrics import (
    align_class_scores,
    evaluate_aligned_scores,
    per_class_metrics,
    top_confused_pairs,
)
from src.analysis.plotting import (
    plot_confusion_matrix_overview,
    plot_top_confused_pairs,
)
from src.data.common import NUM_CLASSES, sha256_file
from src.traditional.freeze_stage2 import (
    ALLOWED_TEST_MODELS,
    CANONICAL_MODEL_SPECS,
    DEFAULT_SELECTION_MANIFEST,
    SELECTION_SCHEMA_VERSION,
    STAGE1_FILES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS = np.arange(NUM_CLASSES, dtype=np.int64)
REQUIRED_FEATURE_KEYS = (
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
    "manifest_sha256",
    "manifest_file_name",
)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_path(value: str | Path, *, relative_to: Path | None = None) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists() or relative_to is None:
        return project_candidate
    return (relative_to / candidate).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=False, lineterminator="\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_confusion_npz(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(handle)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, confusion_matrix=matrix, labels=LABELS)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_bundle(model_path: Path) -> dict[str, Any]:
    try:
        bundle = joblib.load(model_path)
    except Exception as error:  # joblib may surface backend-specific exceptions.
        raise ValueError(f"Could not load model bundle {model_path}: {error}") from error
    if not isinstance(bundle, dict):
        raise ValueError("Model artifact must be a Stage 2 bundle dictionary")
    required = {
        "experiment_name",
        "model_type",
        "estimator",
        "classes",
        "feature_config_path",
        "feature_config_sha256",
        "feature_components",
        "feature_indices",
        "training_time_seconds",
        "feature_set",
        "feature_bindings",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Model bundle is missing keys: {missing}")
    name = str(bundle["experiment_name"])
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
        raise ValueError(f"Unsafe experiment_name in model bundle: {name!r}")
    classes = np.asarray(bundle["classes"], dtype=np.int64)
    if not np.array_equal(classes, LABELS):
        raise ValueError("Model bundle classes must be exactly 0..499")
    estimator_classes = np.asarray(
        getattr(bundle["estimator"], "classes_", []), dtype=np.int64
    )
    if not np.array_equal(estimator_classes, classes):
        raise ValueError("estimator.classes_ does not match the bundle class order")
    return bundle


def _split_metadata(
    config: dict[str, Any], split: str
) -> tuple[str, str, str, str, int]:
    """Read shared-handcrafted or flat BoVW split metadata."""

    try:
        entry = config["split_artifacts"][split]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Feature config has no split_artifacts.{split}") from error
    if not isinstance(entry, dict):
        raise ValueError(f"split_artifacts.{split} must be an object")
    if isinstance(entry.get("feature_file"), dict):
        feature = entry["feature_file"]
        manifest = entry.get("manifest")
        if not isinstance(manifest, dict):
            raise ValueError(f"split_artifacts.{split}.manifest must be an object")
        try:
            return (
                str(feature["file_name"]),
                str(feature["sha256"]),
                str(manifest["file_name"]),
                str(manifest["sha256"]),
                int(manifest["rows"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid nested split metadata for {split}") from error
    try:
        return (
            str(entry["output_file"]),
            str(entry["output_sha256"]),
            str(entry["manifest_file"]),
            str(entry["manifest_sha256"]),
            int(entry["rows"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid flat BoVW split metadata for {split}") from error


def _feature_artifact(config: dict[str, Any], split: str, config_path: Path) -> tuple[Path, str]:
    file_name, expected_hash, _, _, _ = _split_metadata(config, split)
    path = _resolve_path(file_name, relative_to=config_path.parent)
    if not path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Feature cache hash mismatch for {path}: expected {expected_hash}, "
            f"got {actual_hash}"
        )
    return path, actual_hash


def _scalar_text(value: np.ndarray, name: str) -> str:
    if np.asarray(value).shape != ():
        raise ValueError(f"Feature cache {name} must be a scalar")
    return str(np.asarray(value).item())


def _load_features(
    feature_path: Path,
    config: dict[str, Any],
    split: str,
    split_dir: Path,
) -> dict[str, np.ndarray]:
    try:
        with np.load(feature_path, allow_pickle=False) as archive:
            missing = sorted(set(REQUIRED_FEATURE_KEYS) - set(archive.files))
            if missing:
                raise ValueError(f"Feature cache is missing arrays: {missing}")
            arrays = {key: archive[key] for key in REQUIRED_FEATURE_KEYS}
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load feature cache {feature_path}: {error}") from error

    features = np.asarray(arrays["features"])
    labels = np.asarray(arrays["new_labels"], dtype=np.int64)
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.floating):
        raise ValueError("features must be a two-dimensional floating-point array")
    if not np.isfinite(features).all():
        raise ValueError("features contains NaN or infinite values")
    row_count = features.shape[0]
    for key in REQUIRED_FEATURE_KEYS[1:10]:
        if np.asarray(arrays[key]).ndim != 1 or len(arrays[key]) != row_count:
            raise ValueError(f"Feature cache array {key} does not match feature rows")
    if set(labels.tolist()) != set(LABELS.tolist()):
        raise ValueError(f"Feature cache {split} labels are not exactly 0..499")
    split_values = np.asarray(arrays["splits"]).astype(str)
    if not np.all(split_values == split):
        raise ValueError(f"Feature cache contains records outside split {split!r}")

    (
        _,
        _,
        expected_manifest_name,
        expected_manifest_hash,
        expected_rows,
    ) = _split_metadata(config, split)
    manifest_path = split_dir / expected_manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise ValueError(f"Frozen split manifest hash mismatch: {manifest_path}")
    if _scalar_text(arrays["manifest_file_name"], "manifest_file_name") != expected_manifest_name:
        raise ValueError("Feature cache manifest file name does not match config")
    if _scalar_text(arrays["manifest_sha256"], "manifest_sha256") != expected_manifest_hash:
        raise ValueError("Feature cache manifest hash does not match config")
    if row_count != expected_rows:
        raise ValueError(f"Feature row count is {row_count}, expected {expected_rows}")

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    comparisons = {
        "new_labels": manifest["new_label"].to_numpy(dtype=np.int64),
        "image_ids": manifest["image_id"].to_numpy(dtype=np.int64),
        "splits": manifest["split"].astype(str).to_numpy(),
        "source_splits": manifest["source_split"].astype(str).to_numpy(),
        "image_paths": manifest["image_path"].astype(str).to_numpy(),
        "local_image_paths": manifest["local_image_path"].astype(str).to_numpy(),
        "original_category_ids": manifest["original_category_id"].to_numpy(dtype=np.int64),
        "scientific_names": manifest["scientific_name"].astype(str).to_numpy(),
        "common_names": manifest["common_name"].astype(str).to_numpy(),
    }
    for key, expected in comparisons.items():
        actual = np.asarray(arrays[key])
        if actual.dtype.kind in "USO":
            actual = actual.astype(str)
        if not np.array_equal(actual, expected):
            raise ValueError(f"Feature cache {key} does not preserve manifest order")
    arrays["features"] = features
    arrays["new_labels"] = labels
    return arrays


def _prevalidate_selection(
    selection_path: Path,
    *,
    model_path: Path | None,
    split_dir: Path,
) -> tuple[dict[str, Any], str, str | None]:
    """Validate the selection before any test manifest/cache is resolved."""

    selection = _read_json(selection_path)
    if (
        selection.get("status") != "frozen"
        or selection.get("schema_version") != SELECTION_SCHEMA_VERSION
    ):
        raise ValueError("Selection manifest is not a frozen Stage 2 schema-v2 manifest")
    selected = selection.get("selected_models")
    allowed = selection.get("allowed_test_models")
    if (
        not isinstance(selected, list)
        or set(selected) != set(ALLOWED_TEST_MODELS)
        or not isinstance(allowed, list)
        or set(allowed) != set(ALLOWED_TEST_MODELS)
    ):
        raise ValueError("Selection manifest must select exactly the three allowed models")
    grid = selection.get("validation_grid")
    if not isinstance(grid, dict) or int(grid.get("rows", 0)) != 19:
        raise ValueError("Selection manifest does not bind the 19-row validation grid")
    grid_path = _resolve_path(str(grid.get("path", "")))
    if not grid_path.is_file() or sha256_file(grid_path) != grid.get("sha256"):
        raise ValueError("Selected validation grid changed")
    stage1_entries = selection.get("stage1_artifacts")
    if not isinstance(stage1_entries, dict) or set(stage1_entries) != set(STAGE1_FILES):
        raise ValueError(
            "Selection manifest must contain exactly the six frozen Stage 1 artifacts"
        )
    for file_name in STAGE1_FILES:
        entry = stage1_entries[file_name]
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"Malformed selected Stage 1 entry: {file_name}")
        path = split_dir / file_name
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Frozen Stage 1 artifact changed: {file_name}")

    models = selection.get("models")
    if not isinstance(models, dict) or set(models) != set(ALLOWED_TEST_MODELS):
        raise ValueError("Selection manifest has an invalid model set")
    feature_configs = selection.get("feature_configs")
    validation_entries = selection.get("validation_metrics")
    if (
        not isinstance(feature_configs, dict)
        or set(feature_configs) != set(ALLOWED_TEST_MODELS)
        or not isinstance(validation_entries, dict)
        or set(validation_entries) != set(ALLOWED_TEST_MODELS)
    ):
        raise ValueError("Selection feature-config/validation sets are incomplete")
    for name, spec in CANONICAL_MODEL_SPECS.items():
        entry = models.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Selection model entry is malformed: {name}")
        for field in ("model_type", "classifier", "feature_set", "feature_dim"):
            if entry.get(field) != spec[field]:
                raise ValueError(f"Selected {name} violates canonical {field}")
        if tuple(entry.get("feature_components", ())) != tuple(spec["feature_components"]):
            raise ValueError(f"Selected {name} has invalid component order")
        normalisation = entry.get("normalisation")
        if (
            not isinstance(normalisation, dict)
            or not np.isclose(float(normalisation.get("alpha", -1)), 0.3, atol=1e-12)
            or normalisation.get("blockwise") is not True
        ):
            raise ValueError(f"Selected {name} has invalid canonical normalisation")
        selected_model_path = _resolve_path(str(entry.get("path", "")))
        if (
            not selected_model_path.is_file()
            or sha256_file(selected_model_path) != entry.get("sha256")
        ):
            raise ValueError(f"Selected model changed: {name}")
        selected_config_path = _resolve_path(
            str(entry.get("feature_config_path", ""))
        )
        selected_config = feature_configs[name]
        if (
            not isinstance(selected_config, dict)
            or not selected_config_path.is_file()
            or sha256_file(selected_config_path) != entry.get("feature_config_sha256")
            or _resolve_path(str(selected_config.get("path", "")))
            != selected_config_path
            or selected_config.get("sha256") != entry.get("feature_config_sha256")
        ):
            raise ValueError(f"Selected feature config changed: {name}")
        bindings = entry.get("feature_bindings")
        if not isinstance(bindings, list) or not bindings:
            raise ValueError(f"Selected model has no feature bindings: {name}")
        for binding in bindings:
            if not isinstance(binding, dict):
                raise ValueError(f"Selected model has malformed feature bindings: {name}")
            for path_key, hash_key in (
                ("feature_config_path", "feature_config_sha256"),
                ("train_feature_path", "train_feature_sha256"),
                ("vocabulary_path", "vocabulary_sha256"),
            ):
                if hash_key not in binding:
                    continue
                artifact = _resolve_path(str(binding.get(path_key, "")))
                if not artifact.is_file() or sha256_file(artifact) != binding.get(hash_key):
                    raise ValueError(f"Selected feature binding changed: {name}.{path_key}")
            train_manifest = split_dir / "train.csv"
            if (
                binding.get("train_manifest_file_name") != "train.csv"
                or binding.get("train_manifest_sha256") != sha256_file(train_manifest)
            ):
                raise ValueError(f"Selected train-only binding changed: {name}")
        validation_entry = validation_entries[name]
        if not isinstance(validation_entry, dict):
            raise ValueError(f"Selected validation entry is malformed: {name}")
        validation_path = _resolve_path(str(validation_entry.get("path", "")))
        if (
            not validation_path.is_file()
            or sha256_file(validation_path) != validation_entry.get("sha256")
        ):
            raise ValueError(f"Validation metrics changed after selection: {name}")
    if model_path is None:
        return selection, sha256_file(selection_path), None
    resolved_model = model_path.resolve()
    matches = [
        (name, entry)
        for name, entry in models.items()
        if isinstance(entry, dict)
        and _resolve_path(str(entry.get("path", ""))) == resolved_model
    ]
    if len(matches) != 1:
        raise ValueError("Requested model path is not uniquely selected for test")
    experiment_name, model_entry = matches[0]
    model_hash = sha256_file(model_path)
    if model_entry.get("sha256") != model_hash:
        raise ValueError("Model hash does not match the selection manifest")
    validation_entry = validation_entries.get(experiment_name)
    if not isinstance(validation_entry, dict):
        raise ValueError("Selection manifest has no validation metrics for the model")
    validation_path = _resolve_path(str(validation_entry.get("path", "")))
    if not validation_path.is_file() or sha256_file(validation_path) != validation_entry.get("sha256"):
        raise ValueError("Validation metrics changed after selection")
    return selection, sha256_file(selection_path), experiment_name


def _load_feature_set(
    bundle: dict[str, Any],
    *,
    split: str,
    split_dir: Path,
    selection_path: Path | None,
    selection_sha256: str | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Load one or two bound caches and prove identical row order."""

    bindings = bundle["feature_bindings"]
    arrays_by_role: list[tuple[str, dict[str, np.ndarray]]] = []
    cache_provenance: list[dict[str, Any]] = []
    for binding in bindings:
        role = str(binding["role"])
        base_config_path = _resolve_path(str(binding["feature_config_path"]))
        if (
            not base_config_path.is_file()
            or sha256_file(base_config_path) != binding["feature_config_sha256"]
        ):
            raise ValueError(f"{role} feature config changed after training")
        config_path = base_config_path
        if split == "test":
            sidecar_name = (
                "bovw_test_feature_config.json"
                if role == "bovw"
                else "traditional_test_feature_config.json"
            )
            config_path = base_config_path.parent / sidecar_name
        config = _read_json(config_path)
        if split == "test":
            assert selection_path is not None and selection_sha256 is not None
            selected_binding = config.get("selection_manifest")
            method_binding = config.get("method_config")
            if (
                not isinstance(selected_binding, dict)
                or selected_binding.get("sha256") != selection_sha256
                or not isinstance(method_binding, dict)
                or method_binding.get("sha256") != binding["feature_config_sha256"]
            ):
                raise ValueError(f"{role} test cache is not bound to this selection/model")
            if binding.get("vocabulary_sha256") is not None:
                sidecar_vocabulary = config.get("vocabulary")
                bound_vocabulary_path = _resolve_path(
                    str(binding.get("vocabulary_path", ""))
                )
                if (
                    not isinstance(sidecar_vocabulary, dict)
                    or sidecar_vocabulary.get("output_sha256")
                    != binding.get("vocabulary_sha256")
                    or (base_config_path.parent
                    / str(sidecar_vocabulary.get("output_file", ""))).resolve()
                    != bound_vocabulary_path
                    or sidecar_vocabulary.get("training_manifest_sha256")
                    != binding.get("vocabulary_training_manifest_sha256")
                ):
                    raise ValueError(
                        f"{role} test cache uses a different vocabulary/training binding"
                    )
        feature_path, feature_hash = _feature_artifact(config, split, config_path)
        arrays = _load_features(feature_path, config, split, split_dir)
        arrays_by_role.append((role, arrays))
        cache_entry = {
            "role": role,
            "config_path": _portable_path(config_path),
            "config_sha256": sha256_file(config_path),
            "feature_file_path": _portable_path(feature_path),
            "feature_file_sha256": feature_hash,
        }
        if binding.get("vocabulary_sha256") is not None:
            cache_entry["vocabulary_path"] = binding["vocabulary_path"]
            cache_entry["vocabulary_sha256"] = binding["vocabulary_sha256"]
        cache_provenance.append(cache_entry)

    primary = arrays_by_role[0][1]
    for role, arrays in arrays_by_role[1:]:
        for key in REQUIRED_FEATURE_KEYS[1:]:
            if not np.array_equal(np.asarray(primary[key]), np.asarray(arrays[key])):
                raise ValueError(f"Cannot combine caches: {role} differs in row field {key}")
    primary = dict(primary)
    primary["features"] = np.hstack(
        [np.asarray(arrays["features"]) for _, arrays in arrays_by_role]
    ).astype(np.float32, copy=False)
    expected_width = {"handcrafted": 2424, "bovw": 1280, "combined": 3704}[
        str(bundle["feature_set"])
    ]
    if primary["features"].shape[1] != expected_width:
        raise ValueError(
            f"Loaded {bundle['feature_set']} width {primary['features'].shape[1]}, "
            f"expected {expected_width}"
        )
    return primary, cache_provenance


def _score_estimator(estimator: Any, features: np.ndarray) -> tuple[np.ndarray, str, float]:
    start = time.perf_counter()
    if hasattr(estimator, "predict_proba"):
        raw_scores = estimator.predict_proba(features)
        score_type = "predict_proba"
    elif hasattr(estimator, "decision_function"):
        raw_scores = estimator.decision_function(features)
        score_type = "decision_function"
    else:
        raise ValueError("Estimator provides neither predict_proba nor decision_function")
    elapsed = time.perf_counter() - start
    return np.asarray(raw_scores), score_type, float(elapsed)


def _prediction_frame(
    arrays: dict[str, np.ndarray],
    aligned_scores: np.ndarray,
    predicted: np.ndarray,
    order: np.ndarray,
    *,
    score_type: str,
) -> pd.DataFrame:
    truth = np.asarray(arrays["new_labels"], dtype=np.int64)
    rows = np.arange(truth.size)
    top_five = LABELS[order[:, :5]]
    top_five_scores = np.take_along_axis(aligned_scores, order[:, :5], axis=1)
    inverse_order = np.empty_like(order)
    inverse_order[rows[:, None], order] = np.arange(order.shape[1], dtype=np.int64)
    names = {
        int(label): str(name)
        for label, name in zip(truth, np.asarray(arrays["scientific_names"]).astype(str))
    }
    return pd.DataFrame(
        {
            "split": np.asarray(arrays["splits"]).astype(str),
            "image_id": np.asarray(arrays["image_ids"], dtype=np.int64),
            "image_path": np.asarray(arrays["image_paths"]).astype(str),
            "local_image_path": np.asarray(arrays["local_image_paths"]).astype(str),
            "true_new_label": truth,
            "predicted_new_label": predicted,
            "true_scientific_name": np.asarray(arrays["scientific_names"]).astype(str),
            "predicted_scientific_name": [names.get(int(value), "") for value in predicted],
            "is_correct": truth == predicted,
            "is_top5_correct": np.any(top_five == truth[:, None], axis=1),
            "true_class_score": aligned_scores[rows, truth],
            "predicted_class_score": aligned_scores[rows, predicted],
            "true_class_rank": inverse_order[rows, truth] + 1,
            "top5_new_labels": [json.dumps(values.tolist()) for values in top_five],
            "top5_scores": [
                json.dumps([float(value) for value in values], separators=(",", ":"))
                for values in top_five_scores
            ],
            "score_type": score_type,
        }
    )


def _output_paths(name: str, split: str, args: argparse.Namespace) -> dict[str, Path]:
    prefix = f"traditional_{name}"
    if split == "val":
        return {
            "metrics": args.metrics_dir / f"{prefix}_val_metrics.json",
            "predictions": args.predictions_dir / f"{prefix}_val_predictions.csv",
            "confusion": args.metrics_dir / f"{prefix}_val_confusion.npz",
            "per_class": args.tables_dir / f"{prefix}_val_per_class_metrics.csv",
            "pairs": args.tables_dir / f"{prefix}_val_top_confusions.csv",
            "confusion_figure": args.figures_dir / f"{prefix}_val_confusion_matrix.png",
            "pairs_figure": args.figures_dir / f"{prefix}_val_top_confusions.png",
        }
    return {
        "metrics": args.metrics_dir / f"{prefix}_metrics.json",
        "predictions": args.predictions_dir / f"{prefix}_test_predictions.csv",
        "confusion": args.metrics_dir / f"{prefix}_test_confusion.npz",
        "per_class": args.tables_dir / f"{prefix}_test_per_class_metrics.csv",
        "pairs": args.tables_dir / f"{prefix}_top_confusions.csv",
        "confusion_figure": args.figures_dir / f"{prefix}_confusion_matrix.png",
        "pairs_figure": args.figures_dir / f"{prefix}_top_confusions.png",
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test" and not args.include_test:
        # Fail before resolving or opening any held-out test feature artifact.
        raise ValueError("Test evaluation requires the explicit --include-test flag")
    model_path = args.model.resolve()
    selection_path = Path(
        getattr(
            args,
            "selection_manifest",
            getattr(args, "freeze_manifest", DEFAULT_SELECTION_MANIFEST),
        )
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    selection: dict[str, Any] | None = None
    selection_hash: str | None = None
    selected_name: str | None = None
    if args.split == "test":
        # This is deliberately the first substantive test action.  It validates
        # the selection, model, train caches, configs, vocabulary, Stage 1, and
        # validation evidence before any test sidecar/cache/manifest is resolved.
        selection, selection_hash, selected_name = _prevalidate_selection(
            selection_path,
            model_path=model_path,
            split_dir=args.split_dir,
        )
    bundle = _load_bundle(model_path)
    name = str(bundle["experiment_name"])
    if args.split == "test" and (name != selected_name or name not in ALLOWED_TEST_MODELS):
        raise ValueError("Loaded model identity differs from the selected test model")
    config_path = (
        args.feature_config.resolve()
        if args.feature_config is not None
        else _resolve_path(str(bundle["feature_config_path"]))
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Feature config not found: {config_path}")
    config_hash = sha256_file(config_path)
    if config_hash != str(bundle["feature_config_sha256"]):
        raise ValueError("Feature config hash does not match the trained model bundle")
    config = _read_json(config_path)
    outputs = _output_paths(name, args.split, args)
    existing = [path for path in outputs.values() if path.exists()]
    if args.split == "test":
        if existing:
            raise FileExistsError(
                "Test artifacts are immutable and already exist: "
                + ", ".join(str(path) for path in existing)
            )
    elif existing and not bool(getattr(args, "dangerously_overwrite_validation", False)):
        raise FileExistsError(
            "Validation artifacts already exist; pass "
            "--dangerously-overwrite-validation to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    arrays, cache_provenance = _load_feature_set(
        bundle,
        split=args.split,
        split_dir=args.split_dir,
        selection_path=selection_path if args.split == "test" else None,
        selection_sha256=selection_hash,
    )

    indices = np.asarray(bundle["feature_indices"], dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("Model feature_indices must be a non-empty unique 1D array")
    full_features = np.asarray(arrays["features"])
    if indices.min() < 0 or indices.max() >= full_features.shape[1]:
        raise ValueError("Model feature_indices are outside the cached feature dimension")
    selected_features = full_features[:, indices]

    raw_scores, score_type, inference_seconds = _score_estimator(
        bundle["estimator"], selected_features
    )
    aligned = align_class_scores(raw_scores, bundle["classes"], LABELS)
    aggregate, predicted, order, matrix = evaluate_aligned_scores(
        arrays["new_labels"], aligned, LABELS
    )
    direct_predictions = np.asarray(bundle["estimator"].predict(selected_features))
    if not np.array_equal(direct_predictions, predicted):
        raise ValueError(
            "Estimator predict() disagrees with the stable argmax of aligned scores"
        )
    aggregate["inference_time_seconds"] = inference_seconds
    aggregate["seconds_per_image"] = inference_seconds / len(predicted)

    names = {
        int(label): str(scientific)
        for label, scientific in zip(
            np.asarray(arrays["new_labels"], dtype=np.int64),
            np.asarray(arrays["scientific_names"]).astype(str),
        )
    }
    common_names = {
        int(label): str(common)
        for label, common in zip(
            np.asarray(arrays["new_labels"], dtype=np.int64),
            np.asarray(arrays["common_names"]).astype(str),
        )
    }
    original_category_ids = {
        int(label): int(category_id)
        for label, category_id in zip(
            np.asarray(arrays["new_labels"], dtype=np.int64),
            np.asarray(arrays["original_category_ids"], dtype=np.int64),
        )
    }
    predictions = _prediction_frame(
        arrays, aligned, predicted, order, score_type=score_type
    )
    per_class = per_class_metrics(arrays["new_labels"], predicted, LABELS)
    per_class.insert(
        1,
        "original_category_id",
        per_class["new_label"].map(original_category_ids),
    )
    per_class.insert(2, "scientific_name", per_class["new_label"].map(names))
    per_class.insert(3, "common_name", per_class["new_label"].map(common_names))
    pairs = top_confused_pairs(
        matrix, LABELS, limit=args.top_pairs, scientific_names=names
    )

    metadata = bundle.get("metadata", {})
    image_size = config.get("image_size", metadata.get("image_size"))
    manifest_name = f"{args.split}.csv"
    manifest_path = args.split_dir / manifest_name
    manifest_hash = sha256_file(manifest_path)
    base = {
        "schema_version": 2,
        "status": "complete",
        "experiment_name": name,
        "model_name": name,
        "model_type": str(bundle["model_type"]),
        "classifier": str(metadata.get("classifier", bundle["model_type"])),
        "feature_type": str(config.get("feature_type", "handcrafted")),
        "feature_set": str(bundle["feature_set"]),
        "feature_components": [str(value) for value in bundle["feature_components"]],
        "feature_dim": int(indices.size),
        "image_size": image_size,
        "training_time_seconds": float(bundle["training_time_seconds"]),
        "score_type": score_type,
        "normalisation": metadata.get("normalisation"),
        "estimator_parameters": metadata.get("estimator_parameters"),
        "estimator_diagnostics": metadata.get("estimator_diagnostics"),
        "provenance": {
            "model_path": _portable_path(model_path),
            "model_sha256": sha256_file(model_path),
            "feature_config_path": _portable_path(config_path),
            "feature_config_sha256": config_hash,
            "feature_bindings": bundle["feature_bindings"],
            "feature_caches": cache_provenance,
            "split_manifest": manifest_name,
            "split_manifest_sha256": manifest_hash,
        },
    }
    if metadata.get("svm_backend") is not None:
        base["svm_backend"] = str(metadata["svm_backend"])
    if args.split == "val":
        metrics_document = {**base, "split": "val", "metrics": aggregate}
    else:
        assert selection is not None and selection_hash is not None
        validation_entry = selection["validation_metrics"][name]
        val_path = _resolve_path(str(validation_entry["path"]))
        validation_document = _read_json(val_path)
        if (
            validation_document.get("status") != "complete"
            or sha256_file(val_path) != validation_entry["sha256"]
        ):
            raise ValueError("Validation metrics are missing or incomplete")
        metrics_document = {
            **base,
            "validation": validation_document["metrics"],
            "test": aggregate,
            "validation_metrics_path": _portable_path(val_path),
            "validation_metrics_sha256": sha256_file(val_path),
        }
        metrics_document["provenance"].update(
            {
                "selection_manifest_path": _portable_path(selection_path),
                "selection_manifest_sha256": selection_hash,
            }
        )

    if args.split == "test" and sha256_file(selection_path) != selection_hash:
        raise ValueError("Stage 2 selection manifest changed during test evaluation")

    # Build the complete artifact set away from final paths. A plotting or write
    # failure therefore cannot strand immutable partial test results.
    stage_parent = args.metrics_dir.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{name}_{args.split}_evaluation.", dir=stage_parent
    ) as temporary_directory:
        stage_root = Path(temporary_directory)
        staged = {
            key: stage_root / f"{index:02d}_{path.name}"
            for index, (key, path) in enumerate(outputs.items())
        }
        _atomic_csv(staged["predictions"], predictions)
        _atomic_confusion_npz(staged["confusion"], matrix)
        _atomic_csv(staged["per_class"], per_class)
        _atomic_csv(staged["pairs"], pairs)
        plot_confusion_matrix_overview(
            matrix,
            staged["confusion_figure"],
            title=f"{name}: {args.split} confusion matrix (row-normalized)",
        )
        plot_top_confused_pairs(
            pairs,
            staged["pairs_figure"],
            title=f"{name}: top {args.split} confused class pairs",
        )
        _atomic_json(staged["metrics"], metrics_document)

        installed: list[Path] = []
        previous = {
            path: path.read_bytes() if path.is_file() else None
            for path in outputs.values()
        }
        try:
            installation_order = [key for key in outputs if key != "metrics"] + [
                "metrics"
            ]
            for key in installation_order:
                final_path = outputs[key]
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged[key], final_path)
                installed.append(final_path)
        except OSError:
            for installed_path in reversed(installed):
                old = previous[installed_path]
                if old is None:
                    installed_path.unlink(missing_ok=True)
                else:
                    descriptor, name = tempfile.mkstemp(
                        prefix=f".{installed_path.name}.rollback.",
                        suffix=".tmp",
                        dir=installed_path.parent,
                    )
                    temporary = Path(name)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(old)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, installed_path)
            raise
    print(
        f"[PASS] {name} {args.split}: top-1={aggregate['top1_accuracy']:.6f}, "
        f"top-5={aggregate['top5_accuracy']:.6f}, "
        f"macro-F1={aggregate['macro_f1']:.6f}"
    )
    print(f"[PASS] Metrics: {outputs['metrics']}")
    return metrics_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Stage 2 traditional model with strict provenance checks."
    )
    parser.add_argument("--model", type=Path, required=True, help="Trained joblib bundle")
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=None,
        help="Feature config (defaults to the path embedded in the model)",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Required acknowledgement before held-out test evaluation",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument(
        "--predictions-dir", type=Path, default=Path("results/predictions")
    )
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--top-pairs", type=int, default=20)
    parser.add_argument(
        "--dangerously-overwrite-validation",
        action="store_true",
        help="explicitly replace the complete validation output set",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top_pairs < 1:
        parser.error("--top-pairs must be positive")
    try:
        evaluate(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"[FAIL] {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
