"""Create the immutable forward-looking Stage 2 test-selection manifest."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.data.common import sha256_file
from src.traditional.train_traditional import (
    CANONICAL_LINEARSVC_PARAMETERS,
    CANONICAL_RANDOM_FOREST_PARAMETERS,
    estimator_parameters,
    validate_canonical_estimator_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTION_SCHEMA_VERSION = 2
DEFAULT_SELECTION_MANIFEST = Path("results/metrics/stage2_selection_manifest.json")
DEFAULT_COMPATIBILITY_FREEZE = Path("results/metrics/stage2_freeze.json")

STAGE1_FILES = (
    "selected_classes_seed42.txt",
    "class_mapping.json",
    "train.csv",
    "val.csv",
    "test.csv",
    "split_summary.json",
)
FULL_FEATURE_COMPONENTS = ("color", "texture", "hog", "bovw")
HANDCRAFTED_COMPONENTS = ("color", "texture", "hog")
CANONICAL_BASELINES = {
    "linear_svm": "linear_svm",
    "random_forest": "random_forest",
}
OPTIONAL_BOVW_NAME = "sift_bovw_svm"
ALLOWED_TEST_MODELS = frozenset((*CANONICAL_BASELINES, OPTIONAL_BOVW_NAME))
REQUIRED_VALIDATION_METRICS = (
    "top1_accuracy",
    "top5_accuracy",
    "overall_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)

CANONICAL_MODEL_SPECS = {
    "linear_svm": {
        "model_type": "linear_svm",
        "classifier": "LinearSVC",
        "feature_set": "combined",
        "feature_components": FULL_FEATURE_COMPONENTS,
        "feature_dim": 3704,
        "estimator_parameters": CANONICAL_LINEARSVC_PARAMETERS,
    },
    "random_forest": {
        "model_type": "random_forest",
        "classifier": "RandomForestClassifier",
        "feature_set": "handcrafted",
        "feature_components": HANDCRAFTED_COMPONENTS,
        "feature_dim": 2424,
        "estimator_parameters": CANONICAL_RANDOM_FOREST_PARAMETERS,
    },
    "sift_bovw_svm": {
        "model_type": "linear_svm",
        "classifier": "LinearSVC",
        "feature_set": "bovw",
        "feature_components": ("bovw",),
        "feature_dim": 1280,
        "estimator_parameters": CANONICAL_LINEARSVC_PARAMETERS,
    },
}

EXPECTED_VALIDATION_EXPERIMENTS = frozenset(
    {
        "linear_svm",
        "random_forest",
        "sift_bovw_svm",
        "ablation_cb_color_bovw_a03",
        "ablation_cb_color_bovw_a05",
        "ablation_drop_color",
        "ablation_drop_texture",
        "ablation_drop_hog",
        "ablation_norm_blk_a01",
        "ablation_norm_blk_a02",
        "ablation_norm_blk_a03",
        "ablation_norm_blk_a05",
        "ablation_norm_blk_a10",
        "ablation_norm_glb_a03",
        "ablation_norm_glb_a05",
        "ablation_norm_glb_a10",
        "ablation_only_color",
        "ablation_only_texture",
        "ablation_only_hog",
    }
)


def _expected_grid_specs() -> dict[str, tuple[str, str, str, float, bool, int]]:
    handcrafted = "color+texture+hog"
    specs = {
        "linear_svm": ("LinearSVC", "combined", f"{handcrafted}+bovw", 0.3, True, 3704),
        "random_forest": ("RandomForestClassifier", "handcrafted", handcrafted, 0.3, True, 2424),
        "sift_bovw_svm": ("LinearSVC", "bovw", "bovw", 0.3, True, 1280),
        "ablation_cb_color_bovw_a03": ("LinearSVC", "combined", "color+bovw", 0.3, True, 1800),
        "ablation_cb_color_bovw_a05": ("LinearSVC", "combined", "color+bovw", 0.5, True, 1800),
        "ablation_drop_color": ("LinearSVC", "handcrafted", "texture+hog", 0.5, True, 1904),
        "ablation_drop_texture": ("LinearSVC", "handcrafted", "color+hog", 0.5, True, 2284),
        "ablation_drop_hog": ("LinearSVC", "handcrafted", "color+texture", 0.5, True, 660),
        "ablation_only_color": ("LinearSVC", "handcrafted", "color", 0.5, True, 520),
        "ablation_only_texture": ("LinearSVC", "handcrafted", "texture", 0.5, True, 140),
        "ablation_only_hog": ("LinearSVC", "handcrafted", "hog", 0.5, True, 1764),
    }
    for alpha in (0.1, 0.2, 0.3, 0.5, 1.0):
        specs[f"ablation_norm_blk_a{str(alpha).replace('.', '')}"] = (
            "LinearSVC", "handcrafted", handcrafted, alpha, True, 2424
        )
    for alpha in (0.3, 0.5, 1.0):
        specs[f"ablation_norm_glb_a{str(alpha).replace('.', '')}"] = (
            "LinearSVC", "handcrafted", handcrafted, alpha, False, 2424
        )
    return specs


EXPECTED_GRID_SPECS = _expected_grid_specs()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_embedded(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _publish_pair(
    selection_path: Path,
    selection: Mapping[str, Any],
    compatibility_path: Path,
    compatibility: Mapping[str, Any],
) -> None:
    """Stage both documents and roll back either replacement on failure."""

    destinations = (selection_path, compatibility_path)
    documents = (selection, compatibility)
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
    previous = {
        path: path.read_bytes() if path.is_file() else None for path in destinations
    }
    temporaries: list[Path] = []
    published: list[Path] = []
    try:
        for path, document in zip(destinations, documents, strict=True):
            descriptor, name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary = Path(name)
            temporaries.append(temporary)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(_payload(document))
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in zip(temporaries, destinations, strict=True):
            os.replace(temporary, destination)
            published.append(destination)
    except Exception:
        for path in reversed(published):
            old = previous[path]
            if old is None:
                path.unlink(missing_ok=True)
            else:
                descriptor, name = tempfile.mkstemp(
                    prefix=f".{path.name}.rollback.", suffix=".tmp", dir=path.parent
                )
                temporary = Path(name)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(old)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
        raise
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)


def _finite_number(value: Any, field: str, *, minimum: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Validation field {field} is not numeric") from error
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"Validation field {field} is invalid")
    return number


def _strict_grid_bool(value: Any, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if type(value) is str and value in {"true", "True"}:
        return True
    if type(value) is str and value in {"false", "False"}:
        return False
    raise ValueError(f"Validation field {field} must be exactly true or false")


def _strict_grid_dimension(value: Any, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Validation field {field} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Validation field {field} must be an integer") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"Validation field {field} must be an integer")
    return int(number)


def _validate_validation_grid(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation grid not found: {path}")
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read validation grid {path}: {error}") from error
    required = {
        "experiment",
        "classifier",
        "feature_set",
        "components",
        "alpha",
        "blockwise",
        "dim",
        "top1",
        "top5",
        "macro_f1",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Validation grid is missing columns: {sorted(required - set(frame.columns))}"
        )
    if len(frame) != 19 or frame["experiment"].duplicated().any():
        raise ValueError("Validation grid must contain exactly 19 unique experiments")
    observed = frozenset(frame["experiment"].astype(str))
    if observed != EXPECTED_VALIDATION_EXPERIMENTS:
        raise ValueError(
            "Validation grid experiment set differs from the frozen 19-config grid"
        )
    indexed = frame.set_index("experiment", drop=False)
    for name, expected in EXPECTED_GRID_SPECS.items():
        row = indexed.loc[name]
        classifier, feature_set, components, alpha, blockwise, dimension = expected
        observed_blockwise = _strict_grid_bool(
            row["blockwise"], f"{name}.blockwise"
        )
        observed_dimension = _strict_grid_dimension(row["dim"], f"{name}.dim")
        if (
            str(row["classifier"]) != classifier
            or str(row["feature_set"]) != feature_set
            or str(row["components"]) != components
            or not math.isclose(float(row["alpha"]), alpha, abs_tol=1e-12)
            or observed_blockwise is not blockwise
            or observed_dimension != dimension
        ):
            raise ValueError(f"Validation grid row does not match reviewed spec: {name}")
    fingerprints = frame[
        ["classifier", "feature_set", "components", "alpha", "blockwise", "dim"]
    ].astype(str).agg("|".join, axis=1)
    if fingerprints.nunique() != 19:
        raise ValueError("Validation grid contains duplicate scientific configurations")
    for metric in ("top1", "top5", "macro_f1"):
        values = pd.to_numeric(frame[metric], errors="coerce")
        if values.isna().any() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Validation grid has invalid {metric} values")
    return {
        "path": _portable(path),
        "sha256": sha256_file(path),
        "rows": 19,
        "experiments": sorted(observed),
        "unique_configuration_fingerprints": 19,
    }


def _validate_feature_binding(binding: Mapping[str, Any], split_dir: Path) -> None:
    config_path = _resolve_embedded(str(binding.get("feature_config_path", "")))
    cache_path = _resolve_embedded(str(binding.get("train_feature_path", "")))
    if not config_path.is_file() or sha256_file(config_path) != binding.get(
        "feature_config_sha256"
    ):
        raise ValueError("Model feature-config binding is stale")
    if not cache_path.is_file() or sha256_file(cache_path) != binding.get(
        "train_feature_sha256"
    ):
        raise ValueError("Model train-cache binding is stale")
    train_manifest = split_dir / "train.csv"
    if (
        binding.get("train_manifest_file_name") != "train.csv"
        or not train_manifest.is_file()
        or binding.get("train_manifest_sha256") != sha256_file(train_manifest)
    ):
        raise ValueError("Model train-manifest binding is stale")
    vocabulary_hash = binding.get("vocabulary_sha256")
    if vocabulary_hash is not None:
        vocabulary_path = _resolve_embedded(str(binding.get("vocabulary_path", "")))
        if not vocabulary_path.is_file() or sha256_file(vocabulary_path) != vocabulary_hash:
            raise ValueError("Model BoVW vocabulary binding is stale")
        bundle = joblib.load(vocabulary_path)
        if (
            not isinstance(bundle, dict)
            or bundle.get("training_source") != "train.csv only"
            or bundle.get("training_manifest_sha256") != sha256_file(train_manifest)
        ):
            raise ValueError("BoVW vocabulary was not fitted on the frozen train split")


def _validate_validation_metrics(
    document: Mapping[str, Any],
    *,
    name: str,
    model_type: str,
    model_path: Path,
    config_hash: str,
    feature_bindings: Sequence[Mapping[str, Any]],
    split_dir: Path,
) -> None:
    if (
        document.get("schema_version") not in {1, 2}
        or document.get("status") != "complete"
        or document.get("split") != "val"
        or document.get("experiment_name") != name
        or document.get("model_name") != name
        or document.get("model_type") != model_type
    ):
        raise ValueError(f"Validation metrics schema is invalid for {name}")
    metrics = document.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Validation metrics block is missing for {name}")
    if metrics.get("num_samples") != 5000 or metrics.get("num_classes") != 500:
        raise ValueError(f"Validation counts are invalid for {name}")
    values = {
        key: _finite_number(metrics.get(key), f"{name}.{key}")
        for key in REQUIRED_VALIDATION_METRICS
    }
    if any(value > 1.0 for value in values.values()):
        raise ValueError(f"Validation metric is outside [0,1] for {name}")
    if not math.isclose(values["overall_accuracy"], values["top1_accuracy"], abs_tol=1e-12):
        raise ValueError(f"Validation overall/top-1 disagree for {name}")
    if values["top5_accuracy"] + 1e-12 < values["top1_accuracy"]:
        raise ValueError(f"Validation top-5 is below top-1 for {name}")
    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"Validation provenance is missing for {name}")
    if provenance.get("model_sha256") != sha256_file(model_path):
        raise ValueError(f"Validation model provenance is invalid for {name}")
    if provenance.get("feature_config_sha256") != config_hash:
        raise ValueError(f"Validation feature config provenance is invalid for {name}")
    observed_bindings = provenance.get("feature_bindings")
    if observed_bindings != list(feature_bindings):
        raise ValueError(f"Validation feature bindings differ from model {name}")
    val_manifest = split_dir / "val.csv"
    if (
        provenance.get("split_manifest") != "val.csv"
        or provenance.get("split_manifest_sha256") != sha256_file(val_manifest)
    ):
        raise ValueError(f"Validation split provenance is invalid for {name}")


def _existing_test_artifacts(args: argparse.Namespace) -> list[Path]:
    metrics_dir = Path(args.metrics_dir)
    predictions_dir = Path(args.predictions_dir)
    tables_dir = Path(getattr(args, "tables_dir", Path("results/tables")))
    figures_dir = Path(getattr(args, "figures_dir", Path("results/figures")))
    feature_dir = Path(getattr(args, "feature_dir", Path("results/features")))
    found = [
        path
        for path in metrics_dir.glob("traditional_*_metrics.json")
        if not path.name.endswith("_val_metrics.json")
    ]
    found.extend(metrics_dir.glob("traditional_*_test_confusion.npz"))
    found.extend(predictions_dir.glob("traditional_*_test_predictions.csv"))
    found.extend(tables_dir.glob("traditional_*_test_*.csv"))
    found.extend(figures_dir.glob("traditional_*_test_*.png"))
    found.extend(
        path
        for path in tables_dir.glob("traditional_*_top_confusions.csv")
        if "_val_" not in path.name
    )
    found.extend(
        path
        for path in figures_dir.glob("traditional_*_confusion_matrix.png")
        if "_val_" not in path.name
    )
    found.extend(
        path
        for path in figures_dir.glob("traditional_*_top_confusions.png")
        if "_val_" not in path.name
    )
    for relative in (
        "traditional_test_features.npz",
        "traditional_test_feature_config.json",
        "bovw/bovw_test_features.npz",
        "bovw/bovw_test_feature_config.json",
        "bovw/bovw_test_status.json",
    ):
        candidate = feature_dir / relative
        if candidate.exists():
            found.append(candidate)
    return sorted(set(found))


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(getattr(args, "output", DEFAULT_SELECTION_MANIFEST)).resolve()
    compatibility = Path(
        getattr(args, "compatibility_output", DEFAULT_COMPATIBILITY_FREEZE)
    ).resolve()
    dangerous = bool(getattr(args, "dangerously_replace_existing_selection", False))
    existing = [path for path in (output, compatibility) if path.exists()]
    if existing and not dangerous:
        raise FileExistsError(
            "Stage 2 selection/freeze is immutable: "
            + ", ".join(str(path) for path in existing)
        )
    if existing and dangerous:
        expected = str(getattr(args, "expected_existing_selection_sha256", ""))
        if not output.is_file() or len(expected) != 64 or sha256_file(output) != expected:
            raise ValueError(
                "Dangerous replacement requires --expected-existing-selection-sha256 "
                "matching the authoritative manifest"
            )
    test_artifacts = _existing_test_artifacts(args)
    if test_artifacts:
        raise FileExistsError(
            "Test artifacts/caches already exist; selection cannot be created or replaced: "
            + ", ".join(str(path) for path in test_artifacts)
        )

    model_paths = [Path(path).resolve() for path in args.model]
    if len(model_paths) != 3:
        raise ValueError("Selection requires exactly the three allowed Stage 2 models")
    models: dict[str, dict[str, Any]] = {}
    feature_configs: dict[str, dict[str, str]] = {}
    validation_metrics: dict[str, dict[str, str]] = {}
    split_dir = Path(args.split_dir)
    for model_path in model_paths:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict):
            raise ValueError(f"Not a Stage 2 model bundle: {model_path}")
        name = str(bundle.get("experiment_name", ""))
        if name not in ALLOWED_TEST_MODELS or name in models:
            raise ValueError(f"Model is duplicate or not test-eligible: {name!r}")
        metadata = bundle.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Model metadata is missing for {name}")
        spec = CANONICAL_MODEL_SPECS[name]
        for key in ("model_type", "classifier", "feature_set", "feature_dim"):
            if metadata.get(key) != spec[key]:
                raise ValueError(f"Canonical {name} violates {key}={spec[key]!r}")
        if tuple(metadata.get("feature_components", ())) != spec["feature_components"]:
            raise ValueError(f"Canonical {name} has incorrect feature components")
        normalisation = metadata.get("normalisation")
        if (
            not isinstance(normalisation, Mapping)
            or not math.isclose(float(normalisation.get("alpha", -1)), 0.3, abs_tol=1e-12)
            or normalisation.get("blockwise") is not True
        ):
            raise ValueError(f"Canonical {name} must use blockwise alpha=0.3")
        if name != "random_forest" and metadata.get("svm_backend") != "linearsvc":
            raise ValueError(f"Canonical {name} must use LinearSVC")
        diagnostics = metadata.get("estimator_diagnostics")
        if not isinstance(diagnostics, Mapping) or diagnostics.get("converged") is not True:
            raise ValueError(f"Canonical {name} lacks successful convergence evidence")
        parameters = metadata.get("estimator_parameters")
        if not isinstance(parameters, Mapping) or not parameters:
            raise ValueError(f"Canonical {name} lacks complete estimator parameters")
        validate_canonical_estimator_parameters(
            parameters,
            spec["estimator_parameters"],
            context=f"Canonical {name} metadata",
        )
        estimator = bundle.get("estimator")
        if estimator is None or not callable(getattr(estimator, "get_params", None)):
            raise ValueError(f"Canonical {name} bundle lacks its fitted estimator")
        actual_parameters = estimator_parameters(estimator)
        validate_canonical_estimator_parameters(
            actual_parameters,
            spec["estimator_parameters"],
            context=f"Canonical {name} estimator",
        )
        metadata_classifier_parameters = {
            key: value
            for key, value in parameters.items()
            if str(key).startswith("classifier__")
        }
        actual_classifier_parameters = {
            key: value
            for key, value in actual_parameters.items()
            if str(key).startswith("classifier__")
        }
        if metadata_classifier_parameters != actual_classifier_parameters:
            raise ValueError(
                f"Canonical {name} estimator parameters differ from metadata"
            )

        config_path = _resolve_embedded(str(bundle.get("feature_config_path", "")))
        if not config_path.is_file():
            raise FileNotFoundError(f"Feature config not found: {config_path}")
        config_hash = sha256_file(config_path)
        if config_hash != bundle.get("feature_config_sha256"):
            raise ValueError(f"Feature config changed after training {name}")
        bindings = bundle.get("feature_bindings")
        if not isinstance(bindings, list) or not bindings:
            raise ValueError(f"Model {name} lacks feature cache/config bindings")
        for binding in bindings:
            if not isinstance(binding, Mapping):
                raise ValueError(f"Model {name} has an invalid feature binding")
            _validate_feature_binding(binding, split_dir)

        metadata_path = model_path.with_suffix(".metadata.json")
        metadata_document = _read_json(metadata_path)
        if (
            metadata_document.get("model_sha256") != sha256_file(model_path)
            or metadata_document.get("feature_bindings") != bindings
            or metadata_document.get("estimator_parameters") != parameters
        ):
            raise ValueError(f"Metadata sidecar is stale for {name}")
        validation_path = Path(args.metrics_dir) / f"traditional_{name}_val_metrics.json"
        validation = _read_json(validation_path)
        _validate_validation_metrics(
            validation,
            name=name,
            model_type=str(bundle.get("model_type", "")),
            model_path=model_path,
            config_hash=config_hash,
            feature_bindings=bindings,
            split_dir=split_dir,
        )
        entry = {
            "path": _portable(model_path),
            "sha256": sha256_file(model_path),
            "metadata_path": _portable(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "model_type": spec["model_type"],
            "classifier": spec["classifier"],
            "feature_set": spec["feature_set"],
            "feature_components": list(spec["feature_components"]),
            "feature_dim": spec["feature_dim"],
            "feature_config_path": _portable(config_path),
            "feature_config_sha256": config_hash,
            "feature_bindings": bindings,
            "normalisation": dict(normalisation),
            "estimator_parameters": dict(parameters),
            "estimator_diagnostics": dict(diagnostics),
        }
        models[name] = entry
        feature_configs[name] = {
            "path": _portable(config_path),
            "sha256": config_hash,
        }
        validation_metrics[name] = {
            "path": _portable(validation_path),
            "sha256": sha256_file(validation_path),
        }

    if set(models) != set(ALLOWED_TEST_MODELS):
        raise ValueError("Selection must contain exactly the three allowed test models")
    grid = _validate_validation_grid(Path(args.validation_grid))
    stage1: dict[str, dict[str, Any]] = {}
    for file_name in STAGE1_FILES:
        path = split_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Stage 1 artifact not found: {path}")
        stage1[file_name] = {
            "path": _portable(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    document: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "frozen",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Validation-only selection fixed before any held-out test cache access",
        "retrospective_limitation": (
            "This forward-looking manifest gates all new test access. It cannot prove "
            "that historical snapshot authors never viewed earlier test evidence."
        ),
        "selected_models": sorted(models),
        "allowed_test_models": sorted(ALLOWED_TEST_MODELS),
        "stage1_artifacts": stage1,
        "models": dict(sorted(models.items())),
        "feature_configs": dict(sorted(feature_configs.items())),
        "validation_metrics": dict(sorted(validation_metrics.items())),
        "validation_grid": grid,
        "selection_policy": {
            "required_unique_validation_configurations": 19,
            "test_evaluations_allowed": 3,
            "canonical_power_alpha": 0.3,
            "canonical_blockwise_normalisation": True,
        },
    }
    selection_bytes = _payload(document)
    import hashlib

    selection_hash = hashlib.sha256(selection_bytes).hexdigest()
    compatibility_document = {
        **document,
        "schema_version": 1,
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "compatibility_role": "stage2_freeze alias for the authoritative selection",
        "selection_manifest": {
            "path": _portable(output),
            "sha256": selection_hash,
        },
    }
    _publish_pair(output, document, compatibility, compatibility_document)
    print(f"[PASS] Stage 2 selection frozen for 3 models: {output}")
    print(f"[PASS] Compatibility freeze: {compatibility}")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze exactly three validated Stage 2 models before test access."
    )
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--feature-dir", type=Path, default=Path("results/features"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--validation-grid",
        type=Path,
        default=Path("results/tables/traditional_validation_grid.csv"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument(
        "--compatibility-output", type=Path, default=DEFAULT_COMPATIBILITY_FREEZE
    )
    parser.add_argument("--dangerously-replace-existing-selection", action="store_true")
    parser.add_argument("--expected-existing-selection-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        freeze(args)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(f"[FAIL] {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
