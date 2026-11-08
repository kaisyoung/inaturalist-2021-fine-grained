"""Collect completed Stage 2 test metrics into the required comparison table."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.analysis.plotting import plot_model_comparison, plot_validation_ablation
from src.data.common import sha256_file
from src.traditional.freeze_stage2 import (
    ALLOWED_TEST_MODELS,
    CANONICAL_MODEL_SPECS,
    DEFAULT_SELECTION_MANIFEST,
    FULL_FEATURE_COMPONENTS,
    OPTIONAL_BOVW_NAME,
    SELECTION_SCHEMA_VERSION,
    STAGE1_FILES,
)
from src.traditional.features import feature_configuration


COLUMNS = [
    "model_name",
    "feature_type",
    "classifier",
    "val_top1",
    "val_top5",
    "val_macro_precision",
    "val_macro_recall",
    "val_macro_f1",
    "test_top1",
    "test_top5",
    "test_macro_precision",
    "test_macro_recall",
    "test_macro_f1",
    "train_time_seconds",
    "test_time_seconds",
    "feature_dim",
    "image_size",
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HANDCRAFTED_CONFIG = feature_configuration()
ABLATION_COMPONENTS = {
    "ablation_only_color": ("color",),
    "ablation_only_texture": ("texture",),
    "ablation_only_hog": ("hog",),
}
ABLATION_DIMENSIONS = {
    name: int(_HANDCRAFTED_CONFIG["components"][components[0]]["dimension"])
    for name, components in ABLATION_COMPONENTS.items()
}


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _relative_artifact(value: Any, parent: Path, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Test cache sidecar has invalid {field}")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _validate_test_cache_sidecar(
    *,
    cache: dict[str, Any],
    sidecar: dict[str, Any],
    sidecar_path: Path,
    binding: dict[str, Any],
    selected_path: Path,
    selection_hash: str,
    test_manifest: Path,
    train_manifest: Path,
) -> None:
    """Cross-check a collected cache against its immutable test sidecar."""

    selected_binding = sidecar.get("selection_manifest")
    method_binding = sidecar.get("method_config")
    if (
        not isinstance(selected_binding, dict)
        or selected_binding.get("sha256") != selection_hash
        or _resolve(str(selected_binding.get("path", ""))) != selected_path
        or not isinstance(method_binding, dict)
        or method_binding.get("sha256") != binding.get("feature_config_sha256")
        or _resolve(str(method_binding.get("path", "")))
        != _resolve(str(binding.get("feature_config_path", "")))
    ):
        raise ValueError("Test cache sidecar is not bound to the selected method")

    split_artifacts = sidecar.get("split_artifacts")
    artifact = split_artifacts.get("test") if isinstance(split_artifacts, dict) else None
    if not isinstance(artifact, dict):
        raise ValueError("Test cache sidecar lacks split_artifacts.test")
    feature_file = artifact.get("feature_file")
    manifest = artifact.get("manifest")
    if not isinstance(feature_file, dict) or not isinstance(manifest, dict):
        raise ValueError("Test cache sidecar has malformed test artifact metadata")

    declared_cache = _relative_artifact(
        feature_file.get("file_name"), sidecar_path.parent, field="feature_file.file_name"
    )
    collected_cache = _resolve(str(cache.get("feature_file_path", "")))
    declared_hash = feature_file.get("sha256")
    if (
        declared_cache != collected_cache
        or declared_hash != cache.get("feature_file_sha256")
        or not declared_cache.is_file()
        or sha256_file(declared_cache) != declared_hash
    ):
        raise ValueError("Collected test cache differs from split_artifacts.test")
    declared_bytes = feature_file.get("bytes")
    if type(declared_bytes) is not int or declared_bytes != declared_cache.stat().st_size:
        raise ValueError("Test cache byte count differs from split_artifacts.test")

    current_test_hash = sha256_file(test_manifest)
    declared_rows = manifest.get("rows")
    if (
        manifest.get("file_name") != test_manifest.name
        or manifest.get("sha256") != current_test_hash
        or type(declared_rows) is not int
        or declared_rows != len(pd.read_csv(test_manifest, keep_default_na=False))
    ):
        raise ValueError("Test manifest metadata differs from split_artifacts.test")

    selected_vocabulary_hash = binding.get("vocabulary_sha256")
    if selected_vocabulary_hash is None:
        if "vocabulary" in sidecar or cache.get("vocabulary_sha256") is not None:
            raise ValueError("Handcrafted test cache has an unexpected vocabulary")
        return
    vocabulary = sidecar.get("vocabulary")
    if not isinstance(vocabulary, dict):
        raise ValueError("BoVW test sidecar lacks its selected vocabulary")
    selected_vocabulary_path = _resolve(str(binding.get("vocabulary_path", "")))
    sidecar_vocabulary_path = _relative_artifact(
        vocabulary.get("output_file"), sidecar_path.parent, field="vocabulary.output_file"
    )
    training_hash = binding.get("vocabulary_training_manifest_sha256")
    if (
        sidecar_vocabulary_path != selected_vocabulary_path
        or _resolve(str(cache.get("vocabulary_path", ""))) != selected_vocabulary_path
        or vocabulary.get("output_sha256") != selected_vocabulary_hash
        or cache.get("vocabulary_sha256") != selected_vocabulary_hash
        or not selected_vocabulary_path.is_file()
        or sha256_file(selected_vocabulary_path) != selected_vocabulary_hash
        or vocabulary.get("training_manifest_sha256") != training_hash
        or training_hash != sha256_file(train_manifest)
    ):
        raise ValueError("BoVW test cache has a non-selected vocabulary/training binding")


def _metric(block: dict[str, Any], key: str, path: Path) -> float:
    try:
        value = float(block[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} has invalid metric {key}") from error
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{path} metric {key} is outside [0, 1]")
    return value


def _svm_identity(
    document: dict[str, Any],
    context: str,
    *,
    expected_classifier: str | None = None,
    expected_backend: str | None = None,
) -> tuple[str, str | None]:
    """Validate classifier/backend evidence and optionally compare a reference."""

    classifier = str(document.get("classifier", ""))
    if not classifier:
        raise ValueError(f"Missing classifier identity in {context}")
    if expected_classifier is not None and classifier != expected_classifier:
        raise ValueError(
            f"SVM classifier {classifier!r} does not match canonical classifier "
            f"{expected_classifier!r} in {context}"
        )
    backend_value = document.get("svm_backend")
    backend = str(backend_value) if backend_value is not None else None
    backend_classifiers = {"linearsvc": "LinearSVC", "sgd": "SGDClassifier"}
    if backend is not None:
        expected_backend_classifier = backend_classifiers.get(backend)
        if expected_backend_classifier is None or classifier != expected_backend_classifier:
            raise ValueError(f"Inconsistent SVM backend/classifier evidence in {context}")
    if expected_backend is not None and backend != expected_backend:
        raise ValueError(
            f"SVM backend {backend!r} does not match canonical backend "
            f"{expected_backend!r} in {context}"
        )
    return classifier, backend


def _validate_main_experiments(completed: dict[str, dict[str, Any]]) -> None:
    """Require the three selected models with their exact scientific roles."""

    if set(completed) != set(ALLOWED_TEST_MODELS):
        raise ValueError(
            "Completed Stage 2 experiments must be exactly "
            f"{sorted(ALLOWED_TEST_MODELS)}; observed={sorted(completed)}"
        )
    for name, spec in CANONICAL_MODEL_SPECS.items():
        document = completed[name]
        for field in ("model_type", "classifier", "feature_set", "feature_dim"):
            if document.get(field) != spec[field]:
                raise ValueError(
                    f"Canonical {name} violates {field}={spec[field]!r}"
                )
        if tuple(document.get("feature_components", ())) != tuple(
            spec["feature_components"]
        ):
            raise ValueError(f"Canonical {name} has incorrect feature components")
    _svm_identity(
        completed["linear_svm"],
        "canonical linear_svm",
        expected_classifier="LinearSVC",
        expected_backend="linearsvc",
    )
    _svm_identity(
        completed[OPTIONAL_BOVW_NAME],
        f"canonical {OPTIONAL_BOVW_NAME}",
        expected_classifier="LinearSVC",
        expected_backend="linearsvc",
    )


def _image_size(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{int(value[0])}x{int(value[1])}"
    if isinstance(value, int):
        return f"{value}x{value}"
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"Invalid image_size: {value!r}")


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


def _validate_stage1_freeze(freeze: dict[str, Any], split_dir: Path) -> None:
    entries = freeze.get("stage1_artifacts")
    if not isinstance(entries, dict) or set(entries) != set(STAGE1_FILES):
        raise ValueError("Freeze does not contain exactly the six Stage 1 artifacts")
    for file_name in STAGE1_FILES:
        entry = entries[file_name]
        path = split_dir / file_name
        if (
            not isinstance(entry, dict)
            or not path.is_file()
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ValueError(f"Current Stage 1 artifact does not match freeze: {file_name}")


def _validate_result_provenance(
    document: dict[str, Any], path: Path, args: argparse.Namespace
) -> None:
    """Validate selection -> model/config/cache -> validation/test evidence."""

    def require_file(
        container: dict[str, Any], path_key: str, hash_key: str, context: str
    ) -> Path:
        artifact = _resolve(str(container.get(path_key, "")))
        if not artifact.is_file() or sha256_file(artifact) != container.get(hash_key):
            raise ValueError(
                f"Artifact provenance check failed for {context}.{path_key} in {path}"
            )
        return artifact

    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Metrics provenance is missing in {path}")
    selected_path = Path(args.selection_manifest).resolve()
    referenced_selection = require_file(
        provenance,
        "selection_manifest_path",
        "selection_manifest_sha256",
        "test",
    )
    if referenced_selection != selected_path:
        raise ValueError(f"Metrics reference a non-authoritative selection in {path}")
    selection = _read(selected_path)
    if (
        selection.get("schema_version") != SELECTION_SCHEMA_VERSION
        or selection.get("status") != "frozen"
        or set(selection.get("selected_models", ())) != set(ALLOWED_TEST_MODELS)
        or set(selection.get("allowed_test_models", ())) != set(ALLOWED_TEST_MODELS)
    ):
        raise ValueError(f"Invalid authoritative Stage 2 selection in {path}")
    _validate_stage1_freeze(selection, args.split_dir)

    grid = selection.get("validation_grid")
    if not isinstance(grid, dict) or int(grid.get("rows", 0)) != 19:
        raise ValueError(f"Selection does not bind the 19-row validation grid in {path}")
    require_file(grid, "path", "sha256", "selection.validation_grid")

    model_name = str(document.get("model_name", ""))
    models = selection.get("models")
    if not isinstance(models, dict) or set(models) != set(ALLOWED_TEST_MODELS):
        raise ValueError(f"Selection model set is invalid in {path}")
    model_entry = models.get(model_name)
    if not isinstance(model_entry, dict):
        raise ValueError(f"Model is not selected for test: {model_name}")
    spec = CANONICAL_MODEL_SPECS[model_name]
    for field in ("model_type", "classifier", "feature_set", "feature_dim"):
        if model_entry.get(field) != spec[field] or document.get(field) != spec[field]:
            raise ValueError(f"Selected {model_name} has inconsistent {field} in {path}")
    if (
        tuple(model_entry.get("feature_components", ()))
        != tuple(spec["feature_components"])
        or tuple(document.get("feature_components", ()))
        != tuple(spec["feature_components"])
    ):
        raise ValueError(f"Selected {model_name} component order is invalid in {path}")
    if document.get("normalisation") != model_entry.get("normalisation"):
        raise ValueError(f"Selected {model_name} normalisation changed in {path}")

    model_path = require_file(provenance, "model_path", "model_sha256", "test")
    config_path = require_file(
        provenance, "feature_config_path", "feature_config_sha256", "test"
    )
    if (
        model_path != _resolve(str(model_entry.get("path", "")))
        or provenance.get("model_sha256") != model_entry.get("sha256")
        or config_path != _resolve(str(model_entry.get("feature_config_path", "")))
        or provenance.get("feature_config_sha256")
        != model_entry.get("feature_config_sha256")
    ):
        raise ValueError(f"Test artifact is not bound to selected model {model_name}")
    selected_config = selection.get("feature_configs", {}).get(model_name)
    if (
        not isinstance(selected_config, dict)
        or _resolve(str(selected_config.get("path", ""))) != config_path
        or selected_config.get("sha256") != provenance.get("feature_config_sha256")
    ):
        raise ValueError(f"Feature config is not selected for {model_name} in {path}")

    bindings = model_entry.get("feature_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError(f"Selected model has no feature bindings in {path}")
    if provenance.get("feature_bindings") != bindings:
        raise ValueError(f"Test feature bindings differ from selection in {path}")
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError(f"Malformed feature binding in {path}")
        require_file(
            binding, "feature_config_path", "feature_config_sha256", "binding"
        )
        require_file(binding, "train_feature_path", "train_feature_sha256", "binding")
        train_manifest = args.split_dir / "train.csv"
        if (
            binding.get("train_manifest_file_name") != "train.csv"
            or not train_manifest.is_file()
            or binding.get("train_manifest_sha256") != sha256_file(train_manifest)
        ):
            raise ValueError(f"Train-only feature binding failed in {path}")
        if binding.get("vocabulary_sha256") is not None:
            require_file(binding, "vocabulary_path", "vocabulary_sha256", "binding")

    caches = provenance.get("feature_caches")
    if not isinstance(caches, list) or not all(
        isinstance(row, dict) for row in caches
    ):
        raise ValueError(f"Malformed test cache provenance in {path}")
    binding_roles = [str(row["role"]) for row in bindings]
    cache_roles = [str(row.get("role", "")) for row in caches]
    if (
        len(binding_roles) != len(set(binding_roles))
        or len(cache_roles) != len(set(cache_roles))
        or set(cache_roles) != set(binding_roles)
    ):
        raise ValueError(f"Test cache roles differ from selected bindings in {path}")
    binding_by_role = {str(row["role"]): row for row in bindings}
    selection_hash = provenance["selection_manifest_sha256"]
    test_manifest = args.split_dir / "test.csv"
    train_manifest = args.split_dir / "train.csv"
    test_manifest_hash = sha256_file(test_manifest)
    for cache in caches:
        cache_config = require_file(cache, "config_path", "config_sha256", "cache")
        require_file(cache, "feature_file_path", "feature_file_sha256", "cache")
        role = str(cache["role"])
        binding = binding_by_role[role]
        sidecar = _read(cache_config)
        _validate_test_cache_sidecar(
            cache=cache,
            sidecar=sidecar,
            sidecar_path=cache_config,
            binding=binding,
            selected_path=selected_path,
            selection_hash=selection_hash,
            test_manifest=test_manifest,
            train_manifest=train_manifest,
        )

    if (
        provenance.get("split_manifest") != "test.csv"
        or not test_manifest.is_file()
        or provenance.get("split_manifest_sha256") != test_manifest_hash
    ):
        raise ValueError(f"Current test manifest provenance failed in {path}")

    validation_path = _resolve(str(document.get("validation_metrics_path", "")))
    validation_hash = document.get("validation_metrics_sha256")
    frozen_validation = selection.get("validation_metrics", {}).get(model_name)
    if (
        not validation_path.is_file()
        or sha256_file(validation_path) != validation_hash
        or not isinstance(frozen_validation, dict)
        or frozen_validation.get("sha256") != validation_hash
        or _resolve(str(frozen_validation.get("path", ""))) != validation_path
    ):
        raise ValueError(f"Validation metrics are not selected in {path}")
    validation_document = _read(validation_path)
    if (
        validation_document.get("schema_version") not in {1, 2}
        or validation_document.get("status") != "complete"
        or validation_document.get("split") != "val"
        or validation_document.get("experiment_name") != model_name
        or validation_document.get("metrics") != document.get("validation")
    ):
        raise ValueError(f"Embedded validation metrics disagree with {validation_path}")
    validation_provenance = validation_document.get("provenance")
    val_manifest = args.split_dir / "val.csv"
    if (
        not isinstance(validation_provenance, dict)
        or validation_provenance.get("model_sha256") != provenance.get("model_sha256")
        or validation_provenance.get("feature_config_sha256")
        != provenance.get("feature_config_sha256")
        or validation_provenance.get("feature_bindings") != bindings
        or validation_provenance.get("split_manifest") != "val.csv"
        or not val_manifest.is_file()
        or validation_provenance.get("split_manifest_sha256")
        != sha256_file(val_manifest)
    ):
        raise ValueError(f"Current validation provenance failed in {path}")


def _publish_group(staged: dict[Path, Path], *, force: bool) -> None:
    """Publish a set of staged files with rollback if any replacement fails."""

    for final in staged:
        if final.exists() and not force:
            raise FileExistsError(f"Output exists; pass --force to replace: {final}")

    candidates: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        # Prepare a complete candidate beside every destination before mutation.
        for final, source in staged.items():
            final.parent.mkdir(parents=True, exist_ok=True)
            handle, candidate_name = tempfile.mkstemp(
                prefix=f".{final.name}.candidate.", dir=final.parent
            )
            os.close(handle)
            candidate = Path(candidate_name)
            shutil.copyfile(source, candidate)
            with candidate.open("rb") as stream:
                os.fsync(stream.fileno())
            candidates[final] = candidate

        for final, candidate in candidates.items():
            if final.exists():
                handle, backup_name = tempfile.mkstemp(
                    prefix=f".{final.name}.backup.", dir=final.parent
                )
                os.close(handle)
                backup = Path(backup_name)
                backup.unlink()
                os.replace(final, backup)
                backups[final] = backup
            os.replace(candidate, final)
            installed.append(final)
    except OSError:
        for final in reversed(installed):
            final.unlink(missing_ok=True)
        for final, backup in backups.items():
            if backup.exists():
                os.replace(backup, final)
        raise
    finally:
        for candidate in candidates.values():
            candidate.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def collect(args: argparse.Namespace) -> pd.DataFrame:
    paths = sorted(
        path
        for path in args.metrics_dir.glob("traditional_*_metrics.json")
        if not path.name.endswith("_val_metrics.json")
    )
    if not paths:
        raise ValueError(f"No completed test metrics found in {args.metrics_dir}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    completed: dict[str, dict[str, Any]] = {}
    for path in paths:
        document = _read(path)
        if document.get("status") != "complete":
            print(f"[SKIP] Incomplete experiment metrics: {path}")
            continue
        if document.get("schema_version") != SELECTION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported metrics schema: {path}")
        name = str(document.get("model_name", ""))
        if not name or name in seen:
            raise ValueError(f"Missing or duplicate model_name in {path}")
        if name.startswith("ablation_"):
            raise ValueError(f"Validation-only ablation has a test metrics file: {path}")
        if name not in ALLOWED_TEST_MODELS:
            raise ValueError(f"Unknown completed Stage 2 main experiment: {name}")
        seen.add(name)
        completed[name] = document
        validation = document.get("validation")
        test = document.get("test")
        if not isinstance(validation, dict) or not isinstance(test, dict):
            raise ValueError(f"Metrics file lacks validation/test blocks: {path}")
        if int(validation.get("num_samples", -1)) != 5000:
            raise ValueError(f"Validation sample count is not 5,000 in {path}")
        if int(test.get("num_samples", -1)) != 5000:
            raise ValueError(f"Test sample count is not 5,000 in {path}")
        training_time = float(document.get("training_time_seconds", -1))
        test_time = float(test.get("inference_time_seconds", -1))
        feature_dim = int(document.get("feature_dim", 0))
        if (
            not math.isfinite(training_time)
            or training_time < 0
            or not math.isfinite(test_time)
            or test_time < 0
            or feature_dim < 1
        ):
            raise ValueError(f"Invalid timing or feature dimension in {path}")
        _validate_result_provenance(document, path, args)
        rows.append(
            {
                "model_name": name,
                "feature_type": str(document.get("feature_type", "")),
                "classifier": str(document.get("classifier", "")),
                "val_top1": _metric(validation, "top1_accuracy", path),
                "val_top5": _metric(validation, "top5_accuracy", path),
                "val_macro_precision": _metric(validation, "macro_precision", path),
                "val_macro_recall": _metric(validation, "macro_recall", path),
                "val_macro_f1": _metric(validation, "macro_f1", path),
                "test_top1": _metric(test, "top1_accuracy", path),
                "test_top5": _metric(test, "top5_accuracy", path),
                "test_macro_precision": _metric(test, "macro_precision", path),
                "test_macro_recall": _metric(test, "macro_recall", path),
                "test_macro_f1": _metric(test, "macro_f1", path),
                "train_time_seconds": training_time,
                "test_time_seconds": test_time,
                "feature_dim": feature_dim,
                "image_size": _image_size(document.get("image_size")),
            }
        )
    if not rows:
        raise ValueError("No valid completed test metrics were found")
    _validate_main_experiments(completed)
    result = pd.DataFrame(rows, columns=COLUMNS).sort_values(
        "model_name", kind="stable"
    )
    ablations = _collect_ablation_frame(args.metrics_dir)
    output_paths = {args.output, args.figure}
    if ablations is not None:
        output_paths.update({args.ablation_output, args.ablation_figure})
    for output_path in output_paths:
        if output_path.exists() and not args.force:
            raise FileExistsError(
                f"Output exists; pass --force to replace: {output_path}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".stage2_collection.", dir=args.output.parent
    ) as temporary_directory:
        root = Path(temporary_directory)
        staged_csv = root / "main.csv"
        staged_figure = root / "main.png"
        _atomic_csv(staged_csv, result)
        plot_model_comparison(result, staged_figure)
        staged_outputs = {
            args.output: staged_csv,
            args.figure: staged_figure,
        }
        if ablations is not None:
            staged_ablation_csv = root / "ablation.csv"
            staged_ablation_figure = root / "ablation.png"
            _atomic_csv(staged_ablation_csv, ablations)
            plot_validation_ablation(ablations, staged_ablation_figure)
            staged_outputs.update(
                {
                    args.ablation_output: staged_ablation_csv,
                    args.ablation_figure: staged_ablation_figure,
                }
            )
        _publish_group(staged_outputs, force=args.force)

    if ablations is None:
        print("[SKIP] No completed validation-only ablation metrics found")
    else:
        print(
            f"[PASS] Collected {len(ablations)} ablation-table row(s), including "
            f"the canonical full reference: {args.ablation_output}"
        )
    print(f"[PASS] Collected {len(result)} completed model(s): {args.output}")
    return result


def _collect_ablation_frame(metrics_dir: Path) -> pd.DataFrame | None:
    """Collect component ablations with the canonical SVM as the full reference.

    The full row deliberately reuses ``linear_svm`` validation metrics. Training
    an identical ``ablation_full`` model would waste compute and could introduce
    an accidental solver/backend mismatch in what should be a controlled study.
    """

    candidate_documents: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name in ABLATION_COMPONENTS:
        path = metrics_dir / f"traditional_{name}_val_metrics.json"
        if not path.is_file():
            continue
        document = _read(path)
        if document.get("status") != "complete":
            continue
        candidate_documents[name] = (path, document)
    duplicate_full = metrics_dir / "traditional_ablation_full_val_metrics.json"
    if duplicate_full.exists():
        raise ValueError(
            "ablation_full must not be trained; use canonical linear_svm validation "
            "metrics as the full reference"
        )
    if not candidate_documents:
        return None

    canonical_path = metrics_dir / "traditional_linear_svm_val_metrics.json"
    if not canonical_path.is_file():
        raise ValueError(
            "Completed component ablations require canonical linear_svm validation metrics"
        )
    canonical = _read(canonical_path)
    canonical_classifier = str(canonical.get("classifier", ""))
    canonical_backend_value = canonical.get("svm_backend")
    canonical_backend = (
        str(canonical_backend_value) if canonical_backend_value is not None else None
    )
    rows = [
        _ablation_row(
            canonical,
            canonical_path,
            source_name="linear_svm",
            output_name="full",
            expected_components=FULL_FEATURE_COMPONENTS,
            expected_dimension=CANONICAL_MODEL_SPECS["linear_svm"]["feature_dim"],
            expected_classifier=None,
            expected_backend=None,
        )
    ]
    for name in ABLATION_COMPONENTS:
        if name not in candidate_documents:
            continue
        path, document = candidate_documents[name]
        rows.append(
            _ablation_row(
                document,
                path,
                source_name=name,
                output_name=name,
                expected_components=ABLATION_COMPONENTS[name],
                expected_dimension=ABLATION_DIMENSIONS[name],
                expected_classifier=canonical_classifier,
                expected_backend=canonical_backend,
            )
        )
    return pd.DataFrame(rows)


def _ablation_row(
    document: dict[str, Any],
    path: Path,
    *,
    source_name: str,
    output_name: str,
    expected_components: Sequence[str],
    expected_dimension: int,
    expected_classifier: str | None,
    expected_backend: str | None,
) -> dict[str, Any]:
    """Validate one evaluator-produced validation document and return its row."""

    if (
        document.get("schema_version") not in {1, SELECTION_SCHEMA_VERSION}
        or document.get("status") != "complete"
        or document.get("split") != "val"
        or document.get("experiment_name") != source_name
        or document.get("model_name") != source_name
        or document.get("model_type") != "linear_svm"
    ):
        raise ValueError(f"Invalid ablation/reference metrics schema: {path}")
    if tuple(document.get("feature_components", ())) != tuple(expected_components):
        raise ValueError(f"Unexpected feature components in {path}")
    if int(document.get("feature_dim", 0)) != expected_dimension:
        raise ValueError(f"Unexpected feature dimension in {path}")
    _svm_identity(
        document,
        str(path),
        expected_classifier=expected_classifier,
        expected_backend=expected_backend,
    )
    metrics = document.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid ablation/reference metrics block: {path}")
    if metrics.get("num_samples") != 5000 or metrics.get("num_classes") != 500:
        raise ValueError(f"Invalid ablation/reference validation counts: {path}")
    train_time = float(document.get("training_time_seconds", -1))
    validation_time = float(metrics.get("inference_time_seconds", -1))
    if (
        not math.isfinite(train_time)
        or train_time < 0
        or not math.isfinite(validation_time)
        or validation_time < 0
    ):
        raise ValueError(f"Invalid ablation/reference timing in {path}")
    top1 = _metric(metrics, "top1_accuracy", path)
    top5 = _metric(metrics, "top5_accuracy", path)
    if top5 + 1e-12 < top1:
        raise ValueError(f"Validation top-5 is below top-1 in {path}")
    return {
        "experiment_name": output_name,
        "feature_components": "+".join(expected_components),
        "feature_dim": expected_dimension,
        "val_top1": top1,
        "val_top5": top5,
        "val_macro_precision": _metric(metrics, "macro_precision", path),
        "val_macro_recall": _metric(metrics, "macro_recall", path),
        "val_macro_f1": _metric(metrics, "macro_f1", path),
        "train_time_seconds": train_time,
        "val_time_seconds": validation_time,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate completed Stage 2 traditional test metrics."
    )
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
        help="Authoritative frozen Stage 2 selection required by test provenance.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables/stage2_traditional_results.csv"),
    )
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=Path("results/tables/stage2_feature_ablation_results.csv"),
    )
    parser.add_argument(
        "--ablation-figure",
        type=Path,
        default=Path("results/figures/stage2_feature_ablation_comparison.png"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/figures/stage2_traditional_model_comparison.png"),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        collect(args)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        print(f"[FAIL] {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
