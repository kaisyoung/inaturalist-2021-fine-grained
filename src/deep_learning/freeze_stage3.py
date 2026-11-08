"""Freeze validation-qualified Stage 3 models before held-out test access.

The freeze is deliberately stricter than a list of checkpoint names.  It binds
the complete validation evidence for every completed candidate, records the
deterministic augmentation selection, and binds the checkpoint/config/log
artifacts for the exact models that may be evaluated on the test split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.common import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_FREEZE_PATH = (
    PROJECT_ROOT / "results/metrics/deep_learning/stage3_freeze.json"
).resolve()
CANONICAL_ACCESS_RECORD = (
    PROJECT_ROOT / "results/metrics/deep_learning/stage3_test_access.json"
).resolve()
STAGE1_FILES = (
    "selected_classes_seed42.txt",
    "class_mapping.json",
    "train.csv",
    "val.csv",
    "test.csv",
    "split_summary.json",
)
STAGE2_FILES = (
    "STAGE2_FINAL_ACCEPTANCE_REPORT.md",
    "STAGE2_TRADITIONAL_REPORT.md",
    "results/metrics/stage2_selection_manifest.json",
    "results/metrics/stage2_freeze.json",
    "results/tables/stage2_traditional_results.csv",
    "results/tables/stage2_feature_ablation_results.csv",
)
STAGE2_ACCEPTED_HASHES = {
    "STAGE2_FINAL_ACCEPTANCE_REPORT.md": "a2a6aa5df577b3d56da5be1a3329227250d15e978b2f3a912f573b90a014c6d9",
    "STAGE2_TRADITIONAL_REPORT.md": "874130190f8a9f976e84009e0b1459d268626361e3e38382c935de66e51ac9e6",
    # Replaced only by scripts.rehash_stage2_evidence after the authoritative
    # schema-2 selection manifest has been rebuilt and reviewed.
    "results/metrics/stage2_selection_manifest.json": "5b39108a92ab34d052010e99b8cbcb5c7af5ba685889ad5f143ae6d26f8c8f05",
    "results/metrics/stage2_freeze.json": "f5b908ce68b0cfb95b3047c2ba5a875fb975e56026b6337c966d5c326140562d",
    "results/tables/stage2_traditional_results.csv": "b2a2d554975c06e28639a39d44b80e120e36dc5eb8e23bc0d75715a8ac15768c",
    "results/tables/stage2_feature_ablation_results.csv": "ba723a0a362bd50a569a12bbcb6f56578ed37f1a6f351daed793c74ad692e2ce",
}
STAGE2_SELECTION_SCHEMA_VERSION = 2
STAGE2_ALLOWED_TEST_MODELS = frozenset(
    ("linear_svm", "random_forest", "sift_bovw_svm")
)

SCRATCH_EXPERIMENT = "resnet18_scratch_basic"
PRETRAINED_RESNET_EXPERIMENTS = (
    "resnet18_pretrained_basic",
    "resnet18_pretrained_strong_aug",
)
EFFICIENTNET_EXPERIMENT = "efficientnet_b0_pretrained_strong_aug"
FROZEN_HEAD_EXPERIMENT = "resnet18_pretrained_frozen_head"
CANONICAL_EXPERIMENTS = (
    SCRATCH_EXPERIMENT,
    *PRETRAINED_RESNET_EXPERIMENTS,
    EFFICIENTNET_EXPERIMENT,
    FROZEN_HEAD_EXPERIMENT,
)
REQUIRED_EXPERIMENTS = frozenset((SCRATCH_EXPERIMENT, *PRETRAINED_RESNET_EXPERIMENTS))
REQUIRED_METRICS = (
    "loss",
    "top1_accuracy",
    "top5_accuracy",
    "overall_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)
MAX_VALIDATION_METRIC_TOLERANCE = 1e-6


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _portable(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve(value: str | Path, *, relative_to: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_path = (PROJECT_ROOT / path).resolve()
    if project_path.exists() or relative_to is None:
        return project_path
    return (relative_to / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency precheck covers this.
        raise RuntimeError("PyTorch is required to inspect Stage 3 checkpoints") from error
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # PyTorch uses several backend-specific errors.
        raise ValueError(f"Could not load checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    return checkpoint


def _finite(value: Any, field: str, *, minimum: float | None = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not numeric") from error
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{field} is invalid")
    return number


def _stored_metric(metrics: Mapping[str, Any], key: str, field: str) -> float:
    aliases = {
        "loss": ("loss", "val_loss"),
        "top1_accuracy": ("top1_accuracy", "val_top1_accuracy", "top1", "val_top1"),
        "top5_accuracy": ("top5_accuracy", "val_top5_accuracy", "top5", "val_top5"),
        "overall_accuracy": (
            "overall_accuracy",
            "val_overall_accuracy",
            "top1_accuracy",
            "val_top1_accuracy",
        ),
        "macro_precision": ("macro_precision", "val_macro_precision"),
        "macro_recall": ("macro_recall", "val_macro_recall"),
        "macro_f1": ("macro_f1", "val_macro_f1"),
    }
    for alias in aliases[key]:
        if alias in metrics:
            return _finite(metrics[alias], f"{field}.{alias}")
    raise ValueError(f"{field} has no value for {key}")


def _verify_validation_agreement(
    observed: Mapping[str, float],
    persisted_best: Mapping[str, Any],
    checkpoint_best: Mapping[str, Any],
    *,
    tolerance: float,
    name: str,
) -> None:
    if tolerance > MAX_VALIDATION_METRIC_TOLERANCE:
        raise ValueError(
            f"{name} validation recheck tolerance {tolerance} exceeds the fixed "
            f"maximum {MAX_VALIDATION_METRIC_TOLERANCE}"
        )
    for key in REQUIRED_METRICS:
        fresh = float(observed[key])
        persisted = _stored_metric(persisted_best, key, f"{name}.stored_best")
        checkpoint = _stored_metric(checkpoint_best, key, f"{name}.checkpoint_best")
        if not (
            math.isclose(fresh, persisted, rel_tol=tolerance, abs_tol=tolerance)
            and math.isclose(fresh, checkpoint, rel_tol=tolerance, abs_tol=tolerance)
            and math.isclose(persisted, checkpoint, rel_tol=tolerance, abs_tol=tolerance)
        ):
            raise ValueError(
                f"{name} independent validation disagrees for {key}: "
                f"observed={fresh}, persisted={persisted}, checkpoint={checkpoint}"
            )


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return {
        "path": _portable(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _parse_named_paths(
    raw: Any,
    *,
    selected: Sequence[str] = (),
    suffixes: Sequence[str] = (),
) -> dict[str, Path]:
    """Accept NAME=PATH CLI values, mappings, or selected-order plain paths."""

    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(name): Path(path) for name, path in raw.items()}
    values = [raw] if isinstance(raw, (str, Path)) else list(raw)
    result: dict[str, Path] = {}
    plain: list[Path] = []
    for value in values:
        text = str(value)
        if "=" in text:
            name, path_text = text.split("=", 1)
            if not name or not path_text or name in result:
                raise ValueError(f"Invalid or duplicate NAME=PATH value: {text!r}")
            result[name] = Path(path_text)
        else:
            plain.append(Path(text))
    if plain:
        if len(plain) == len(selected) and not result:
            result.update(zip(selected, plain))
        else:
            for path in plain:
                name = path.name
                matched = False
                for suffix in suffixes:
                    if name.endswith(suffix):
                        name = name[: -len(suffix)]
                        matched = True
                        break
                if not matched or not name or name in result:
                    raise ValueError(
                        "Plain artifact paths must either align with --selected order "
                        "or use a canonical experiment suffix"
                    )
                result[name] = path
    return result


def _stage1_artifacts(split_dir: Path) -> dict[str, dict[str, Any]]:
    split_dir = split_dir.resolve()
    entries = {name: _artifact(split_dir / name) for name in STAGE1_FILES}
    summary = _read_json(split_dir / "split_summary.json")
    if summary.get("schema_version") != 2:
        raise ValueError("Stage 1 split_summary.json is not schema version 2")
    frozen = summary.get("artifacts")
    if not isinstance(frozen, dict):
        raise ValueError("Stage 1 split summary has no artifact hash block")
    for name in STAGE1_FILES:
        if name == "split_summary.json":
            continue
        if frozen.get(name, {}).get("sha256") != entries[name]["sha256"]:
            raise ValueError(f"Stage 1 split-summary hash disagrees for {name}")
    return entries


def _stage2_artifacts(stage2_paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    entries = {_portable(path): _artifact(path.resolve()) for path in stage2_paths}
    selection_path = next(
        (
            path.resolve()
            for path in stage2_paths
            if path.name == "stage2_selection_manifest.json"
        ),
        None,
    )
    freeze_path = next(
        (path.resolve() for path in stage2_paths if path.name == "stage2_freeze.json"),
        None,
    )
    if selection_path is None:
        raise ValueError(
            "Frozen Stage 2 evidence must include stage2_selection_manifest.json"
        )
    if freeze_path is None:
        raise ValueError("Frozen Stage 2 evidence must include stage2_freeze.json")
    selection = _read_json(selection_path)
    freeze = _read_json(freeze_path)
    if (
        selection.get("schema_version") != STAGE2_SELECTION_SCHEMA_VERSION
        or selection.get("status") != "frozen"
        or set(selection.get("selected_models", ())) != STAGE2_ALLOWED_TEST_MODELS
        or set(selection.get("allowed_test_models", ())) != STAGE2_ALLOWED_TEST_MODELS
        or not isinstance(selection.get("models"), Mapping)
        or set(selection["models"]) != STAGE2_ALLOWED_TEST_MODELS
        or not isinstance(selection.get("feature_configs"), Mapping)
        or set(selection["feature_configs"]) != STAGE2_ALLOWED_TEST_MODELS
        or not isinstance(selection.get("validation_metrics"), Mapping)
        or set(selection["validation_metrics"]) != STAGE2_ALLOWED_TEST_MODELS
    ):
        raise ValueError(
            "stage2_selection_manifest.json is not the authoritative schema-2 "
            "three-model selection"
        )
    grid = selection.get("validation_grid")
    if not isinstance(grid, Mapping) or int(grid.get("rows", 0)) != 19:
        raise ValueError("Stage 2 selection does not bind the 19-row validation grid")

    selection_binding = freeze.get("selection_manifest")
    expected_freeze_keys = set(selection) | {
        "selection_schema_version",
        "compatibility_role",
        "selection_manifest",
    }
    if (
        freeze.get("schema_version") != 1
        or freeze.get("selection_schema_version") != STAGE2_SELECTION_SCHEMA_VERSION
        or freeze.get("status") != "frozen"
        or freeze.get("compatibility_role")
        != "stage2_freeze alias for the authoritative selection"
        or set(freeze) != expected_freeze_keys
        or not isinstance(selection_binding, Mapping)
        or _resolve(str(selection_binding.get("path", ""))) != selection_path
        or selection_binding.get("sha256") != sha256_file(selection_path)
    ):
        raise ValueError(
            "stage2_freeze.json is not a valid alias of the authoritative "
            "Stage 2 selection"
        )
    for key, value in selection.items():
        if key != "schema_version" and freeze.get(key) != value:
            raise ValueError(
                f"stage2_freeze.json disagrees with the Stage 2 selection: {key}"
            )
    return dict(sorted(entries.items()))


def _canonical_stage2_paths(raw_stage2: Sequence[Path] | None) -> list[Path]:
    paths = [(PROJECT_ROOT / value).resolve() for value in STAGE2_FILES]
    if raw_stage2:
        supplied = {Path(value).resolve() for value in raw_stage2}
        if supplied != set(paths):
            raise ValueError("Stage 2 evidence paths are fixed and cannot be overridden")
    return paths


def _validate_stage1_in_checkpoint(
    checkpoint: Mapping[str, Any], stage1: Mapping[str, Mapping[str, Any]], name: str
) -> None:
    embedded = checkpoint.get("stage1_artifacts")
    if not isinstance(embedded, Mapping) or set(embedded) != set(STAGE1_FILES):
        raise ValueError(f"{name} checkpoint does not bind exactly six Stage 1 artifacts")
    for file_name, current in stage1.items():
        entry = embedded[file_name]
        embedded_hash = entry.get("sha256") if isinstance(entry, Mapping) else entry
        if embedded_hash != current["sha256"]:
            raise ValueError(f"{name} checkpoint Stage 1 hash mismatch: {file_name}")


def _validation_artifacts(document: Mapping[str, Any], metrics_path: Path) -> dict[str, Any]:
    raw = document.get("artifacts")
    if not isinstance(raw, Mapping):
        raise ValueError(f"Validation metrics has no artifact manifest: {metrics_path}")
    verified: dict[str, Any] = {}
    for key, entry in raw.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Malformed validation artifact {key!r}")
        artifact_path = _resolve(str(entry.get("path", "")), relative_to=metrics_path.parent)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Validation artifact not found: {artifact_path}")
        actual_hash = sha256_file(artifact_path)
        if actual_hash != entry.get("sha256"):
            raise ValueError(f"Validation artifact changed after evaluation: {artifact_path}")
        verified[str(key)] = {
            "path": _portable(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": actual_hash,
        }
    return dict(sorted(verified.items()))


def _validate_candidate(
    name: str,
    *,
    checkpoint_path: Path,
    metrics_path: Path,
    best_metrics_path: Path,
    resolved_config_path: Path,
    source_config_override: Path | None,
    log_path: Path,
    stage1: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if name not in CANONICAL_EXPERIMENTS:
        raise ValueError(f"Unknown Stage 3 experiment: {name}")
    for path in (
        checkpoint_path,
        metrics_path,
        best_metrics_path,
        resolved_config_path,
        log_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required {name} artifact not found: {path}")

    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    required_checkpoint_fields = {
        "experiment_name",
        "checkpoint_kind",
        "epoch",
        "model_name",
        "pretrained",
        "freeze_backbone",
        "num_classes",
        "image_size",
        "model_state_dict",
        "best_epoch",
        "best_metrics",
        "resolved_config",
        "resolved_config_sha256",
        "source_config_path",
        "source_config_sha256",
        "stage1_artifacts",
        "versions",
        "device",
        "parameter_counts",
        "training_time_seconds",
    }
    missing = sorted(required_checkpoint_fields - set(checkpoint))
    if missing:
        raise ValueError(f"{name} checkpoint is missing keys: {missing}")
    if (
        checkpoint.get("experiment_name") != name
        or checkpoint.get("checkpoint_kind") != "best"
        or checkpoint.get("num_classes") != 500
        or checkpoint.get("best_epoch") != checkpoint.get("epoch")
    ):
        raise ValueError(f"{name} is not a canonical best 500-class checkpoint")
    _validate_stage1_in_checkpoint(checkpoint, stage1, name)

    resolved_hash = sha256_file(resolved_config_path)
    if checkpoint.get("resolved_config_sha256") != resolved_hash:
        raise ValueError(f"{name} resolved config changed after training")
    resolved = _read_json(resolved_config_path)
    if checkpoint.get("resolved_config") != resolved:
        raise ValueError(f"{name} embedded and persisted resolved configs differ")

    source_config_path = (
        source_config_override.resolve()
        if source_config_override is not None
        else _resolve(
            str(checkpoint.get("source_config_path", "")),
            relative_to=resolved_config_path.parent,
        )
    )
    if not source_config_path.is_file():
        raise FileNotFoundError(f"{name} source config not found: {source_config_path}")
    source_hash = sha256_file(source_config_path)
    if source_hash != checkpoint.get("source_config_sha256"):
        raise ValueError(f"{name} source config changed after training")

    metrics = _read_json(metrics_path)
    if (
        metrics.get("schema_version") != 1
        or metrics.get("status") != "complete"
        or metrics.get("split") != "val"
        or metrics.get("experiment_name") != name
    ):
        raise ValueError(f"{name} independent validation metrics are invalid")
    values = metrics.get("metrics")
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} validation metrics block is missing")
    normalized = {key: _finite(values.get(key), f"{name}.{key}") for key in REQUIRED_METRICS}
    if any(normalized[key] > 1.0 for key in REQUIRED_METRICS if key != "loss"):
        raise ValueError(f"{name} validation classification metric is outside [0, 1]")
    if not math.isclose(
        normalized["top1_accuracy"], normalized["overall_accuracy"], abs_tol=1e-12
    ):
        raise ValueError(f"{name} validation top-1 and overall accuracy disagree")
    if normalized["top5_accuracy"] + 1e-12 < normalized["top1_accuracy"]:
        raise ValueError(f"{name} validation top-5 is below top-1")
    if int(values.get("num_classes", 0)) != 500 or int(values.get("num_samples", 0)) != 5000:
        raise ValueError(f"{name} validation counts must be 500 classes and 5000 samples")
    agreement = metrics.get("validation_recheck")
    if not isinstance(agreement, Mapping) or agreement.get("agrees_with_stored_best") is not True:
        raise ValueError(f"{name} has no successful independent validation recheck")
    provenance = metrics.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{name} validation provenance is missing")
    expected_provenance = {
        "checkpoint_sha256": checkpoint_hash,
        "resolved_config_sha256": resolved_hash,
        "source_config_sha256": source_hash,
        "split_manifest_sha256": stage1["val.csv"]["sha256"],
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(f"{name} validation provenance mismatch: {key}")

    best_document = _read_json(best_metrics_path)
    if (
        best_document.get("schema_version") != 1
        or best_document.get("experiment_name") != name
        or best_document.get("checkpoint_sha256") != checkpoint_hash
        or best_document.get("resolved_config_sha256") != resolved_hash
    ):
        raise ValueError(f"{name} stored best-epoch metrics provenance is invalid")
    if int(best_document.get("best_epoch", -1)) != int(checkpoint["best_epoch"]):
        raise ValueError(f"{name} stored best epoch disagrees with checkpoint")
    best_values = best_document.get("metrics")
    if not isinstance(best_values, Mapping):
        raise ValueError(f"{name} stored best metrics block is missing")
    checkpoint_best = checkpoint.get("best_metrics")
    if not isinstance(checkpoint_best, Mapping):
        raise ValueError(f"{name} checkpoint best metrics block is missing")
    tolerance = _finite(agreement.get("tolerance", 0.0), f"{name}.recheck_tolerance")
    _verify_validation_agreement(
        normalized,
        best_values,
        checkpoint_best,
        tolerance=tolerance,
        name=name,
    )

    artifacts = _validation_artifacts(metrics, metrics_path)
    return {
        "experiment_name": name,
        "model_name": str(checkpoint["model_name"]),
        "pretrained": bool(checkpoint["pretrained"]),
        "freeze_backbone": bool(checkpoint["freeze_backbone"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "validation_metrics": {**normalized, "num_samples": 5000, "num_classes": 500},
        "checkpoint": _artifact(checkpoint_path),
        "resolved_config": _artifact(resolved_config_path),
        "source_config": _artifact(source_config_path),
        "training_log": _artifact(log_path),
        "stored_best_metrics": _artifact(best_metrics_path),
        "independent_validation_metrics": _artifact(metrics_path),
        "validation_artifacts": artifacts,
        "versions": checkpoint["versions"],
        "device": checkpoint["device"],
        "parameter_counts": checkpoint["parameter_counts"],
        "training_time_seconds": _finite(
            checkpoint["training_time_seconds"], f"{name}.training_time_seconds"
        ),
    }


def _ranking_key(entry: Mapping[str, Any]) -> tuple[float, float, float, float, int, str]:
    metrics = entry["validation_metrics"]
    return (
        -float(metrics["macro_f1"]),
        -float(metrics["top1_accuracy"]),
        -float(metrics["top5_accuracy"]),
        float(metrics["loss"]),
        int(entry["best_epoch"]),
        str(entry["experiment_name"]),
    )


def _expected_selection(candidates: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], str, dict[str, Any]]:
    missing = sorted(REQUIRED_EXPERIMENTS - set(candidates))
    if missing:
        raise ValueError(f"Stage 3 freeze requires completed D1 and D2 validation: {missing}")
    ranked_resnets = sorted(
        (candidates[name] for name in PRETRAINED_RESNET_EXPERIMENTS), key=_ranking_key
    )
    pretrained_choice = str(ranked_resnets[0]["experiment_name"])
    selected = [SCRATCH_EXPERIMENT, pretrained_choice]
    if EFFICIENTNET_EXPERIMENT in candidates:
        selected.append(EFFICIENTNET_EXPERIMENT)
    pretrained_selected = [
        candidates[name]
        for name in selected
        if bool(candidates[name].get("pretrained", False))
    ]
    analysis_model = str(sorted(pretrained_selected, key=_ranking_key)[0]["experiment_name"])
    decision = {
        "policy": (
            "macro_f1 descending, top1_accuracy descending, top5_accuracy descending, "
            "loss ascending, best_epoch ascending, experiment_name ascending"
        ),
        "pretrained_resnet_ranking": [
            str(entry["experiment_name"]) for entry in ranked_resnets
        ],
        "selected_pretrained_resnet": pretrained_choice,
        "analysis_model": analysis_model,
        "analysis_model_policy": "highest locked validation ranking among selected pretrained models",
        "scratch_required": SCRATCH_EXPERIMENT,
        "efficientnet_included_if_completed": EFFICIENTNET_EXPERIMENT in candidates,
        "frozen_head_validation_only": FROZEN_HEAD_EXPERIMENT in candidates,
    }
    return selected, analysis_model, decision


def _existing_test_evidence(args: argparse.Namespace) -> list[Path]:
    canonical_directories = (
        PROJECT_ROOT / "results/metrics/deep_learning",
        PROJECT_ROOT / "results/predictions/deep_learning",
        PROJECT_ROOT / "results/tables/deep_learning",
        PROJECT_ROOT / "results/figures/deep_learning",
    )
    supplied_directories = (
        Path(getattr(args, "metrics_dir", canonical_directories[0])),
        Path(getattr(args, "predictions_dir", canonical_directories[1])),
        Path(getattr(args, "tables_dir", canonical_directories[2])),
        Path(getattr(args, "figures_dir", canonical_directories[3])),
    )
    directory_sets = {
        tuple(path.resolve() for path in canonical_directories),
        tuple(path.resolve() for path in supplied_directories),
    }
    found = [CANONICAL_ACCESS_RECORD] if CANONICAL_ACCESS_RECORD.exists() else []
    supplied_access = getattr(args, "access_record", None)
    if supplied_access is not None and Path(supplied_access).resolve().exists():
        found.append(Path(supplied_access).resolve())
    for directory, pattern in (
        *((directories[0], "*_test_metrics.json") for directories in directory_sets),
        *((directories[0], "*_test_confusion.npz") for directories in directory_sets),
        *((directories[1], "*_test_predictions.csv") for directories in directory_sets),
        *((directories[2], "*_test_*.csv") for directories in directory_sets),
        *((directories[3], "*_test_*.png") for directories in directory_sets),
    ):
        found.extend(directory.glob(pattern))
    return sorted(set(path.resolve() for path in found))


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    """Validate all candidate evidence and atomically write ``stage3_freeze.json``."""

    output = Path(
        getattr(args, "output", CANONICAL_FREEZE_PATH)
    ).resolve()
    if output != CANONICAL_FREEZE_PATH:
        raise ValueError(
            f"Canonical Stage 3 freeze path is fixed: {CANONICAL_FREEZE_PATH}"
        )
    supplied_access = getattr(args, "access_record", None)
    if supplied_access is not None and Path(supplied_access).resolve() != CANONICAL_ACCESS_RECORD:
        raise ValueError(
            f"Canonical Stage 3 test-access path is fixed: {CANONICAL_ACCESS_RECORD}"
        )
    selected = list(
        getattr(args, "selected_experiment", None)
        or getattr(args, "selected", None)
        or []
    )
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Repeat --selected-experiment with each unique frozen model")

    existing_test = _existing_test_evidence(args)
    if existing_test:
        raise FileExistsError(
            "Stage 3 freeze cannot be created or replaced after test access: "
            + ", ".join(str(path) for path in existing_test)
        )
    predecessor: dict[str, Any] | None = None
    if output.exists():
        if not bool(getattr(args, "dangerous_replace", False)):
            raise FileExistsError(f"Stage 3 freeze already exists and is immutable: {output}")
        current_hash = sha256_file(output)
        supplied_hash = getattr(args, "current_freeze_sha256", None)
        if supplied_hash != current_hash:
            raise ValueError("Dangerous freeze replacement requires the current freeze hash")
        old = _read_json(output)
        predecessor = {
            "path": _portable(output),
            "sha256": current_hash,
            "created_at_utc": old.get("created_at_utc"),
        }

    split_dir = Path(getattr(args, "split_dir", "data/splits")).resolve()
    stage1 = _stage1_artifacts(split_dir)
    raw_stage2 = getattr(args, "stage2_artifact", None)
    stage2_paths = _canonical_stage2_paths(raw_stage2)
    stage2 = _stage2_artifacts(stage2_paths)
    if set(STAGE2_ACCEPTED_HASHES) != set(STAGE2_FILES):
        raise RuntimeError("Stage 2 accepted-hash allowlist is incomplete")
    for relative, expected_hash in STAGE2_ACCEPTED_HASHES.items():
        entry = stage2.get(relative)
        if not isinstance(entry, Mapping) or entry.get("sha256") != expected_hash:
            raise ValueError(f"Frozen Stage 2 evidence changed: {relative}")
    stage2_selection = _read_json(
        next(
            path
            for path in stage2_paths
            if path.name == "stage2_selection_manifest.json"
        )
    )
    for file_name, current in stage1.items():
        frozen = stage2_selection.get("stage1_artifacts", {}).get(file_name, {})
        if frozen.get("sha256") != current["sha256"]:
            raise ValueError(f"Stage 1 changed relative to frozen Stage 2: {file_name}")

    checkpoint_overrides = _parse_named_paths(
        getattr(args, "checkpoint", None), selected=selected, suffixes=("_best.pt",)
    )
    metric_overrides = _parse_named_paths(
        getattr(args, "metric", None), selected=selected, suffixes=("_val_metrics.json",)
    )
    candidate_metric_overrides = _parse_named_paths(
        getattr(args, "candidate_metric", None), suffixes=("_val_metrics.json",)
    )
    metric_overrides.update(candidate_metric_overrides)
    best_metric_overrides = _parse_named_paths(
        getattr(args, "best_metric", None), suffixes=("_best_val_metrics.json",)
    )
    resolved_overrides = _parse_named_paths(
        getattr(args, "resolved_config", None),
        selected=selected,
        suffixes=("_resolved_config.json",),
    )
    source_overrides = _parse_named_paths(
        getattr(args, "source_config", None), selected=selected, suffixes=(".yaml", ".yml")
    )
    log_overrides = _parse_named_paths(
        getattr(args, "log", None), selected=selected, suffixes=("_train_log.jsonl",)
    )

    models_dir = Path(getattr(args, "models_dir", "results/models/deep_learning"))
    metrics_dir = Path(getattr(args, "metrics_dir", "results/metrics/deep_learning"))
    logs_dir = Path(getattr(args, "logs_dir", "results/logs/deep_learning"))
    candidate_names = {
        path.name[: -len("_val_metrics.json")]
        for path in metrics_dir.glob("*_val_metrics.json")
        if not path.name.endswith("_best_val_metrics.json")
    }
    candidate_names.update(metric_overrides)
    candidate_names.intersection_update(CANONICAL_EXPERIMENTS)
    candidate_names.update(REQUIRED_EXPERIMENTS)

    candidates: dict[str, dict[str, Any]] = {}
    for name in sorted(candidate_names):
        metrics_path = metric_overrides.get(name, metrics_dir / f"{name}_val_metrics.json").resolve()
        # Optional experiments are candidates only after a complete independent eval exists.
        if name not in REQUIRED_EXPERIMENTS and not metrics_path.is_file():
            continue
        candidates[name] = _validate_candidate(
            name,
            checkpoint_path=checkpoint_overrides.get(
                name, models_dir / f"{name}_best.pt"
            ).resolve(),
            metrics_path=metrics_path,
            best_metrics_path=best_metric_overrides.get(
                name, metrics_dir / f"{name}_best_val_metrics.json"
            ).resolve(),
            resolved_config_path=resolved_overrides.get(
                name, metrics_dir / f"{name}_resolved_config.json"
            ).resolve(),
            source_config_override=(
                source_overrides[name].resolve() if name in source_overrides else None
            ),
            log_path=log_overrides.get(name, logs_dir / f"{name}_train_log.jsonl").resolve(),
            stage1=stage1,
        )

    expected_selected, analysis_model, decision = _expected_selection(candidates)
    if set(selected) != set(expected_selected) or len(selected) != len(expected_selected):
        raise ValueError(
            "Selected experiments do not match the locked validation decision; "
            f"expected {expected_selected}, got {selected}"
        )
    # Canonicalize order so argument order cannot alter the freeze hash semantics.
    selected = expected_selected
    frozen_models = {
        name: {
            **candidates[name],
            "selection_reason": (
                "required scratch baseline"
                if name == SCRATCH_EXPERIMENT
                else "validation-ranked pretrained ResNet18 augmentation"
                if name in PRETRAINED_RESNET_EXPERIMENTS
                else "completed distinct efficient architecture"
            ),
        }
        for name in selected
    }
    candidate_summary = {
        name: {
            "experiment_name": name,
            "model_name": entry["model_name"],
            "pretrained": entry["pretrained"],
            "freeze_backbone": entry["freeze_backbone"],
            "best_epoch": entry["best_epoch"],
            "validation_metrics": entry["validation_metrics"],
            "checkpoint_sha256": entry["checkpoint"]["sha256"],
            "independent_validation_metrics": entry["independent_validation_metrics"],
            "selected_for_test": name in selected,
        }
        for name, entry in sorted(candidates.items())
    }

    document: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen",
        "policy": "Validation-ranked model set fixed before any Stage 3 test access",
        "created_at_utc": _utc_now(),
        "stage1_artifacts": stage1,
        "stage2_artifacts": stage2,
        "completed_candidates": candidate_summary,
        "selection_decision": decision,
        "selected_experiments": selected,
        "analysis_model": analysis_model,
        "models": frozen_models,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "selected_model_versions": {
                name: candidates[name]["versions"] for name in selected
            },
            "selected_model_devices": {
                name: candidates[name]["device"] for name in selected
            },
        },
    }
    if predecessor is not None:
        document["predecessor"] = predecessor
    _atomic_json(output, document)
    print(f"[PASS] Stage 3 frozen for {len(selected)} model(s): {output}")
    print(f"[PASS] Analysis model: {analysis_model}")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze validation-qualified Stage 3 models before test access."
    )
    parser.add_argument(
        "--selected-experiment",
        "--selected",
        action="append",
        dest="selected_experiment",
        required=True,
        help="Experiment selected for test; repeat for the complete frozen set",
    )
    parser.add_argument("--checkpoint", action="append", help="Optional NAME=PATH")
    parser.add_argument("--metric", action="append", help="Optional selected NAME=PATH")
    parser.add_argument(
        "--candidate-metric", action="append", help="Optional candidate NAME=PATH"
    )
    parser.add_argument("--best-metric", action="append", help="Optional NAME=PATH")
    parser.add_argument("--resolved-config", action="append", help="Optional NAME=PATH")
    parser.add_argument("--source-config", action="append", help="Optional NAME=PATH")
    parser.add_argument("--log", action="append", help="Optional NAME=PATH")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--stage2-artifact",
        type=Path,
        action="append",
        help="Override the frozen Stage 2 evidence set; repeat for every artifact",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("results/models/deep_learning"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics/deep_learning"))
    parser.add_argument("--logs-dir", type=Path, default=Path("results/logs/deep_learning"))
    parser.add_argument(
        "--predictions-dir", type=Path, default=Path("results/predictions/deep_learning")
    )
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables/deep_learning"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures/deep_learning"))
    parser.add_argument("--access-record", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics/deep_learning/stage3_freeze.json"),
    )
    parser.add_argument("--dangerous-replace", action="store_true")
    parser.add_argument("--current-freeze-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.access_record is None:
        args.access_record = args.metrics_dir / "stage3_test_access.json"
    try:
        freeze(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"[FAIL] {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
