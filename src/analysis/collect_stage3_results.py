"""Collect completed Stage 3 metrics without modifying frozen Stage 2 evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.analysis.plotting import plot_stage3_validation_comparison
from src.data.common import sha256_file
from src.deep_learning.train_utils import atomic_write_bytes


ACCEPTED_STAGE2_HASHES = {
    "results/tables/stage2_traditional_results.csv": (
        "b2a2d554975c06e28639a39d44b80e120e36dc5eb8e23bc0d75715a8ac15768c"
    )
}
ACCEPTED_STAGE2_RESULTS_SHA256 = ACCEPTED_STAGE2_HASHES[
    "results/tables/stage2_traditional_results.csv"
]
ACCEPTED_VAL_SPLIT_SHA256 = (
    "4adf287996d35d9e57f829426ff7480bbaf0b74e1c319a29d57fe74b3f0366bb"
)


COLUMNS = [
    "model_name",
    "method_family",
    "architecture",
    "pretrained",
    "augmentation",
    "freeze_backbone",
    "train_top1_best",
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
    "training_time_seconds",
    "test_time_seconds",
    "num_parameters",
    "trainable_parameters",
    "notes",
]

ABLATION_COLUMNS = [
    "model_name",
    "architecture",
    "pretrained",
    "augmentation",
    "freeze_backbone",
    "train_top1_best",
    "val_top1",
    "val_top5",
    "val_macro_precision",
    "val_macro_recall",
    "val_macro_f1",
    "training_time_seconds",
    "num_parameters",
    "trainable_parameters",
    "notes",
]

STAGE2_TABLE_SPECS = {
    "linear_svm": {
        "feature_type": "color_texture_hog_sift_bovw",
        "classifier": "LinearSVC",
        "feature_dim": 3704,
    },
    "random_forest": {
        "feature_type": "color_texture_hog",
        "classifier": "RandomForestClassifier",
        "feature_dim": 2424,
    },
    "sift_bovw_svm": {
        "feature_type": "sift_bovw",
        "classifier": "LinearSVC",
        "feature_dim": 1280,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _metrics(document: dict[str, Any]) -> dict[str, Any]:
    value = document.get("metrics", document)
    if not isinstance(value, dict):
        raise ValueError("metrics must be a JSON object")
    return value


def _metric(metrics: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in metrics:
            value = float(metrics[name])
            if not np.isfinite(value):
                raise ValueError(f"metric {name} is not finite")
            return value
    raise ValueError(f"none of the required metrics is present: {names}")


def _optional_metric(document: dict[str, Any] | None, *names: str) -> float | str:
    if document is None:
        return ""
    metrics = _metrics(document)
    try:
        return _metric(metrics, *names)
    except ValueError:
        for name in names:
            if name in document:
                value = float(document[name])
                if np.isfinite(value):
                    return value
        return ""


def _validate_score(value: float, label: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{label} must be within [0, 1], got {value}")


def _read_history(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"invalid JSONL record at {path}:{line_number}")
        records.append(value)
    return records


def _verify_validation_document(
    *,
    experiment: str,
    val_path: Path,
    document: Mapping[str, Any],
    status: Mapping[str, Any],
    best: Mapping[str, Any],
    resolved_path: Path,
    best_path: Path,
    log_path: Path,
    freeze: Mapping[str, Any] | None,
) -> None:
    provenance = document.get("provenance")
    recheck = document.get("validation_recheck")
    artifacts = document.get("artifacts")
    if (
        document.get("schema_version") != 1
        or document.get("status") != "complete"
        or document.get("split") != "val"
        or document.get("experiment_name") != experiment
        or not isinstance(provenance, Mapping)
        or not isinstance(recheck, Mapping)
        or recheck.get("agrees_with_stored_best") is not True
        or float(recheck.get("tolerance", 1.0)) > 1e-6
        or not isinstance(artifacts, Mapping)
    ):
        raise ValueError(f"invalid independent validation evidence for {experiment}")
    required_artifacts = {
        "predictions",
        "confusion",
        "per_class",
        "pairs",
        "confusion_figure",
        "pairs_figure",
    }
    if set(artifacts) != required_artifacts:
        raise ValueError(f"incomplete validation artifact set for {experiment}")
    for key, entry in artifacts.items():
        _artifact_path(entry, label=f"{experiment}.val.{key}")

    bound_files = {
        "checkpoint": (
            provenance.get("checkpoint_path"),
            provenance.get("checkpoint_sha256"),
        ),
        "resolved_config": (
            provenance.get("resolved_config_path"),
            provenance.get("resolved_config_sha256"),
        ),
        "source_config": (
            provenance.get("source_config_path"),
            provenance.get("source_config_sha256"),
        ),
        "stored_best_metrics": (
            provenance.get("stored_best_metrics_path"),
            provenance.get("stored_best_metrics_sha256"),
        ),
    }
    verified: dict[str, Path] = {}
    for label, (path_value, digest) in bound_files.items():
        verified[label] = _artifact_path(
            {"path": path_value, "sha256": digest},
            label=f"{experiment}.{label}",
        )
    if verified["resolved_config"] != resolved_path.resolve():
        raise ValueError(f"resolved-config path mismatch for {experiment}")
    if verified["stored_best_metrics"] != best_path.resolve():
        raise ValueError(f"stored-best path mismatch for {experiment}")
    if provenance.get("split_manifest") != "val.csv" or provenance.get(
        "split_manifest_sha256"
    ) != ACCEPTED_VAL_SPLIT_SHA256:
        raise ValueError(f"validation split provenance mismatch for {experiment}")
    if best.get("checkpoint_sha256") != provenance.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint binding mismatch for {experiment}")
    if best.get("resolved_config_sha256") != provenance.get("resolved_config_sha256"):
        raise ValueError(f"resolved-config binding mismatch for {experiment}")
    if status.get("resolved_config_sha256") != provenance.get("resolved_config_sha256"):
        raise ValueError(f"status/config binding mismatch for {experiment}")
    if not log_path.is_file() or not _read_history(log_path):
        raise ValueError(f"training history is missing for {experiment}")

    val_metrics = _metrics(dict(document))
    stored_metrics = best.get("metrics")
    if not isinstance(stored_metrics, Mapping):
        raise ValueError(f"stored best metrics are missing for {experiment}")
    for key in (
        "loss",
        "top1_accuracy",
        "top5_accuracy",
        "overall_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        observed = _metric(val_metrics, key)
        stored = _metric(dict(stored_metrics), key)
        if not math.isclose(observed, stored, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"validation/stored metric mismatch for {experiment}: {key}")

    if freeze is not None:
        candidates = freeze.get("completed_candidates")
        if not isinstance(candidates, Mapping) or experiment not in candidates:
            raise ValueError(f"completed validation is absent from the freeze: {experiment}")
        frozen_entry = candidates[experiment]
        if not isinstance(frozen_entry, Mapping):
            raise ValueError(f"malformed frozen candidate: {experiment}")
        binding = frozen_entry.get("independent_validation_metrics")
        if not isinstance(binding, Mapping) or binding.get("sha256") != sha256_file(val_path):
            raise ValueError(f"validation metrics changed after freeze: {experiment}")


def _artifact_path(entry: Any, *, label: str) -> Path:
    if not isinstance(entry, Mapping):
        raise ValueError(f"malformed artifact entry: {label}")
    path = Path(str(entry.get("path", "")))
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"artifact hash mismatch: {label}")
    if entry.get("bytes") is not None and path.stat().st_size != int(entry["bytes"]):
        raise ValueError(f"artifact size mismatch: {label}")
    return path


def _selected_names(freeze: Mapping[str, Any]) -> list[str]:
    selected = freeze.get("selected_experiments")
    models = freeze.get("models")
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status") != "frozen"
        or not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(selected))
        or not isinstance(models, Mapping)
        or set(selected) != set(models)
    ):
        raise ValueError("invalid Stage 3 freeze membership")
    return [str(value) for value in selected]


def _verified_test_documents(
    metrics_dir: Path,
    *,
    freeze_path: Path,
    access_record: Path,
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    """Return frozen membership and test documents only after the batch is complete."""

    discovered = {
        path.name.removesuffix("_test_metrics.json"): path
        for path in metrics_dir.glob("*_test_metrics.json")
        if "__debug" not in path.name
    }
    if not freeze_path.is_file():
        if discovered or access_record.exists():
            raise ValueError("Stage 3 test evidence exists without the canonical freeze")
        return set(), {}

    freeze = _load_json(freeze_path)
    selected_list = _selected_names(freeze)
    selected = set(selected_list)
    freeze_hash = sha256_file(freeze_path)
    extras = sorted(set(discovered) - selected)
    if extras:
        raise ValueError(f"unselected experiments have held-out test metrics: {extras}")
    if not access_record.is_file():
        if discovered:
            raise ValueError("test metrics exist without a completed test-access record")
        return selected, {}

    access = _load_json(access_record)
    if access.get("freeze_manifest_sha256") != freeze_hash:
        raise ValueError("test-access record does not match the current freeze")
    if list(access.get("selected_experiments", [])) != selected_list:
        raise ValueError("test-access membership does not match the freeze")
    complete = access.get("status") == "completed" and access.get("outcome") == "complete"
    if not complete:
        if discovered:
            raise ValueError("test metrics exist but the frozen-set access did not complete")
        return selected, {}
    attempts = access.get("attempts")
    if not isinstance(attempts, Mapping) or set(attempts) != selected:
        raise ValueError("completed test-access record has incomplete model attempts")
    if any(
        not isinstance(attempts[name], Mapping)
        or attempts[name].get("status") != "complete"
        for name in selected_list
    ):
        raise ValueError("completed test-access record contains a failed model attempt")
    if set(discovered) != selected:
        missing = sorted(selected - set(discovered))
        raise ValueError(f"completed test batch is missing model metrics: {missing}")

    documents: dict[str, dict[str, Any]] = {}
    for name in selected_list:
        document = _load_json(discovered[name])
        provenance = document.get("provenance")
        access_artifacts = attempts[name].get("artifacts")
        if (
            document.get("status") != "complete"
            or document.get("split") != "test"
            or document.get("experiment_name") != name
            or not isinstance(provenance, Mapping)
            or provenance.get("freeze_manifest_sha256") != freeze_hash
            or document.get("metrics") != attempts[name].get("metrics")
            or not isinstance(access_artifacts, Mapping)
        ):
            raise ValueError(f"test metrics provenance is invalid for {name}")
        artifacts = document.get("artifacts")
        if not isinstance(artifacts, Mapping) or not artifacts:
            raise ValueError(f"test metrics has no bound artifacts for {name}")
        required_access = {"metrics", *artifacts.keys()}
        if set(access_artifacts) != required_access:
            raise ValueError(f"test-access artifact membership is invalid for {name}")
        for key, entry in access_artifacts.items():
            verified_path = _artifact_path(entry, label=f"{name}.test.{key}")
            if key == "metrics" and verified_path != discovered[name].resolve():
                raise ValueError(f"test-access metrics path is invalid for {name}")
            if key != "metrics" and entry != artifacts.get(key):
                raise ValueError(f"test artifact disagrees with access record: {name}.{key}")
        documents[name] = document
    return selected, documents


def _deep_rows(metrics_dir: Path, log_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    freeze_path = metrics_dir / "stage3_freeze.json"
    selected, verified_tests = _verified_test_documents(
        metrics_dir,
        freeze_path=freeze_path,
        access_record=metrics_dir / "stage3_test_access.json",
    )
    freeze_document = _load_json(freeze_path) if freeze_path.is_file() else None
    validation_paths = sorted(metrics_dir.glob("*_val_metrics.json"))
    validation_paths = [
        path
        for path in validation_paths
        if "_best_val_metrics" not in path.name and "__debug" not in path.name
    ]
    for val_path in validation_paths:
        val_document = _load_json(val_path)
        experiment = str(
            val_document.get(
                "experiment_name", val_path.name.removesuffix("_val_metrics.json")
            )
        )
        status_path = metrics_dir / f"{experiment}_status.json"
        status = _load_json(status_path)
        if status.get("status") != "COMPLETE":
            continue
        best_path = metrics_dir / f"{experiment}_best_val_metrics.json"
        resolved_path = metrics_dir / f"{experiment}_resolved_config.json"
        log_path = log_dir / f"{experiment}_train_log.jsonl"
        best = _load_json(best_path)
        resolved_document = _load_json(resolved_path)
        config = resolved_document.get("config", resolved_document)
        if not isinstance(config, dict):
            raise ValueError(f"invalid resolved configuration: {resolved_path}")
        _verify_validation_document(
            experiment=experiment,
            val_path=val_path,
            document=val_document,
            status=status,
            best=best,
            resolved_path=resolved_path,
            best_path=best_path,
            log_path=log_path,
            freeze=freeze_document,
        )
        history = _read_history(log_path)
        best_epoch = int(best["best_epoch"])
        best_records = [record for record in history if int(record.get("epoch", -1)) == best_epoch]
        if len(best_records) != 1:
            raise ValueError(f"cannot identify best epoch {best_epoch} in {log_path}")
        val = _metrics(val_document)
        if int(val.get("num_samples", -1)) != 5000 or int(val.get("num_classes", -1)) != 500:
            raise ValueError(f"{experiment} validation artifact has the wrong contract")
        val_values = {
            "val_top1": _metric(val, "top1_accuracy", "val_top1"),
            "val_top5": _metric(val, "top5_accuracy", "val_top5"),
            "val_macro_precision": _metric(val, "macro_precision", "val_macro_precision"),
            "val_macro_recall": _metric(val, "macro_recall", "val_macro_recall"),
            "val_macro_f1": _metric(val, "macro_f1", "val_macro_f1"),
        }
        for key, value in val_values.items():
            _validate_score(value, f"{experiment} {key}")

        test_document = verified_tests.get(experiment)
        if test_document is not None:
            test_block = _metrics(test_document)
            if int(test_block.get("num_samples", -1)) != 5000 or int(
                test_block.get("num_classes", -1)
            ) != 500:
                raise ValueError(f"{experiment} test artifact has the wrong row count")
        parameters = best.get("parameter_counts", config.get("parameter_counts", {}))
        if not isinstance(parameters, dict):
            parameters = {}
        test_values: dict[str, float | str] = {
            "test_top1": _optional_metric(test_document, "top1_accuracy", "test_top1"),
            "test_top5": _optional_metric(test_document, "top5_accuracy", "test_top5"),
            "test_macro_precision": _optional_metric(test_document, "macro_precision", "test_macro_precision"),
            "test_macro_recall": _optional_metric(test_document, "macro_recall", "test_macro_recall"),
            "test_macro_f1": _optional_metric(test_document, "macro_f1", "test_macro_f1"),
        }
        for key, value in test_values.items():
            if test_document is not None and value == "":
                raise ValueError(f"{experiment} completed test artifact is missing {key}")
            if value != "":
                _validate_score(float(value), f"{experiment} {key}")
        notes = ["completed validation"]
        if selected and experiment in selected:
            notes.append("selected before held-out test")
        elif freeze_path.is_file():
            notes.append("validation-only; not selected for test")
        if test_document is not None:
            notes.append("frozen test complete")
        train_top1 = float(best_records[0]["train_top1"])
        _validate_score(train_top1, f"{experiment} train_top1_best")
        training_time = float(best["training_time_seconds"])
        if not np.isfinite(training_time) or training_time < 0:
            raise ValueError(f"{experiment} training time must be finite and nonnegative")
        total_parameters = parameters.get("total", parameters.get("total_parameters", ""))
        trainable_parameters = parameters.get(
            "trainable", parameters.get("trainable_parameters", "")
        )
        if (
            isinstance(total_parameters, bool)
            or isinstance(trainable_parameters, bool)
            or int(total_parameters) <= 0
            or int(trainable_parameters) <= 0
            or int(trainable_parameters) > int(total_parameters)
        ):
            raise ValueError(f"{experiment} parameter counts are invalid")
        test_time = _optional_metric(
            test_document,
            "model_forward_time_seconds",
            "forward_time_seconds",
            "test_time_seconds",
        )
        if test_document is not None and (
            test_time == "" or not np.isfinite(float(test_time)) or float(test_time) < 0
        ):
            raise ValueError(f"{experiment} test time must be finite and nonnegative")
        row = {
            "model_name": experiment,
            "method_family": "deep_learning",
            "architecture": config.get("model_name", ""),
            "pretrained": bool(config.get("pretrained", False)),
            "augmentation": config.get("augmentation", ""),
            "freeze_backbone": bool(config.get("freeze_backbone", False)),
            "train_top1_best": train_top1,
            **val_values,
            **test_values,
            "training_time_seconds": training_time,
            "test_time_seconds": test_time,
            "num_parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
            "notes": "; ".join(notes),
        }
        rows.append(row)
    if not rows:
        raise ValueError("no completed canonical Stage 3 validation results were found")
    observed_rows = {str(row["model_name"]) for row in rows}
    if selected and not selected <= observed_rows:
        raise ValueError(
            "frozen selected models are missing completed validation rows: "
            + ", ".join(sorted(selected - observed_rows))
        )
    return sorted(rows, key=lambda row: str(row["model_name"]))


def _traditional_rows(stage2_path: Path) -> list[dict[str, Any]]:
    if not stage2_path.is_file():
        raise FileNotFoundError(stage2_path)
    observed_hash = sha256_file(stage2_path)
    if observed_hash != ACCEPTED_STAGE2_RESULTS_SHA256:
        raise ValueError(
            "Stage 2 result table is not the accepted frozen artifact: "
            f"expected {ACCEPTED_STAGE2_RESULTS_SHA256}, got {observed_hash}"
        )
    frame = pd.read_csv(stage2_path, keep_default_na=False)
    expected_columns = [
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
    if list(frame.columns) != expected_columns:
        raise ValueError(
            "Stage 2 table columns do not match the frozen schema: "
            f"expected {expected_columns}, got {list(frame.columns)}"
        )
    required_models = set(STAGE2_TABLE_SPECS)
    observed_models = set(frame["model_name"].astype(str))
    if observed_models != required_models or len(frame) != 3:
        raise ValueError(
            "frozen Stage 2 comparison must contain exactly linear_svm, "
            "random_forest, and sift_bovw_svm"
        )
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        model_name = str(record["model_name"])
        spec = STAGE2_TABLE_SPECS[model_name]
        if (
            str(record["feature_type"]) != spec["feature_type"]
            or str(record["classifier"]) != spec["classifier"]
            or isinstance(record["feature_dim"], bool)
            or int(record["feature_dim"]) != spec["feature_dim"]
            or not str(record["image_size"])
        ):
            raise ValueError(
                f"Stage 2 table row violates the rewritten contract: {model_name}"
            )
        metric_values: dict[str, float] = {}
        for key in (
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
        ):
            try:
                value = float(record[key])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Stage 2 table has a nonnumeric metric: {model_name}.{key}"
                ) from error
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Stage 2 table metric is outside [0, 1]: {model_name}.{key}"
                )
            metric_values[key] = value
        if (
            metric_values["val_top5"] + 1e-12 < metric_values["val_top1"]
            or metric_values["test_top5"] + 1e-12 < metric_values["test_top1"]
        ):
            raise ValueError(f"Stage 2 top-5 is below top-1 for {model_name}")
        for key in ("train_time_seconds", "test_time_seconds"):
            try:
                value = float(record[key])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Stage 2 table has invalid timing: {model_name}.{key}"
                ) from error
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Stage 2 table timing is negative/nonfinite: {model_name}.{key}"
                )
        row = {column: "" for column in COLUMNS}
        row.update(
            {
                "model_name": model_name,
                "method_family": "traditional",
                "architecture": record["classifier"],
                "pretrained": False,
                "augmentation": "none",
                "val_top1": record["val_top1"],
                "val_top5": record["val_top5"],
                "val_macro_precision": record["val_macro_precision"],
                "val_macro_recall": record["val_macro_recall"],
                "val_macro_f1": record["val_macro_f1"],
                "test_top1": record["test_top1"],
                "test_top5": record["test_top5"],
                "test_macro_precision": record["test_macro_precision"],
                "test_macro_recall": record["test_macro_recall"],
                "test_macro_f1": record["test_macro_f1"],
                "training_time_seconds": record["train_time_seconds"],
                "test_time_seconds": record["test_time_seconds"],
                "notes": (
                    "frozen rewritten Stage 2; "
                    f"features={record['feature_type']}; dim={record['feature_dim']}"
                ),
            }
        )
        rows.append(row)
    return rows


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _install_derived_set(
    staged: Mapping[Path, Path], *, force: bool, backup_dir: Path
) -> None:
    existing = [destination for destination in staged if destination.exists()]
    if existing and not force:
        raise FileExistsError(
            "derived Stage 3 outputs already exist; use --force to regenerate: "
            + ", ".join(str(path) for path in existing)
        )
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for index, destination in enumerate(existing):
            backup = backup_dir / f"backup_{index:02d}_{destination.name}"
            os.replace(destination, backup)
            backups[destination] = backup
        for destination, source in staged.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            installed.append(destination)
    except Exception:
        for destination in installed:
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise


def collect(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    deep_path = output_dir / "stage3_deep_learning_results.csv"
    ablation_path = output_dir / "stage3_ablation_results.csv"
    final_path = output_dir / "final_model_comparison.csv"
    figure_path = Path(args.figure)
    outputs = [deep_path, ablation_path, final_path, figure_path]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "derived Stage 3 outputs already exist; use --force to regenerate: "
            + ", ".join(str(path) for path in existing)
        )
    deep_rows = _deep_rows(Path(args.metrics_dir), Path(args.log_dir))
    deep_frame = pd.DataFrame(deep_rows, columns=COLUMNS)
    ablation_frame = deep_frame.loc[:, ABLATION_COLUMNS].copy()
    traditional = pd.DataFrame(_traditional_rows(Path(args.stage2_results)), columns=COLUMNS)
    final_frame = pd.concat([traditional, deep_frame], ignore_index=True).loc[:, COLUMNS]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".stage3_aggregation.", dir=output_dir.parent
    ) as temporary:
        stage_root = Path(temporary)
        staged = {
            deep_path: stage_root / deep_path.name,
            ablation_path: stage_root / ablation_path.name,
            final_path: stage_root / final_path.name,
            figure_path: stage_root / figure_path.name,
        }
        atomic_write_bytes(staged[deep_path], _csv_bytes(deep_frame))
        atomic_write_bytes(staged[ablation_path], _csv_bytes(ablation_frame))
        atomic_write_bytes(staged[final_path], _csv_bytes(final_frame))
        plot_stage3_validation_comparison(
            deep_frame,
            staged[figure_path],
            title="Stage 3 deep-learning validation comparison",
        )
        _install_derived_set(staged, force=bool(args.force), backup_dir=stage_root)
    return {
        "deep_rows": len(deep_frame),
        "final_rows": len(final_frame),
        "stage2_results_sha256": sha256_file(Path(args.stage2_results)),
        "outputs": {
            path.as_posix(): sha256_file(path) for path in outputs
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-dir", type=Path, default=Path("results/metrics/deep_learning")
    )
    parser.add_argument(
        "--log-dir", type=Path, default=Path("results/logs/deep_learning")
    )
    parser.add_argument(
        "--stage2-results",
        type=Path,
        default=Path("results/tables/stage2_traditional_results.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/figures/deep_learning/stage3_validation_comparison.png"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = collect(_parse_args(argv))
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"Collected {result['deep_rows']} Stage 3 rows and "
        f"{result['final_rows']} total comparison rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
