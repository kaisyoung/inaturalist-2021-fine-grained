"""Validate and aggregate the frozen Stage 4 robustness experiment matrix.

The collector is intentionally strict: the protocol is the source of truth,
every canonical cell must exist exactly once, and metrics are recomputed from
the hash-bound prediction CSV before any table is written.  Clean rows are
synthesised from frozen Stage 2/3 evidence; clean inference is never rerun.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.data.common import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODEL_COUNT = 4
EXPECTED_CORRUPTION_COUNT = 4
EXPECTED_SEVERITIES_PER_CORRUPTION = 4
EXPECTED_NUM_IMAGES = 5_000
EXPECTED_NUM_CLASSES = 500
CANONICAL_RUN_TYPE = "canonical_full"

RESULT_COLUMNS = [
    "model_name",
    "method_family",
    "architecture",
    "corruption_type",
    "severity",
    "severity_value",
    "severity_parameter",
    "source",
    "top1",
    "top5",
    "overall_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "inference_time_seconds",
    "num_images",
]


@dataclass(frozen=True)
class Protocol:
    path: Path
    sha256: str
    document: dict[str, Any]
    models: dict[str, dict[str, Any]]
    corruptions: dict[str, dict[str, Any]]
    test_manifest: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _resolve(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _artifact(entry: Any, *, label: str, require_hash: bool = True) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError(f"Malformed artifact binding: {label}")
    path = _resolve(entry.get("path", ""))
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    observed_hash = sha256_file(path)
    if require_hash and observed_hash != entry.get("sha256"):
        raise ValueError(f"Artifact hash mismatch: {label}")
    if entry.get("bytes") is not None and path.stat().st_size != int(entry["bytes"]):
        raise ValueError(f"Artifact byte-size mismatch: {label}")
    return {
        "path": path,
        "sha256": observed_hash,
        "bytes": path.stat().st_size,
    }


def _same_artifact(left: Any, right: Any, *, label: str) -> None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError(f"Malformed artifact provenance: {label}")
    left_path = _resolve(left.get("path", ""))
    right_path = _resolve(right.get("path", ""))
    if (
        left_path != right_path
        or str(left.get("sha256", "")) != str(right.get("sha256", ""))
        or int(left.get("bytes", -1)) != int(right.get("bytes", -2))
    ):
        raise ValueError(f"Artifact provenance mismatch: {label}")


def _find_test_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    stage1 = document.get("stage1_artifacts")
    if not isinstance(stage1, Mapping):
        raise ValueError("Protocol does not bind Stage 1 artifacts")
    for key in ("test.csv", "test", "test_split"):
        if key in stage1:
            entry = stage1[key]
            if not isinstance(entry, Mapping):
                break
            return dict(entry)
    raise ValueError("Protocol does not bind data/splits/test.csv")


def load_protocol(path: Path) -> Protocol:
    """Load and fully verify the immutable robustness protocol."""

    path = path.resolve()
    document = _load_json(path)
    policy = document.get("policy")
    models = document.get("models")
    corruptions = document.get("corruptions")
    if (
        document.get("schema_version") != 1
        or document.get("stage") != "stage4_robustness"
        or document.get("status") != "frozen"
        or document.get("split") != "test"
        or not isinstance(policy, Mapping)
        or policy.get("corruptions_applied_on_the_fly") is not True
        or policy.get("no_retraining_or_weight_updates") is not True
        or policy.get("model_selection_locked_before_robustness") is not True
        or policy.get("no_test_tuning") is not True
    ):
        raise ValueError("Invalid or non-frozen Stage 4 robustness protocol")
    if not isinstance(models, Mapping) or len(models) != EXPECTED_MODEL_COUNT:
        raise ValueError(f"Protocol must contain exactly {EXPECTED_MODEL_COUNT} models")
    model_order = document.get("model_order")
    if (
        not isinstance(model_order, list)
        or len(model_order) != EXPECTED_MODEL_COUNT
        or len(set(model_order)) != len(model_order)
        or set(model_order) != set(models)
    ):
        raise ValueError("Protocol model_order must exactly enumerate the four models")
    if not isinstance(corruptions, Mapping) or len(corruptions) != EXPECTED_CORRUPTION_COUNT:
        raise ValueError(
            f"Protocol must contain exactly {EXPECTED_CORRUPTION_COUNT} corruptions"
        )
    normalized_models: dict[str, dict[str, Any]] = {}
    for raw_name in model_order:
        raw_entry = models[raw_name]
        name = str(raw_name)
        if not name or not isinstance(raw_entry, Mapping):
            raise ValueError("Protocol model entries must be named objects")
        entry = dict(raw_entry)
        if not entry.get("method_family") or not entry.get("architecture"):
            raise ValueError(f"Protocol model metadata is incomplete: {name}")
        for artifact_name in ("artifact", "clean_metrics", "clean_predictions"):
            _artifact(entry.get(artifact_name), label=f"protocol.models.{name}.{artifact_name}")
        support_key = "model_metadata" if entry["method_family"] == "traditional" else "resolved_config"
        if support_key not in entry:
            raise ValueError(f"Protocol model support metadata is incomplete: {name}")
        entry["metadata_or_resolved_config"] = entry[support_key]
        _artifact(entry[support_key], label=f"protocol.models.{name}.{support_key}")
        normalized_models[name] = entry
    normalized_corruptions: dict[str, dict[str, Any]] = {}
    expected_runs = 0
    for raw_name, raw_entry in corruptions.items():
        name = str(raw_name)
        if not name or not isinstance(raw_entry, Mapping):
            raise ValueError("Protocol corruption entries must be named objects")
        entry = dict(raw_entry)
        severities = entry.get("severities")
        if not entry.get("parameter") or not isinstance(severities, list):
            raise ValueError(f"Malformed corruption protocol: {name}")
        if len(severities) != EXPECTED_SEVERITIES_PER_CORRUPTION:
            raise ValueError(
                f"{name} must define exactly {EXPECTED_SEVERITIES_PER_CORRUPTION} severities"
            )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in severities):
            raise ValueError(f"{name} severity values must be numeric")
        if len({float(value) for value in severities}) != len(severities):
            raise ValueError(f"{name} severity values must be unique")
        normalized_corruptions[name] = entry
        expected_runs += len(severities) * len(models)
    if int(document.get("expected_run_count", -1)) != expected_runs or expected_runs != 64:
        raise ValueError("Protocol expected_run_count does not match the fixed 4x4x4 matrix")
    test_manifest = _find_test_manifest(document)
    _artifact(test_manifest, label="protocol.stage1_artifacts.test.csv")
    # Every frozen upstream artifact recorded by the protocol must remain unchanged.
    for section_name in ("stage1_artifacts",):
        for name, entry in dict(document[section_name]).items():
            _artifact(entry, label=f"protocol.{section_name}.{name}")
    for section_name in ("stage2_evidence", "stage3_evidence"):
        section = document.get(section_name)
        if not isinstance(section, Mapping) or not section:
            raise ValueError(f"Protocol does not bind {section_name} evidence")
        for name, entry in section.items():
            _artifact(entry, label=f"protocol.{section_name}.{name}")
    return Protocol(
        path=path,
        sha256=sha256_file(path),
        document=document,
        models=normalized_models,
        corruptions=normalized_corruptions,
        test_manifest=test_manifest,
    )


def expected_run_keys(protocol: Protocol) -> set[tuple[str, str, int]]:
    return {
        (model, corruption, severity_index)
        for model in protocol.models
        for corruption, definition in protocol.corruptions.items()
        for severity_index in range(1, len(definition["severities"]) + 1)
    }


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} is invalid: {value!r}")
    return result


def _score(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be within [0, 1]")
    return result


def _parse_bool(series: pd.Series, label: str) -> np.ndarray:
    mapped = series.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"{label} contains invalid booleans")
    return mapped.to_numpy(dtype=bool)


def _parse_list(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, list):
        return value
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f"Could not parse {label}") from error
    if not isinstance(parsed, list):
        raise ValueError(f"{label} must contain a list")
    return parsed


def _validate_predictions(
    path: Path,
    *,
    manifest: pd.DataFrame,
    model_name: str,
    corruption: str,
    severity_index: int,
    parameter: str,
    severity_value: float,
) -> dict[str, float]:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "manifest_index",
        "image_id",
        "local_image_path",
        "true_label",
        "predicted_label",
        "correct",
        "top5_labels",
        "top5_scores",
        "true_class_score",
        "true_class_rank",
        "corruption_type",
        "severity_index",
        "severity_parameter",
        "severity_value",
        "scientific_name",
        "common_name",
        "score_type",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction CSV is missing columns for {model_name}: {missing}")
    if len(frame) != EXPECTED_NUM_IMAGES:
        raise ValueError(f"Expected {EXPECTED_NUM_IMAGES} predictions, got {len(frame)}")
    for column in ("manifest_index", "image_id", "true_label", "predicted_label", "severity_index"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not bool((frame[column] % 1 == 0).all()):
            raise ValueError(f"Prediction {column} must contain integers")
        frame[column] = frame[column].astype("int64")
    if frame["manifest_index"].tolist() != list(range(EXPECTED_NUM_IMAGES)):
        raise ValueError("Predictions are not in frozen manifest order")
    if frame["image_id"].duplicated().any() or frame["local_image_path"].duplicated().any():
        raise ValueError("Prediction identifiers and paths must be unique")
    if not frame["true_label"].between(0, EXPECTED_NUM_CLASSES - 1).all() or not frame[
        "predicted_label"
    ].between(0, EXPECTED_NUM_CLASSES - 1).all():
        raise ValueError("Prediction labels must remain within 0..499")
    manifest_ids = pd.to_numeric(manifest["image_id"], errors="raise").astype("int64")
    manifest_labels = pd.to_numeric(manifest["new_label"], errors="raise").astype("int64")
    comparisons = (
        frame["image_id"].reset_index(drop=True).equals(manifest_ids.reset_index(drop=True)),
        frame["true_label"].reset_index(drop=True).equals(manifest_labels.reset_index(drop=True)),
        frame["local_image_path"].astype(str).reset_index(drop=True).equals(
            manifest["local_image_path"].astype(str).reset_index(drop=True)
        ),
        frame["scientific_name"].astype(str).reset_index(drop=True).equals(
            manifest["scientific_name"].astype(str).reset_index(drop=True)
        ),
        frame["common_name"].astype(str).reset_index(drop=True).equals(
            manifest["common_name"].astype(str).reset_index(drop=True)
        ),
    )
    if not all(comparisons):
        raise ValueError("Predictions disagree with the frozen test manifest")
    correct = _parse_bool(frame["correct"], "prediction.correct")
    expected_correct = frame["true_label"].to_numpy() == frame["predicted_label"].to_numpy()
    if not np.array_equal(correct, expected_correct):
        raise ValueError("Prediction correctness disagrees with labels")
    if set(frame["corruption_type"].astype(str)) != {corruption}:
        raise ValueError("Prediction corruption_type is inconsistent")
    if set(frame["severity_index"].astype(int)) != {severity_index}:
        raise ValueError("Prediction severity_index is inconsistent")
    if set(frame["severity_parameter"].astype(str)) != {parameter}:
        raise ValueError("Prediction severity_parameter is inconsistent")
    values = pd.to_numeric(frame["severity_value"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.allclose(values, severity_value, rtol=0, atol=1e-12):
        raise ValueError("Prediction severity_value is inconsistent")
    if frame["score_type"].astype(str).eq("").any():
        raise ValueError("Prediction score_type cannot be blank")

    top5_hits: list[bool] = []
    for row_number, row in frame.iterrows():
        labels = _parse_list(row["top5_labels"], label=f"top5_labels row {row_number}")
        scores = _parse_list(row["top5_scores"], label=f"top5_scores row {row_number}")
        if len(labels) != 5 or len(scores) != 5:
            raise ValueError("Every prediction must contain exactly five ranked classes")
        try:
            labels_int = [int(value) for value in labels]
            scores_float = [float(value) for value in scores]
        except (TypeError, ValueError) as error:
            raise ValueError("Top-five labels/scores must be numeric") from error
        if (
            len(set(labels_int)) != 5
            or not all(0 <= value < EXPECTED_NUM_CLASSES for value in labels_int)
            or not all(math.isfinite(value) for value in scores_float)
            or labels_int[0] != int(row["predicted_label"])
        ):
            raise ValueError("Invalid top-five ranking")
        expected_order = sorted(range(5), key=lambda idx: (-scores_float[idx], labels_int[idx]))
        if expected_order != list(range(5)):
            raise ValueError("Top-five ranking is not stable descending with lower-label ties")
        top5_hits.append(int(row["true_label"]) in labels_int)
    true_score = pd.to_numeric(frame["true_class_score"], errors="raise").to_numpy(float)
    true_rank = pd.to_numeric(frame["true_class_rank"], errors="raise").to_numpy(float)
    if (
        not np.isfinite(true_score).all()
        or not np.isfinite(true_rank).all()
        or not np.equal(true_rank, np.floor(true_rank)).all()
        or not ((1 <= true_rank) & (true_rank <= EXPECTED_NUM_CLASSES)).all()
    ):
        raise ValueError("Invalid true-class score/rank")
    precision, recall, f1, _ = precision_recall_fscore_support(
        frame["true_label"].to_numpy(),
        frame["predicted_label"].to_numpy(),
        labels=np.arange(EXPECTED_NUM_CLASSES),
        average="macro",
        zero_division=0,
    )
    top1 = float(correct.mean())
    return {
        "top1_accuracy": top1,
        "top5_accuracy": float(np.mean(top5_hits)),
        "overall_accuracy": top1,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def _document_key(document: Mapping[str, Any], path: Path) -> tuple[str, str, int]:
    corruption = document.get("corruption")
    if not isinstance(corruption, Mapping):
        raise ValueError(f"Malformed corruption descriptor: {path}")
    try:
        severity = int(corruption.get("severity_index"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid severity index: {path}") from error
    return str(document.get("model_name", "")), str(corruption.get("type", "")), severity


def _discover_documents(metrics_dir: Path) -> dict[tuple[str, str, int], tuple[Path, dict[str, Any]]]:
    discovered: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    if not metrics_dir.is_dir():
        raise FileNotFoundError(metrics_dir)
    for path in sorted(metrics_dir.glob("*.json")):
        document = _load_json(path)
        if document.get("run_type") != CANONICAL_RUN_TYPE:
            continue
        key = _document_key(document, path)
        if key in discovered:
            raise ValueError(f"Duplicate canonical robustness run: {key}")
        discovered[key] = (path, document)
    return discovered


def _clean_metrics(model_name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    binding = _artifact(entry.get("clean_metrics"), label=f"{model_name}.clean_metrics")
    document = _load_json(binding["path"])
    if entry.get("method_family") == "traditional":
        metrics = document.get("test")
    else:
        if document.get("status") != "complete" or document.get("split") != "test":
            raise ValueError(f"Invalid frozen clean test metrics: {model_name}")
        metrics = document.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"Clean metrics are missing: {model_name}")
    num_samples = int(metrics.get("num_samples", -1))
    if num_samples != EXPECTED_NUM_IMAGES:
        raise ValueError(f"Clean metrics sample count is invalid: {model_name}")
    result = {
        key: _score(metrics.get(key), f"{model_name}.clean.{key}")
        for key in (
            "top1_accuracy",
            "top5_accuracy",
            "overall_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
    }
    if not math.isclose(result["top1_accuracy"], result["overall_accuracy"], abs_tol=1e-12):
        raise ValueError(f"Clean top-1/overall mismatch: {model_name}")
    time_value = metrics.get("test_time_seconds", metrics.get("inference_time_seconds"))
    result["inference_time_seconds"] = _number(
        time_value, f"{model_name}.clean.inference_time_seconds", minimum=0.0
    )
    return result


def collect_rows(protocol_path: Path, metrics_dir: Path) -> pd.DataFrame:
    """Return the validated clean + corruption table without writing outputs."""

    protocol = load_protocol(protocol_path)
    expected = expected_run_keys(protocol)
    discovered = _discover_documents(metrics_dir)
    missing = sorted(expected - set(discovered))
    unknown = sorted(set(discovered) - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError("Canonical robustness matrix is incomplete: " + "; ".join(details))

    manifest_binding = _artifact(protocol.test_manifest, label="protocol.test_manifest")
    manifest = pd.read_csv(manifest_binding["path"], keep_default_na=False)
    required_manifest = {
        "image_id",
        "new_label",
        "local_image_path",
        "scientific_name",
        "common_name",
    }
    if len(manifest) != EXPECTED_NUM_IMAGES or not required_manifest <= set(manifest.columns):
        raise ValueError("Frozen test manifest has invalid schema/count")

    rows: list[dict[str, Any]] = []
    for model_name, entry in protocol.models.items():
        clean = _clean_metrics(model_name, entry)
        rows.append(
            {
                "model_name": model_name,
                "method_family": entry["method_family"],
                "architecture": entry["architecture"],
                "corruption_type": "clean",
                "severity": 0,
                "severity_value": 0,
                "severity_parameter": "none",
                "source": "frozen_clean_test_metrics",
                "top1": clean["top1_accuracy"],
                "top5": clean["top5_accuracy"],
                "overall_accuracy": clean["overall_accuracy"],
                "macro_precision": clean["macro_precision"],
                "macro_recall": clean["macro_recall"],
                "macro_f1": clean["macro_f1"],
                "inference_time_seconds": clean["inference_time_seconds"],
                "num_images": EXPECTED_NUM_IMAGES,
            }
        )

    for key in sorted(discovered):
        path, document = discovered[key]
        model_name, corruption_name, severity_index = key
        entry = protocol.models[model_name]
        definition = protocol.corruptions[corruption_name]
        descriptor = document.get("corruption")
        inputs = document.get("inputs")
        metrics = document.get("metrics")
        timing = document.get("timing")
        artifacts = document.get("artifacts")
        protocol_binding = document.get("protocol")
        if (
            document.get("schema_version") != 1
            or document.get("stage") != "stage4_robustness"
            or document.get("status") != "complete"
            or document.get("run_type") != CANONICAL_RUN_TYPE
            or document.get("split") != "test"
            or document.get("method_family") != entry["method_family"]
            or document.get("architecture") != entry["architecture"]
            or not isinstance(descriptor, Mapping)
            or not isinstance(inputs, Mapping)
            or not isinstance(metrics, Mapping)
            or not isinstance(timing, Mapping)
            or not isinstance(artifacts, Mapping)
            or not isinstance(protocol_binding, Mapping)
        ):
            raise ValueError(f"Malformed canonical robustness result: {path}")
        if (
            _resolve(protocol_binding.get("path", "")) != protocol.path
            or protocol_binding.get("sha256") != protocol.sha256
        ):
            raise ValueError(f"Protocol provenance mismatch: {path}")
        expected_value = float(definition["severities"][severity_index - 1])
        if (
            descriptor.get("parameter") != definition["parameter"]
            or not math.isclose(
                _number(descriptor.get("parameter_value"), "corruption.parameter_value"),
                expected_value,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or int(descriptor.get("seed", -1)) != int(protocol.document.get("seed", -2))
        ):
            raise ValueError(f"Corruption definition disagrees with protocol: {path}")
        _same_artifact(inputs.get("manifest"), protocol.test_manifest, label=f"{key}.manifest")
        _same_artifact(inputs.get("model_artifact"), entry["artifact"], label=f"{key}.model")
        _same_artifact(inputs.get("clean_metrics"), entry["clean_metrics"], label=f"{key}.clean_metrics")
        _same_artifact(
            inputs.get("clean_predictions"),
            entry["clean_predictions"],
            label=f"{key}.clean_predictions",
        )
        support = inputs.get("model_support_artifacts")
        expected_support = entry["metadata_or_resolved_config"]
        if isinstance(support, Mapping) and "path" in support:
            _same_artifact(support, expected_support, label=f"{key}.model_support")
        elif isinstance(support, Mapping):
            if not any(
                isinstance(candidate, Mapping)
                and _resolve(candidate.get("path", "")) == _resolve(expected_support["path"])
                and candidate.get("sha256") == expected_support.get("sha256")
                for candidate in support.values()
            ):
                raise ValueError(f"Model support provenance mismatch: {path}")
        elif isinstance(support, list):
            if not any(
                isinstance(candidate, Mapping)
                and _resolve(candidate.get("path", "")) == _resolve(expected_support["path"])
                and candidate.get("sha256") == expected_support.get("sha256")
                for candidate in support
            ):
                raise ValueError(f"Model support provenance mismatch: {path}")
        else:
            raise ValueError(f"Model support provenance is missing: {path}")

        prediction_entry = artifacts.get("predictions")
        prediction = _artifact(prediction_entry, label=f"{key}.predictions")
        expected_prediction_path = (
            PROJECT_ROOT
            / "results/predictions/robustness"
            / f"{model_name}_{corruption_name}_s{severity_index}_predictions.csv"
        ).resolve()
        if prediction["path"] != expected_prediction_path:
            raise ValueError(f"Non-canonical prediction path: {prediction['path']}")
        recomputed = _validate_predictions(
            prediction["path"],
            manifest=manifest,
            model_name=model_name,
            corruption=corruption_name,
            severity_index=severity_index,
            parameter=str(definition["parameter"]),
            severity_value=expected_value,
        )
        declared: dict[str, float] = {}
        for metric_name in recomputed:
            declared[metric_name] = _score(metrics.get(metric_name), f"{key}.{metric_name}")
            if not math.isclose(
                declared[metric_name], recomputed[metric_name], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"Prediction/metric mismatch for {key}: {metric_name}")
        if int(metrics.get("num_samples", -1)) != EXPECTED_NUM_IMAGES or int(
            metrics.get("num_classes", -1)
        ) != EXPECTED_NUM_CLASSES:
            raise ValueError(f"Metric counts are invalid: {path}")
        full_wall = _number(
            timing.get("full_evaluation_wall_seconds"),
            f"{key}.full_evaluation_wall_seconds",
            minimum=0.0,
        )
        forward = _number(
            timing.get("model_forward_seconds"),
            f"{key}.model_forward_seconds",
            minimum=0.0,
        )
        feature = _number(
            timing.get("feature_extraction_seconds"),
            f"{key}.feature_extraction_seconds",
            minimum=0.0,
        )
        if full_wall + 1e-12 < forward + feature:
            raise ValueError(f"Timing components exceed evaluation wall time: {path}")
        rows.append(
            {
                "model_name": model_name,
                "method_family": entry["method_family"],
                "architecture": entry["architecture"],
                "corruption_type": corruption_name,
                "severity": severity_index,
                "severity_value": expected_value,
                "severity_parameter": definition["parameter"],
                "source": "stage4_on_the_fly_corruption",
                "top1": declared["top1_accuracy"],
                "top5": declared["top5_accuracy"],
                "overall_accuracy": declared["overall_accuracy"],
                "macro_precision": declared["macro_precision"],
                "macro_recall": declared["macro_recall"],
                "macro_f1": declared["macro_f1"],
                "inference_time_seconds": full_wall,
                "num_images": EXPECTED_NUM_IMAGES,
            }
        )
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    validate_result_matrix(frame, protocol)
    return frame.sort_values(
        ["model_name", "corruption_type", "severity"], kind="stable"
    ).reset_index(drop=True)


def validate_result_matrix(frame: pd.DataFrame, protocol: Protocol) -> None:
    missing_columns = sorted(set(RESULT_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Robustness table is missing columns: {missing_columns}")
    clean = frame.loc[frame["corruption_type"] == "clean"]
    corrupted = frame.loc[frame["corruption_type"] != "clean"]
    if len(clean) != len(protocol.models) or set(clean["model_name"]) != set(protocol.models):
        raise ValueError("Robustness table must contain one clean row per protocol model")
    keys = list(
        zip(
            corrupted["model_name"].astype(str),
            corrupted["corruption_type"].astype(str),
            pd.to_numeric(corrupted["severity"], errors="raise").astype(int),
        )
    )
    if len(keys) != len(set(keys)):
        raise ValueError("Robustness table contains duplicate matrix cells")
    expected = expected_run_keys(protocol)
    if set(keys) != expected:
        raise ValueError(
            f"Robustness table matrix mismatch; missing={sorted(expected-set(keys))}; "
            f"unknown={sorted(set(keys)-expected)}"
        )
    for column in ("top1", "top5", "overall_accuracy", "macro_precision", "macro_recall", "macro_f1"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all() or not values.between(0.0, 1.0).all():
            raise ValueError(f"Robustness {column} values must be finite and within [0,1]")
    if not np.allclose(frame["top1"], frame["overall_accuracy"], rtol=0, atol=1e-12):
        raise ValueError("Robustness top1 and overall_accuracy disagree")
    if not (pd.to_numeric(frame["num_images"], errors="raise") == EXPECTED_NUM_IMAGES).all():
        raise ValueError("Every robustness row must represent exactly 5000 images")
    timings = pd.to_numeric(frame["inference_time_seconds"], errors="raise")
    if not np.isfinite(timings).all() or (timings < 0).any():
        raise ValueError("Robustness timing values must be finite and nonnegative")


def _atomic_csv(path: Path, frame: pd.DataFrame, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output already exists; use --force deliberately: {path}")
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/metrics/stage4_robustness_protocol.json"),
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=Path("results/metrics/robustness")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/tables/robustness_results.csv")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        frame = collect_rows(args.protocol, args.metrics_dir)
        _atomic_csv(args.output, frame, force=bool(args.force))
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"Wrote {len(frame)} rows ({len(frame)-EXPECTED_MODEL_COUNT} corrupted, "
        f"{EXPECTED_MODEL_COUNT} frozen clean) to {args.output}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
