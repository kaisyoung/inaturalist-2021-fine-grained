"""Leakage-safe validation and frozen-set test evaluation for Stage 3.

Validation accepts one best checkpoint and independently verifies its stored
best-epoch metrics.  Test evaluation accepts no ad-hoc checkpoint: it verifies
the entire Stage 3 freeze before loading ``test.csv`` and then attempts every
frozen model as one auditable batch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis.metrics import evaluate_aligned_scores, per_class_metrics, top_confused_pairs
from src.analysis.plotting import plot_confusion_matrix_overview, plot_top_confused_pairs
from src.data.common import NUM_CLASSES, sha256_file
from src.deep_learning.freeze_stage3 import (
    CANONICAL_ACCESS_RECORD,
    CANONICAL_FREEZE_PATH,
    MAX_VALIDATION_METRIC_TOLERANCE,
    PROJECT_ROOT,
    STAGE1_FILES,
    _expected_selection,
    _portable,
    _read_json,
    _resolve,
)


LABELS = np.arange(NUM_CLASSES, dtype=np.int64)
SCORE_TYPE = "softmax_probability"
SAFE_EXPERIMENT = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
CANONICAL_TEST_DIRECTORIES = {
    "metrics_dir": (PROJECT_ROOT / "results/metrics/deep_learning").resolve(),
    "predictions_dir": (PROJECT_ROOT / "results/predictions/deep_learning").resolve(),
    "tables_dir": (PROJECT_ROOT / "results/tables/deep_learning").resolve(),
    "figures_dir": (PROJECT_ROOT / "results/figures/deep_learning").resolve(),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
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


def _atomic_confusion(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(handle)
    temporary = Path(name)
    try:
        np.savez_compressed(
            temporary,
            confusion_matrix=np.asarray(matrix, dtype=np.int64),
            labels=LABELS,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_canonical_test_gate(args: argparse.Namespace) -> None:
    freeze_value = getattr(args, "freeze_manifest", CANONICAL_FREEZE_PATH)
    if _resolve(freeze_value) != CANONICAL_FREEZE_PATH:
        raise ValueError(f"Stage 3 test freeze path is fixed: {CANONICAL_FREEZE_PATH}")
    access_value = getattr(args, "access_record", None)
    if access_value is not None and _resolve(access_value) != CANONICAL_ACCESS_RECORD:
        raise ValueError(
            f"Stage 3 test-access record path is fixed: {CANONICAL_ACCESS_RECORD}"
        )
    for attribute, expected in CANONICAL_TEST_DIRECTORIES.items():
        supplied = getattr(args, attribute, expected)
        if _resolve(supplied) != expected:
            raise ValueError(f"Stage 3 test {attribute} is fixed: {expected}")


def _require_unchanged_freeze(path: Path, expected_sha256: str) -> dict[str, Any]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Stage 3 freeze changed during test evaluation: expected "
            f"{expected_sha256}, got {observed}"
        )
    return _read_json(path)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": _portable(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency precheck covers this.
        raise RuntimeError("PyTorch is required for deep-learning evaluation") from error
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"Could not load checkpoint {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {path}")
    return value


def _select_device(requested: str | None = None) -> Any:
    import torch

    requested = requested or "auto"
    available = {
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "cpu": True,
    }
    if requested != "auto":
        if requested not in available or not available[requested]:
            raise ValueError(f"Requested evaluation device is unavailable: {requested}")
        return torch.device(requested)
    for name in ("cuda", "mps", "cpu"):
        if available[name]:
            return torch.device(name)
    raise RuntimeError("No PyTorch evaluation device is available")


def _synchronize(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _seed_worker(worker_id: int) -> None:
    import torch

    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _resolve_context_path(
    explicit: Path | None,
    checkpoint: Mapping[str, Any],
    key: str,
    default: Path,
) -> Path:
    if explicit is not None:
        return explicit.resolve()
    embedded = checkpoint.get(key)
    if embedded:
        return _resolve(str(embedded), relative_to=default.parent)
    return default.resolve()


def _checkpoint_metric(metrics: Mapping[str, Any], key: str) -> float:
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
            try:
                value = float(metrics[alias])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Stored best metric {alias} is not numeric") from error
            if not math.isfinite(value):
                raise ValueError(f"Stored best metric {alias} is not finite")
            return value
    raise ValueError(f"Stored best metrics has no value for {key}")


def _stage1_hashes(
    split_dir: Path, *, verify_test_manifest: bool = False
) -> dict[str, dict[str, Any]]:
    """Bind Stage 1 while leaving test.csv unopened on validation paths."""

    summary = _read_json(split_dir / "split_summary.json")
    if summary.get("schema_version") != 2:
        raise ValueError("Stage 1 split summary must use schema version 2")
    result: dict[str, dict[str, Any]] = {}
    for name in STAGE1_FILES:
        path = split_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Stage 1 artifact not found: {path}")
        if name == "test.csv" and not verify_test_manifest:
            expected = summary.get("artifacts", {}).get(name, {}).get("sha256")
            if not expected:
                raise ValueError("Stage 1 split summary has no accepted test.csv hash")
            result[name] = {
                "path": _portable(path),
                "bytes": path.stat().st_size,
                "sha256": expected,
                "verification": "accepted precheck digest; test.csv not opened",
            }
            continue
        result[name] = {
            "path": _portable(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "verification": "verified current bytes",
        }
    for name in STAGE1_FILES:
        if name == "split_summary.json":
            continue
        expected = summary.get("artifacts", {}).get(name, {}).get("sha256")
        if expected != result[name]["sha256"]:
            raise ValueError(f"Stage 1 split-summary hash mismatch: {name}")
    return result


def _validate_checkpoint_context(
    checkpoint_path: Path,
    *,
    split_dir: Path,
    metrics_dir: Path,
    resolved_config_path: Path | None = None,
    source_config_path: Path | None = None,
    best_metrics_path: Path | None = None,
    expected_name: str | None = None,
    frozen_entry: Mapping[str, Any] | None = None,
    verify_test_manifest: bool = False,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    required = {
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
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {missing}")
    name = str(checkpoint["experiment_name"])
    if not SAFE_EXPERIMENT.fullmatch(name):
        raise ValueError(f"Unsafe checkpoint experiment name: {name!r}")
    if expected_name is not None and name != expected_name:
        raise ValueError(f"Frozen experiment {expected_name} loaded checkpoint for {name}")
    if (
        checkpoint["checkpoint_kind"] != "best"
        or int(checkpoint["num_classes"]) != NUM_CLASSES
        or int(checkpoint["epoch"]) != int(checkpoint["best_epoch"])
        or int(checkpoint["image_size"]) != 224
    ):
        raise ValueError("Evaluator requires a canonical best 500-class 224px checkpoint")

    stage1 = _stage1_hashes(
        split_dir, verify_test_manifest=verify_test_manifest
    )
    embedded_stage1 = checkpoint["stage1_artifacts"]
    if not isinstance(embedded_stage1, Mapping) or set(embedded_stage1) != set(STAGE1_FILES):
        raise ValueError("Checkpoint must bind exactly the six Stage 1 artifacts")
    for file_name, current in stage1.items():
        entry = embedded_stage1[file_name]
        embedded_hash = entry.get("sha256") if isinstance(entry, Mapping) else entry
        if embedded_hash != current["sha256"]:
            raise ValueError(f"Checkpoint Stage 1 hash mismatch: {file_name}")

    resolved_path = _resolve_context_path(
        resolved_config_path,
        checkpoint,
        "resolved_config_path",
        metrics_dir / f"{name}_resolved_config.json",
    )
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Resolved config not found: {resolved_path}")
    resolved_hash = sha256_file(resolved_path)
    resolved = _read_json(resolved_path)
    if (
        checkpoint["resolved_config_sha256"] != resolved_hash
        or checkpoint["resolved_config"] != resolved
    ):
        raise ValueError("Persisted and checkpoint-embedded resolved configs differ")
    source_path = _resolve_context_path(
        source_config_path,
        checkpoint,
        "source_config_path",
        Path(str(checkpoint["source_config_path"])),
    )
    if not source_path.is_file():
        raise FileNotFoundError(f"Source config not found: {source_path}")
    source_hash = sha256_file(source_path)
    if checkpoint["source_config_sha256"] != source_hash:
        raise ValueError("Source config changed after training")

    best_path = (
        best_metrics_path.resolve()
        if best_metrics_path is not None
        else (metrics_dir / f"{name}_best_val_metrics.json").resolve()
    )
    if not best_path.is_file():
        raise FileNotFoundError(f"Stored best-epoch metrics not found: {best_path}")
    best_document = _read_json(best_path)
    if (
        best_document.get("schema_version") != 1
        or best_document.get("experiment_name") != name
        or int(best_document.get("best_epoch", -1)) != int(checkpoint["best_epoch"])
        or best_document.get("checkpoint_sha256") != checkpoint_hash
        or best_document.get("resolved_config_sha256") != resolved_hash
    ):
        raise ValueError("Stored best-epoch metrics provenance is invalid")
    if not isinstance(best_document.get("metrics"), Mapping):
        raise ValueError("Stored best-epoch metrics block is missing")

    if frozen_entry is not None:
        bindings = {
            "checkpoint": (checkpoint_path, checkpoint_hash),
            "resolved_config": (resolved_path, resolved_hash),
            "source_config": (source_path, source_hash),
            "stored_best_metrics": (best_path, sha256_file(best_path)),
        }
        for key, (path, digest) in bindings.items():
            entry = frozen_entry.get(key)
            if (
                not isinstance(entry, Mapping)
                or entry.get("sha256") != digest
                or _resolve(str(entry.get("path", ""))) != path.resolve()
            ):
                raise ValueError(f"Frozen {name} binding is invalid: {key}")

    return {
        "name": name,
        "checkpoint_path": checkpoint_path.resolve(),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint": checkpoint,
        "resolved_config_path": resolved_path,
        "resolved_config_sha256": resolved_hash,
        "resolved_config": resolved,
        "source_config_path": source_path,
        "source_config_sha256": source_hash,
        "best_metrics_path": best_path,
        "best_metrics_sha256": sha256_file(best_path),
        "best_metrics": best_document,
        "stage1": stage1,
    }


def _model_for_context(context: Mapping[str, Any], device: Any) -> Any:
    import torch

    os.environ.setdefault(
        "TORCH_HOME", str((PROJECT_ROOT / "results/models/torch_cache").resolve())
    )
    from src.deep_learning.models import create_model

    checkpoint = context["checkpoint"]
    result = create_model(
        str(checkpoint["model_name"]),
        pretrained=bool(checkpoint["pretrained"]),
        freeze_backbone=bool(checkpoint["freeze_backbone"]),
        num_classes=NUM_CLASSES,
    )
    model = result[0] if isinstance(result, tuple) else result
    metadata = result[1] if isinstance(result, tuple) and len(result) == 2 else {}
    expected_metadata = context["resolved_config"].get("model_metadata", {})
    if bool(checkpoint["pretrained"]):
        for key in ("weights_enum", "weight_file_name", "weight_file_sha256"):
            if metadata.get(key) != expected_metadata.get(key):
                raise ValueError(f"Pretrained weight provenance mismatch: {key}")
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, ValueError, KeyError) as error:
        raise ValueError(f"Checkpoint state is incompatible with its model: {error}") from error
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model


def _class_names(split_dir: Path) -> dict[int, dict[str, Any]]:
    document = _read_json(split_dir / "class_mapping.json")
    classes = document.get("classes")
    if (
        document.get("schema_version") != 2
        or document.get("num_classes") != NUM_CLASSES
        or not isinstance(classes, list)
        or len(classes) != NUM_CLASSES
    ):
        raise ValueError("class_mapping.json is not the canonical 500-class schema")
    result: dict[int, dict[str, Any]] = {}
    for item in classes:
        if not isinstance(item, Mapping):
            raise ValueError("Malformed class mapping entry")
        label = int(item.get("new_label", -1))
        if label in result or label not in range(NUM_CLASSES):
            raise ValueError("Class mapping labels must be exactly 0..499")
        result[label] = {
            "scientific_name": str(item.get("scientific_name", "")),
            "common_name": str(item.get("common_name", "")),
            "original_category_id": int(item.get("original_category_id", -1)),
        }
    if set(result) != set(range(NUM_CLASSES)):
        raise ValueError("Class mapping labels must be exactly 0..499")
    return result


def _metadata_values(metadata: Mapping[str, Any], key: str, count: int) -> list[Any]:
    if key not in metadata:
        return [None] * count
    value = metadata[key]
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif not isinstance(value, (list, tuple)):
        value = [value] * count
    result = list(value)
    if len(result) != count:
        raise ValueError(f"Collated metadata field {key} has the wrong batch length")
    return result


def _make_loader(
    manifest_path: Path,
    *,
    image_root: Path,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    from src.deep_learning.dataset import ManifestImageDataset
    from src.deep_learning.transforms import get_transform

    dataset = ManifestImageDataset(
        manifest_path,
        image_root,
        get_transform("eval", image_size=image_size),
        split,
    )
    generator = torch.Generator()
    generator.manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _run_inference(
    context: Mapping[str, Any],
    *,
    loader: Any,
    device: Any,
    split: str,
    names: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    model = _model_for_context(context, device)
    logits_batches: list[np.ndarray] = []
    truth_batches: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []
    loss_sum = 0.0
    forward_seconds = 0.0
    sample_count = 0
    wall_start = time.perf_counter()
    with torch.inference_mode():
        for images, labels, metadata in loader:
            if not isinstance(metadata, Mapping):
                raise ValueError("Dataset metadata must collate to a dictionary")
            labels = labels.to(device=device, dtype=torch.long, non_blocking=False)
            images = images.to(device=device, dtype=torch.float32, non_blocking=False)
            _synchronize(device)
            forward_start = time.perf_counter()
            logits = model(images)
            _synchronize(device)
            forward_seconds += time.perf_counter() - forward_start
            if logits.ndim != 2 or tuple(logits.shape) != (len(labels), NUM_CLASSES):
                raise ValueError(
                    f"Model logits must have shape batch x {NUM_CLASSES}, got {tuple(logits.shape)}"
                )
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("Model produced NaN or infinite logits")
            loss_sum += float(functional.cross_entropy(logits, labels, reduction="sum").item())
            logits_batches.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
            truth = labels.detach().cpu().numpy().astype(np.int64, copy=False)
            truth_batches.append(truth)
            count = len(truth)
            fields = {
                key: _metadata_values(metadata, key, count)
                for key in (
                    "split",
                    "image_id",
                    "image_path",
                    "local_image_path",
                    "original_category_id",
                    "scientific_name",
                    "common_name",
                )
            }
            for index in range(count):
                metadata_rows.append({key: value[index] for key, value in fields.items()})
            sample_count += count
    wall_seconds = time.perf_counter() - wall_start
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    if sample_count == 0:
        raise ValueError(f"{split} evaluation dataset is empty")

    logits = np.concatenate(logits_batches, axis=0)
    truth = np.concatenate(truth_batches, axis=0)
    # Stable softmax in float64 makes score export deterministic and prevents overflow.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    aggregate, predicted, order, matrix = evaluate_aligned_scores(truth, probabilities, LABELS)
    aggregate.update(
        {
            "loss": loss_sum / sample_count,
            "model_forward_time_seconds": float(forward_seconds),
            "evaluation_wall_time_seconds": float(wall_seconds),
            "seconds_per_image": float(forward_seconds / sample_count),
        }
    )
    if split == "test":
        aggregate["test_time_seconds"] = float(forward_seconds)

    rows = np.arange(sample_count)
    top5 = LABELS[order[:, :5]]
    top5_scores = np.take_along_axis(probabilities, order[:, :5], axis=1)
    inverse_order = np.empty_like(order)
    inverse_order[rows[:, None], order] = np.arange(NUM_CLASSES, dtype=np.int64)
    records: list[dict[str, Any]] = []
    for index, (true_label, predicted_label) in enumerate(zip(truth, predicted)):
        true_label = int(true_label)
        predicted_label = int(predicted_label)
        metadata = metadata_rows[index]
        true_info = names[true_label]
        predicted_info = names[predicted_label]
        correct = true_label == predicted_label
        records.append(
            {
                "split": str(metadata.get("split") or split),
                "image_id": int(metadata["image_id"]),
                "image_path": str(metadata["image_path"]),
                "local_image_path": str(metadata["local_image_path"]),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": correct,
                "top1_score": float(probabilities[index, predicted_label]),
                "true_new_label": true_label,
                "predicted_new_label": predicted_label,
                "original_category_id": int(
                    metadata.get("original_category_id")
                    if metadata.get("original_category_id") is not None
                    else true_info["original_category_id"]
                ),
                "true_scientific_name": str(
                    metadata.get("scientific_name") or true_info["scientific_name"]
                ),
                "true_common_name": str(metadata.get("common_name") or true_info["common_name"]),
                "predicted_scientific_name": predicted_info["scientific_name"],
                "predicted_common_name": predicted_info["common_name"],
                "is_correct": correct,
                "is_top5_correct": true_label in top5[index],
                "true_class_score": float(probabilities[index, true_label]),
                "predicted_class_score": float(probabilities[index, predicted_label]),
                "true_class_rank": int(inverse_order[index, true_label] + 1),
                "top5_new_labels": json.dumps(top5[index].tolist(), separators=(",", ":")),
                "top5_scientific_names": json.dumps(
                    [names[int(label)]["scientific_name"] for label in top5[index]],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "top5_scores": json.dumps(
                    [float(value) for value in top5_scores[index]], separators=(",", ":")
                ),
                "score_type": SCORE_TYPE,
            }
        )
    predictions = pd.DataFrame.from_records(records)
    per_class = per_class_metrics(truth, predicted, LABELS)
    per_class.insert(
        1,
        "original_category_id",
        per_class["new_label"].map(lambda label: names[int(label)]["original_category_id"]),
    )
    per_class.insert(
        2,
        "scientific_name",
        per_class["new_label"].map(lambda label: names[int(label)]["scientific_name"]),
    )
    per_class.insert(
        3,
        "common_name",
        per_class["new_label"].map(lambda label: names[int(label)]["common_name"]),
    )
    pairs = top_confused_pairs(
        matrix,
        LABELS,
        limit=50,
        scientific_names={label: value["scientific_name"] for label, value in names.items()},
    )
    return {
        "metrics": aggregate,
        "predictions": predictions,
        "confusion": matrix,
        "per_class": per_class,
        "pairs": pairs,
    }


def _output_paths(name: str, split: str, args: argparse.Namespace) -> dict[str, Path]:
    metrics_dir = Path(getattr(args, "metrics_dir", "results/metrics/deep_learning"))
    predictions_dir = Path(
        getattr(args, "predictions_dir", "results/predictions/deep_learning")
    )
    tables_dir = Path(getattr(args, "tables_dir", "results/tables/deep_learning"))
    figures_dir = Path(getattr(args, "figures_dir", "results/figures/deep_learning"))
    prefix = f"{name}_{split}"
    return {
        "metrics": metrics_dir / f"{prefix}_metrics.json",
        "predictions": predictions_dir / f"{prefix}_predictions.csv",
        "confusion": metrics_dir / f"{prefix}_confusion.npz",
        "per_class": tables_dir / f"{prefix}_per_class_metrics.csv",
        "pairs": tables_dir / f"{prefix}_top_confusions.csv",
        "confusion_figure": figures_dir / f"{prefix}_confusion_matrix.png",
        "pairs_figure": figures_dir / f"{prefix}_top_confusions.png",
    }


def _compare_validation(
    context: Mapping[str, Any], observed: Mapping[str, Any], tolerance: float
) -> dict[str, Any]:
    if not math.isfinite(tolerance) or not 0.0 <= tolerance <= MAX_VALIDATION_METRIC_TOLERANCE:
        raise ValueError(
            f"Validation metric tolerance must be between 0 and "
            f"{MAX_VALIDATION_METRIC_TOLERANCE}"
        )
    checkpoint_best = context["checkpoint"]["best_metrics"]
    persisted_best = context["best_metrics"]["metrics"]
    if not isinstance(checkpoint_best, Mapping):
        raise ValueError("Checkpoint best_metrics must be an object")
    differences: dict[str, dict[str, float]] = {}
    for key in (
        "loss",
        "top1_accuracy",
        "top5_accuracy",
        "overall_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
    ):
        fresh = float(observed[key])
        checkpoint_value = _checkpoint_metric(checkpoint_best, key)
        persisted_value = _checkpoint_metric(persisted_best, key)
        differences[key] = {
            "observed": fresh,
            "checkpoint_best": checkpoint_value,
            "persisted_best": persisted_value,
            "max_absolute_difference": max(
                abs(fresh - checkpoint_value), abs(fresh - persisted_value)
            ),
        }
        if not (
            math.isclose(fresh, checkpoint_value, rel_tol=tolerance, abs_tol=tolerance)
            and math.isclose(fresh, persisted_value, rel_tol=tolerance, abs_tol=tolerance)
        ):
            raise ValueError(
                f"Independent validation disagrees with stored best metric {key}: "
                f"fresh={fresh}, checkpoint={checkpoint_value}, persisted={persisted_value}"
            )
    return {
        "agrees_with_stored_best": True,
        "tolerance": tolerance,
        "stored_best_epoch": int(context["checkpoint"]["best_epoch"]),
        "differences": differences,
    }


def _base_metrics_document(
    context: Mapping[str, Any],
    *,
    split: str,
    aggregate: Mapping[str, Any],
    recheck: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    checkpoint = context["checkpoint"]
    document: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "split": split,
        "experiment_name": context["name"],
        "model_name": str(checkpoint["model_name"]),
        "pretrained": bool(checkpoint["pretrained"]),
        "freeze_backbone": bool(checkpoint["freeze_backbone"]),
        "num_classes": NUM_CLASSES,
        "image_size": int(checkpoint["image_size"]),
        "best_epoch": int(checkpoint["best_epoch"]),
        "training_time_seconds": float(checkpoint["training_time_seconds"]),
        "score_type": SCORE_TYPE,
        "metrics": dict(aggregate),
        "validation_recheck": dict(recheck),
        "provenance": {
            "checkpoint_path": _portable(context["checkpoint_path"]),
            "checkpoint_sha256": context["checkpoint_sha256"],
            "resolved_config_path": _portable(context["resolved_config_path"]),
            "resolved_config_sha256": context["resolved_config_sha256"],
            "source_config_path": _portable(context["source_config_path"]),
            "source_config_sha256": context["source_config_sha256"],
            "stored_best_metrics_path": _portable(context["best_metrics_path"]),
            "stored_best_metrics_sha256": context["best_metrics_sha256"],
            "split_manifest": f"{split}.csv",
            "split_manifest_sha256": context["stage1"][f"{split}.csv"]["sha256"],
            "device": str(device),
            "versions": checkpoint["versions"],
        },
    }
    return document


def _stage_artifacts(
    result: Mapping[str, Any],
    document: dict[str, Any],
    outputs: Mapping[str, Path],
    stage_root: Path,
) -> dict[str, Path]:
    staged = {
        key: stage_root / f"{index:02d}_{path.name}"
        for index, (key, path) in enumerate(outputs.items())
    }
    _atomic_csv(staged["predictions"], result["predictions"])
    _atomic_confusion(staged["confusion"], result["confusion"])
    _atomic_csv(staged["per_class"], result["per_class"])
    _atomic_csv(staged["pairs"], result["pairs"])
    plot_confusion_matrix_overview(
        result["confusion"],
        staged["confusion_figure"],
        title=f"{document['experiment_name']}: {document['split']} confusion matrix",
    )
    plot_top_confused_pairs(
        result["pairs"],
        staged["pairs_figure"],
        title=f"{document['experiment_name']}: top {document['split']} confused pairs",
    )
    document["artifacts"] = {
        key: {
            "path": _portable(outputs[key]),
            "bytes": staged[key].stat().st_size,
            "sha256": sha256_file(staged[key]),
        }
        for key in outputs
        if key != "metrics"
    }
    _atomic_json(staged["metrics"], document)
    return staged


def _access_artifact_bindings(
    staged: Mapping[str, Path],
    outputs: Mapping[str, Path],
    document: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Staged metrics document has no artifact hash map")
    return {
        "metrics": {
            "path": _portable(outputs["metrics"]),
            "bytes": staged["metrics"].stat().st_size,
            "sha256": sha256_file(staged["metrics"]),
        },
        **{str(key): dict(value) for key, value in artifacts.items()},
    }


def _install_staged(
    staged_sets: Sequence[tuple[Mapping[str, Path], Mapping[str, Path]]],
    *,
    allow_replace: bool,
    defer_commit: bool = False,
) -> dict[str, Any]:
    installed: list[Path] = []
    backups: dict[Path, Path] = {}
    try:
        for staged, outputs in staged_sets:
            for key in [key for key in outputs if key != "metrics"] + ["metrics"]:
                final = outputs[key]
                if final.exists() and not allow_replace:
                    raise FileExistsError(f"Evaluation artifact already exists: {final}")
                final.parent.mkdir(parents=True, exist_ok=True)
                if allow_replace and final.exists():
                    handle, backup_name = tempfile.mkstemp(
                        prefix=f".{final.name}.stage3-backup.", dir=final.parent
                    )
                    os.close(handle)
                    backup = Path(backup_name)
                    backup.unlink()
                    os.replace(final, backup)
                    backups[final] = backup
                os.replace(staged[key], final)
                installed.append(final)
    except Exception:
        for path in reversed(installed):
            path.unlink(missing_ok=True)
        restoration_errors: list[str] = []
        for final, backup in backups.items():
            if not backup.exists():
                continue
            try:
                os.replace(backup, final)
            except OSError as error:
                restoration_errors.append(f"{final}: {error}")
        if restoration_errors:
            raise RuntimeError(
                "Stage 3 artifact transaction failed and rollback was incomplete: "
                + "; ".join(restoration_errors)
            )
        raise
    else:
        transaction = {"installed": list(installed), "backups": dict(backups)}
        if not defer_commit:
            _commit_installation(transaction)
        return transaction


def _commit_installation(transaction: Mapping[str, Any]) -> None:
    for backup in transaction.get("backups", {}).values():
        try:
            Path(backup).unlink(missing_ok=True)
        except OSError:
            # The new complete set is already installed and its completed
            # access record is committed.  A hidden stale backup is safer than
            # attempting a destructive rollback after a cleanup-only failure.
            pass


def _rollback_installation(transaction: Mapping[str, Any]) -> None:
    for path in reversed(transaction.get("installed", [])):
        Path(path).unlink(missing_ok=True)
    errors: list[str] = []
    for final, backup in transaction.get("backups", {}).items():
        final_path = Path(final)
        backup_path = Path(backup)
        if not backup_path.exists():
            continue
        try:
            os.replace(backup_path, final_path)
        except OSError as error:
            errors.append(f"{final_path}: {error}")
    if errors:
        raise RuntimeError(
            "Stage 3 artifact rollback was incomplete: " + "; ".join(errors)
        )


def evaluate_validation(args: argparse.Namespace) -> dict[str, Any]:
    metric_tolerance = float(
        getattr(args, "metric_tolerance", MAX_VALIDATION_METRIC_TOLERANCE)
    )
    if (
        not math.isfinite(metric_tolerance)
        or metric_tolerance < 0
        or metric_tolerance > MAX_VALIDATION_METRIC_TOLERANCE
    ):
        raise ValueError(
            f"Validation metric tolerance must be between 0 and "
            f"{MAX_VALIDATION_METRIC_TOLERANCE}"
        )
    checkpoint_arg = getattr(args, "checkpoint", None)
    if checkpoint_arg is None:
        raise ValueError("Validation evaluation requires exactly one --checkpoint")
    checkpoint_path = Path(checkpoint_arg).resolve()
    split_dir = Path(getattr(args, "split_dir", "data/splits")).resolve()
    metrics_dir = Path(getattr(args, "metrics_dir", "results/metrics/deep_learning"))
    context = _validate_checkpoint_context(
        checkpoint_path,
        split_dir=split_dir,
        metrics_dir=metrics_dir,
        resolved_config_path=getattr(args, "resolved_config", None),
        source_config_path=getattr(args, "source_config", None),
        best_metrics_path=getattr(args, "best_metrics", None),
    )
    outputs = _output_paths(context["name"], "val", args)
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not bool(getattr(args, "force", False)):
        raise FileExistsError(
            "Validation artifacts already exist; pass --force to replace them: "
            + ", ".join(str(path) for path in existing)
        )
    device = _select_device(getattr(args, "device", None))
    loader = _make_loader(
        split_dir / "val.csv",
        image_root=Path(getattr(args, "image_root", "data/processed/images")).resolve(),
        split="val",
        image_size=int(context["checkpoint"]["image_size"]),
        batch_size=int(getattr(args, "batch_size", 32)),
        num_workers=int(getattr(args, "num_workers", 4)),
    )
    result = _run_inference(
        context,
        loader=loader,
        device=device,
        split="val",
        names=_class_names(split_dir),
    )
    recheck = _compare_validation(
        context, result["metrics"], metric_tolerance
    )
    document = _base_metrics_document(
        context,
        split="val",
        aggregate=result["metrics"],
        recheck=recheck,
        device=device,
    )
    metrics_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{context['name']}_val_eval.", dir=metrics_dir.parent
    ) as temporary:
        staged = _stage_artifacts(result, document, outputs, Path(temporary))
        _install_staged([(staged, outputs)], allow_replace=bool(existing))
    print(
        f"[PASS] {context['name']} val: "
        f"top-1={result['metrics']['top1_accuracy']:.6f}, "
        f"top-5={result['metrics']['top5_accuracy']:.6f}, "
        f"macro-F1={result['metrics']['macro_f1']:.6f}"
    )
    return document


def _verify_artifact_entry(entry: Any, label: str) -> Path:
    if not isinstance(entry, Mapping):
        raise ValueError(f"Malformed frozen artifact entry: {label}")
    path = _resolve(str(entry.get("path", "")))
    if not path.is_file() or sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"Frozen artifact changed: {label}")
    if entry.get("bytes") is not None and int(entry["bytes"]) != path.stat().st_size:
        raise ValueError(f"Frozen artifact size changed: {label}")
    return path


def _verify_freeze(
    freeze_path: Path,
    *,
    split_dir: Path,
    verify_test_manifest: bool,
) -> dict[str, Any]:
    freeze = _read_json(freeze_path)
    if freeze.get("schema_version") != 1 or freeze.get("status") != "frozen":
        raise ValueError("Not a valid frozen Stage 3 manifest")
    selected = freeze.get("selected_experiments")
    models = freeze.get("models")
    if (
        not isinstance(selected, list)
        or not selected
        or len(selected) != len(set(selected))
        or not isinstance(models, Mapping)
        or set(selected) != set(models)
    ):
        raise ValueError("Freeze membership is malformed or incomplete")
    analysis_model = freeze.get("analysis_model")
    if analysis_model not in selected:
        raise ValueError("Frozen analysis model is not in the test set")
    candidates = freeze.get("completed_candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("Freeze has no completed-candidate selection evidence")
    expected_selected, expected_analysis, _ = _expected_selection(candidates)
    if selected != expected_selected or analysis_model != expected_analysis:
        raise ValueError(
            "Freeze membership does not match the deterministic validation ranking"
        )
    stage1 = freeze.get("stage1_artifacts")
    if not isinstance(stage1, Mapping) or set(stage1) != set(STAGE1_FILES):
        raise ValueError("Freeze must bind exactly six Stage 1 artifacts")
    for name in STAGE1_FILES:
        if name == "test.csv" and not verify_test_manifest:
            continue
        entry = stage1[name]
        path = split_dir / name
        if (
            not isinstance(entry, Mapping)
            or not path.is_file()
            or sha256_file(path) != entry.get("sha256")
        ):
            raise ValueError(f"Frozen Stage 1 artifact changed: {name}")
    stage2 = freeze.get("stage2_artifacts")
    if not isinstance(stage2, Mapping) or not stage2:
        raise ValueError("Freeze has no bound Stage 2 evidence")
    for name, entry in stage2.items():
        _verify_artifact_entry(entry, f"Stage 2 {name}")
    for name in selected:
        entry = models[name]
        if not isinstance(entry, Mapping) or entry.get("experiment_name") != name:
            raise ValueError(f"Malformed frozen model entry: {name}")
        for key in (
            "checkpoint",
            "resolved_config",
            "source_config",
            "training_log",
            "stored_best_metrics",
            "independent_validation_metrics",
        ):
            _verify_artifact_entry(entry.get(key), f"{name}.{key}")
        validation_artifacts = entry.get("validation_artifacts")
        if not isinstance(validation_artifacts, Mapping) or not validation_artifacts:
            raise ValueError(f"Frozen validation artifacts missing for {name}")
        for key, artifact_entry in validation_artifacts.items():
            _verify_artifact_entry(artifact_entry, f"{name}.validation_artifacts.{key}")
    return freeze


def _test_existing_outputs(
    selected: Sequence[str], args: argparse.Namespace, access_record: Path
) -> list[Path]:
    del selected
    paths = [access_record] if access_record.exists() else []
    for directory, pattern in (
        (CANONICAL_TEST_DIRECTORIES["metrics_dir"], "*_test_metrics.json"),
        (CANONICAL_TEST_DIRECTORIES["metrics_dir"], "*_test_confusion.npz"),
        (CANONICAL_TEST_DIRECTORIES["predictions_dir"], "*_test_predictions.csv"),
        (CANONICAL_TEST_DIRECTORIES["tables_dir"], "*_test_*.csv"),
        (CANONICAL_TEST_DIRECTORIES["figures_dir"], "*_test_*.png"),
    ):
        paths.extend(directory.glob(pattern))
    return sorted(set(paths))


def evaluate_frozen_test_set(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(getattr(args, "include_test", False)):
        # This check intentionally precedes every freeze/checkpoint/manifest read.
        raise ValueError("Test evaluation requires the explicit --include-test flag")
    if getattr(args, "checkpoint", None) is not None:
        raise ValueError("Frozen-set test evaluation does not accept an ad-hoc checkpoint")
    _require_canonical_test_gate(args)
    freeze_path = CANONICAL_FREEZE_PATH
    split_dir = Path(getattr(args, "split_dir", "data/splits")).resolve()
    # First verify every non-test binding.  test.csv remains unopened here.
    freeze = _verify_freeze(
        freeze_path, split_dir=split_dir, verify_test_manifest=False
    )
    freeze_hash = sha256_file(freeze_path)
    selected = list(freeze["selected_experiments"])
    metrics_dir = CANONICAL_TEST_DIRECTORIES["metrics_dir"]
    access_record = CANONICAL_ACCESS_RECORD
    dangerous = bool(getattr(args, "dangerous_overwrite_test", False))
    existing = _test_existing_outputs(selected, args, access_record)
    previous_access: dict[str, Any] | None = None
    if existing and not dangerous:
        raise FileExistsError(
            "Frozen Stage 3 test artifacts/access record already exist: "
            + ", ".join(str(path) for path in existing)
        )
    if dangerous:
        if access_record.is_file():
            previous_access = _read_json(access_record)
            if previous_access.get("freeze_manifest_sha256") != freeze_hash:
                raise ValueError("Dangerous test overwrite must preserve the same freeze")
        elif existing:
            raise ValueError("Dangerous overwrite requires the prior test-access record")

    contexts: dict[str, dict[str, Any]] = {}
    device = _select_device(getattr(args, "device", None))
    # Parse and load every checkpoint before test.csv is touched.
    for name in selected:
        entry = freeze["models"][name]
        checkpoint_path = _resolve(entry["checkpoint"]["path"])
        contexts[name] = _validate_checkpoint_context(
            checkpoint_path,
            split_dir=split_dir,
            metrics_dir=metrics_dir,
            resolved_config_path=_resolve(entry["resolved_config"]["path"]),
            source_config_path=_resolve(entry["source_config"]["path"]),
            best_metrics_path=_resolve(entry["stored_best_metrics"]["path"]),
            expected_name=name,
            frozen_entry=entry,
        )
        # Constructing now catches state/model mismatches without test access.
        model = _model_for_context(contexts[name], device)
        del model

    attempt_number = (
        int(previous_access.get("attempt_number", 1)) + 1 if previous_access else 1
    )
    started = {
        "schema_version": 1,
        "status": "started",
        "outcome": "in_progress",
        "attempt_number": attempt_number,
        "started_at_utc": _utc_now(),
        "freeze_manifest_path": _portable(freeze_path),
        "freeze_manifest_sha256": freeze_hash,
        "selected_experiments": selected,
        "analysis_model": freeze["analysis_model"],
    }
    if previous_access is not None:
        started["predecessor_access_sha256"] = sha256_file(access_record)
    _atomic_json(access_record, started)

    attempts: dict[str, dict[str, Any]] = {}
    batch_started = time.perf_counter()
    completed_document: dict[str, Any]
    installation_transaction: dict[str, Any] | None = None
    try:
        # The access record now exists; verify the frozen test manifest hash before loading it.
        reverified_freeze = _verify_freeze(
            freeze_path, split_dir=split_dir, verify_test_manifest=True
        )
        _require_unchanged_freeze(freeze_path, freeze_hash)
        if reverified_freeze != freeze:
            raise ValueError("Stage 3 freeze content changed during test setup")
        loader = _make_loader(
            split_dir / "test.csv",
            image_root=Path(getattr(args, "image_root", "data/processed/images")).resolve(),
            split="test",
            image_size=224,
            batch_size=int(getattr(args, "batch_size", 32)),
            num_workers=int(getattr(args, "num_workers", 4)),
        )
        names = _class_names(split_dir)
        stage_parent = metrics_dir.parent
        stage_parent.mkdir(parents=True, exist_ok=True)
        staged_sets: list[tuple[Mapping[str, Path], Mapping[str, Path]]] = []
        with tempfile.TemporaryDirectory(prefix=".stage3_test_batch.", dir=stage_parent) as temp:
            stage_root = Path(temp)
            for index, name in enumerate(selected):
                attempt_start = time.perf_counter()
                try:
                    result = _run_inference(
                        contexts[name],
                        loader=loader,
                        device=device,
                        split="test",
                        names=names,
                    )
                    recheck = {
                        "agrees_with_stored_best": True,
                        "source": "frozen independent validation",
                        "validation_metrics_sha256": freeze["models"][name][
                            "independent_validation_metrics"
                        ]["sha256"],
                    }
                    document = _base_metrics_document(
                        contexts[name],
                        split="test",
                        aggregate=result["metrics"],
                        recheck=recheck,
                        device=device,
                    )
                    document["validation"] = freeze["models"][name]["validation_metrics"]
                    document["provenance"].update(
                        {
                            "freeze_manifest_path": _portable(freeze_path),
                            "freeze_manifest_sha256": freeze_hash,
                            "test_access_record_path": _portable(access_record),
                        }
                    )
                    outputs = _output_paths(name, "test", args)
                    model_stage = stage_root / f"{index:02d}_{name}"
                    model_stage.mkdir()
                    staged = _stage_artifacts(result, document, outputs, model_stage)
                    staged_sets.append((staged, outputs))
                    output_bindings = _access_artifact_bindings(
                        staged, outputs, document
                    )
                    attempts[name] = {
                        "status": "complete",
                        "wall_time_seconds": time.perf_counter() - attempt_start,
                        "metrics": result["metrics"],
                        "artifacts": output_bindings,
                    }
                except Exception as error:  # Continue across the complete frozen set.
                    attempts[name] = {
                        "status": "failed",
                        "wall_time_seconds": time.perf_counter() - attempt_start,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
            if set(attempts) != set(selected):
                missing_attempts = sorted(set(selected) - set(attempts))
                raise RuntimeError(
                    "Frozen test batch did not attempt every selected model: "
                    + ", ".join(missing_attempts)
                )
            failed = [name for name, value in attempts.items() if value["status"] != "complete"]
            if failed:
                raise RuntimeError(
                    "Frozen test batch failed for: "
                    + ", ".join(f"{name} ({attempts[name]['error']})" for name in failed)
                )
            # Rehash immediately before publication so a changed freeze can
            # never authorize a staged result set computed under old bindings.
            _require_unchanged_freeze(freeze_path, freeze_hash)
            final_freeze = _verify_freeze(
                freeze_path, split_dir=split_dir, verify_test_manifest=True
            )
            _require_unchanged_freeze(freeze_path, freeze_hash)
            if final_freeze != freeze:
                raise ValueError("Stage 3 freeze content changed before test publication")
            installation_transaction = _install_staged(
                staged_sets,
                allow_replace=dangerous,
                defer_commit=True,
            )
            installed_freeze = _verify_freeze(
                freeze_path, split_dir=split_dir, verify_test_manifest=True
            )
            _require_unchanged_freeze(freeze_path, freeze_hash)
            if installed_freeze != freeze:
                raise ValueError("Stage 3 freeze content changed during test publication")
        completed_document = {
            **started,
            "status": "completed",
            "outcome": "complete",
            "completed_at_utc": _utc_now(),
            "batch_wall_time_seconds": time.perf_counter() - batch_started,
            "attempts": attempts,
        }
        _atomic_json(access_record, completed_document)
        assert installation_transaction is not None
        _commit_installation(installation_transaction)
    except Exception as error:
        rollback_error: str | None = None
        if installation_transaction is not None:
            try:
                _rollback_installation(installation_transaction)
            except Exception as rollback_exception:
                rollback_error = str(rollback_exception)
        completed_document = {
            **started,
            "status": "completed",
            "outcome": "failed",
            "completed_at_utc": _utc_now(),
            "batch_wall_time_seconds": time.perf_counter() - batch_started,
            "attempts": attempts,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if rollback_error is not None:
            completed_document["rollback_error"] = rollback_error
        _atomic_json(access_record, completed_document)
        raise

    print(f"[PASS] Frozen Stage 3 test batch completed for {len(selected)} model(s)")
    print(f"[PASS] Test access record: {access_record}")
    return completed_document


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    split = str(getattr(args, "split", "val"))
    if split == "val":
        return evaluate_validation(args)
    if split == "test":
        return evaluate_frozen_test_set(args)
    raise ValueError(f"Unknown evaluation split: {split}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one Stage 3 checkpoint on validation or the complete frozen test set."
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--best-metrics", type=Path)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("results/metrics/deep_learning/stage3_freeze.json"),
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--image-root", type=Path, default=Path("data/processed/images"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics/deep_learning"))
    parser.add_argument(
        "--predictions-dir", type=Path, default=Path("results/predictions/deep_learning")
    )
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables/deep_learning"))
    parser.add_argument("--figures-dir", type=Path, default=Path("results/figures/deep_learning"))
    parser.add_argument("--access-record", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--metric-tolerance", type=float, default=1e-6)
    parser.add_argument("--force", action="store_true", help="Replace validation artifacts")
    parser.add_argument("--dangerous-overwrite-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers non-negative")
    if (
        not math.isfinite(args.metric_tolerance)
        or args.metric_tolerance < 0
        or args.metric_tolerance > MAX_VALIDATION_METRIC_TOLERANCE
    ):
        parser.error(
            "--metric-tolerance must be between 0 and "
            f"{MAX_VALIDATION_METRIC_TOLERANCE}"
        )
    try:
        evaluate(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"[FAIL] {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
