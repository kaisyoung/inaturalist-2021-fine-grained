"""Leakage-safe robustness evaluation for frozen Stage 2/3 models.

Canonical execution evaluates the complete protocol matrix on held-out test
images only after ``--include-test``.  ``--max-samples`` is a smoke mode whose
outputs are forced into isolated directories and can never be mistaken for
full protocol evidence.  No optimizer or training API is used in this module.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import joblib
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.analysis.metrics import align_class_scores, evaluate_aligned_scores
from src.data.common import CSV_COLUMNS, NUM_CLASSES, deterministic_json_bytes, sha256_file
from src.robustness.corruptions import (
    CORRUPTION_SEVERITIES,
    CorruptionTransform,
    apply_corruption,
    corruption_spec,
    prepare_rgb,
)
from src.robustness.freeze_protocol import (
    CANONICAL_METRICS_DIR,
    CANONICAL_PREDICTIONS_DIR,
    CANONICAL_PROTOCOL,
    EXPECTED_MODEL_NAMES,
    PROJECT_ROOT,
    ProtocolError,
    load_and_verify_protocol,
)
from src.traditional.features import extract_features_from_array, feature_dimension


# The corruption pipeline recomputes only the per-image handcrafted descriptor.
# Its width belongs to the traditional feature contract and must not be copied as
# a literal into this downstream evaluator.
TRADITIONAL_FEATURE_DIM = feature_dimension()


LABELS = np.arange(NUM_CLASSES, dtype=np.int64)
PREDICTION_COLUMNS = (
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
    "predicted_scientific_name",
    "predicted_common_name",
    "score_type",
)
SMOKE_METRICS_DIR = Path("results/metrics/robustness_smoke")
SMOKE_PREDICTIONS_DIR = Path("results/predictions/robustness_smoke")


class RobustnessEvaluationError(RuntimeError):
    """Raised when evaluation or its provenance is unsafe/inconsistent."""


def _resolve(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _portable(root: Path, value: Path) -> str:
    try:
        return value.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise RobustnessEvaluationError(
            f"Output/artifact is outside the project root: {value.resolve()}"
        ) from error


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RobustnessEvaluationError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise RobustnessEvaluationError(f"Expected a JSON object in {path}")
    return value


def _artifact(root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RobustnessEvaluationError(f"Required artifact is missing: {path}")
    return {
        "path": _portable(root, path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _require_bound_artifact(root: Path, entry: Mapping[str, Any], label: str) -> Path:
    if not {"path", "bytes", "sha256"}.issubset(entry):
        raise RobustnessEvaluationError(f"Protocol {label} entry is incomplete")
    path = _resolve(root, str(entry["path"]))
    if not path.is_file():
        raise RobustnessEvaluationError(f"Protocol {label} is missing: {path}")
    if path.stat().st_size != int(entry["bytes"]):
        raise RobustnessEvaluationError(f"Protocol {label} byte count changed")
    observed = sha256_file(path)
    if observed != str(entry["sha256"]):
        raise RobustnessEvaluationError(
            f"Protocol {label} hash changed: expected {entry['sha256']}, got {observed}"
        )
    return path


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package, key in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("Pillow", "Pillow"),
        ("scikit-learn", "scikit_learn"),
        ("scikit-image", "scikit_image"),
        ("opencv-python-headless", "opencv_python_headless"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("joblib", "joblib"),
    ):
        try:
            versions[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "not_installed"
    return versions


def _load_manifest(root: Path, max_samples: int | None) -> pd.DataFrame:
    """Validate the complete test contract before optionally taking a smoke prefix."""

    from src.deep_learning.dataset import ManifestImageDataset
    from torchvision import transforms

    manifest_path = root / "data/splits/test.csv"
    image_root = root / "data/processed/images"
    # Construction performs exact schema/count/path/provenance/existence checks
    # without decoding image pixels.
    ManifestImageDataset(
        manifest_path,
        image_root,
        transforms.ToTensor(),
        "test",
        strict=True,
    )
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if list(frame.columns) != CSV_COLUMNS or len(frame) != 5000:
        raise RobustnessEvaluationError("Test manifest changed after strict validation")
    frame["image_id"] = pd.to_numeric(frame["image_id"], errors="raise").astype(np.int64)
    frame["new_label"] = pd.to_numeric(frame["new_label"], errors="raise").astype(np.int64)
    frame.insert(0, "manifest_index", np.arange(len(frame), dtype=np.int64))
    if max_samples is not None:
        frame = frame.iloc[:max_samples].copy()
    return frame.reset_index(drop=True)


def _resolve_image_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.parts[:1] != ("test",):
        raise RobustnessEvaluationError(f"Unsafe test local_image_path: {relative!r}")
    image_root = (root / "data/processed/images").resolve()
    resolved = image_root.joinpath(*pure.parts).resolve(strict=False)
    try:
        resolved.relative_to(image_root)
    except ValueError as error:
        raise RobustnessEvaluationError(f"Test image path escapes image root: {relative}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"Test image is missing: {resolved}")
    return resolved


def _traditional_feature_worker(
    task: tuple[str, int, str, int, int],
) -> np.ndarray:
    path_value, image_id, corruption_type, severity, seed = task
    path = Path(path_value)
    try:
        with Image.open(path) as opened:
            rgb = prepare_rgb(opened)
        corrupted = apply_corruption(
            rgb,
            corruption_type,
            severity,
            seed=seed,
            image_key=image_id,
        )
        resized = corrupted.resize((128, 128), resample=Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.uint8).copy()
        feature = extract_features_from_array(array)
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise RobustnessEvaluationError(
            f"Could not extract corrupted features for image_id={image_id}, path={path}: {error}"
        ) from error
    if (
        feature.shape != (TRADITIONAL_FEATURE_DIM,)
        or feature.dtype != np.float32
        or not np.isfinite(feature).all()
    ):
        raise RobustnessEvaluationError(
            f"Invalid traditional feature for image_id={image_id}: shape={feature.shape}, dtype={feature.dtype}"
        )
    return feature


def _extract_traditional_features(
    frame: pd.DataFrame,
    *,
    root: Path,
    corruption_type: str,
    severity: int,
    seed: int,
    workers: int,
) -> np.ndarray:
    tasks = [
        (
            str(_resolve_image_path(root, str(row.local_image_path))),
            int(row.image_id),
            corruption_type,
            severity,
            seed,
        )
        for row in frame.itertuples(index=False)
    ]
    matrix = np.empty((len(tasks), TRADITIONAL_FEATURE_DIM), dtype=np.float32)
    if workers == 1:
        iterator: Iterable[np.ndarray] = (
            _traditional_feature_worker(task)
            for task in tqdm(tasks, desc="Corrupted handcrafted features", unit="image")
        )
        for index, feature in enumerate(iterator):
            matrix[index] = feature
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            features = executor.map(_traditional_feature_worker, tasks, chunksize=16)
            for index, feature in enumerate(
                tqdm(features, total=len(tasks), desc="Corrupted handcrafted features", unit="image")
            ):
                matrix[index] = feature
    if not np.isfinite(matrix).all():
        raise RobustnessEvaluationError("Traditional feature matrix contains non-finite values")
    return matrix


def _load_traditional_runtime(
    root: Path, model_name: str, model_entry: Mapping[str, Any]
) -> dict[str, Any]:
    model_path = _require_bound_artifact(root, model_entry["artifact"], f"{model_name} model")
    _require_bound_artifact(root, model_entry["model_metadata"], f"{model_name} metadata")
    _require_bound_artifact(root, model_entry["feature_config"], f"{model_name} feature config")
    try:
        bundle = joblib.load(model_path)
    except Exception as error:
        raise RobustnessEvaluationError(f"Could not load frozen Random Forest: {error}") from error
    if not isinstance(bundle, dict) or bundle.get("experiment_name") != "random_forest":
        raise RobustnessEvaluationError("Frozen Random Forest bundle has an invalid experiment name")
    if bundle.get("model_type") != "random_forest":
        raise RobustnessEvaluationError("Frozen traditional bundle is not the Random Forest")
    estimator = bundle.get("estimator")
    classes = np.asarray(getattr(estimator, "classes_", []), dtype=np.int64)
    if not np.array_equal(classes, LABELS):
        raise RobustnessEvaluationError("Frozen Random Forest classes are not exactly 0..499")
    indices = np.asarray(bundle.get("feature_indices"), dtype=np.int64)
    if not np.array_equal(
        indices, np.arange(TRADITIONAL_FEATURE_DIM, dtype=np.int64)
    ):
        raise RobustnessEvaluationError(
            "Frozen Random Forest is not bound to the full "
            f"{TRADITIONAL_FEATURE_DIM}-D handcrafted feature vector"
        )
    return {"estimator": estimator, "classes": classes, "model_path": model_path}


def _traditional_inference(
    runtime: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    root: Path,
    corruption_type: str,
    severity: int,
    seed: int,
    workers: int,
) -> tuple[np.ndarray, float, float, float, str]:
    full_start = time.perf_counter()
    feature_start = time.perf_counter()
    features = _extract_traditional_features(
        frame,
        root=root,
        corruption_type=corruption_type,
        severity=severity,
        seed=seed,
        workers=workers,
    )
    feature_seconds = time.perf_counter() - feature_start
    estimator = runtime["estimator"]
    model_start = time.perf_counter()
    if hasattr(estimator, "predict_proba"):
        raw_scores = estimator.predict_proba(features)
        score_type = "predict_proba"
    elif hasattr(estimator, "decision_function"):
        raw_scores = estimator.decision_function(features)
        score_type = "decision_function"
    else:
        raise RobustnessEvaluationError("Traditional estimator exposes no score method")
    model_seconds = time.perf_counter() - model_start
    aligned = align_class_scores(raw_scores, runtime["classes"], LABELS)
    full_seconds = time.perf_counter() - full_start
    return aligned, full_seconds, model_seconds, feature_seconds, score_type


def _load_deep_runtime(
    root: Path,
    model_name: str,
    model_entry: Mapping[str, Any],
    *,
    device_name: str,
) -> dict[str, Any]:
    import torch

    from src.deep_learning.models import create_model
    from src.deep_learning.train_utils import select_device

    checkpoint_path = _require_bound_artifact(root, model_entry["artifact"], f"{model_name} checkpoint")
    config_path = _require_bound_artifact(root, model_entry["resolved_config"], f"{model_name} resolved config")
    source_path = _require_bound_artifact(root, model_entry["source_config"], f"{model_name} source config")
    config = _read_json(config_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RobustnessEvaluationError(f"Could not load checkpoint {checkpoint_path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise RobustnessEvaluationError(f"Checkpoint is not a mapping: {checkpoint_path}")
    expected = {
        "experiment_name": model_name,
        "model_name": model_entry["architecture"],
        "num_classes": NUM_CLASSES,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RobustnessEvaluationError(
                f"Checkpoint {key} mismatch for {model_name}: expected {value!r}, got {checkpoint.get(key)!r}"
            )
    if checkpoint.get("resolved_config_sha256") != model_entry["resolved_config"]["sha256"]:
        raise RobustnessEvaluationError(f"Checkpoint/resolved-config hash mismatch for {model_name}")
    if config.get("source_config_sha256") != model_entry["source_config"]["sha256"]:
        raise RobustnessEvaluationError(f"Resolved/source-config hash mismatch for {model_name}")
    if int(config.get("image_size", -1)) != 224:
        raise RobustnessEvaluationError(f"Stage 4 requires the frozen 224-pixel eval transform for {model_name}")

    # Evaluation deliberately constructs the architecture without downloading
    # ImageNet weights; the complete frozen Stage 3 state dict is then loaded.
    model, metadata = create_model(
        str(model_entry["architecture"]),
        pretrained=False,
        freeze_backbone=False,
        num_classes=NUM_CLASSES,
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as error:
        raise RobustnessEvaluationError(f"Could not restore {model_name} weights: {error}") from error
    if metadata["total_parameters"] != checkpoint.get("parameter_counts", {}).get("total"):
        raise RobustnessEvaluationError(f"Parameter-count mismatch for {model_name}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    device = select_device(device_name)
    model.to(device)
    return {
        "model": model,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "config_path": config_path,
        "source_path": source_path,
        "image_size": 224,
    }


def _metadata_values(metadata: Mapping[str, Any], key: str, count: int) -> list[Any]:
    value = metadata[key]
    if hasattr(value, "detach"):
        result = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        result = list(value)
    else:
        result = [value]
    if len(result) != count:
        raise RobustnessEvaluationError(f"Collated metadata field {key!r} has wrong length")
    return result


def _deep_inference(
    runtime: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    root: Path,
    corruption_type: str,
    severity: int,
    seed: int,
    batch_size: int,
    workers: int,
) -> tuple[np.ndarray, float, float, float, str]:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    from src.deep_learning.dataset import ManifestImageDataset
    from src.deep_learning.train_utils import seed_worker, synchronize_device
    from src.deep_learning.transforms import get_transform

    eval_transform = get_transform("eval", image_size=int(runtime["image_size"]))
    transform = transforms.Compose(
        [CorruptionTransform(corruption_type, severity, seed=seed), *eval_transform.transforms]
    )
    full_dataset = ManifestImageDataset(
        root / "data/splits/test.csv",
        root / "data/processed/images",
        transform,
        "test",
        strict=True,
    )
    dataset: Any = full_dataset
    if len(frame) != len(full_dataset):
        expected_indices = frame["manifest_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(expected_indices, np.arange(len(frame), dtype=np.int64)):
            raise RobustnessEvaluationError("Smoke subset must be the deterministic manifest prefix")
        dataset = Subset(full_dataset, expected_indices.tolist())
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    model = runtime["model"]
    device = runtime["device"]
    logits_batches: list[np.ndarray] = []
    observed_ids: list[int] = []
    model_seconds = 0.0
    full_start = time.perf_counter()
    with torch.inference_mode():
        for images, labels, metadata in tqdm(loader, desc="Corrupted deep inference", unit="batch"):
            if not isinstance(metadata, Mapping):
                raise RobustnessEvaluationError("Dataset metadata did not collate to a mapping")
            count = int(labels.shape[0])
            observed_ids.extend(int(value) for value in _metadata_values(metadata, "image_id", count))
            images = images.to(device=device, dtype=torch.float32, non_blocking=False)
            synchronize_device(device)
            model_start = time.perf_counter()
            logits = model(images)
            synchronize_device(device)
            model_seconds += time.perf_counter() - model_start
            if tuple(logits.shape) != (count, NUM_CLASSES) or not bool(torch.isfinite(logits).all()):
                raise RobustnessEvaluationError(
                    f"Deep model returned invalid logits shape/values: {tuple(logits.shape)}"
                )
            logits_batches.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
    full_seconds = time.perf_counter() - full_start
    expected_ids = frame["image_id"].astype(int).tolist()
    if observed_ids != expected_ids:
        raise RobustnessEvaluationError("Deep inference order differs from test manifest order")
    if not logits_batches:
        raise RobustnessEvaluationError("Deep robustness dataset is empty")
    logits = np.concatenate(logits_batches, axis=0)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    scores = np.exp(shifted)
    scores /= scores.sum(axis=1, keepdims=True)
    return scores, full_seconds, model_seconds, 0.0, "softmax_probability"


def _prediction_frame(
    frame: pd.DataFrame,
    scores: np.ndarray,
    predicted: np.ndarray,
    order: np.ndarray,
    *,
    score_type: str,
    corruption_type: str,
    severity: int,
    root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    spec = corruption_spec(corruption_type, severity)
    truth = frame["new_label"].to_numpy(dtype=np.int64)
    rows = np.arange(len(frame), dtype=np.int64)
    top5 = LABELS[order[:, :5]]
    top5_scores = np.take_along_axis(scores, order[:, :5], axis=1)
    inverse = np.empty_like(order)
    inverse[rows[:, None], order] = np.arange(NUM_CLASSES, dtype=np.int64)
    names = (
        frame.loc[:, ["new_label", "scientific_name", "common_name"]]
        .drop_duplicates("new_label")
        .set_index("new_label")
        .to_dict("index")
    )
    # Smoke prefixes do not contain every label, so fill predicted names from
    # the immutable class mapping rather than guessing from the subset.
    mapping_path = root / "data/splits/class_mapping.json"
    mapping = _read_json(mapping_path)
    for entry in mapping.get("classes", []):
        if isinstance(entry, Mapping):
            label = int(entry.get("new_label", -1))
            if 0 <= label < NUM_CLASSES:
                names[label] = {
                    "scientific_name": str(entry.get("scientific_name", "")),
                    "common_name": str(entry.get("common_name", "")),
                }
    if set(names) != set(range(NUM_CLASSES)):
        raise RobustnessEvaluationError("Class mapping does not define all 500 labels")

    records: list[dict[str, Any]] = []
    for index in range(len(frame)):
        true_label = int(truth[index])
        predicted_label = int(predicted[index])
        records.append(
            {
                "manifest_index": int(frame.iloc[index]["manifest_index"]),
                "image_id": int(frame.iloc[index]["image_id"]),
                "local_image_path": str(frame.iloc[index]["local_image_path"]),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": true_label == predicted_label,
                "top5_labels": json.dumps(top5[index].tolist(), separators=(",", ":")),
                "top5_scores": json.dumps(
                    [float(value) for value in top5_scores[index]], separators=(",", ":")
                ),
                "true_class_score": float(scores[index, true_label]),
                "true_class_rank": int(inverse[index, true_label] + 1),
                "corruption_type": spec.corruption_type,
                "severity_index": spec.severity_index,
                "severity_parameter": spec.parameter,
                "severity_value": spec.value,
                "scientific_name": str(frame.iloc[index]["scientific_name"]),
                "common_name": str(frame.iloc[index]["common_name"]),
                "predicted_scientific_name": str(names[predicted_label]["scientific_name"]),
                "predicted_common_name": str(names[predicted_label]["common_name"]),
                "score_type": score_type,
            }
        )
    result = pd.DataFrame.from_records(records, columns=PREDICTION_COLUMNS)
    return result


def _prefix(model_name: str, corruption_type: str, severity: int) -> str:
    return f"{model_name}_{corruption_type}_s{severity}"


def _output_paths(
    root: Path,
    model_name: str,
    corruption_type: str,
    severity: int,
    *,
    max_samples: int | None,
    output_dir: Path | None,
) -> tuple[Path, Path]:
    prefix = _prefix(model_name, corruption_type, severity)
    if output_dir is not None:
        base = _resolve(root, output_dir)
        metrics_dir = base / "metrics"
        predictions_dir = base / "predictions"
    elif max_samples is None:
        metrics_dir = _resolve(root, CANONICAL_METRICS_DIR)
        predictions_dir = _resolve(root, CANONICAL_PREDICTIONS_DIR)
    else:
        metrics_dir = _resolve(root, SMOKE_METRICS_DIR)
        predictions_dir = _resolve(root, SMOKE_PREDICTIONS_DIR)
    if max_samples is not None:
        canonical_metrics = _resolve(root, CANONICAL_METRICS_DIR)
        canonical_predictions = _resolve(root, CANONICAL_PREDICTIONS_DIR)
        if metrics_dir == canonical_metrics or predictions_dir == canonical_predictions:
            raise RobustnessEvaluationError("Smoke outputs may not use canonical full-output directories")
        prefix += f"_smoke_n{max_samples}"
    return metrics_dir / f"{prefix}.json", predictions_dir / f"{prefix}_predictions.csv"


def _validate_existing(
    metrics_path: Path,
    predictions_path: Path,
    *,
    protocol_sha256: str,
    model_name: str,
    corruption_type: str,
    severity: int,
    expected_samples: int,
) -> dict[str, Any] | None:
    if not metrics_path.exists() and not predictions_path.exists():
        return None
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise RobustnessEvaluationError(
            f"Stale/partial output pair exists for {_prefix(model_name, corruption_type, severity)}"
        )
    document = _read_json(metrics_path)
    spec = corruption_spec(corruption_type, severity)
    expected = {
        "schema_version": 1,
        "stage": "stage4_robustness",
        "status": "complete",
        "model_name": model_name,
        "split": "test",
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise RobustnessEvaluationError(f"Stale metrics output: {metrics_path} has wrong {key}")
    if document.get("protocol", {}).get("sha256") != protocol_sha256:
        raise RobustnessEvaluationError(f"Stale metrics output uses a different protocol: {metrics_path}")
    if document.get("corruption") != {
        **spec.to_dict(),
        "seed": 42,
        "determinism": (
            "per-image PCG64 base realization shared across severities"
            if spec.corruption_type == "gaussian_noise"
            else "per-image angle shared across severities"
            if spec.corruption_type == "motion_blur"
            else "deterministic operation"
        ),
    }:
        raise RobustnessEvaluationError(f"Stale metrics output has a different corruption: {metrics_path}")
    metrics = document.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("num_samples") != expected_samples:
        raise RobustnessEvaluationError(f"Stale metrics output has wrong sample count: {metrics_path}")
    prediction_entry = document.get("artifacts", {}).get("predictions")
    if not isinstance(prediction_entry, Mapping):
        raise RobustnessEvaluationError(f"Stale metrics output lacks prediction provenance: {metrics_path}")
    if predictions_path.stat().st_size != int(prediction_entry.get("bytes", -1)):
        raise RobustnessEvaluationError(f"Existing prediction byte count changed: {predictions_path}")
    if sha256_file(predictions_path) != prediction_entry.get("sha256"):
        raise RobustnessEvaluationError(f"Existing prediction hash changed: {predictions_path}")
    prediction_frame = pd.read_csv(predictions_path)
    if list(prediction_frame.columns) != list(PREDICTION_COLUMNS) or len(prediction_frame) != expected_samples:
        raise RobustnessEvaluationError(f"Existing prediction schema/count is stale: {predictions_path}")
    return document


def _atomic_output_pair(
    metrics_path: Path,
    predictions_path: Path,
    predictions: pd.DataFrame,
    document_without_artifact: MutableMapping[str, Any],
    *,
    root: Path,
    protocol_path: Path,
    protocol_sha256: str,
    model_path: Path,
    model_sha256: str,
) -> dict[str, Any]:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError("Robustness outputs appeared during evaluation; refusing replacement")

    pred_descriptor, pred_name = tempfile.mkstemp(
        prefix=f".{predictions_path.name}.", suffix=".tmp", dir=predictions_path.parent
    )
    prediction_tmp = Path(pred_name)
    metrics_descriptor, metrics_name = tempfile.mkstemp(
        prefix=f".{metrics_path.name}.", suffix=".tmp", dir=metrics_path.parent
    )
    metrics_tmp = Path(metrics_name)
    try:
        with os.fdopen(pred_descriptor, "w", encoding="utf-8", newline="") as stream:
            predictions.to_csv(stream, index=False, lineterminator="\n")
            stream.flush()
            os.fsync(stream.fileno())
        prediction_entry = {
            "path": _portable(root, predictions_path),
            "bytes": int(prediction_tmp.stat().st_size),
            "sha256": sha256_file(prediction_tmp),
        }
        document_without_artifact["artifacts"] = {"predictions": prediction_entry}
        with os.fdopen(metrics_descriptor, "wb") as stream:
            stream.write(deterministic_json_bytes(document_without_artifact))
            stream.flush()
            os.fsync(stream.fileno())

        if sha256_file(protocol_path) != protocol_sha256:
            raise RobustnessEvaluationError("Robustness protocol changed during evaluation")
        if sha256_file(model_path) != model_sha256:
            raise RobustnessEvaluationError("Frozen model artifact changed during evaluation")
        if metrics_path.exists() or predictions_path.exists():
            raise FileExistsError("Robustness outputs appeared during evaluation")
        os.replace(prediction_tmp, predictions_path)
        os.replace(metrics_tmp, metrics_path)
        return dict(document_without_artifact)
    finally:
        prediction_tmp.unlink(missing_ok=True)
        metrics_tmp.unlink(missing_ok=True)


def evaluate_one(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
    model_name: str,
    corruption_type: str,
    severity: int,
    project_root: Path = PROJECT_ROOT,
    batch_size: int = 32,
    workers: int = 4,
    device: str = "auto",
    max_samples: int | None = None,
    output_dir: Path | None = None,
    runtime_cache: MutableMapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one protocol cell, returning a fresh or exact-valid document."""

    root = Path(project_root).resolve()
    if model_name not in EXPECTED_MODEL_NAMES:
        raise RobustnessEvaluationError(f"Model is not in the frozen protocol: {model_name}")
    spec = corruption_spec(corruption_type, severity)
    model_entry = protocol.get("models", {}).get(model_name)
    if not isinstance(model_entry, Mapping):
        raise RobustnessEvaluationError(f"Protocol has no model entry for {model_name}")
    if int(protocol.get("seed", -1)) != 42:
        raise RobustnessEvaluationError("Protocol seed is not the fixed value 42")
    model_path = _require_bound_artifact(root, model_entry["artifact"], f"{model_name} artifact")
    model_sha = str(model_entry["artifact"]["sha256"])
    metrics_path, predictions_path = _output_paths(
        root,
        model_name,
        spec.corruption_type,
        spec.severity_index,
        max_samples=max_samples,
        output_dir=output_dir,
    )
    expected_samples = 5000 if max_samples is None else max_samples
    existing = _validate_existing(
        metrics_path,
        predictions_path,
        protocol_sha256=protocol_sha256,
        model_name=model_name,
        corruption_type=spec.corruption_type,
        severity=spec.severity_index,
        expected_samples=expected_samples,
    )
    if existing is not None:
        print(f"[SKIP] exact-valid robustness cache: {metrics_path}")
        return existing

    frame = _load_manifest(root, max_samples)
    if len(frame) != expected_samples:
        raise RobustnessEvaluationError(
            f"Evaluation selected {len(frame)} images; expected {expected_samples}"
        )
    cache = runtime_cache if runtime_cache is not None else {}
    if model_name not in cache:
        if model_entry["method_family"] == "traditional":
            cache[model_name] = _load_traditional_runtime(root, model_name, model_entry)
        elif model_entry["method_family"] == "deep_learning":
            cache[model_name] = _load_deep_runtime(
                root, model_name, model_entry, device_name=device
            )
        else:
            raise RobustnessEvaluationError(f"Unknown method family for {model_name}")
    runtime = cache[model_name]
    if model_entry["method_family"] == "traditional":
        scores, full_seconds, model_seconds, feature_seconds, score_type = _traditional_inference(
            runtime,
            frame,
            root=root,
            corruption_type=spec.corruption_type,
            severity=spec.severity_index,
            seed=42,
            workers=workers,
        )
    else:
        scores, full_seconds, model_seconds, feature_seconds, score_type = _deep_inference(
            runtime,
            frame,
            root=root,
            corruption_type=spec.corruption_type,
            severity=spec.severity_index,
            seed=42,
            batch_size=batch_size,
            workers=workers,
        )
    truth = frame["new_label"].to_numpy(dtype=np.int64)
    metrics, predicted, order, _ = evaluate_aligned_scores(truth, scores, LABELS)
    metrics.update(
        {
            "inference_time_seconds": float(model_seconds),
            "model_forward_time_seconds": float(model_seconds),
            "evaluation_wall_time_seconds": float(full_seconds),
            "feature_extraction_time_seconds": float(feature_seconds),
            "seconds_per_image": float(model_seconds / len(frame)),
        }
    )
    predictions = _prediction_frame(
        frame,
        scores,
        predicted,
        order,
        score_type=score_type,
        corruption_type=spec.corruption_type,
        severity=spec.severity_index,
        root=root,
    )
    determinism = (
        "per-image PCG64 base realization shared across severities"
        if spec.corruption_type == "gaussian_noise"
        else "per-image angle shared across severities"
        if spec.corruption_type == "motion_blur"
        else "deterministic operation"
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "stage": "stage4_robustness",
        "status": "complete",
        "run_type": "canonical_full" if max_samples is None else "smoke",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name,
        "method_family": str(model_entry["method_family"]),
        "architecture": str(model_entry["architecture"]),
        "split": "test",
        "protocol": {
            "path": _portable(root, protocol_path),
            "sha256": protocol_sha256,
        },
        "corruption": {
            **spec.to_dict(),
            "seed": 42,
            "determinism": determinism,
        },
        "inputs": {
            "manifest": _artifact(root, root / "data/splits/test.csv"),
            "model_artifact": dict(model_entry["artifact"]),
            "model_support_artifacts": {
                key: dict(value)
                for key, value in model_entry.items()
                if key
                in {
                    "model_metadata",
                    "feature_config",
                    "resolved_config",
                    "source_config",
                }
                and isinstance(value, Mapping)
            },
            "clean_metrics": dict(model_entry["clean_metrics"]),
            "clean_predictions": dict(model_entry["clean_predictions"]),
        },
        "metrics": metrics,
        "timing": {
            "full_evaluation_wall_seconds": float(full_seconds),
            "model_forward_seconds": float(model_seconds),
            "feature_extraction_seconds": float(feature_seconds),
            "definition": (
                "full wall includes image decode, on-the-fly corruption, preprocessing, "
                "feature extraction/transfer, and one model score call; serialization and metrics excluded. "
                "model_forward_seconds covers only predict_proba/decision_function or synchronized neural forward passes."
            ),
        },
        "evaluation": {
            "batch_size": int(batch_size),
            "num_workers": int(workers),
            "max_samples": max_samples,
            "sample_selection": (
                "complete manifest" if max_samples is None else "deterministic manifest prefix"
            ),
            "score_type": score_type,
            "labels": "exactly 0..499",
            "stable_tie_break": "lower new_label first",
            "weight_update_operations": 0,
        },
        "package_versions": _package_versions(),
    }
    return _atomic_output_pair(
        metrics_path,
        predictions_path,
        predictions,
        document,
        root=root,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha256,
        model_path=model_path,
        model_sha256=model_sha,
    )


def run_protocol(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    protocol_sha256: str,
    project_root: Path = PROJECT_ROOT,
    batch_size: int = 32,
    workers: int = 4,
    device: str = "auto",
    max_samples: int | None = None,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run model-major order over the complete frozen corruption matrix."""

    records: list[dict[str, Any]] = []
    cache: dict[str, Any] = {}
    for model_name in EXPECTED_MODEL_NAMES:
        for corruption_type in CORRUPTION_SEVERITIES:
            for severity in range(1, 5):
                records.append(
                    evaluate_one(
                        protocol=protocol,
                        protocol_path=protocol_path,
                        protocol_sha256=protocol_sha256,
                        model_name=model_name,
                        corruption_type=corruption_type,
                        severity=severity,
                        project_root=project_root,
                        batch_size=batch_size,
                        workers=workers,
                        device=device,
                        max_samples=max_samples,
                        output_dir=output_dir,
                        runtime_cache=cache,
                    )
                )
        cache.pop(model_name, None)
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Stage 2/3 models under the fixed Stage 4 corruptions."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--protocol-run", action="store_true", help="Run every frozen model/corruption/severity")
    mode.add_argument("--model-name", choices=EXPECTED_MODEL_NAMES)
    parser.add_argument("--model-type", choices=("traditional", "deep"))
    parser.add_argument("--model-artifact", type=Path, help="Optional assertion; must equal protocol artifact")
    parser.add_argument("--config", type=Path, help="Optional assertion; must equal bound feature/resolved config")
    parser.add_argument("--corruption", choices=tuple(CORRUPTION_SEVERITIES))
    parser.add_argument("--severity", type=int, choices=range(1, 5))
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--include-test", action="store_true", help="Required before any held-out test access")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-samples", type=int, help="Isolated smoke mode using the manifest prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--protocol", type=Path, default=CANONICAL_PROTOCOL)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.include_test:
        parser.error("held-out robustness evaluation requires --include-test")
    if args.seed != 42:
        parser.error("the frozen Stage 4 protocol requires --seed 42")
    if args.batch_size < 1 or args.num_workers < 1:
        parser.error("--batch-size and --num-workers must be positive")
    if args.max_samples is not None and not 1 <= args.max_samples <= 5000:
        parser.error("--max-samples must be between 1 and 5000")
    if args.max_samples == 5000:
        parser.error("--max-samples 5000 is not smoke mode; omit it for a canonical full run")
    if args.protocol_run:
        if args.corruption is not None or args.severity is not None:
            parser.error("--protocol-run may not be combined with --corruption/--severity")
        if args.model_type or args.model_artifact or args.config:
            parser.error("--protocol-run uses model metadata only from the frozen protocol")
    elif args.corruption is None or args.severity is None:
        parser.error("an individual run requires --corruption and --severity")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    root = Path(args.project_root).resolve()
    protocol_path = _resolve(root, args.protocol)
    try:
        protocol, protocol_sha = load_and_verify_protocol(
            protocol_path, project_root=root
        )
        if not args.protocol_run:
            entry = protocol["models"][args.model_name]
            expected_type = "traditional" if entry["method_family"] == "traditional" else "deep"
            if args.model_type is not None and args.model_type != expected_type:
                raise RobustnessEvaluationError("--model-type contradicts the frozen protocol")
            if args.model_artifact is not None:
                if _resolve(root, args.model_artifact) != _resolve(root, entry["artifact"]["path"]):
                    raise RobustnessEvaluationError("--model-artifact differs from the frozen protocol")
            if args.config is not None:
                key = "feature_config" if expected_type == "traditional" else "resolved_config"
                if _resolve(root, args.config) != _resolve(root, entry[key]["path"]):
                    raise RobustnessEvaluationError("--config differs from the frozen protocol")
            evaluate_one(
                protocol=protocol,
                protocol_path=protocol_path,
                protocol_sha256=protocol_sha,
                model_name=args.model_name,
                corruption_type=args.corruption,
                severity=args.severity,
                project_root=root,
                batch_size=args.batch_size,
                workers=args.num_workers,
                device=args.device,
                max_samples=args.max_samples,
                output_dir=args.output_dir,
            )
            print(f"[PASS] Completed {_prefix(args.model_name, args.corruption, args.severity)}")
        else:
            records = run_protocol(
                protocol=protocol,
                protocol_path=protocol_path,
                protocol_sha256=protocol_sha,
                project_root=root,
                batch_size=args.batch_size,
                workers=args.num_workers,
                device=args.device,
                max_samples=args.max_samples,
                output_dir=args.output_dir,
            )
            print(f"[PASS] Completed/verified {len(records)} robustness protocol cells")
    except (OSError, ValueError, TypeError, KeyError, ProtocolError, RobustnessEvaluationError) as error:
        print(f"[FAIL] {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PREDICTION_COLUMNS",
    "RobustnessEvaluationError",
    "build_parser",
    "evaluate_one",
    "run_protocol",
]
