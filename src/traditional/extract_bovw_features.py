"""Pilot and optionally materialize the train-only SIFT-BoVW feature cache."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import joblib
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm

from src.data.common import (
    CSV_COLUMNS,
    NUM_CLASSES,
    SCHEMA_VERSION as STAGE1_SCHEMA_VERSION,
    TEST_IMAGES_PER_CLASS,
    TRAIN_IMAGES_PER_CLASS,
    VAL_IMAGES_PER_CLASS,
    deterministic_json_bytes,
    is_safe_relative_posix_path,
    sha256_file,
)
from src.traditional.bovw import (
    BoVWConfig,
    bovw_configuration,
    create_sift,
    deterministic_balanced_indices,
    encode_descriptors,
    extract_sift_descriptors,
    extract_sift_result,
    fit_vocabulary,
    load_grayscale,
    make_pilot_gate,
    run_descriptor_pass,
)


EXPECTED_ROWS = {"train": 20_000, "val": 5_000, "test": 5_000}
EXPECTED_PER_CLASS = {
    "train": TRAIN_IMAGES_PER_CLASS,
    "val": VAL_IMAGES_PER_CLASS,
    "test": TEST_IMAGES_PER_CLASS,
}
EXPECTED_SOURCE_SPLIT = {"train": "train_mini", "val": "train_mini", "test": "val"}
BASE_STAGE1_ARTIFACTS = {
    "selected_classes_seed42.txt",
    "class_mapping.json",
}
PRE_SELECTION_SPLITS = ("train", "val")
BOVW_CACHE_SCHEMA_VERSION = 3
CONFIG_NAME = "bovw_feature_config.json"
TEST_CONFIG_NAME = "bovw_test_feature_config.json"
DEFAULT_SELECTION_MANIFEST = Path("results/metrics/stage2_selection_manifest.json")
GATE_NAME = "bovw_pilot_gate.json"
STATUS_NAME = "bovw_status.json"
LEGACY_FAILURE_NAME = "bovw_failure.json"


def _atomic_json(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(deterministic_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(arrays: Mapping[str, np.ndarray], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_joblib(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        joblib.dump(value, temporary, compress=3)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_stage1_provenance(
    split_dir: Path, manifest_hashes: Mapping[str, str]
) -> dict[str, Any]:
    summary_path = split_dir / "split_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"Missing frozen Stage 1 summary: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read Stage 1 summary {summary_path}: {exc}") from exc
    if summary.get("schema_version") != STAGE1_SCHEMA_VERSION:
        raise ValueError(
            f"Stage 1 summary schema must be {STAGE1_SCHEMA_VERSION}"
        )
    if summary.get("csv_columns") != CSV_COLUMNS:
        raise ValueError("Stage 1 summary CSV schema does not match src.data.common")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Stage 1 summary has no artifacts mapping")
    required = BASE_STAGE1_ARTIFACTS | {
        f"{split}.csv" for split in manifest_hashes
    }
    missing = sorted(required - set(artifacts))
    if missing:
        raise ValueError(f"Stage 1 summary is missing frozen artifacts: {missing}")
    frozen_hashes: dict[str, str] = {}
    for name in sorted(required):
        path = split_dir / name
        if not path.is_file():
            raise ValueError(f"Missing frozen Stage 1 artifact: {path}")
        details = artifacts[name]
        if not isinstance(details, Mapping) or not isinstance(details.get("sha256"), str):
            raise ValueError(f"Invalid Stage 1 artifact metadata for {name}")
        actual_hash = sha256_file(path)
        if actual_hash != details["sha256"]:
            raise ValueError(
                f"Frozen Stage 1 artifact hash mismatch for {name}: "
                f"expected {details['sha256']}, got {actual_hash}"
            )
        if details.get("bytes") is not None and int(details["bytes"]) != path.stat().st_size:
            raise ValueError(f"Frozen Stage 1 artifact size mismatch for {name}")
        frozen_hashes[name] = actual_hash
    for split, digest in manifest_hashes.items():
        if frozen_hashes[f"{split}.csv"] != digest:
            raise ValueError(f"Stage 1 manifest hash mismatch for {split}.csv")
    return {
        "schema_version": int(summary["schema_version"]),
        "seed": int(summary["seed"]),
        "split_summary_file_name": summary_path.name,
        "split_summary_sha256": sha256_file(summary_path),
        "frozen_artifact_hashes": frozen_hashes,
    }


def load_manifests(
    split_dir: Path,
    image_root: Path,
    splits: Sequence[str] = PRE_SELECTION_SPLITS,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[Path]], dict[str, Any]]:
    """Load the immutable Stage 1 CSVs and resolve safe materialized paths."""

    frames: dict[str, pd.DataFrame] = {}
    paths: dict[str, list[Path]] = {}
    manifest_hashes: dict[str, str] = {}
    image_root = image_root.resolve()
    requested = tuple(str(split).strip().lower() for split in splits)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("splits must contain unique split names")
    unknown = sorted(set(requested) - set(EXPECTED_ROWS))
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}")
    for split in requested:
        expected_rows = EXPECTED_ROWS[split]
        manifest_path = split_dir / f"{split}.csv"
        if not manifest_path.is_file():
            raise ValueError(f"Missing Stage 1 manifest: {manifest_path}")
        frame = pd.read_csv(manifest_path, keep_default_na=False)
        if list(frame.columns) != CSV_COLUMNS:
            raise ValueError(
                f"{manifest_path} columns do not exactly match the frozen CSV schema"
            )
        if len(frame) != expected_rows:
            raise ValueError(
                f"{manifest_path} has {len(frame)} rows; expected {expected_rows}"
            )
        try:
            labels = frame["new_label"].to_numpy(dtype=np.int64)
            image_ids = frame["image_id"].to_numpy(dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{manifest_path} has invalid integer columns") from exc
        if not np.array_equal(np.unique(labels), np.arange(NUM_CLASSES, dtype=np.int64)):
            raise ValueError(f"{manifest_path} must contain every new_label 0..499")
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        if counts.shape != (NUM_CLASSES,) or not np.all(
            counts == EXPECTED_PER_CLASS[split]
        ):
            raise ValueError(
                f"{manifest_path} must contain exactly {EXPECTED_PER_CLASS[split]} "
                "records per class"
            )
        if set(frame["split"].astype(str)) != {split}:
            raise ValueError(f"{manifest_path} has an incorrect split column")
        if set(frame["source_split"].astype(str)) != {EXPECTED_SOURCE_SPLIT[split]}:
            raise ValueError(f"{manifest_path} has an incorrect source_split column")
        for column in ("image_id", "image_path", "local_image_path"):
            if frame[column].duplicated().any():
                raise ValueError(f"{manifest_path} contains duplicate {column} values")
        filenames = frame["image_path"].astype(str).map(lambda value: Path(value).name)
        if filenames.duplicated().any():
            raise ValueError(f"{manifest_path} contains duplicate image UUID filenames")
        if len(np.unique(image_ids)) != expected_rows:
            raise ValueError(f"{manifest_path} contains duplicate image IDs")
        resolved: list[Path] = []
        for value in frame["local_image_path"].astype(str):
            if not is_safe_relative_posix_path(value):
                raise ValueError(f"Unsafe local_image_path in {manifest_path}: {value!r}")
            candidate = (image_root / value).resolve()
            try:
                candidate.relative_to(image_root)
            except ValueError as exc:
                raise ValueError(f"Image path escapes image root: {value!r}") from exc
            resolved.append(candidate)
        frames[split] = frame
        paths[split] = resolved
        manifest_hashes[split] = sha256_file(manifest_path)

    for left_index, left in enumerate(requested):
        for right in requested[left_index + 1 :]:
            for column in ("image_id", "image_path", "local_image_path"):
                overlap = set(frames[left][column]).intersection(frames[right][column])
                if overlap:
                    raise ValueError(
                        f"Cross-split duplicate {column} between {left} and {right}"
                    )
            left_names = {Path(value).name for value in frames[left]["image_path"].astype(str)}
            right_names = {Path(value).name for value in frames[right]["image_path"].astype(str)}
            if left_names.intersection(right_names):
                raise ValueError(
                    f"Cross-split duplicate image UUID filename between {left} and {right}"
                )
    stage1 = _load_stage1_provenance(split_dir, manifest_hashes)
    return frames, paths, stage1


def _progress(description: str):
    return lambda values: tqdm(values, desc=description, unit="image")


def encode_split(
    paths: Sequence[Path],
    vocabulary: Any,
    *,
    config: BoVWConfig,
    description: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode one split in manifest order, failing on any decode error."""

    detector = create_sift(config.sift_nfeatures)
    features = np.empty((len(paths), config.feature_dim), dtype=np.float32)
    failures: list[dict[str, Any]] = []
    zero_descriptor_images = 0
    start = time.perf_counter()
    for index, path in enumerate(tqdm(paths, desc=description, unit="image")):
        try:
            image = load_grayscale(path, config.max_side)
            descriptors = extract_sift_result(
                image, nfeatures=config.sift_nfeatures, sift=detector
            )
            if descriptors.is_empty:
                zero_descriptor_images += 1
            features[index] = encode_descriptors(descriptors, vocabulary)
        except Exception as exc:
            failures.append({"index": index, "path": path.as_posix(), "error": str(exc)})
    elapsed = float(time.perf_counter() - start)
    if failures:
        first = failures[0]
        raise ValueError(
            f"{len(failures)} images failed during {description}; first: "
            f"{first['path']}: {first['error']}"
        )
    return features, {
        "rows": len(paths),
        "zero_descriptor_images": zero_descriptor_images,
        "encoding_seconds": elapsed,
    }


def _npz_arrays(
    features: np.ndarray,
    frame: pd.DataFrame,
    *,
    manifest_path: Path,
) -> dict[str, np.ndarray]:
    return {
        "features": np.asarray(features, dtype=np.float32),
        "new_labels": frame["new_label"].to_numpy(dtype=np.int64),
        "image_ids": frame["image_id"].to_numpy(dtype=np.int64),
        "splits": frame["split"].astype(str).to_numpy(dtype=str),
        "source_splits": frame["source_split"].astype(str).to_numpy(dtype=str),
        "image_paths": frame["image_path"].astype(str).to_numpy(dtype=str),
        "local_image_paths": frame["local_image_path"].astype(str).to_numpy(dtype=str),
        "original_category_ids": frame["original_category_id"].to_numpy(
            dtype=np.int64
        ),
        "scientific_names": frame["scientific_name"].astype(str).to_numpy(dtype=str),
        "common_names": frame["common_name"].astype(str).to_numpy(dtype=str),
        "manifest_sha256": np.asarray(sha256_file(manifest_path)),
        "manifest_file_name": np.asarray(manifest_path.name),
    }


def build_split_artifact_metadata(
    manifest_path: Path,
    feature_path: Path,
    *,
    rows: int,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the evaluator-compatible artifact record plus legacy aliases."""

    manifest_hash = sha256_file(manifest_path)
    output_hash = sha256_file(feature_path)
    output_bytes = feature_path.stat().st_size
    return {
        "manifest": {
            "file_name": manifest_path.name,
            "sha256": manifest_hash,
            "rows": int(rows),
        },
        "feature_file": {
            "file_name": feature_path.name,
            "sha256": output_hash,
            "bytes": output_bytes,
        },
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest_hash,
        "rows": int(rows),
        "output_file": feature_path.name,
        "output_sha256": output_hash,
        "output_bytes": output_bytes,
        **dict(statistics),
    }


def _validate_bovw_npz(
    path: Path,
    frame: pd.DataFrame,
    *,
    split: str,
    manifest_sha256: str,
    feature_dim: int,
) -> bool:
    """Validate shape, order, labels, and manifest binding without pickle."""

    try:
        with np.load(path, allow_pickle=False) as cache:
            required = {
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
            }
            if set(cache.files) != required:
                return False
            if cache["features"].dtype != np.float32 or cache["features"].shape != (
                len(frame),
                feature_dim,
            ):
                return False
            if not np.isfinite(cache["features"]).all():
                return False
            comparisons = {
                "new_labels": frame["new_label"].to_numpy(dtype=np.int64),
                "image_ids": frame["image_id"].to_numpy(dtype=np.int64),
                "splits": frame["split"].astype(str).to_numpy(dtype=str),
                "source_splits": frame["source_split"].astype(str).to_numpy(dtype=str),
                "image_paths": frame["image_path"].astype(str).to_numpy(dtype=str),
                "local_image_paths": frame["local_image_path"].astype(str).to_numpy(dtype=str),
                "original_category_ids": frame["original_category_id"].to_numpy(dtype=np.int64),
                "scientific_names": frame["scientific_name"].astype(str).to_numpy(dtype=str),
                "common_names": frame["common_name"].astype(str).to_numpy(dtype=str),
            }
            if any(
                not np.array_equal(cache[name], expected)
                for name, expected in comparisons.items()
            ):
                return False
            if str(np.asarray(cache["manifest_sha256"]).item()) != manifest_sha256:
                return False
            if str(np.asarray(cache["manifest_file_name"]).item()) != f"{split}.csv":
                return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def _validate_completed_cache(
    config_path: Path,
    status_path: Path,
    desired: BoVWConfig,
    split_dir: Path,
) -> bool:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("schema_version") != BOVW_CACHE_SCHEMA_VERSION:
            return False
        if config.get("configuration") != asdict(desired):
            return False
        stage1 = config["stage1"]
        summary_path = split_dir / stage1["split_summary_file_name"]
        if (
            not summary_path.is_file()
            or sha256_file(summary_path) != stage1["split_summary_sha256"]
        ):
            return False
        for name, expected_hash in stage1["frozen_artifact_hashes"].items():
            frozen_path = split_dir / name
            if not frozen_path.is_file() or sha256_file(frozen_path) != expected_hash:
                return False
        artifacts = config["split_artifacts"]
        if set(artifacts) != set(PRE_SELECTION_SPLITS):
            return False
        for split in PRE_SELECTION_SPLITS:
            entry = artifacts[split]
            manifest = entry["manifest"]
            feature_file = entry["feature_file"]
            manifest_path = split_dir / f"{split}.csv"
            if (
                not manifest_path.is_file()
                or manifest["file_name"] != manifest_path.name
                or manifest["rows"] != EXPECTED_ROWS[split]
                or sha256_file(manifest_path) != manifest["sha256"]
            ):
                return False
            path = config_path.parent / feature_file["file_name"]
            if (
                not path.is_file()
                or path.stat().st_size != feature_file["bytes"]
                or sha256_file(path) != feature_file["sha256"]
            ):
                return False
            frame = pd.read_csv(manifest_path, keep_default_na=False)
            if not _validate_bovw_npz(
                path,
                frame,
                split=split,
                manifest_sha256=manifest["sha256"],
                feature_dim=desired.feature_dim,
            ):
                return False
        vocabulary = config_path.parent / config["vocabulary"]["output_file"]
        if not vocabulary.is_file() or sha256_file(vocabulary) != config["vocabulary"][
            "output_sha256"
        ]:
            return False
        vocabulary_bundle = joblib.load(vocabulary)
        if (
            not isinstance(vocabulary_bundle, dict)
            or vocabulary_bundle.get("configuration") != asdict(desired)
            or vocabulary_bundle.get("training_source") != "train.csv only"
            or vocabulary_bundle.get("training_manifest_sha256")
            != config["split_artifacts"]["train"]["manifest"]["sha256"]
        ):
            return False
        fitted = vocabulary_bundle.get("vocabulary")
        if (
            getattr(fitted, "n_clusters", None) != desired.vocabulary_size
            or not np.isfinite(float(getattr(fitted, "sigma", np.nan)))
            or float(getattr(fitted, "sigma", 0.0)) <= 0
        ):
            return False
        gate = config["pilot_gate"]
        gate_path = config_path.parent / gate["file_name"]
        if (
            gate["status"] != "PASS"
            or not gate_path.is_file()
            or sha256_file(gate_path) != gate["sha256"]
        ):
            return False
        status = json.loads(status_path.read_text(encoding="utf-8"))
        return bool(
            status.get("status") == "COMPLETE"
            and status.get("feature_config_sha256") == sha256_file(config_path)
            and status.get("pilot_gate_sha256") == gate["sha256"]
            and status.get("configuration") == asdict(desired)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _status_document(
    status: str,
    config: BoVWConfig,
    *,
    gate_path: Path | None = None,
    feature_config_path: Path | None = None,
    error: BaseException | None = None,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "configuration": asdict(config),
    }
    if gate_path is not None and gate_path.is_file():
        document.update(
            {
                "pilot_gate_file": gate_path.name,
                "pilot_gate_sha256": sha256_file(gate_path),
            }
        )
    if feature_config_path is not None and feature_config_path.is_file():
        document.update(
            {
                "feature_config_file": feature_config_path.name,
                "feature_config_sha256": sha256_file(feature_config_path),
            }
        )
    if error is not None:
        document.update(
            {"error_type": type(error).__name__, "error": str(error)}
        )
    if arguments is not None:
        document["arguments"] = dict(arguments)
    return document


def _remove_forced_artifacts(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.is_dir():
            raise IsADirectoryError(f"Expected an artifact file, found directory: {path}")
    for path in paths:
        path.unlink(missing_ok=True)


def _extract_test_bovw(args: argparse.Namespace, config: BoVWConfig) -> tuple[str, dict[str, Any]]:
    """Encode test only after validating every immutable pre-selection binding."""

    selection_path = Path(
        getattr(args, "selection_manifest", DEFAULT_SELECTION_MANIFEST)
    ).resolve()
    try:
        from src.traditional.evaluate_traditional import _prevalidate_selection

        selection, selection_hash, _ = _prevalidate_selection(
            selection_path,
            model_path=None,
            split_dir=Path(args.split_dir),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "A valid Stage 2 selection manifest is required before any test "
            f"manifest/cache/image access: {error}"
        ) from error

    output_dir = Path(args.output_dir)
    base_config_path = output_dir / CONFIG_NAME
    if not base_config_path.is_file():
        raise ValueError(f"Pre-selection BoVW config is missing: {base_config_path}")
    base_hash = sha256_file(base_config_path)
    models = selection.get("models")
    selected_names = selection.get("selected_models")
    if (
        not isinstance(models, dict)
        or not isinstance(selected_names, list)
        or "sift_bovw_svm" not in selected_names
        or not isinstance(models.get("sift_bovw_svm"), dict)
        or models["sift_bovw_svm"].get("feature_config_sha256") != base_hash
    ):
        raise ValueError(
            "Selection manifest does not bind the current sift_bovw_svm config"
        )
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    if base_config.get("configuration") != asdict(config):
        raise ValueError("Current BoVW CLI configuration differs from the frozen config")
    vocabulary_entry = base_config.get("vocabulary")
    if not isinstance(vocabulary_entry, dict):
        raise ValueError("Frozen BoVW config has no vocabulary binding")
    vocabulary_path = output_dir / str(vocabulary_entry.get("output_file", ""))
    if (
        not vocabulary_path.is_file()
        or sha256_file(vocabulary_path) != vocabulary_entry.get("output_sha256")
    ):
        raise ValueError("Frozen BoVW vocabulary hash does not match")
    bundle = joblib.load(vocabulary_path)
    if (
        not isinstance(bundle, dict)
        or bundle.get("configuration") != asdict(config)
        or bundle.get("training_source") != "train.csv only"
    ):
        raise ValueError("BoVW vocabulary is not bound to the train-only configuration")
    vocabulary = bundle.get("vocabulary")
    if getattr(vocabulary, "n_clusters", None) != config.vocabulary_size:
        raise ValueError("BoVW vocabulary size does not match the frozen configuration")

    # Only now may the test manifest and image paths be resolved.
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_dir)
    frames, paths, stage1 = load_manifests(
        split_dir, Path(args.image_root), splits=("test",)
    )
    destination = output_dir / "bovw_test_features.npz"
    test_config_path = output_dir / TEST_CONFIG_NAME
    status_path = output_dir / "bovw_test_status.json"
    existing = [path for path in (destination, test_config_path, status_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Existing test BoVW artifacts are immutable without --force: "
            + ", ".join(str(path) for path in existing)
        )
    if args.force:
        _remove_forced_artifacts(existing)

    vocabulary_hash = sha256_file(vocabulary_path)
    encoded, statistics = encode_split(
        paths["test"], vocabulary, config=config, description="BoVW encode test"
    )
    if (
        sha256_file(selection_path) != selection_hash
        or sha256_file(base_config_path) != base_hash
        or sha256_file(vocabulary_path) != vocabulary_hash
    ):
        raise ValueError("A frozen Stage 2 binding changed during test extraction")
    manifest_path = split_dir / "test.csv"
    _atomic_npz(
        _npz_arrays(encoded, frames["test"], manifest_path=manifest_path), destination
    )
    artifact = build_split_artifact_metadata(
        manifest_path,
        destination,
        rows=len(frames["test"]),
        statistics=statistics,
    )
    document = {
        "schema_version": BOVW_CACHE_SCHEMA_VERSION,
        **bovw_configuration(config),
        "training_source": "train.csv only",
        "stage1": stage1,
        "method_config": {
            "path": base_config_path.resolve().as_posix(),
            "sha256": base_hash,
        },
        "selection_manifest": {
            "path": selection_path.as_posix(),
            "sha256": selection_hash,
        },
        "vocabulary": {
            "output_file": vocabulary_path.name,
            "output_sha256": vocabulary_hash,
            "training_manifest_sha256": bundle.get("training_manifest_sha256"),
        },
        "split_artifacts": {"test": artifact},
        "artifacts": {
            "test": {
                "path": destination.name,
                "sha256": artifact["feature_file"]["sha256"],
                "bytes": artifact["feature_file"]["bytes"],
                "rows": len(frames["test"]),
            }
        },
    }
    _atomic_json(document, test_config_path)
    status = {
        "schema_version": 1,
        "status": "COMPLETE",
        "configuration": asdict(config),
        "test_feature_config_file": test_config_path.name,
        "test_feature_config_sha256": sha256_file(test_config_path),
        "selection_manifest_sha256": selection_hash,
    }
    _atomic_json(status, status_path)
    return "COMPLETE", document


def extract_bovw(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    config = BoVWConfig(
        max_side=args.max_side,
        sift_nfeatures=args.sift_nfeatures,
        vocabulary_size=args.vocabulary_size,
        vocabulary_images_per_class=args.vocabulary_images_per_class,
        pilot_images_per_class=args.pilot_images_per_class,
        vocabulary_descriptors_per_image=args.descriptors_per_image,
        batch_size=args.batch_size,
        n_init=args.n_init,
        soft_assign_k=getattr(args, "soft_assign_k", 3),
        pyramid_grid=getattr(args, "pyramid_grid", 2),
        use_spatial_pyramid=not getattr(args, "no_spatial_pyramid", False),
        seed=args.seed,
    )
    config.validate()
    requested = tuple(
        part.strip().lower()
        for part in str(getattr(args, "splits", "train,val")).split(",")
        if part.strip()
    )
    if requested == ("test",):
        return _extract_test_bovw(args, config)
    if requested != PRE_SELECTION_SPLITS:
        raise ValueError("pre-selection BoVW extraction requires --splits train,val")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = Path(args.split_dir)
    config_path = output_dir / CONFIG_NAME
    vocabulary_path = output_dir / "bovw_vocabulary.joblib"
    gate_path = output_dir / GATE_NAME
    status_path = output_dir / STATUS_NAME
    legacy_failure_path = output_dir / LEGACY_FAILURE_NAME
    output_paths = {
        split: output_dir / f"bovw_{split}_features.npz" for split in requested
    }
    if (
        config_path.exists()
        and status_path.exists()
        and _validate_completed_cache(config_path, status_path, config, split_dir)
        and not args.force
    ):
        return "SKIP_VALID_CACHE", json.loads(config_path.read_text(encoding="utf-8"))
    partial = [
        path
        for path in (
            vocabulary_path,
            *output_paths.values(),
            config_path,
            gate_path,
            status_path,
            legacy_failure_path,
        )
        if path.exists()
    ]
    if partial and not args.force:
        raise FileExistsError(
            "Stale or partial BoVW cache exists; inspect it and rerun with --force: "
            + ", ".join(str(path) for path in partial)
        )
    if args.force:
        _remove_forced_artifacts(partial)

    frames, paths, stage1 = load_manifests(
        split_dir, Path(args.image_root), splits=requested
    )
    train_labels = frames["train"]["new_label"].to_numpy(dtype=np.int64)
    gate, _pilot_descriptors = make_pilot_gate(
        paths["train"],
        train_labels,
        config=config,
        output_dir=output_dir,
        total_encoding_images=sum(EXPECTED_ROWS[split] for split in requested),
        progress=_progress("BoVW pilot"),
    )
    gate.update(
        {
            "train_manifest_file": "train.csv",
            "train_manifest_sha256": sha256_file(split_dir / "train.csv"),
        }
    )
    _atomic_json(gate, gate_path)
    if args.pilot_only:
        status = _status_document("PILOT ONLY", config, gate_path=gate_path)
        status["pilot_gate_status"] = gate["status"]
        _atomic_json(status, status_path)
        return "PILOT ONLY", status
    if not gate["passed"]:
        status = _status_document("NOT RUN", config, gate_path=gate_path)
        status["reason"] = "SIFT-BoVW feasibility gate did not pass"
        status["pilot_gate_status"] = gate["status"]
        _atomic_json(status, status_path)
        return "NOT RUN", status

    running_status = _status_document("RUNNING", config, gate_path=gate_path)
    _atomic_json(running_status, status_path)

    vocabulary_indices = deterministic_balanced_indices(
        train_labels,
        config.vocabulary_images_per_class,
        seed=config.seed,
        source_tag=4,
    )
    vocabulary_descriptors, vocabulary_stats = run_descriptor_pass(
        paths["train"],
        train_labels,
        vocabulary_indices,
        config=config,
        descriptors_per_image=config.vocabulary_descriptors_per_image,
        progress=_progress("BoVW vocabulary descriptors"),
    )
    if vocabulary_stats["decode_failures"]:
        first = vocabulary_stats["decode_failures"][0]
        raise ValueError(
            f"Vocabulary pass had {len(vocabulary_stats['decode_failures'])} decode "
            f"failure(s); first: {first['path']}: {first['error']}"
        )
    if vocabulary_stats["classes_with_descriptors"] != 500:
        raise ValueError(
            "Vocabulary descriptors did not cover every training class: "
            f"{vocabulary_stats['classes_with_descriptors']}/500"
        )
    start = time.perf_counter()
    vocabulary = fit_vocabulary(
        vocabulary_descriptors,
        vocabulary_size=config.vocabulary_size,
        batch_size=config.batch_size,
        n_init=config.n_init,
        seed=config.seed,
        config=config,
    )
    clustering_seconds = float(time.perf_counter() - start)
    vocabulary_bundle = {
        "schema_version": 2,
        "vocabulary": vocabulary,
        "configuration": asdict(config),
        "training_source": "train.csv only",
        "training_manifest_sha256": sha256_file(split_dir / "train.csv"),
        "vocabulary_indices": vocabulary_indices,
        "statistics": {**vocabulary_stats, "clustering_seconds": clustering_seconds},
    }
    _atomic_joblib(vocabulary_bundle, vocabulary_path)

    split_artifacts: dict[str, Any] = {}
    artifacts_alias: dict[str, Any] = {}
    for split in requested:
        encoded, stats = encode_split(
            paths[split], vocabulary, config=config, description=f"BoVW encode {split}"
        )
        manifest_path = split_dir / f"{split}.csv"
        destination = output_paths[split]
        _atomic_npz(
            _npz_arrays(encoded, frames[split], manifest_path=manifest_path), destination
        )
        split_artifacts[split] = build_split_artifact_metadata(
            manifest_path,
            destination,
            rows=len(frames[split]),
            statistics=stats,
        )
        feature_file = split_artifacts[split]["feature_file"]
        artifacts_alias[split] = {
            "path": destination.name,
            "sha256": feature_file["sha256"],
            "bytes": feature_file["bytes"],
            "rows": int(len(frames[split])),
        }

    feature_config = {
        "schema_version": BOVW_CACHE_SCHEMA_VERSION,
        **bovw_configuration(config),
        "training_source": "train.csv only",
        "stage1": stage1,
        "pilot_gate_file": gate_path.name,
        "pilot_gate_status": gate["status"],
        "pilot_gate": {
            "file_name": gate_path.name,
            "sha256": sha256_file(gate_path),
            "status": gate["status"],
        },
        "vocabulary": {
            "output_file": vocabulary_path.name,
            "output_sha256": sha256_file(vocabulary_path),
            "descriptor_count": vocabulary_stats["descriptor_count"],
            "classes_with_descriptors": vocabulary_stats["classes_with_descriptors"],
            "clustering_seconds": clustering_seconds,
            "sigma": vocabulary.sigma,
            "training_manifest_sha256": sha256_file(split_dir / "train.csv"),
        },
        "split_artifacts": split_artifacts,
        "artifacts": artifacts_alias,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "opencv": cv2.__version__,
            "joblib": joblib.__version__,
        },
    }
    _atomic_json(feature_config, config_path)
    status = _status_document(
        "COMPLETE",
        config,
        gate_path=gate_path,
        feature_config_path=config_path,
    )
    _atomic_json(status, status_path)
    return "COMPLETE", feature_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the train-only SIFT-BoVW feasibility gate and, if it passes, "
            "extract cached BoVW features."
        )
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument(
        "--image-root", type=Path, default=Path("data/processed/images")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/features/bovw")
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="train,val before selection; test alone after selection",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
        help="immutable selection manifest required before --splits test",
    )
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--sift-nfeatures", type=int, default=300)
    parser.add_argument("--vocabulary-size", type=int, default=256)
    parser.add_argument("--vocabulary-images-per-class", type=int, default=4)
    parser.add_argument("--pilot-images-per-class", type=int, default=2)
    parser.add_argument("--descriptors-per-image", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--soft-assign-k", type=int, default=3)
    parser.add_argument("--pyramid-grid", type=int, default=2)
    parser.add_argument(
        "--no-spatial-pyramid",
        action="store_true",
        help="validation-only ablation: encode only the global histogram",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Write the feasibility report but do not start the full extraction.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace a stale or completed cache."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        status, report = extract_bovw(args)
    except (ValueError, FileExistsError, OSError, RuntimeError) as exc:
        config = BoVWConfig(
            max_side=args.max_side,
            sift_nfeatures=args.sift_nfeatures,
            vocabulary_size=args.vocabulary_size,
            vocabulary_images_per_class=args.vocabulary_images_per_class,
            pilot_images_per_class=args.pilot_images_per_class,
            vocabulary_descriptors_per_image=args.descriptors_per_image,
            batch_size=args.batch_size,
            n_init=args.n_init,
            soft_assign_k=args.soft_assign_k,
            pyramid_grid=args.pyramid_grid,
            use_spatial_pyramid=not args.no_spatial_pyramid,
            seed=args.seed,
        )
        arguments = {
            key: (value.as_posix() if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        output_dir = Path(args.output_dir)
        status_path = output_dir / STATUS_NAME
        gate_path = output_dir / GATE_NAME
        failure = _status_document(
            "FAILED",
            config,
            gate_path=gate_path,
            error=exc,
            arguments=arguments,
        )
        try:
            # A refusal caused by existing artifacts must not overwrite the
            # canonical status of the previous run.  --force removes that
            # status before new work starts, so failures during a forced retry
            # are recorded normally.
            replace_running = False
            if status_path.is_file():
                try:
                    existing_status = json.loads(status_path.read_text(encoding="utf-8"))
                    replace_running = bool(
                        existing_status.get("status") == "RUNNING"
                        and existing_status.get("configuration") == asdict(config)
                    )
                except (OSError, json.JSONDecodeError, AttributeError):
                    replace_running = False
            if args.force or not status_path.exists() or replace_running:
                _atomic_json(failure, status_path)
        except OSError:
            pass
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"BoVW status: {status}")
    if "passed" in report:
        print(f"pilot_gate_passed: {report['passed']}")
    elif "pilot_gate_status" in report:
        print(f"pilot_gate_status: {report['pilot_gate_status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
