"""Create and verify the immutable Stage 4 robustness protocol.

The protocol binds the already-selected Stage 2/3 models, their clean test
evidence, the accepted Stage 1 split, and the complete 64-run corruption
matrix.  It never loads image pixels or trains a model.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from src.data.common import deterministic_json_bytes, sha256_file
from src.deep_learning.train_utils import STAGE1_EXPECTED_HASHES
from src.robustness.corruptions import DEFAULT_SEED, protocol_corruption_matrix


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_PROTOCOL = Path("results/metrics/stage4_robustness_protocol.json")
CANONICAL_METRICS_DIR = Path("results/metrics/robustness")
CANONICAL_PREDICTIONS_DIR = Path("results/predictions/robustness")
EXPECTED_MODEL_NAMES = (
    "traditional_random_forest",
    "resnet18_scratch_basic",
    "resnet18_pretrained_strong_aug",
    "efficientnet_b0_pretrained_strong_aug",
)
EXPECTED_EVIDENCE_HASHES = {
    "results/metrics/stage2_freeze.json": (
        "f5b908ce68b0cfb95b3047c2ba5a875fb975e56026b6337c966d5c326140562d"
    ),
    "results/tables/stage2_traditional_results.csv": (
        "b2a2d554975c06e28639a39d44b80e120e36dc5eb8e23bc0d75715a8ac15768c"
    ),
    "results/tables/stage2_feature_ablation_results.csv": (
        "ba723a0a362bd50a569a12bbcb6f56578ed37f1a6f351daed793c74ad692e2ce"
    ),
    "results/metrics/deep_learning/stage3_freeze.json": (
        "bea68b307fb7ff5db4f066b47ff00ade462bd71609ed9735c4faff125ce2257f"
    ),
    "results/metrics/deep_learning/stage3_test_access.json": (
        "04420ab5eb9bca3f5a6e56f6015fd2ce158ba9121d0e328b076912bac2292f2b"
    ),
    "results/tables/stage3_deep_learning_results.csv": (
        "5f5434e5a571f9e5f033594cdb25031483ccd6f2d3efe8e8166bdbc5038a911d"
    ),
}
METRIC_NAMES = (
    "top1_accuracy",
    "top5_accuracy",
    "overall_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "full_evaluation_wall_seconds",
    "model_forward_seconds",
    "num_samples",
)


class ProtocolError(RuntimeError):
    """Raised when frozen robustness provenance is absent or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _portable(root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ProtocolError(f"Artifact lies outside project root: {resolved}") from error


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"Could not read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON artifact must contain an object: {path}")
    return value


def _artifact(root: Path, relative: str | Path, *, expected: str | None = None) -> dict[str, Any]:
    path = _resolve(root, relative)
    if not path.is_file():
        raise ProtocolError(f"Required frozen artifact is missing: {path}")
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ProtocolError(
            f"Frozen artifact changed: {_portable(root, path)}; expected {expected}, got {digest}"
        )
    return {
        "path": _portable(root, path),
        "bytes": int(path.stat().st_size),
        "sha256": digest,
    }


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(deterministic_json_bytes(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_deep_model(
    root: Path,
    freeze: Mapping[str, Any],
    test_access: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    model_entry = freeze.get("models", {}).get(name)
    if not isinstance(model_entry, Mapping):
        raise ProtocolError(f"Stage 3 freeze has no selected model {name!r}")
    if model_entry.get("experiment_name") != name:
        raise ProtocolError(f"Stage 3 freeze experiment mismatch for {name}")
    if model_entry.get("selected_for_test") is False:
        raise ProtocolError(f"Stage 3 model was not selected for test: {name}")

    access_models = test_access.get("models")
    if not isinstance(access_models, Mapping) or name not in access_models:
        # Stage 3 access schema also stores selected models under attempts.
        access_models = test_access.get("attempts")
    if not isinstance(access_models, Mapping) or name not in access_models:
        raise ProtocolError(f"Stage 3 completed test access does not bind {name}")

    checkpoint_info = model_entry.get("checkpoint")
    resolved_info = model_entry.get("resolved_config")
    source_info = model_entry.get("source_config")
    for label, entry in (
        ("checkpoint", checkpoint_info),
        ("resolved config", resolved_info),
        ("source config", source_info),
    ):
        if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
            raise ProtocolError(f"Stage 3 freeze has invalid {label} for {name}")

    metrics_path = Path(f"results/metrics/deep_learning/{name}_test_metrics.json")
    predictions_path = Path(f"results/predictions/deep_learning/{name}_test_predictions.csv")
    metrics = _read_json(_resolve(root, metrics_path))
    if metrics.get("status") != "complete" or metrics.get("split") != "test":
        raise ProtocolError(f"Clean Stage 3 test metrics are incomplete for {name}")
    artifacts = metrics.get("artifacts")
    prediction_binding = artifacts.get("predictions") if isinstance(artifacts, Mapping) else None
    if not isinstance(prediction_binding, Mapping):
        raise ProtocolError(f"Clean Stage 3 metrics do not bind predictions for {name}")

    checkpoint = _artifact(root, str(checkpoint_info["path"]), expected=str(checkpoint_info["sha256"]))
    resolved_config = _artifact(root, str(resolved_info["path"]), expected=str(resolved_info["sha256"]))
    source_config = _artifact(root, str(source_info["path"]), expected=str(source_info["sha256"]))
    clean_predictions = _artifact(
        root, predictions_path, expected=str(prediction_binding.get("sha256"))
    )
    if clean_predictions["path"] != str(prediction_binding.get("path")):
        raise ProtocolError(f"Clean prediction path mismatch for {name}")
    return {
        "method_family": "deep_learning",
        "architecture": str(model_entry.get("model_name")),
        "pretrained": bool(model_entry.get("pretrained")),
        "artifact": checkpoint,
        "resolved_config": resolved_config,
        "source_config": source_config,
        "clean_metrics": _artifact(root, metrics_path),
        "clean_predictions": clean_predictions,
    }


def _verify_random_forest(root: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    frozen_model = freeze.get("models", {}).get("random_forest")
    if not isinstance(frozen_model, Mapping):
        raise ProtocolError("Stage 2 freeze has no random_forest model")
    model_path = str(frozen_model.get("path", ""))
    model_hash = str(frozen_model.get("sha256", ""))
    config_hash = str(frozen_model.get("feature_config_sha256", ""))
    if not model_path or not model_hash or not config_hash:
        raise ProtocolError("Stage 2 random_forest freeze entry is incomplete")
    metadata_path = Path("results/models/traditional/random_forest.metadata.json")
    metadata = _read_json(_resolve(root, metadata_path))
    if metadata.get("model_sha256") != model_hash:
        raise ProtocolError("Random Forest metadata and freeze model hashes disagree")
    config_path = Path(str(metadata.get("feature_config_path", "")))
    if not config_path.as_posix() or metadata.get("feature_config_sha256") != config_hash:
        raise ProtocolError("Random Forest feature-config provenance is inconsistent")

    metrics_path = Path("results/metrics/traditional_random_forest_metrics.json")
    predictions_path = Path("results/predictions/traditional_random_forest_test_predictions.csv")
    metrics = _read_json(_resolve(root, metrics_path))
    if metrics.get("status") != "complete" or not isinstance(metrics.get("test"), Mapping):
        raise ProtocolError("Clean Random Forest test metrics are incomplete")
    provenance = metrics.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ProtocolError("Clean Random Forest metrics lack provenance")
    if provenance.get("model_sha256") != model_hash:
        raise ProtocolError("Clean Random Forest metrics use a different model")
    if provenance.get("split_manifest_sha256") != STAGE1_EXPECTED_HASHES["test.csv"]:
        raise ProtocolError("Clean Random Forest metrics use a different test split")

    return {
        "method_family": "traditional",
        "architecture": "random_forest",
        "pretrained": False,
        "artifact": _artifact(root, model_path, expected=model_hash),
        "model_metadata": _artifact(root, metadata_path),
        "feature_config": _artifact(root, config_path, expected=config_hash),
        "clean_metrics": _artifact(root, metrics_path),
        "clean_predictions": _artifact(root, predictions_path),
    }


def build_protocol_document(
    project_root: Path = PROJECT_ROOT,
    *,
    enforce_accepted_hashes: bool = True,
) -> dict[str, Any]:
    """Build and fully verify a protocol document without writing it."""

    root = Path(project_root).resolve()
    stage1: dict[str, Any] = {}
    for name, expected in STAGE1_EXPECTED_HASHES.items():
        stage1[name] = _artifact(root, Path("data/splits") / name, expected=expected)

    evidence: dict[str, dict[str, Any]] = {}
    for relative, accepted in EXPECTED_EVIDENCE_HASHES.items():
        evidence[relative] = _artifact(
            root, relative, expected=accepted if enforce_accepted_hashes else None
        )
    stage2_freeze = _read_json(_resolve(root, "results/metrics/stage2_freeze.json"))
    if stage2_freeze.get("status") != "frozen":
        raise ProtocolError("Stage 2 freeze status is not frozen")
    stage3_freeze = _read_json(
        _resolve(root, "results/metrics/deep_learning/stage3_freeze.json")
    )
    stage3_access = _read_json(
        _resolve(root, "results/metrics/deep_learning/stage3_test_access.json")
    )
    if stage3_freeze.get("status") != "frozen":
        raise ProtocolError("Stage 3 freeze status is not frozen")
    if not (
        stage3_access.get("status") == "completed"
        and stage3_access.get("outcome") == "complete"
    ):
        raise ProtocolError("Stage 3 test-access batch is not complete")

    models = {
        "traditional_random_forest": _verify_random_forest(root, stage2_freeze),
        "resnet18_scratch_basic": _verify_deep_model(
            root, stage3_freeze, stage3_access, "resnet18_scratch_basic"
        ),
        "resnet18_pretrained_strong_aug": _verify_deep_model(
            root, stage3_freeze, stage3_access, "resnet18_pretrained_strong_aug"
        ),
        "efficientnet_b0_pretrained_strong_aug": _verify_deep_model(
            root, stage3_freeze, stage3_access, "efficientnet_b0_pretrained_strong_aug"
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "stage4_robustness",
        "status": "frozen",
        "created_at_utc": _utc_now(),
        "seed": DEFAULT_SEED,
        "split": "test",
        "expected_num_images_per_run": 5000,
        "expected_run_count": len(models) * 4 * 4,
        "policy": {
            "corruptions_applied_on_the_fly": True,
            "corrupted_images_persisted": False,
            "model_selection_locked_before_robustness": True,
            "no_retraining_or_weight_updates": True,
            "no_test_tuning": True,
            "image_order": "unchanged Stage 1 test.csv manifest order",
        },
        "metrics": list(METRIC_NAMES),
        "corruptions": protocol_corruption_matrix(),
        "model_order": list(EXPECTED_MODEL_NAMES),
        "stage1_artifacts": stage1,
        "stage2_evidence": {
            key: value for key, value in evidence.items() if "stage2" in key
        },
        "stage3_evidence": {
            key: value for key, value in evidence.items() if "stage3" in key
        },
        "models": models,
    }


def _iter_artifact_entries(value: Any, prefix: str = "root") -> Iterator[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            yield prefix, value
            return
        for key, child in value.items():
            yield from _iter_artifact_entries(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_artifact_entries(child, f"{prefix}[{index}]")


def validate_protocol_document(
    document: Mapping[str, Any], project_root: Path = PROJECT_ROOT
) -> None:
    """Validate schema, fixed matrix, model set, and every bound artifact byte."""

    if document.get("schema_version") != SCHEMA_VERSION:
        raise ProtocolError("Unsupported Stage 4 protocol schema version")
    if document.get("stage") != "stage4_robustness" or document.get("status") != "frozen":
        raise ProtocolError("Stage 4 protocol is not frozen")
    if document.get("seed") != DEFAULT_SEED or document.get("split") != "test":
        raise ProtocolError("Stage 4 protocol seed/split differs from the fixed contract")
    if document.get("corruptions") != protocol_corruption_matrix():
        raise ProtocolError("Stage 4 corruption matrix differs from the fixed contract")
    models = document.get("models")
    if (
        not isinstance(models, Mapping)
        or set(models) != set(EXPECTED_MODEL_NAMES)
        or document.get("model_order") != list(EXPECTED_MODEL_NAMES)
    ):
        raise ProtocolError(
            f"Stage 4 protocol model order/set must be {list(EXPECTED_MODEL_NAMES)}"
        )
    if document.get("expected_run_count") != 64:
        raise ProtocolError("Stage 4 protocol must declare exactly 64 full runs")
    policy = document.get("policy")
    required_true = (
        "corruptions_applied_on_the_fly",
        "model_selection_locked_before_robustness",
        "no_retraining_or_weight_updates",
        "no_test_tuning",
    )
    if not isinstance(policy, Mapping) or not all(policy.get(key) is True for key in required_true):
        raise ProtocolError("Stage 4 protocol safety policy is incomplete")

    root = Path(project_root).resolve()
    found = 0
    for label, entry in _iter_artifact_entries(document):
        found += 1
        path = _resolve(root, str(entry["path"]))
        if not path.is_file():
            raise ProtocolError(f"Protocol artifact is missing ({label}): {path}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ProtocolError(f"Protocol artifact byte count changed ({label})")
        observed = sha256_file(path)
        if observed != str(entry["sha256"]):
            raise ProtocolError(
                f"Protocol artifact hash changed ({label}): expected {entry['sha256']}, got {observed}"
            )
    if found < 30:
        raise ProtocolError("Stage 4 protocol does not bind enough provenance artifacts")


def load_and_verify_protocol(
    path: Path = CANONICAL_PROTOCOL,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], str]:
    root = Path(project_root).resolve()
    resolved = _resolve(root, path)
    if not resolved.is_file():
        raise ProtocolError(f"Stage 4 robustness protocol is missing: {resolved}")
    document = _read_json(resolved)
    validate_protocol_document(document, root)
    return document, sha256_file(resolved)


def canonical_full_outputs_exist(project_root: Path = PROJECT_ROOT) -> list[Path]:
    root = Path(project_root).resolve()
    found: list[Path] = []
    for relative in (CANONICAL_METRICS_DIR, CANONICAL_PREDICTIONS_DIR):
        directory = _resolve(root, relative)
        if directory.is_dir():
            found.extend(path for path in directory.iterdir() if path.is_file() and not path.name.startswith("."))
    return sorted(found)


def freeze_protocol(
    output_path: Path = CANONICAL_PROTOCOL,
    *,
    project_root: Path = PROJECT_ROOT,
    dangerous_replace: bool = False,
    current_protocol_sha256: str | None = None,
    enforce_accepted_hashes: bool = True,
) -> tuple[dict[str, Any], str]:
    """Create the protocol, or safely return an existing valid protocol."""

    root = Path(project_root).resolve()
    output = _resolve(root, output_path)
    if output.exists():
        observed_hash = sha256_file(output)
        try:
            existing = _read_json(output)
            validate_protocol_document(existing, root)
        except ProtocolError:
            existing = None
        if existing is not None and not dangerous_replace:
            return existing, observed_hash
        if not dangerous_replace:
            raise FileExistsError(
                f"Protocol exists but is not valid/current: {output}; replacement requires --dangerous-replace"
            )
        if current_protocol_sha256 != observed_hash:
            raise ProtocolError(
                "Dangerous replacement requires --current-protocol-sha256 equal to the current file hash"
            )
        full_outputs = canonical_full_outputs_exist(root)
        if full_outputs:
            raise ProtocolError(
                "Protocol replacement is permanently forbidden after canonical full outputs exist: "
                + ", ".join(_portable(root, path) for path in full_outputs[:5])
            )
    elif dangerous_replace:
        raise ProtocolError("--dangerous-replace is invalid when no protocol exists")

    document = build_protocol_document(root, enforce_accepted_hashes=enforce_accepted_hashes)
    _atomic_json(output, document)
    validate_protocol_document(document, root)
    return document, sha256_file(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the immutable Stage 4 fixed-model robustness protocol."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=CANONICAL_PROTOCOL)
    parser.add_argument("--dangerous-replace", action="store_true")
    parser.add_argument("--current-protocol-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _, digest = freeze_protocol(
            args.output,
            project_root=args.project_root,
            dangerous_replace=args.dangerous_replace,
            current_protocol_sha256=args.current_protocol_sha256,
        )
    except (OSError, ValueError, TypeError, KeyError, ProtocolError) as error:
        print(f"[FAIL] {error}", file=os.sys.stderr)
        return 1
    print(f"[PASS] Stage 4 robustness protocol is frozen: {args.output}")
    print(f"[PASS] SHA-256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_METRICS_DIR",
    "CANONICAL_PREDICTIONS_DIR",
    "CANONICAL_PROTOCOL",
    "EXPECTED_MODEL_NAMES",
    "ProtocolError",
    "build_protocol_document",
    "canonical_full_outputs_exist",
    "freeze_protocol",
    "load_and_verify_protocol",
    "validate_protocol_document",
]
