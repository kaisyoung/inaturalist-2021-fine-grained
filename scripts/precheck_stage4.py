#!/usr/bin/env python3
"""Read-only Stage 4 readiness audit and evidence report generator."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.common import sha256_file
EXPECTED_STAGE1 = {
    "selected_classes_seed42.txt": "3c2af417a5c174ef14f2be117a00880b0a08f8dce0c24fad6aaa8ea5dcde8141",
    "class_mapping.json": "2d6e60104b133d2758dffd585d373b085ddb42845d0b4b2c93bf5277ef9a0a1b",
    "train.csv": "d8f4abccc568484f6efaff243c409b8ddcbc8fe46d2146a6922230b4428706c7",
    "val.csv": "4adf287996d35d9e57f829426ff7480bbaf0b74e1c319a29d57fe74b3f0366bb",
    "test.csv": "45ba7673b6b0e27cab8ef5a482ea8142de54e35be05a5b1ba9bb3fe20d5b8f91",
    "split_summary.json": "324413fb3d8d3affe611f8fd1454358cf54f2e6fcd81ef961994be0cc19f0b76",
}
EXPECTED_STAGE2 = {
    "results/metrics/stage2_freeze.json": "f5b908ce68b0cfb95b3047c2ba5a875fb975e56026b6337c966d5c326140562d",
    "results/tables/stage2_traditional_results.csv": "b2a2d554975c06e28639a39d44b80e120e36dc5eb8e23bc0d75715a8ac15768c",
    "results/tables/stage2_feature_ablation_results.csv": "ba723a0a362bd50a569a12bbcb6f56578ed37f1a6f351daed793c74ad692e2ce",
}
EXPECTED_STAGE3 = {
    "results/metrics/deep_learning/stage3_freeze.json": "bea68b307fb7ff5db4f066b47ff00ade462bd71609ed9735c4faff125ce2257f",
    "results/metrics/deep_learning/stage3_test_access.json": "04420ab5eb9bca3f5a6e56f6015fd2ce158ba9121d0e328b076912bac2292f2b",
    "results/tables/stage3_deep_learning_results.csv": "5f5434e5a571f9e5f033594cdb25031483ccd6f2d3efe8e8166bdbc5038a911d",
    "results/tables/stage3_ablation_results.csv": "79cf56f87fef800632adbaec8d9a4ab80686ff4f89db75dc25321cd47fc35b93",
    "results/tables/final_model_comparison.csv": "322f88ce28f9c919b09a068ee55f9c87b537e4d36c9995994386f618d3ab39d5",
}
EXPECTED_MODEL_HASHES = {
    "results/models/traditional/random_forest.joblib": (
        "346c4d48e505b71f019765d30bc2c54dc73a0e566a971cfc6ad65cf4a0946fda"
    ),
}
EXPECTED_PROTOCOL_MODEL_ARTIFACTS = {
    "traditional_random_forest": "results/models/traditional/random_forest.joblib",
    "resnet18_scratch_basic": "results/models/deep_learning/resnet18_scratch_basic_best.pt",
    "resnet18_pretrained_strong_aug": "results/models/deep_learning/resnet18_pretrained_strong_aug_best.pt",
    "efficientnet_b0_pretrained_strong_aug": "results/models/deep_learning/efficientnet_b0_pretrained_strong_aug_best.pt",
}
EXPECTED_PROTOCOL_CORRUPTIONS = {
    "gaussian_noise": {"parameter": "sigma", "severities": [0.02, 0.05, 0.1, 0.2]},
    "gaussian_blur": {"parameter": "kernel_size", "severities": [3, 5, 9, 15]},
    "motion_blur": {"parameter": "kernel_length", "severities": [5, 9, 15, 25]},
    "jpeg_compression": {"parameter": "quality", "severities": [80, 50, 30, 10]},
}
STAGE4_MODELS = {
    "random_forest": {
        "family": "traditional",
        "artifact": "results/models/traditional/random_forest.joblib",
        "sha256": EXPECTED_MODEL_HASHES[
            "results/models/traditional/random_forest.joblib"
        ],
        "validation_metrics": "results/metrics/traditional_random_forest_val_metrics.json",
        "test_metrics": "results/metrics/traditional_random_forest_metrics.json",
        "test_predictions": "results/predictions/traditional_random_forest_test_predictions.csv",
    },
    "resnet18_scratch_basic": {
        "family": "deep_learning",
        "artifact": "results/models/deep_learning/resnet18_scratch_basic_best.pt",
        "sha256": "509dd62c39597ad868b9527ae8b2fc7e42f25fcb239f067310a3ad6e0bfa8bd6",
        "validation_metrics": "results/metrics/deep_learning/resnet18_scratch_basic_val_metrics.json",
        "test_metrics": "results/metrics/deep_learning/resnet18_scratch_basic_test_metrics.json",
        "test_predictions": "results/predictions/deep_learning/resnet18_scratch_basic_test_predictions.csv",
    },
    "resnet18_pretrained_strong_aug": {
        "family": "deep_learning",
        "artifact": "results/models/deep_learning/resnet18_pretrained_strong_aug_best.pt",
        "sha256": "b8a371f0e9aa4c8b412b34884bf1a3ddfd9d2694346c23d29b6249b22ba6e4e5",
        "validation_metrics": "results/metrics/deep_learning/resnet18_pretrained_strong_aug_val_metrics.json",
        "test_metrics": "results/metrics/deep_learning/resnet18_pretrained_strong_aug_test_metrics.json",
        "test_predictions": "results/predictions/deep_learning/resnet18_pretrained_strong_aug_test_predictions.csv",
    },
    "efficientnet_b0_pretrained_strong_aug": {
        "family": "deep_learning",
        "artifact": "results/models/deep_learning/efficientnet_b0_pretrained_strong_aug_best.pt",
        "sha256": "787bc18b24b02e7671eea4197af75be8f193c969331351d8173a84aad7cd7a97",
        "validation_metrics": "results/metrics/deep_learning/efficientnet_b0_pretrained_strong_aug_val_metrics.json",
        "test_metrics": "results/metrics/deep_learning/efficientnet_b0_pretrained_strong_aug_test_metrics.json",
        "test_predictions": "results/predictions/deep_learning/efficientnet_b0_pretrained_strong_aug_test_predictions.csv",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing report without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _canonical_output_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for relative in (
        "results/metrics/robustness",
        "results/predictions/robustness",
    ):
        directory = root / relative
        if directory.is_dir():
            found.extend(
                path
                for path in directory.iterdir()
                if path.is_file() and not path.name.startswith(".")
            )
    return sorted(found)


def _replacement_protocol_context_errors(document: dict[str, Any]) -> list[str]:
    """Validate old-protocol identity without revalidating its stale artifact hashes."""

    errors: list[str] = []
    required_fields = {
        "schema_version": 1,
        "stage": "stage4_robustness",
        "status": "frozen",
        "seed": 42,
        "split": "test",
        "expected_num_images_per_run": 5000,
        "expected_run_count": 64,
    }
    for field, expected in required_fields.items():
        if document.get(field) != expected:
            errors.append(f"{field} must be {expected!r}")
    model_order = list(EXPECTED_PROTOCOL_MODEL_ARTIFACTS)
    if document.get("model_order") != model_order:
        errors.append("model_order differs from the four-model Stage 4 contract")
    models = document.get("models")
    if not isinstance(models, dict) or set(models) != set(model_order):
        errors.append("models differs from the four-model Stage 4 contract")
    else:
        for name, expected_path in EXPECTED_PROTOCOL_MODEL_ARTIFACTS.items():
            entry = models.get(name)
            artifact = entry.get("artifact") if isinstance(entry, dict) else None
            if not isinstance(artifact, dict) or artifact.get("path") != expected_path:
                errors.append(f"{name} artifact path differs from the Stage 4 contract")
    if document.get("corruptions") != EXPECTED_PROTOCOL_CORRUPTIONS:
        errors.append("corruption matrix differs from the fixed 4x4 contract")
    policy = document.get("policy")
    required_true = (
        "corruptions_applied_on_the_fly",
        "model_selection_locked_before_robustness",
        "no_retraining_or_weight_updates",
        "no_test_tuning",
    )
    if not isinstance(policy, dict) or not all(
        policy.get(field) is True for field in required_true
    ):
        errors.append("safety policy is incomplete")
    for field in ("stage1_artifacts", "stage2_evidence", "stage3_evidence"):
        if not isinstance(document.get(field), dict) or not document[field]:
            errors.append(f"{field} provenance context is missing")
    return errors


def run_precheck(
    root: Path, *, replacement_rerun: bool = False
) -> tuple[list[tuple[str, str, str]], list[str]]:
    checks: list[tuple[str, str, str]] = []
    blockers: list[str] = []

    def record(label: str, passed: bool, detail: str) -> None:
        status = "PASS" if passed else "FAIL"
        checks.append((status, label, detail))
        if not passed:
            blockers.append(f"{label}: {detail}")

    for name, expected in EXPECTED_STAGE1.items():
        path = root / "data/splits" / name
        actual = sha256_file(path) if path.is_file() else "MISSING"
        record(f"Stage 1 {name}", actual == expected, f"expected {expected}; observed {actual}")
    for relative, expected in {**EXPECTED_STAGE2, **EXPECTED_STAGE3}.items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else "MISSING"
        record(relative, actual == expected, f"expected {expected}; observed {actual}")

    freeze_path = root / "results/metrics/deep_learning/stage3_freeze.json"
    access_path = root / "results/metrics/deep_learning/stage3_test_access.json"
    if freeze_path.is_file():
        freeze = _read_json(freeze_path)
        expected_selected = {
            "resnet18_scratch_basic",
            "resnet18_pretrained_strong_aug",
            "efficientnet_b0_pretrained_strong_aug",
        }
        record(
            "Stage 3 selected set",
            set(freeze.get("selected_experiments", [])) == expected_selected,
            str(freeze.get("selected_experiments")),
        )
        record(
            "Stage 3 analysis model",
            freeze.get("analysis_model") == "efficientnet_b0_pretrained_strong_aug",
            str(freeze.get("analysis_model")),
        )
    if access_path.is_file():
        access = _read_json(access_path)
        record(
            "Stage 3 test access complete",
            access.get("status") == "completed" and access.get("outcome") == "complete",
            f"status={access.get('status')}; outcome={access.get('outcome')}",
        )

    for name, model in STAGE4_MODELS.items():
        artifact = root / str(model["artifact"])
        actual = sha256_file(artifact) if artifact.is_file() else "MISSING"
        record(f"Model artifact {name}", actual == model["sha256"], f"{artifact.relative_to(root)}; {actual}")
        for key in ("validation_metrics", "test_metrics", "test_predictions"):
            path = root / str(model[key])
            exists = path.is_file()
            detail = str(path.relative_to(root))
            if exists and key == "test_predictions":
                try:
                    frame = pd.read_csv(path)
                    exists = len(frame) == 5000 and "image_id" in frame and "local_image_path" in frame
                    detail += f"; rows={len(frame)}"
                except Exception as error:
                    exists = False
                    detail += f"; unreadable={error}"
            record(f"{name} {key}", exists, detail)

    mapping_path = root / "data/splits/class_mapping.json"
    try:
        mapping = _read_json(mapping_path)
        classes = mapping.get("classes")
        mapping_ok = mapping.get("schema_version") == 2 and isinstance(classes, list) and len(classes) == 500
    except Exception as error:
        mapping_ok = False
        classes = None
        mapping_detail = str(error)
    else:
        mapping_detail = f"schema={mapping.get('schema_version')}; classes={len(classes) if isinstance(classes, list) else 'invalid'}"
    record("Class mapping contract", mapping_ok, mapping_detail)

    protocol = root / "results/metrics/stage4_robustness_protocol.json"
    if replacement_rerun:
        protocol_ok = False
        protocol_detail = f"missing: {protocol.relative_to(root)}"
        if protocol.is_file():
            try:
                document = _read_json(protocol)
                context_errors = _replacement_protocol_context_errors(document)
                protocol_ok = not context_errors
                protocol_detail = (
                    f"{protocol.relative_to(root)}; sha256={sha256_file(protocol)}"
                )
                if not protocol_ok:
                    protocol_detail += "; " + "; ".join(context_errors)
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
                protocol_detail = f"unreadable replacement token: {error}"
        record(
            "Existing Stage 4 protocol replacement token",
            protocol_ok,
            protocol_detail,
        )
        canonical_outputs = _canonical_output_files(root)
        record(
            "Canonical Stage 4 cell directories are empty",
            not canonical_outputs,
            (
                "no canonical cell files remain"
                if not canonical_outputs
                else f"{len(canonical_outputs)} file(s), first={canonical_outputs[0].relative_to(root)}"
            ),
        )
    else:
        record(
            "No pre-existing Stage 4 protocol",
            not protocol.exists(),
            (
                "absent as expected"
                if not protocol.exists()
                else f"already exists: {protocol.relative_to(root)}"
            ),
        )
    return checks, blockers


def render_report(
    checks: list[tuple[str, str, str]],
    blockers: list[str],
    *,
    replacement_rerun: bool = False,
) -> str:
    status = "PASS" if not blockers else "BLOCKED"
    lines = [
        "# Stage 4 Precheck Report",
        "",
        f"**Status: {status}**",
        "",
        "This audit is read-only apart from this report. It does not retrain a model,",
        "run corrupted inference, or modify frozen Stage 1/2/3 evidence.",
        "",
        "## Command",
        "",
        "```text",
        (
            ".venv/bin/python scripts/precheck_stage4.py --replacement-rerun --force"
            if replacement_rerun
            else ".venv/bin/python scripts/precheck_stage4.py"
        ),
        "```",
        "",
        "## Checks",
        "",
        "| Result | Check | Evidence |",
        "|---|---|---|",
    ]
    for result, label, detail in checks:
        lines.append(f"| {result} | `{label}` | `{detail}` |")
    lines.extend(
        [
            "",
            "## Locked model-selection summary",
            "",
            "- Traditional robustness comparator: the fixed 2,424-D handcrafted-only",
            "  `random_forest`; it is distinct from the primary 3,704-D Stage 2 LinearSVC.",
            "- Scratch CNN: `resnet18_scratch_basic` from the Stage 3 freeze.",
            "- Matched pretrained CNN: `resnet18_pretrained_strong_aug` from the Stage 3 freeze.",
            "- Best pretrained/Grad-CAM model: `efficientnet_b0_pretrained_strong_aug`,",
            "  the immutable Stage 3 analysis model.",
            "- No ResNet50 artifact exists and none is required.",
            "",
            "## Readiness",
            "",
            f"- Robustness evaluation may begin: **{'YES' if not blockers else 'NO'}**",
            f"- Grad-CAM may begin: **{'YES' if not blockers else 'NO'}**",
            "- Stage 4 model training required: **NO**",
            "",
            "## Blockers",
            "",
        ]
    )
    if blockers:
        lines.extend(f"- {item}" for item in blockers)
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "STAGE4_PRECHECK_REPORT.md")
    parser.add_argument("--force", action="store_true", help="Replace an existing precheck report only.")
    parser.add_argument(
        "--replacement-rerun",
        action="store_true",
        help=(
            "authorize a rerun precheck: require the old canonical protocol as the "
            "replacement token and require both canonical cell directories to be empty"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    checks, blockers = run_precheck(
        root, replacement_rerun=bool(args.replacement_rerun)
    )
    _atomic_text(
        args.output.expanduser().resolve(),
        render_report(
            checks,
            blockers,
            replacement_rerun=bool(args.replacement_rerun),
        ),
        args.force,
    )
    for result, label, detail in checks:
        print(f"[{result}] {label} - {detail}")
    print(f"Stage 4 precheck: {'PASS' if not blockers else 'BLOCKED'}")
    print(f"Report: {args.output.expanduser().resolve()}")
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
