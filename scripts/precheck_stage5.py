#!/usr/bin/env python3
"""Read-only Stage 5 evidence precheck with immutable hash snapshot output."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.stage5_contract import (  # noqa: E402
    FROZEN_ARTIFACTS,
    MAIN_SOURCE_COLUMNS,
    MAIN_SOURCE_MODELS,
    STAGE4_PROTOCOL_MODELS,
    compare_read_only_evidence,
    load_json,
    require_columns,
    require_unique_labels,
    verify_frozen_artifacts,
    verify_nested_artifact_bindings,
    snapshot_read_only_evidence,
)
from src.data.common import sha256_file  # noqa: E402


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _command_path(path: Path) -> str:
    """Keep the invoked venv/script spelling instead of resolving symlinks."""

    absolute = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        return absolute.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return absolute.as_posix()


def _check(condition: bool, label: str, details: str) -> dict[str, Any]:
    return {"passed": bool(condition), "label": label, "details": details}


def run_checks() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    artifact_rows = verify_frozen_artifacts()
    checks = [
        _check(
            bool(row["passed"]),
            f"frozen hash: {row['path']}",
            row["observed_sha256"] if row["exists"] else "MISSING",
        )
        for row in artifact_rows
    ]
    nested_rows: list[dict[str, Any]] = []
    try:
        nested_rows = verify_nested_artifact_bindings()
        checks.extend(
            _check(
                True,
                f"nested manifest binding: {row['path']}",
                f"sha256={row['sha256']}, bytes={row['bytes']}, bindings={row['binding_count']}",
            )
            for row in nested_rows
        )
        checks.append(
            _check(
                True,
                "Nested manifest binding summary",
                f"deduplicated_artifacts={len(nested_rows)}, all paths safe/in-root and all hashes/sizes matched",
            )
        )
    except (FileNotFoundError, ValueError) as error:
        checks.append(_check(False, "Nested manifest binding verification", str(error)))
    read_only_rows: list[dict[str, Any]] = []
    try:
        read_only_rows = snapshot_read_only_evidence()
        checks.append(
            _check(
                True,
                "Complete Stage 1--4 read-only evidence inventory",
                f"files={len(read_only_rows)}, bytes={sum(int(row['bytes']) for row in read_only_rows)}",
            )
        )
    except (FileNotFoundError, ValueError) as error:
        checks.append(_check(False, "Complete Stage 1--4 read-only evidence inventory", str(error)))
    try:
        split_expectations = {"train.csv": 20_000, "val.csv": 5_000, "test.csv": 5_000}
        for name, expected in split_expectations.items():
            frame = pd.read_csv(PROJECT_ROOT / "data/splits" / name, usecols=["new_label"])
            labels = set(pd.to_numeric(frame["new_label"], errors="raise").astype(int))
            checks.append(
                _check(
                    len(frame) == expected and labels == set(range(500)),
                    f"Stage 1 split contract: {name}",
                    f"rows={len(frame)}, classes={len(labels)}, expected_rows={expected}",
                )
            )
    except (FileNotFoundError, ValueError) as error:
        checks.append(_check(False, "Stage 1 split contract", str(error)))

    try:
        stage2 = load_json(PROJECT_ROOT / "results/metrics/stage2_freeze.json")
        checks.append(_check(stage2.get("status") == "frozen", "Stage 2 freeze status", str(stage2.get("status"))))
        stage3 = load_json(PROJECT_ROOT / "results/metrics/deep_learning/stage3_freeze.json")
        checks.append(_check(stage3.get("status") == "frozen", "Stage 3 freeze status", str(stage3.get("status"))))
        access = load_json(PROJECT_ROOT / "results/metrics/deep_learning/stage3_test_access.json")
        checks.append(
            _check(
                access.get("status") == "completed" and access.get("outcome") == "complete",
                "Stage 3 test-access status",
                f"status={access.get('status')}, outcome={access.get('outcome')}",
            )
        )
        protocol = load_json(PROJECT_ROOT / "results/metrics/stage4_robustness_protocol.json")
        protocol_models = tuple(protocol.get("model_order", []))
        protocol_ok = (
            protocol.get("status") == "frozen"
            and protocol.get("expected_run_count") == 64
            and protocol.get("expected_num_images_per_run") == 5_000
            and protocol_models == STAGE4_PROTOCOL_MODELS
        )
        checks.append(
            _check(
                protocol_ok,
                "Stage 4 robustness protocol freeze",
                f"status={protocol.get('status')}, runs={protocol.get('expected_run_count')}, models={list(protocol_models)}",
            )
        )
        selection = load_json(PROJECT_ROOT / "results/metrics/explainability/gradcam_selection_manifest.json")
        cam = load_json(PROJECT_ROOT / "results/metrics/explainability/gradcam_manifest.json")
        checks.append(_check(selection.get("status") == "selected", "Grad-CAM selection status", str(selection.get("status"))))
        checks.append(_check(cam.get("status") == "complete", "Grad-CAM run status", str(cam.get("status"))))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        checks.append(_check(False, "Freeze/status documents", str(error)))

    try:
        main = pd.read_csv(PROJECT_ROOT / "results/tables/final_model_comparison.csv")
        require_columns(main, MAIN_SOURCE_COLUMNS, "final_model_comparison.csv")
        require_unique_labels(main, "model_name", MAIN_SOURCE_MODELS, "final_model_comparison.csv")
        table_counts = {
            "stage2_traditional_results.csv": 3,
            "stage2_feature_ablation_results.csv": 4,
            "stage3_deep_learning_results.csv": 5,
            "stage3_ablation_results.csv": 5,
            "robustness_results.csv": 68,
            "robustness_drop_summary.csv": 64,
            "robustness_model_ranking.csv": 4,
            "gradcam_selected_examples.csv": 33,
            "stage4_summary.csv": 4,
        }
        for name, expected in table_counts.items():
            frame = pd.read_csv(PROJECT_ROOT / "results/tables" / name)
            checks.append(
                _check(len(frame) == expected, f"Result-table rows: {name}", f"rows={len(frame)}, expected={expected}")
            )
    except (FileNotFoundError, ValueError) as error:
        checks.append(_check(False, "Result-table schemas", str(error)))

    checks.append(
        _check(
            all(row["passed"] for row in artifact_rows),
            "No new model training or test inference required",
            "All accepted reports, result tables, freezes, and advanced-analysis evidence are present and unchanged.",
        )
    )
    return checks, artifact_rows, nested_rows, read_only_rows


def snapshot_document(
    artifact_rows: list[dict[str, Any]],
    nested_rows: list[dict[str, Any]],
    read_only_rows: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "snapshot_label": label,
        "scope": "accepted Stage 1-4 evidence before/after Stage 5 derivation",
        "artifacts": {
            row["path"]: {
                "stage": row["stage"],
                "artifact_type": row["artifact_type"],
                "bytes": (PROJECT_ROOT / row["path"]).stat().st_size if row["exists"] else None,
                "sha256": row["observed_sha256"] or None,
                "accepted_sha256": row["expected_sha256"],
                "matches_accepted": bool(row["passed"]),
            }
            for row in artifact_rows
        },
        "manifest_bound_artifacts": {
            row["path"]: {
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "binding_count": row["binding_count"],
                "origins": row["origins"],
            }
            for row in nested_rows
        },
        "read_only_evidence": {
            row["path"]: {
                "bytes": int(row["bytes"]),
                "sha256": row["sha256"],
            }
            for row in read_only_rows
        },
    }


def report_markdown(
    checks: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    nested_rows: list[dict[str, Any]],
    read_only_rows: list[dict[str, Any]],
    *,
    command: str,
    snapshot_path: Path,
) -> str:
    passed = sum(bool(row["passed"]) for row in checks)
    failed = len(checks) - passed
    status = "PASS" if failed == 0 else "FAIL"
    lines = [
        "# Stage 5 Precheck Report",
        "",
        f"**Status: {status}**",
        "",
        "**Scope:** read-only verification of frozen Stage 1--4 evidence before final report and packaging work.",
        "",
        "## Command",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Exact command output",
        "",
        "```text",
    ]
    for row in checks:
        marker = "PASS" if row["passed"] else "FAIL"
        lines.append(f"[{marker}] {row['label']} :: {row['details']}")
    lines.extend(
        [
            f"SUMMARY: {passed} PASS, {failed} FAIL, {len(checks)} TOTAL",
            "```",
            "",
            "## Frozen evidence snapshot",
            "",
            f"The machine-readable snapshot is `{_portable(snapshot_path)}`. It records {len(artifact_rows)} accepted contract artifacts, {len(nested_rows)} deduplicated manifest-bound artifacts, and a closed inventory of {len(read_only_rows)} Stage 1--4 evidence files with exact SHA-256 and byte-size evidence.",
            "",
            "Stage 4's immutable freeze is `results/metrics/stage4_robustness_protocol.json`; no separate `stage4_test_access.json` was part of the accepted Stage 4 design.",
            "",
            "## Readiness",
            "",
            (
                "All required evidence is present and unchanged. Stage 5 may derive report tables, report-only figure copies, LaTeX sources, and packaging manifests without retraining or rerunning test inference."
                if failed == 0
                else "Stage 5 is blocked until every failed or missing frozen-evidence check is resolved without fabricating results."
            ),
            "",
            "The CVPR report template, consolidated Stage 5 tables, report-ready figure copies, and submission ZIP are Stage 5 derived deliverables; their absence at precheck is expected.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_outputs(outputs: Mapping[Path, bytes], *, force: bool) -> None:
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Precheck outputs already exist; use --force deliberately: "
            + ", ".join(str(path) for path in existing)
        )
    with tempfile.TemporaryDirectory(prefix=".stage5_precheck.", dir=PROJECT_ROOT) as name:
        stage = Path(name)
        staged: dict[Path, Path] = {}
        for index, (destination, payload) in enumerate(outputs.items()):
            temporary = stage / f"{index:02d}_{destination.name}"
            temporary.write_bytes(payload)
            staged[destination] = temporary
        backups: dict[Path, Path] = {}
        installed: list[Path] = []
        try:
            for index, destination in enumerate(existing):
                backup = stage / f"backup_{index:02d}_{destination.name}"
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-output", type=Path, default=Path("STAGE5_PRECHECK_REPORT.md"))
    parser.add_argument(
        "--snapshot-output", type=Path,
        default=Path("results/metrics/stage5_frozen_hashes_before.json"),
    )
    parser.add_argument("--snapshot-label", default="before_stage5")
    parser.add_argument(
        "--baseline-snapshot",
        type=Path,
        help="Compare the current closed evidence set with a prior Stage 5 snapshot",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = _parse_args(argv)
    checks, artifact_rows, nested_rows, read_only_rows = run_checks()
    if args.baseline_snapshot is not None:
        baseline_path = (
            (PROJECT_ROOT / args.baseline_snapshot).resolve()
            if not args.baseline_snapshot.is_absolute()
            else args.baseline_snapshot.resolve()
        )
        try:
            baseline = load_json(baseline_path)
            passed, details = compare_read_only_evidence(read_only_rows, baseline)
            checks.append(_check(passed, "Before/after Stage 1--4 evidence equality", details))
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            checks.append(_check(False, "Before/after Stage 1--4 evidence equality", str(error)))
    snapshot_path = (PROJECT_ROOT / args.snapshot_output).resolve() if not args.snapshot_output.is_absolute() else args.snapshot_output.resolve()
    report_path = (PROJECT_ROOT / args.report_output).resolve() if not args.report_output.is_absolute() else args.report_output.resolve()
    command = shlex.join(
        [
            _command_path(Path(sys.executable)),
            _command_path(Path(__file__)),
            *raw_argv,
        ]
    )
    snapshot = snapshot_document(
        artifact_rows, nested_rows, read_only_rows, str(args.snapshot_label)
    )
    report = report_markdown(
        checks,
        artifact_rows,
        nested_rows,
        read_only_rows,
        command=command,
        snapshot_path=snapshot_path,
    )
    outputs = {
        snapshot_path: (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        report_path: report.encode("utf-8"),
    }
    try:
        _atomic_outputs(outputs, force=bool(args.force))
    except FileExistsError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for row in checks:
        marker = "PASS" if row["passed"] else "FAIL"
        print(f"[{marker}] {row['label']} :: {row['details']}")
    failed = sum(not row["passed"] for row in checks)
    print(f"SUMMARY: {len(checks)-failed} PASS, {failed} FAIL, {len(checks)} TOTAL")
    print(f"REPORT: {_portable(report_path)}")
    print(f"SNAPSHOT: {_portable(snapshot_path)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
