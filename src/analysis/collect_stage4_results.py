"""Validate all Stage 4 evidence and create the two final comparison tables."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.analysis.collect_robustness_results import (
    collect_rows,
    load_protocol,
    validate_result_matrix,
)
from src.analysis.robustness_analysis import analyse_robustness
from src.analysis.stage4_failure_analysis import (
    ANALYSIS_MODEL,
    OUTPUT_NAMES,
    _artifact,
    _find_bound_artifact,
    _load_json,
    _verify_inputs,
    build_failure_tables,
)
from src.data.common import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUMMARY_COLUMNS = [
    "model_name",
    "method_family",
    "architecture",
    "clean_test_top1",
    "clean_test_top5",
    "clean_test_macro_f1",
    "average_corrupted_top1",
    "average_corrupted_top5",
    "average_corrupted_macro_f1",
    "worst_corruption_type",
    "worst_severity",
    "worst_corruption_top1",
    "worst_corruption_macro_f1",
    "average_robustness_score",
    "robustness_rank",
    "robustness_runs",
    "gradcam_selected_examples",
    "failure_analysis_model",
    "protocol_sha256",
]

ROBUSTNESS_COLUMNS = [
    "average_corrupted_top1",
    "average_corrupted_top5",
    "average_corrupted_macro_f1",
    "worst_corruption_type",
    "worst_severity",
    "worst_corruption_top1",
    "worst_corruption_macro_f1",
    "average_robustness_score",
    "robustness_rank",
]


def _assert_frames(actual: pd.DataFrame, expected: pd.DataFrame, *, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(f"Stale or inconsistent {label}: {error}") from error


def _all_artifact_bindings(value: Any) -> list[Mapping[str, Any]]:
    bindings: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            bindings.append(value)
        for nested in value.values():
            bindings.extend(_all_artifact_bindings(nested))
    elif isinstance(value, list):
        for nested in value:
            bindings.extend(_all_artifact_bindings(nested))
    return bindings


def _require_selection_manifest_status(document: Mapping[str, Any]) -> None:
    """Require the terminal state emitted by the deterministic selector.

    Selection and CAM generation are separate transactions: the selector's
    completed state is deliberately named ``selected``, while the later CAM
    run uses ``complete``.  Keeping those states distinct prevents a selected
    CSV from being mistaken for completed visualization evidence.
    """

    if document.get("status") != "selected":
        raise ValueError("Grad-CAM selection manifest is not selected")


def _verify_gradcam(
    *,
    selection_path: Path,
    selection_manifest_path: Path,
    run_manifest_path: Path,
    analysis_path: Path,
) -> pd.DataFrame:
    selection_manifest = _load_json(selection_manifest_path)
    run_manifest = _load_json(run_manifest_path)
    _require_selection_manifest_status(selection_manifest)
    if run_manifest.get("status") != "complete":
        raise ValueError("Grad-CAM run manifest is not complete")
    selection_binding = _find_bound_artifact(selection_manifest, selection_path)
    if selection_binding is None or _artifact(selection_binding, label="gradcam.selection") != selection_path.resolve():
        raise ValueError("Grad-CAM selection manifest does not bind its CSV")
    run_selection_binding = _find_bound_artifact(run_manifest, selection_path)
    if run_selection_binding is None or _artifact(run_selection_binding, label="gradcam.run.selection") != selection_path.resolve():
        raise ValueError("Grad-CAM run manifest does not bind the frozen selection")
    if not analysis_path.is_file() or analysis_path.stat().st_size == 0:
        raise FileNotFoundError(f"Grad-CAM interpretation markdown is missing/empty: {analysis_path}")

    bindings = _all_artifact_bindings(run_manifest)
    bound_paths: set[Path] = set()
    for index, binding in enumerate(bindings):
        bound_paths.add(_artifact(binding, label=f"gradcam.run.artifact[{index}]").resolve())
    required_grids = {
        (PROJECT_ROOT / "results/figures/gradcam/correct_high_confidence_grid.png").resolve(),
        (PROJECT_ROOT / "results/figures/gradcam/incorrect_high_confidence_grid.png").resolve(),
        (PROJECT_ROOT / "results/figures/gradcam/confused_pairs_grid.png").resolve(),
    }
    if not required_grids <= bound_paths:
        raise ValueError("Grad-CAM run manifest does not bind all three summary grids")
    selection = pd.read_csv(selection_path, keep_default_na=False)
    required = {"image_id", "selection_type", "analysis_experiment"}
    if selection.empty or not required <= set(selection.columns):
        raise ValueError("Grad-CAM selection table is empty or malformed")
    if set(selection["analysis_experiment"].astype(str)) != {ANALYSIS_MODEL}:
        raise ValueError("Grad-CAM selection uses the wrong analysis model")
    counts = selection["selection_type"].astype(str).value_counts()
    if counts.get("correct_high_confidence", 0) < 10 or counts.get(
        "incorrect_high_confidence", 0
    ) < 10:
        raise ValueError("Grad-CAM selection lacks the required correct/error examples")
    figure_paths = {path for path in bound_paths if path.suffix.lower() == ".png"}
    for image_id in pd.to_numeric(selection["image_id"], errors="raise").astype("int64"):
        if not any(f"/{image_id}__" in path.as_posix() for path in figure_paths):
            raise ValueError(f"Grad-CAM figure is missing for selected image {image_id}")
    return selection


def _model_source_row(
    model_name: str,
    entry: Mapping[str, Any],
    clean: pd.Series,
    existing: pd.DataFrame,
) -> pd.Series:
    source_name = model_name
    if model_name == "traditional_random_forest":
        source_name = "random_forest"
    matches = existing.loc[existing["model_name"] == source_name]
    if len(matches) != 1:
        raise ValueError(f"Frozen final comparison has no unique row for {model_name}")
    row = matches.iloc[0].copy()
    for column, expected in (
        ("test_top1", clean["top1"]),
        ("test_top5", clean["top5"]),
        ("test_macro_f1", clean["macro_f1"]),
    ):
        if not math.isclose(float(row[column]), float(expected), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Frozen clean metric mismatch for {model_name}: {column}")
    row["model_name"] = model_name
    row["method_family"] = entry["method_family"]
    row["architecture"] = entry["architecture"]
    return row


def collect_stage4(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol = load_protocol(Path(args.protocol))

    # Independently reproduce aggregation from all 64 JSON/CSV pairs and then
    # require byte-decoded table content to agree with that reproduction.
    expected_robustness = collect_rows(Path(args.protocol), Path(args.metrics_dir))
    if not Path(args.robustness_results).is_file():
        raise FileNotFoundError(args.robustness_results)
    observed_robustness = pd.read_csv(args.robustness_results, keep_default_na=False)
    validate_result_matrix(observed_robustness, protocol)
    _assert_frames(observed_robustness, expected_robustness, label="robustness_results.csv")
    expected_drops, expected_ranking = analyse_robustness(expected_robustness, protocol)
    for path_value, expected, label in (
        (args.drop_summary, expected_drops, "robustness_drop_summary.csv"),
        (args.ranking, expected_ranking, "robustness_model_ranking.csv"),
    ):
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = pd.read_csv(path, keep_default_na=False)
        # Empty CSV fields represent mathematically unavailable relative drops.
        for column in expected.columns:
            if expected[column].dtype.kind == "f" and column in observed:
                observed[column] = pd.to_numeric(observed[column], errors="coerce")
        _assert_frames(observed, expected, label=label)

    selection = _verify_gradcam(
        selection_path=Path(args.gradcam_selection).resolve(),
        selection_manifest_path=Path(args.gradcam_selection_manifest).resolve(),
        run_manifest_path=Path(args.gradcam_manifest).resolve(),
        analysis_path=Path(args.gradcam_analysis).resolve(),
    )
    verification_args = SimpleNamespace(
        freeze_manifest=Path(args.freeze_manifest),
        test_access=Path(args.test_access),
        gradcam_selection=Path(args.gradcam_selection),
        gradcam_selection_manifest=Path(args.gradcam_selection_manifest),
    )
    inputs = _verify_inputs(verification_args)
    expected_failure_tables = build_failure_tables(*inputs)
    for name, expected in expected_failure_tables.items():
        path = Path(args.failure_dir) / OUTPUT_NAMES[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = pd.read_csv(path, keep_default_na=False)
        _assert_frames(observed, expected, label=path.name)

    clean = expected_robustness.loc[
        expected_robustness["corruption_type"] == "clean"
    ].set_index("model_name")
    ranking = expected_ranking.set_index("model_name")
    summary_rows: list[dict[str, Any]] = []
    for model_name, entry in protocol.models.items():
        baseline = clean.loc[model_name]
        robust = ranking.loc[model_name]
        summary_rows.append(
            {
                "model_name": model_name,
                "method_family": entry["method_family"],
                "architecture": entry["architecture"],
                "clean_test_top1": baseline["top1"],
                "clean_test_top5": baseline["top5"],
                "clean_test_macro_f1": baseline["macro_f1"],
                "average_corrupted_top1": robust["mean_corrupted_top1"],
                "average_corrupted_top5": robust["mean_corrupted_top5"],
                "average_corrupted_macro_f1": robust["mean_corrupted_macro_f1"],
                "worst_corruption_type": robust["worst_corruption_type"],
                "worst_severity": robust["worst_severity"],
                "worst_corruption_top1": robust["worst_condition_top1"],
                "worst_corruption_macro_f1": robust["worst_condition_macro_f1"],
                "average_robustness_score": robust["average_robustness_score"],
                "robustness_rank": robust["robustness_rank"],
                "robustness_runs": robust["num_corrupted_conditions"],
                "gradcam_selected_examples": len(selection) if model_name == ANALYSIS_MODEL else "",
                "failure_analysis_model": True if model_name == ANALYSIS_MODEL else "",
                "protocol_sha256": protocol.sha256,
            }
        )
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS).sort_values(
        "robustness_rank", kind="stable"
    ).reset_index(drop=True)

    existing_path = Path(args.clean_comparison)
    if not existing_path.is_file():
        raise FileNotFoundError(existing_path)
    existing = pd.read_csv(existing_path, keep_default_na=False)
    comparison_rows: list[pd.Series] = []
    summary_lookup = summary.set_index("model_name")
    for model_name, entry in protocol.models.items():
        row = _model_source_row(model_name, entry, clean.loc[model_name], existing)
        robust = summary_lookup.loc[model_name]
        for column in ROBUSTNESS_COLUMNS:
            row[column] = robust[column]
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison = comparison[[*existing.columns, *ROBUSTNESS_COLUMNS]].sort_values(
        "robustness_rank", kind="stable"
    ).reset_index(drop=True)
    if len(summary) != 4 or len(comparison) != 4:
        raise ValueError("Final Stage 4 outputs must contain exactly four protocol models")
    return summary, comparison


def _install(outputs: Mapping[Path, pd.DataFrame], *, force: bool) -> None:
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Stage 4 final outputs already exist; use --force deliberately: "
            + ", ".join(str(path) for path in existing)
        )
    parent = next(iter(outputs)).parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage4_final_collection.", dir=parent) as name:
        stage = Path(name)
        staged: dict[Path, Path] = {}
        for index, (destination, frame) in enumerate(outputs.items()):
            temporary = stage / f"{index:02d}_{destination.name}"
            frame.to_csv(temporary, index=False, lineterminator="\n")
            staged[destination] = temporary
        backups: dict[Path, Path] = {}
        installed: list[Path] = []
        try:
            for index, destination in enumerate(existing):
                backup = stage / f"backup_{index:02d}_{destination.name}"
                os.replace(destination, backup)
                backups[destination] = backup
            for destination, source in staged.items():
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
    parser.add_argument("--protocol", type=Path, default=Path("results/metrics/stage4_robustness_protocol.json"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics/robustness"))
    parser.add_argument("--robustness-results", type=Path, default=Path("results/tables/robustness_results.csv"))
    parser.add_argument("--drop-summary", type=Path, default=Path("results/tables/robustness_drop_summary.csv"))
    parser.add_argument("--ranking", type=Path, default=Path("results/tables/robustness_model_ranking.csv"))
    parser.add_argument("--gradcam-selection", type=Path, default=Path("results/tables/gradcam_selected_examples.csv"))
    parser.add_argument("--gradcam-selection-manifest", type=Path, default=Path("results/metrics/explainability/gradcam_selection_manifest.json"))
    parser.add_argument("--gradcam-manifest", type=Path, default=Path("results/metrics/explainability/gradcam_manifest.json"))
    parser.add_argument("--gradcam-analysis", type=Path, default=Path("results/tables/gradcam_analysis.md"))
    parser.add_argument("--freeze-manifest", type=Path, default=Path("results/metrics/deep_learning/stage3_freeze.json"))
    parser.add_argument("--test-access", type=Path, default=Path("results/metrics/deep_learning/stage3_test_access.json"))
    parser.add_argument("--failure-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--clean-comparison", type=Path, default=Path("results/tables/final_model_comparison.csv"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/tables/stage4_summary.csv"))
    parser.add_argument("--comparison-output", type=Path, default=Path("results/tables/final_model_comparison_with_robustness.csv"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        summary, comparison = collect_stage4(args)
        _install(
            {args.summary_output: summary, args.comparison_output: comparison},
            force=bool(args.force),
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {len(summary)} Stage 4 summary/comparison rows for the frozen protocol models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
