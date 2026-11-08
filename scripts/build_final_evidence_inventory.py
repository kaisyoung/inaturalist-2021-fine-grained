#!/usr/bin/env python3
"""Create the final Stage 1--5 evidence inventory with file hashes and policy flags."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "final_evidence_inventory.csv"
INVENTORY_COLUMNS = (
    "stage",
    "artifact_type",
    "file_path",
    "exists",
    "file_size_bytes",
    "sha256",
    "include_in_report_source",
    "include_in_code_zip",
    "reason",
)

ROOT_REPORTS = (
    "DATASET_SMOKE_TEST_REPORT.md",
    "DEEP_LEARNING_SMOKE_TEST_REPORT.md",
    "STAGE1_ACCEPTANCE_REPORT.md",
    "STAGE1_FIX_REPORT.md",
    "STAGE2_TRADITIONAL_REPORT.md",
    "STAGE2_FINAL_ACCEPTANCE_REPORT.md",
    "TRADITIONAL_REWRITE_INTEGRATION_AUDIT.md",
    "STAGE4_TRADITIONAL_REWRITE_RERUN_REPORT.md",
    "STAGE3_PRECHECK_REPORT.md",
    "STAGE3_DEEP_LEARNING_REPORT.md",
    "STAGE4_PRECHECK_REPORT.md",
    "STAGE4_ADVANCED_ANALYSIS_REPORT.md",
    "STAGE5_PRECHECK_REPORT.md",
    "STAGE5_IMMUTABILITY_REPORT.md",
    "STAGE5_RESUME_AUDIT.md",
    "STAGE5_RESUME_REPORT.md",
    "REPORT_REFERENCE_AUDIT.md",
    "REPORT_CONSISTENCY_AUDIT.md",
    "LATEX_BUILD_REPORT.md",
    "FINAL_REPRODUCIBILITY_AUDIT.md",
    "SUBMISSION_PACKAGE_AUDIT.md",
    "STAGE5_FINAL_SUBMISSION_REPORT.md",
    "FINAL_POST_TRADITIONAL_REWRITE_ACCEPTANCE.md",
)

HISTORICAL_REPORTS = frozenset(
    {
        "STAGE2_TRADITIONAL_REPORT.md",
        "STAGE2_FINAL_ACCEPTANCE_REPORT.md",
        "STAGE4_ADVANCED_ANALYSIS_REPORT.md",
        "FINAL_PROJECT_ACCEPTANCE_REPORT.md",
        "FINAL_INDEPENDENT_REVIEW.md",
    }
)

SPLIT_FILES = (
    "data/splits/selected_classes_seed42.txt",
    "data/splits/selected_classes_readable.csv",
    "data/splits/class_mapping.json",
    "data/splits/train.csv",
    "data/splits/val.csv",
    "data/splits/test.csv",
    "data/splits/split_summary.json",
)

SOURCE_FIGURES = (
    "results/figures/dataset_sample_grid.png",
    "results/figures/stage2_traditional_model_comparison.png",
    "results/figures/deep_learning/resnet18_scratch_basic_training_curves.png",
    "results/figures/deep_learning/efficientnet_b0_pretrained_strong_aug_training_curves.png",
    "results/figures/robustness/robustness_top1_curves.png",
    "results/figures/robustness/robustness_macro_f1_curves.png",
    "results/figures/gradcam/correct_high_confidence_grid.png",
    "results/figures/gradcam/incorrect_high_confidence_grid.png",
    "results/figures/gradcam/confused_pairs_grid.png",
)

FINAL_DELIVERABLES = (
    "README.md",
    "requirements.txt",
    ".gitignore",
    "report/COMP9517_Group_Project_Report.pdf",
    "presentation/COMP9517_Final_Presentation_10min.pptx",
    "submission/COMP9517_Group_Project_Code.zip",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage(relative: str) -> str:
    lower = relative.lower()
    if "stage4_traditional_rewrite_rerun" in lower:
        return "stage4"
    if any(
        token in lower
        for token in (
            "traditional_rewrite_integration",
            "final_post_traditional_rewrite",
            "presentation/comp9517_final_presentation",
        )
    ):
        return "stage5"
    if "stage1" in lower or lower.startswith("data/splits/"):
        return "stage1"
    if "stage2" in lower or "traditional_" in lower:
        return "stage2"
    if "stage3" in lower or "deep_learning" in lower:
        return "stage3"
    if any(token in lower for token in ("stage4", "robustness", "gradcam", "explainability")):
        return "stage4"
    if any(
        token in lower
        for token in (
            "stage5",
            "final_report",
            "submission",
            "cvpr_latex",
            "reproducibility",
            "report_reference",
            "report_consistency",
            "latex_build",
        )
    ):
        return "stage5"
    return "all"


def _artifact_type(relative: str) -> str:
    path = Path(relative)
    lower = relative.lower()
    if lower.endswith(".zip"):
        return "code_zip"
    if lower.endswith(".pptx"):
        return "presentation_deck"
    if lower.endswith(".pdf"):
        return "report_pdf"
    if lower.startswith("src/"):
        return "source_code"
    if lower.startswith("scripts/"):
        return "script"
    if lower.startswith("tests/"):
        return "test"
    if lower.startswith("configs/"):
        return "configuration"
    if lower.startswith("data/splits/"):
        return "split_manifest"
    if "figures" in path.parts:
        return "figure"
    if "metrics" in path.parts:
        return "metric_or_freeze"
    if "models" in path.parts:
        return "model_or_checkpoint"
    if "predictions" in path.parts:
        return "prediction"
    if "features" in path.parts:
        return "feature_cache_or_config"
    if "logs" in path.parts:
        return "experiment_log"
    if "tables" in path.parts:
        return "result_table"
    if "cvpr_latex" in path.parts:
        return "report_source"
    if (
        path.name.endswith("REPORT.md")
        or "AUDIT" in path.name
        or "ACCEPTANCE" in path.name
    ):
        return "report_or_audit"
    if path.name == "README.md":
        return "documentation"
    if path.name == "requirements.txt":
        return "environment"
    return "project_metadata"


def _report_source_flag(relative: str) -> bool:
    return (
        relative.startswith("report/cvpr_latex/")
        or relative.startswith("results/final_report_figures/")
        or relative.startswith("results/tables/final_report_")
    )


def _discover(root: Path, output: Path) -> list[str]:
    candidates = (
        set(ROOT_REPORTS)
        | set(HISTORICAL_REPORTS)
        | set(SPLIT_FILES)
        | set(SOURCE_FIGURES)
        | set(FINAL_DELIVERABLES)
    )
    patterns = (
        "src/**/*.py",
        "scripts/*.py",
        "tests/*.py",
        "configs/*",
        "results/metrics/**/*.json",
        "results/tables/**/*.csv",
        "results/tables/**/*.md",
        "results/tables/**/*.json",
        "results/final_report_figures/*",
        "report/cvpr_latex/**/*.tex",
        "report/cvpr_latex/**/*.bib",
        "report/cvpr_latex/**/*.sty",
        "report/cvpr_latex/**/*.bst",
        "report/cvpr_latex/**/*.md",
    )
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                candidates.add(path.relative_to(root).as_posix())
    # Use the same closed Stage 1--4 evidence set as the immutability audit so
    # the final inventory cannot silently omit a frozen model, prediction,
    # feature cache, log, figure, binary metric array, or smoke report.
    baseline_path = root / "results/metrics/stage5_frozen_hashes_before.json"
    if baseline_path.is_file():
        import json

        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        frozen = baseline.get("read_only_evidence", {})
        if not isinstance(frozen, dict):
            raise ValueError("Stage 5 baseline has no read_only_evidence mapping")
        candidates.update(str(path) for path in frozen)
    output_relative = output.relative_to(root).as_posix()
    candidates.discard(output_relative)  # a file cannot truthfully contain its own digest
    return sorted(candidates)


def _code_zip_paths(root: Path) -> set[str]:
    try:
        from scripts.make_code_zip import collect_allowed_files
    except (ImportError, AttributeError):
        return set()
    values = collect_allowed_files(root)
    paths: set[str] = set()
    for value in values:
        path = value[0] if isinstance(value, tuple) else value
        candidate = Path(path)
        if candidate.is_absolute():
            candidate = candidate.relative_to(root)
        paths.add(candidate.as_posix())
    return paths


def build_inventory(root: Path, output: Path) -> list[dict[str, str | int]]:
    code_paths = _code_zip_paths(root)
    rows: list[dict[str, str | int]] = []
    for relative in _discover(root, output):
        path = root / relative
        exists = path.is_file()
        in_report = _report_source_flag(relative)
        in_code = relative in code_paths
        if relative in HISTORICAL_REPORTS:
            reason = (
                "Retained pre-rewrite historical evidence; superseded by the "
                "post-rewrite acceptance record and excluded from the code ZIP."
            )
        elif in_code:
            reason = "Included by the deterministic code-ZIP allowlist."
        elif in_report:
            reason = "Used by the CVPR report; excluded from the code ZIP when it is a figure or build output."
        elif relative in SOURCE_FIGURES:
            reason = "Frozen source evidence for a report-optimized derivative; original is excluded from the code ZIP."
        elif relative.endswith(".pdf"):
            reason = "Separate written-report deliverable; PDFs are forbidden in the code ZIP."
        elif relative.endswith(".pptx"):
            reason = "Separate presentation deliverable; Office decks are excluded from the code ZIP."
        elif relative.endswith(".zip"):
            reason = "Final code archive; an archive is not included inside itself."
        elif relative in ROOT_REPORTS:
            reason = "Internal acceptance/audit evidence retained outside the code submission."
        elif relative.startswith("results/models/"):
            reason = "Frozen model/checkpoint evidence; weight artifacts are forbidden in the code ZIP."
        elif relative.startswith("results/predictions/"):
            reason = "Frozen per-image prediction evidence; excluded from the conservative code ZIP."
        elif relative.startswith("results/features/"):
            reason = "Frozen feature cache/config evidence; feature caches are excluded from the code ZIP."
        elif relative.startswith("results/logs/"):
            reason = "Frozen experiment log evidence; generated logs are excluded from the code ZIP."
        elif relative.endswith((".npz", ".npy")):
            reason = "Frozen binary result array; binary arrays are forbidden in the code ZIP."
        elif relative.startswith("results/figures/"):
            reason = "Frozen source figure evidence; generated figures are excluded from the code ZIP."
        else:
            reason = "Evidence inventory item retained outside the conservative code-ZIP allowlist."
        rows.append(
            {
                "stage": _stage(relative),
                "artifact_type": _artifact_type(relative),
                "file_path": relative,
                "exists": "yes" if exists else "no",
                "file_size_bytes": path.stat().st_size if exists else "",
                "sha256": _sha256(path) if exists else "",
                "include_in_report_source": "yes" if in_report else "no",
                "include_in_code_zip": "yes" if in_code else "no",
                "reason": reason if exists else "Required or expected artifact is missing.",
            }
        )
    return rows


def _atomic_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=INVENTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    output = args.output.resolve()
    rows = build_inventory(root, output)
    _atomic_csv(output, rows)
    missing = [row for row in rows if row["exists"] == "no"]
    print(f"Wrote {len(rows)} evidence rows to {output}")
    if missing:
        print(f"WARNING: {len(missing)} expected artifact(s) are missing")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
