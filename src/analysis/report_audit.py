"""Static and evidence-consistency audit for the final CVPR LaTeX report."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

import pandas as pd

from src.analysis.stage5_contract import PROJECT_ROOT, verify_frozen_artifacts
from src.data.common import sha256_file


DEFAULT_TEX_DIR = PROJECT_ROOT / "report" / "cvpr_latex"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "results" / "final_report_figures"
DEFAULT_REFERENCE_REPORT = PROJECT_ROOT / "REPORT_REFERENCE_AUDIT.md"
DEFAULT_CONSISTENCY_REPORT = PROJECT_ROOT / "REPORT_CONSISTENCY_AUDIT.md"
EXPECTED_TITLE = (
    "Fine-Grained Species Recognition at Scale: Traditional Baselines, "
    "Transfer Learning, Robustness, and Grad-CAM"
)
OFFICIAL_TEMPLATE_HASHES = {
    "cvpr.sty": "2602473285d1a7df2a445ac89b76e1afa0acab78e056f0369d19770245190153",
    "ieeenat_fullname.bst": "e38e6166bd7b1e6d23a1b79dcdb55c656e4fcdbe91bdf6b50d827e6b5d1aacfc",
}
OFFICIAL_ARCHIVE_HASH = "ef672bf4ad1c6801237b957d57b0087d4412e7dee01ad3b512abe82960c6466c"

REFERENCE_CHECK_NAMES = {
    "Required report sources",
    "CVPR submission mode",
    "Official CVPR 2026 template provenance",
    "Exact report title",
    "Required report sections",
    "Author placeholders",
    "Non-author placeholders",
    "Main-text page boundary",
    "Citation/reference consistency",
    "Required reference coverage and provenance",
    "LaTeX labels/references",
    "Direct dependency citations",
    "Known software-reference metadata",
    "Referenced figures exist",
    "Referenced figure hashes",
    "Required report visuals",
}

CONSISTENCY_CHECK_NAMES = {
    "Required report sources",
    "Report identity contract",
    "Generated tables and source hashes",
    "Frozen split schema and counts",
    "Validation/test isolation",
    "Stage 4 non-selection wording",
    "Grad-CAM non-causal wording",
    "Traditional rewrite method contract",
    "Observed headline claims",
    "Causal/protocol wording guard",
    "Frozen Stage 1-4 evidence",
}

REQUIRED_FILES = (
    "main.tex",
    "preamble.tex",
    "references.bib",
    "cvpr.sty",
    "ieeenat_fullname.bst",
    "OFFICIAL_TEMPLATE_SOURCE.md",
    "generated/result_macros.tex",
    "generated/main_results_table.tex",
    "generated/detailed_test_table.tex",
    "generated/ablation_results_table.tex",
    "generated/robustness_summary_table.tex",
    "sections/01_introduction.tex",
    "sections/02_literature_review.tex",
    "sections/03_dataset_protocol.tex",
    "sections/04_methods.tex",
    "sections/05_experimental_setup.tex",
    "sections/06_results.tex",
    "sections/07_discussion.tex",
    "sections/08_conclusion.tex",
)

REQUIRED_SECTION_NAMES = (
    "Introduction",
    "Literature Review",
    "Dataset and Protocol",
    "Methods",
    "Experimental Setup",
    "Experimental Results",
    "Discussion",
    "Conclusion",
)

DIRECT_DEPENDENCY_CITATIONS = {
    "numpy": "harris2020array",
    "pandas": "mckinney2010data",
    "tqdm": "costaluis2019tqdm",
    "scikit-learn": "pedregosa2011scikit",
    "scikit-image": "vanderwalt2014scikitimage",
    "pillow": "clark2026pillow",
    "matplotlib": "hunter2007matplotlib",
    "joblib": "joblib2025",
    "opencv-python-headless": "bradski2000opencv",
    "torch": "paszke2019pytorch",
    "torchvision": "marcel2010torchvision",
    "pyyaml": "pyyaml2025",
}

# Curated from the original papers and official software release/project
# records used to write the report.  Requiring the bibliography to match this
# whitelist prevents a plausible-looking, unreviewed reference from entering
# the submission unnoticed.
REFERENCE_COVERAGE = {
    "dataset": {"vanhorn2018inaturalist", "vanhorn2021benchmarking"},
    "traditional methods": {
        "lowe2004distinctive",
        "csurka2004visual",
        "dalal2005histograms",
        "ojala2002multiresolution",
        "cortes1995support",
        "breiman2001random",
        "arandjelovic2012three",
        "perronnin2010improving",
        "vangemert2010visual",
        "lazebnik2006beyond",
    },
    "deep learning": {
        "deng2009imagenet",
        "he2016deep",
        "tan2019efficientnet",
        "yosinski2014transferable",
        "shorten2019survey",
        "zhong2020random",
    },
    "advanced studies": {"hendrycks2019benchmarking", "selvaraju2017gradcam"},
    "software acknowledgements": set(DIRECT_DEPENDENCY_CITATIONS.values()),
}


@dataclass(frozen=True)
class AuditCheck:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class ObservedClaim:
    macro_name: str
    relative_source: str
    model_name: str
    column: str
    decimals: int = 2


OBSERVED_CLAIMS = (
    ObservedClaim(
        "PrimaryTraditionalTestTopOnePercent",
        "results/tables/final_model_comparison.csv",
        "linear_svm",
        "test_top1",
    ),
    ObservedClaim(
        "PrimaryTraditionalTestTopFivePercent",
        "results/tables/final_model_comparison.csv",
        "linear_svm",
        "test_top5",
    ),
    ObservedClaim(
        "PrimaryTraditionalTestMacroFOnePercent",
        "results/tables/final_model_comparison.csv",
        "linear_svm",
        "test_macro_f1",
    ),
    ObservedClaim(
        "RandomForestTestTopOnePercent",
        "results/tables/final_model_comparison.csv",
        "random_forest",
        "test_top1",
    ),
    ObservedClaim(
        "RandomForestTestTopFivePercent",
        "results/tables/final_model_comparison.csv",
        "random_forest",
        "test_top5",
    ),
    ObservedClaim(
        "RandomForestTestMacroFOnePercent",
        "results/tables/final_model_comparison.csv",
        "random_forest",
        "test_macro_f1",
    ),
    ObservedClaim(
        "BoVWOnlyTestTopOnePercent",
        "results/tables/final_model_comparison.csv",
        "sift_bovw_svm",
        "test_top1",
    ),
    ObservedClaim(
        "BoVWOnlyTestTopFivePercent",
        "results/tables/final_model_comparison.csv",
        "sift_bovw_svm",
        "test_top5",
    ),
    ObservedClaim(
        "BoVWOnlyTestMacroFOnePercent",
        "results/tables/final_model_comparison.csv",
        "sift_bovw_svm",
        "test_macro_f1",
    ),
    ObservedClaim(
        "BestTestTopOnePercent",
        "results/tables/final_model_comparison.csv",
        "efficientnet_b0_pretrained_strong_aug",
        "test_top1",
    ),
    ObservedClaim(
        "BestTestTopFivePercent",
        "results/tables/final_model_comparison.csv",
        "efficientnet_b0_pretrained_strong_aug",
        "test_top5",
    ),
    ObservedClaim(
        "BestTestMacroFOnePercent",
        "results/tables/final_model_comparison.csv",
        "efficientnet_b0_pretrained_strong_aug",
        "test_macro_f1",
    ),
    ObservedClaim(
        "BestMeanCorruptedTopOnePercent",
        "results/tables/robustness_model_ranking.csv",
        "efficientnet_b0_pretrained_strong_aug",
        "mean_corrupted_top1",
    ),
    ObservedClaim(
        "BestMeanCorruptedMacroFOnePercent",
        "results/tables/robustness_model_ranking.csv",
        "efficientnet_b0_pretrained_strong_aug",
        "mean_corrupted_macro_f1",
    ),
    ObservedClaim(
        "BestWorstCorruptedTopOnePercent",
        "results/tables/robustness_model_ranking.csv",
        "efficientnet_b0_pretrained_strong_aug",
        "worst_condition_top1",
    ),
    ObservedClaim(
        "BestWorstCorruptedMacroFOnePercent",
        "results/tables/robustness_model_ranking.csv",
        "efficientnet_b0_pretrained_strong_aug",
        "worst_condition_macro_f1",
    ),
    ObservedClaim(
        "RandomForestMeanCorruptedTopOnePercent",
        "results/tables/robustness_model_ranking.csv",
        "traditional_random_forest",
        "mean_corrupted_top1",
    ),
    ObservedClaim(
        "RandomForestMeanCorruptedMacroFOnePercent",
        "results/tables/robustness_model_ranking.csv",
        "traditional_random_forest",
        "mean_corrupted_macro_f1",
    ),
    ObservedClaim(
        "RandomForestWorstCorruptedTopOnePercent",
        "results/tables/robustness_model_ranking.csv",
        "traditional_random_forest",
        "worst_condition_top1",
    ),
    ObservedClaim(
        "RandomForestWorstCorruptedMacroFOnePercent",
        "results/tables/robustness_model_ranking.csv",
        "traditional_random_forest",
        "worst_condition_macro_f1",
    ),
)


def _strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        match = re.search(r"(?<!\\)%", line)
        lines.append(line[: match.start()] if match else line)
    return "\n".join(lines)


def _tex_sources(tex_dir: Path, *, include_generated: bool = True) -> dict[Path, str]:
    paths = [tex_dir / "main.tex", tex_dir / "preamble.tex"]
    paths.extend(sorted((tex_dir / "sections").glob("*.tex")))
    if include_generated:
        paths.extend(sorted((tex_dir / "generated").glob("*.tex")))
    return {
        path: _strip_comments(path.read_text(encoding="utf-8"))
        for path in paths
        if path.is_file()
    }


def citation_keys(tex_text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite[a-zA-Z*]*\s*\{([^}]*)\}", tex_text):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def bibliography_keys(bib_text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text)]


def _reference_coverage_check(cited: set[str], bib_text: str) -> AuditCheck:
    curated = set().union(*REFERENCE_COVERAGE.values())
    bib_keys = set(bibliography_keys(bib_text))
    failures: list[str] = []
    if cited != curated:
        failures.append(
            f"citation whitelist missing={sorted(curated - cited)} unexpected={sorted(cited - curated)}"
        )
    if bib_keys != curated:
        failures.append(
            f"bibliography whitelist missing={sorted(curated - bib_keys)} unexpected={sorted(bib_keys - curated)}"
        )
    blocks = {
        match.group(1).strip(): match.group(2)
        for match in re.finditer(
            r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)(?=\n@\w+\s*\{|\Z)",
            bib_text,
            re.DOTALL,
        )
    }
    for key in sorted(curated):
        block = blocks.get(key, "")
        missing_core = [
            field
            for field in ("title", "author", "year")
            if not re.search(rf"\b{field}\s*=", block, re.IGNORECASE)
        ]
        has_primary_locator = bool(
            re.search(
                r"\b(?:journal|booktitle|doi|url|howpublished)\s*=",
                block,
                re.IGNORECASE,
            )
        )
        if missing_core or not has_primary_locator:
            failures.append(
                f"{key}: missing metadata={missing_core}, primary_locator={has_primary_locator}"
            )
    uncovered = [
        topic for topic, keys in REFERENCE_COVERAGE.items() if not keys.issubset(cited)
    ]
    if uncovered:
        failures.append(f"uncovered topics={uncovered}")
    return AuditCheck(
        "Required reference coverage and provenance",
        "FAIL" if failures else "PASS",
        "; ".join(failures)
        if failures
        else f"All {len(curated)} curated primary-paper/official-release entries cover the dataset, traditional methods, deep learning, advanced studies, and 12 direct software dependencies",
    )


def label_and_reference_keys(tex_text: str) -> tuple[list[str], list[str]]:
    labels = re.findall(r"\\label\s*\{([^}]+)\}", tex_text)
    refs = re.findall(r"\\(?:ref|eqref|autoref)\s*\{([^}]+)\}", tex_text)
    return labels, refs


def _author_identity_state(main_text: str) -> tuple[str, str]:
    author_match = re.search(r"\\author\s*\{(.*?)\}\s*\\begin\{document\}", main_text, re.DOTALL)
    if not author_match:
        return "invalid", "Could not parse the author block"

    author = author_match.group(1)
    identity_text = re.sub(
        r"\\(?:quad|qquad|and)\b|\\\\(?:\[[^]]*\])?",
        " ",
        author,
    )
    identity_text = " ".join(identity_text.split())
    footer = "COMP9517 Computer Vision Group Project"
    if identity_text.endswith(f" {footer}"):
        identity_text = identity_text[: -len(footer)].strip()

    expected_placeholders = " ".join(
        f"TODO Author {index} Name (TODO zID {index})" for index in range(1, 6)
    )
    if re.search(r"\bTODO\b", identity_text, re.IGNORECASE):
        if identity_text == expected_placeholders:
            return "handoff", "Exactly five visible TODO author/zID pairs remain"
        return "invalid", "TODO author identities do not exactly match the five-pair handoff state"

    pair_pattern = re.compile(r"\s*(?P<name>[^()]+?)\s*\(\s*(?P<zid>z\d{7})\s*\)")
    completed: list[tuple[str, str]] = []
    position = 0
    while position < len(identity_text):
        pair_match = pair_pattern.match(identity_text, position)
        if not pair_match:
            return "invalid", "Completed identities must be plain-text names paired with z plus seven digits"
        name = " ".join(pair_match.group("name").split())
        if not name or not any(character.isalpha() for character in name) or not all(
            character.isalpha() or character.isspace() or character in ".-'\N{RIGHT SINGLE QUOTATION MARK}"
            for character in name
        ):
            return "invalid", f"Completed author name is not plain text: {name!r}"
        completed.append((name, pair_match.group("zid")))
        position = pair_match.end()

    if len(completed) != 5:
        return "invalid", f"Expected five completed name/zID pairs; observed {len(completed)}"
    zids = [zid for _, zid in completed]
    if len(set(zids)) != len(zids):
        return "invalid", "Completed author zIDs must be unique"
    return "completed", "Five completed plain-text author names have five unique valid zIDs"


def placeholder_check(main_text: str, *, allow_placeholders: bool) -> AuditCheck:
    identity_state, detail = _author_identity_state(main_text)
    if identity_state == "completed":
        return AuditCheck("Author placeholders", "PASS", detail)
    if identity_state == "handoff":
        status = "WARN" if allow_placeholders else "FAIL"
        return AuditCheck("Author placeholders", status, f"{detail} and must be replaced before submission")
    return AuditCheck("Author placeholders", "FAIL", detail)


def _normalise_title(main_text: str) -> str | None:
    match = re.search(r"\\title\s*\{(.*?)\}\s*\\author", main_text, re.DOTALL)
    if not match:
        return None
    value = match.group(1).replace(r"\\", " ")
    return " ".join(value.split())


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="")
    return buffer.getvalue().encode("utf-8")


def _table_consistency_check(project_root: Path) -> AuditCheck:
    """Rebuild every derived table in memory and verify its bound hashes."""

    from src.analysis.build_final_report_tables import (
        ablation_table_tex,
        build_ablation_results,
        build_main_results,
        build_robustness_summary,
        detailed_test_table_tex,
        main_table_tex,
        result_macros_tex,
        robustness_table_tex,
    )

    source_paths = (
        "results/tables/final_model_comparison.csv",
        "results/tables/stage2_feature_ablation_results.csv",
        "results/tables/stage3_ablation_results.csv",
        "results/tables/stage4_summary.csv",
        "results/tables/robustness_model_ranking.csv",
        "results/tables/robustness_drop_summary.csv",
    )
    output_dir = project_root / "results" / "tables"
    manifest_path = output_dir / "final_report_tables_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = {path: pd.read_csv(project_root / path) for path in source_paths}
        main = build_main_results(frames[source_paths[0]])
        ablation = build_ablation_results(frames[source_paths[1]], frames[source_paths[2]])
        robustness = build_robustness_summary(
            frames[source_paths[3]], frames[source_paths[4]], frames[source_paths[5]]
        )
        expected: dict[str, bytes] = {
            "results/tables/final_report_main_results.csv": _csv_bytes(main),
            "results/tables/final_report_ablation_results.csv": _csv_bytes(ablation),
            "results/tables/final_report_robustness_summary.csv": _csv_bytes(robustness),
            "report/cvpr_latex/generated/result_macros.tex": result_macros_tex(main, robustness).encode("utf-8"),
            "report/cvpr_latex/generated/main_results_table.tex": main_table_tex(main).encode("utf-8"),
            "report/cvpr_latex/generated/detailed_test_table.tex": detailed_test_table_tex(main).encode("utf-8"),
            "report/cvpr_latex/generated/ablation_results_table.tex": ablation_table_tex(ablation).encode("utf-8"),
            "report/cvpr_latex/generated/robustness_summary_table.tex": robustness_table_tex(robustness).encode("utf-8"),
        }
        failures: list[str] = []
        for relative, payload in expected.items():
            path = project_root / relative
            if not path.is_file():
                failures.append(f"missing {relative}")
                continue
            if path.read_bytes() != payload:
                failures.append(f"content mismatch {relative}")
            digest = hashlib.sha256(payload).hexdigest()
            if manifest.get("outputs", {}).get(relative) != digest:
                failures.append(f"output hash mismatch {relative}")
        for relative in source_paths:
            if manifest.get("sources", {}).get(relative) != sha256_file(project_root / relative):
                failures.append(f"source hash mismatch {relative}")
        expected_counts = {
            "main_results": len(main),
            "ablation_results": len(ablation),
            "robustness_summary": len(robustness),
        }
        if manifest.get("row_counts") != expected_counts:
            failures.append("row-count manifest mismatch")
        if set(manifest.get("outputs", {})) != set(expected):
            failures.append("manifest output set mismatch")
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return AuditCheck("Generated tables and source hashes", "FAIL", f"Could not verify: {error}")
    return AuditCheck(
        "Generated tables and source hashes",
        "FAIL" if failures else "PASS",
        "; ".join(failures)
        if failures
        else f"Recomputed 3 CSVs and 5 LaTeX fragments from 6 hash-bound frozen CSV sources",
    )


def _safe_posix_path(value: object) -> bool:
    text = str(value)
    path = PurePosixPath(text)
    return bool(text) and not path.is_absolute() and ".." not in path.parts and "\\" not in text


def _split_contract_check(project_root: Path) -> AuditCheck:
    expected_columns = [
        "split",
        "source_split",
        "image_path",
        "local_image_path",
        "image_id",
        "new_label",
        "original_category_id",
        "scientific_name",
        "common_name",
    ]
    specifications = {
        "train": (20_000, 40, "train_mini"),
        "val": (5_000, 10, "train_mini"),
        "test": (5_000, 10, "val"),
    }
    frames: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    try:
        for split, (rows, per_class, source) in specifications.items():
            frame = pd.read_csv(project_root / "data" / "splits" / f"{split}.csv")
            frames[split] = frame
            if list(frame.columns) != expected_columns:
                failures.append(f"{split}: schema={list(frame.columns)}")
                continue
            if len(frame) != rows:
                failures.append(f"{split}: rows={len(frame)} expected={rows}")
            if set(frame["split"].astype(str)) != {split}:
                failures.append(f"{split}: split values={sorted(set(frame['split'].astype(str)))}")
            if set(frame["source_split"].astype(str)) != {source}:
                failures.append(f"{split}: source_split values={sorted(set(frame['source_split'].astype(str)))}")
            labels = pd.to_numeric(frame["new_label"], errors="raise").astype(int)
            if set(labels) != set(range(500)):
                failures.append(f"{split}: labels are not exactly 0..499")
            counts = labels.value_counts()
            if len(counts) != 500 or not counts.eq(per_class).all():
                failures.append(f"{split}: per-class counts are not {per_class}")
            if frame.isna().any().any():
                failures.append(f"{split}: null values present")
            if not frame["local_image_path"].map(_safe_posix_path).all():
                failures.append(f"{split}: unsafe local_image_path")
            if not frame["local_image_path"].astype(str).str.startswith(f"{split}/").all():
                failures.append(f"{split}: local_image_path prefix mismatch")
        for column in ("image_path", "local_image_path", "image_id"):
            sets = {name: set(frame[column].astype(str)) for name, frame in frames.items() if column in frame}
            for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
                overlap = sets.get(left, set()) & sets.get(right, set())
                if overlap:
                    failures.append(f"{column}: {left}/{right} overlap={len(overlap)}")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        failures.append(str(error))
    return AuditCheck(
        "Frozen split schema and counts",
        "FAIL" if failures else "PASS",
        "; ".join(failures)
        if failures
        else "Schema-v2 new_label manifests are 20,000/5,000/5,000 with 40/10/10 images per each of 500 classes and disjoint records",
    )


def _test_isolation_check(project_root: Path, report_text: str) -> AuditCheck:
    failures: list[str] = []
    try:
        stage3 = pd.read_csv(project_root / "results" / "tables" / "stage3_deep_learning_results.csv")
        main = pd.read_csv(project_root / "results" / "tables" / "final_report_main_results.csv")
        ablation = pd.read_csv(project_root / "results" / "tables" / "final_report_ablation_results.csv")
        test_columns = [
            "test_top1",
            "test_top5",
            "test_macro_precision",
            "test_macro_recall",
            "test_macro_f1",
            "test_time_seconds",
        ]
        validation_only = {
            "resnet18_pretrained_basic",
            "resnet18_pretrained_frozen_head",
        }
        indexed = stage3.set_index("model_name", verify_integrity=True)
        for name in validation_only:
            if name not in indexed.index:
                failures.append(f"missing validation-only row {name}")
            elif not indexed.loc[name, test_columns].isna().all():
                failures.append(f"validation-only test fields populated for {name}")
        selected = set(indexed.index) - validation_only
        if selected != {
            "resnet18_scratch_basic",
            "resnet18_pretrained_strong_aug",
            "efficientnet_b0_pretrained_strong_aug",
        }:
            failures.append(f"unexpected Stage 3 selected set {sorted(selected)}")
        elif indexed.loc[sorted(selected), test_columns].isna().any().any():
            failures.append("selected Stage 3 test fields are incomplete")
        if any(column.startswith("test_") for column in ablation.columns):
            failures.append("validation-only ablation table contains test columns")
        if main[test_columns].isna().any().any():
            failures.append("main comparison has missing required held-out test values")
        if "validation-only" not in report_text.lower():
            failures.append("report does not qualify ablations as validation-only")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        failures.append(str(error))
    return AuditCheck(
        "Validation/test isolation",
        "FAIL" if failures else "PASS",
        "; ".join(failures)
        if failures
        else "Two validation-only Stage 3 rows retain blank test fields; three frozen Stage 3 and six main-table rows have complete test metrics; ablations expose no test columns",
    )


def _wording_checks(report_text: str) -> list[AuditCheck]:
    lower = " ".join(report_text.lower().split())
    stage4_phrases = (
        "robustness and grad-cam operate only after this clean-test evaluation and never feed changes back into a model",
        "no model is retrained or adapted",
    )
    missing_stage4 = [phrase for phrase in stage4_phrases if phrase not in lower]
    gradcam_phrases = (
        "not causal explanations",
        "do not prove causal feature",
    )
    missing_gradcam = [phrase for phrase in gradcam_phrases if phrase not in lower]
    return [
        AuditCheck(
            "Stage 4 non-selection wording",
            "FAIL" if missing_stage4 else "PASS",
            f"Missing required wording: {missing_stage4}"
            if missing_stage4
            else "Report states that post-test robustness/Grad-CAM cannot select, retrain, or adapt models",
        ),
        AuditCheck(
            "Grad-CAM non-causal wording",
            "FAIL" if missing_gradcam or "grad-cam proves" in lower else "PASS",
            f"Missing={missing_gradcam}; forbidden_proof_claim={'grad-cam proves' in lower}"
            if missing_gradcam or "grad-cam proves" in lower
            else "Report labels Grad-CAM as qualitative diagnostic evidence rather than causal proof",
        ),
    ]


def _traditional_rewrite_contract_check(report_text: str) -> AuditCheck:
    """Reject stale Stage 2 prose and require the rewritten method contract."""

    normalised = report_text.lower().replace("{,}", ",").replace("~", " ")
    normalised = " ".join(normalised.split())
    required_patterns = {
        "3,704-D primary LinearSVC role": (
            r"\bprimary\b.{0,160}\b3,?704\s*-\s*d\b.{0,80}\blinearsvc\b"
        ),
        "2,424-D Random Forest robustness role": (
            r"\b2,?424\s*-\s*d\b.{0,140}\brandom\s+forest\b"
            r".{0,180}\brobustness\s+comparator\b"
        ),
        "1,280-D BoVW block": r"\b1,?280\s*-\s*d\b",
        "top-3 soft assignment": r"\btop\s*-\s*3\s+soft\s+assignment\b",
        "19-configuration validation grid": (
            r"\b19(?:\s*-\s*configuration|\s+(?:recorded\s+)?validation\s+configurations?)\b"
        ),
        "300-tree Random Forest": r"\b300(?:\s*-\s*tree|\s+trees?)\b",
    }
    forbidden_patterns = {
        "stale 1,830-D descriptor": r"\b1,?830\s*-\s*d\b",
        "retired hinge-SGD model": r"\bhinge\s*-\s*sgd\b",
        "retired SGDClassifier backend": r"\bsgdclassifier\b",
        "stale 200-tree Random Forest": r"\b200(?:\s*-\s*tree|\s+trees?)\b",
    }
    missing = [
        name
        for name, pattern in required_patterns.items()
        if not re.search(pattern, normalised)
    ]
    stale = [
        name
        for name, pattern in forbidden_patterns.items()
        if re.search(pattern, normalised)
    ]
    return AuditCheck(
        "Traditional rewrite method contract",
        "FAIL" if missing or stale else "PASS",
        f"missing={missing}; stale={stale}"
        if missing or stale
        else (
            "Report binds the 3,704-D primary LinearSVC, 2,424-D/300-tree "
            "Random Forest robustness comparator, 1,280-D top-3 soft-assigned "
            "BoVW block, and 19-configuration validation grid with no retired contract"
        ),
    )


def _requirements_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[]", value, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def _manifest_outputs(figure_dir: Path) -> tuple[dict[str, dict[str, object]], str | None]:
    path = figure_dir / "figure_manifest.json"
    if not path.is_file():
        return {}, f"Missing {path}"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        outputs = manifest["outputs"]
        if not isinstance(outputs, list):
            raise TypeError("outputs is not a list")
        return {str(row["file"]): row for row in outputs}, None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return {}, f"Invalid figure manifest: {error}"


def _figure_checks(tex_text: str, figure_dir: Path) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    includes = re.findall(r"\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}", tex_text)
    outputs, error = _manifest_outputs(figure_dir)
    if error:
        return [AuditCheck("Figure manifest", "FAIL", error)]
    missing: list[str] = []
    stale: list[str] = []
    unbound: list[str] = []
    for include in includes:
        candidates = [include] if Path(include).suffix else [include + suffix for suffix in (".pdf", ".png", ".jpg", ".jpeg")]
        matched = next((name for name in candidates if (figure_dir / name).is_file()), None)
        if matched is None:
            missing.append(include)
            continue
        if matched not in outputs:
            unbound.append(matched)
            continue
        if sha256_file(figure_dir / matched) != outputs[matched].get("sha256"):
            stale.append(matched)
    if missing:
        checks.append(AuditCheck("Referenced figures exist", "FAIL", f"Missing: {sorted(missing)}"))
    else:
        checks.append(AuditCheck("Referenced figures exist", "PASS", f"Resolved {len(includes)} includegraphics commands"))
    if unbound or stale:
        checks.append(AuditCheck("Referenced figure hashes", "FAIL", f"Unbound={sorted(unbound)} stale={sorted(stale)}"))
    else:
        checks.append(AuditCheck("Referenced figure hashes", "PASS", "Every referenced figure is bound to the current manifest hash"))
    required_visuals = {
        "method_overview.pdf",
        "robustness_curves.png",
        "gradcam_correct_examples.jpg",
        "gradcam_incorrect_examples.jpg",
        "gradcam_confused_pairs.jpg",
    }
    absent = sorted(required_visuals - set(includes))
    checks.append(
        AuditCheck(
            "Required report visuals",
            "FAIL" if absent else "PASS",
            f"Missing includes: {absent}" if absent else "Vector pipeline, robustness, and all three Grad-CAM categories are included",
        )
    )
    return checks


def _claim_checks(report_text: str, project_root: Path) -> list[AuditCheck]:
    failures: list[str] = []
    cache: dict[Path, pd.DataFrame] = {}
    unavailable: set[Path] = set()
    for claim in OBSERVED_CLAIMS:
        source = project_root / claim.relative_source
        if source in unavailable:
            continue
        if source not in cache:
            try:
                cache[source] = pd.read_csv(source)
            except (OSError, TypeError, ValueError, pd.errors.ParserError) as error:
                failures.append(f"{claim.relative_source}: could not read source: {error}")
                unavailable.add(source)
                continue
        frame = cache[source]
        missing_columns = [
            column
            for column in ("model_name", claim.column)
            if column not in frame.columns
        ]
        if missing_columns:
            failures.append(
                f"{claim.relative_source}: missing columns {missing_columns}"
            )
            unavailable.add(source)
            continue
        row = frame[frame["model_name"] == claim.model_name]
        if len(row) != 1:
            failures.append(f"{claim.macro_name}: source row count={len(row)}")
            continue
        observed = f"{float(row.iloc[0][claim.column]) * 100:.{claim.decimals}f}"
        invocation = f"\\{claim.macro_name}"
        definition = f"\\newcommand{{{invocation}}}{{{observed}}}"
        if definition not in report_text:
            failures.append(
                f"{claim.macro_name}: generated definition does not match {observed}"
            )
        elif report_text.count(invocation) < 2:
            failures.append(f"{claim.macro_name}: macro is defined but not used")
    return [
        AuditCheck(
            "Observed headline claims",
            "FAIL" if failures else "PASS",
            "; ".join(failures) if failures else f"Verified {len(OBSERVED_CLAIMS)} claims against frozen CSV values",
        )
    ]


def audit_report(
    tex_dir: Path = DEFAULT_TEX_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    *,
    project_root: Path = PROJECT_ROOT,
    allow_placeholders: bool = False,
    verify_frozen: bool = True,
) -> list[AuditCheck]:
    checks: list[AuditCheck] = []
    missing_files = [name for name in REQUIRED_FILES if not (tex_dir / name).is_file()]
    checks.append(
        AuditCheck(
            "Required report sources",
            "FAIL" if missing_files else "PASS",
            f"Missing: {missing_files}" if missing_files else f"All {len(REQUIRED_FILES)} required files exist",
        )
    )
    if missing_files:
        return checks

    sources = _tex_sources(tex_dir)
    report_text = "\n".join(sources.values())
    main_text = sources[tex_dir / "main.tex"]
    preamble_text = sources[tex_dir / "preamble.tex"]

    exact_mode = main_text.count(r"\usepackage[pagenumbers]{cvpr}") == 1
    forbidden_mode = bool(re.search(r"\\usepackage\s*\[(?:review|rebuttal)\]\s*\{cvpr\}|\\paperID", main_text))
    checks.append(
        AuditCheck(
            "CVPR submission mode",
            "PASS" if exact_mode and not forbidden_mode else "FAIL",
            "Exact pagenumbers mode; no review/rebuttal/paperID" if exact_mode and not forbidden_mode else "Required exact mode missing or forbidden review metadata present",
        )
    )
    provenance_text = (tex_dir / "OFFICIAL_TEMPLATE_SOURCE.md").read_text(encoding="utf-8")
    provenance_failures: list[str] = []
    if OFFICIAL_ARCHIVE_HASH not in provenance_text:
        provenance_failures.append("official archive hash absent from provenance record")
    for filename, expected_hash in OFFICIAL_TEMPLATE_HASHES.items():
        observed_hash = sha256_file(tex_dir / filename)
        if observed_hash != expected_hash:
            provenance_failures.append(f"{filename}={observed_hash}, expected={expected_hash}")
        if expected_hash not in provenance_text:
            provenance_failures.append(f"{filename} hash absent from provenance record")
    checks.append(
        AuditCheck(
            "Official CVPR 2026 template provenance",
            "FAIL" if provenance_failures else "PASS",
            "; ".join(provenance_failures)
            if provenance_failures
            else "Official kit archive provenance and byte-identical CVPR style/BibTeX assets are hash-bound",
        )
    )
    observed_title = _normalise_title(main_text)
    checks.append(
        AuditCheck(
            "Exact report title",
            "PASS" if observed_title == EXPECTED_TITLE else "FAIL",
            f"Exact required title: {EXPECTED_TITLE}"
            if observed_title == EXPECTED_TITLE
            else f"Observed={observed_title!r}; expected={EXPECTED_TITLE!r}",
        )
    )
    observed_sections = re.findall(r"\\section\{([^}]+)\}", report_text)
    missing_sections = [name for name in REQUIRED_SECTION_NAMES if name not in observed_sections]
    checks.append(
        AuditCheck(
            "Required report sections",
            "FAIL" if missing_sections else "PASS",
            f"Missing: {missing_sections}" if missing_sections else "All rubric sections are present",
        )
    )
    checks.append(placeholder_check(main_text, allow_placeholders=allow_placeholders))

    identity_state, identity_detail = _author_identity_state(main_text)
    valid_identity = identity_state in {"handoff", "completed"}
    identity_description = (
        "the exact five-pair TODO handoff"
        if identity_state == "handoff"
        else "five completed name/unique-zID pairs"
    )
    checks.append(
        AuditCheck(
            "Report identity contract",
            "PASS" if observed_title == EXPECTED_TITLE and valid_identity else "FAIL",
            f"Exact title plus {identity_description}"
            if observed_title == EXPECTED_TITLE and valid_identity
            else f"title_ok={observed_title == EXPECTED_TITLE} identity_state={identity_state}: {identity_detail}",
        )
    )

    author_match = re.search(r"\\author\s*\{.*?\}\s*\\begin\{document\}", main_text, re.DOTALL)
    main_without_author = main_text[: author_match.start()] + main_text[author_match.end() :] if author_match else main_text
    todo_locations = [str(path.relative_to(tex_dir)) for path, text in sources.items() if "TODO" in (main_without_author if path == tex_dir / "main.tex" else text)]
    checks.append(
        AuditCheck(
            "Non-author placeholders",
            "FAIL" if todo_locations else "PASS",
            f"TODO found in: {todo_locations}" if todo_locations else "No report-content placeholders",
        )
    )

    barrier = main_text.find(r"\FloatBarrier")
    last_page = main_text.find(r"\label{LastPageMainText}")
    clear_page = main_text.find(r"\clearpage", last_page)
    bibliography = main_text.find(r"\bibliography", last_page)
    float_order_ok = -1 < barrier < last_page < clear_page < bibliography
    checks.append(
        AuditCheck(
            "Main-text page boundary",
            "PASS" if float_order_ok and r"\usepackage{placeins}" in preamble_text else "FAIL",
            "FloatBarrier -> LastPageMainText -> clearpage -> bibliography" if float_order_ok else "Deferred floats or references may be miscounted",
        )
    )

    bib_text = (tex_dir / "references.bib").read_text(encoding="utf-8")
    cited = citation_keys(report_text)
    bib_keys = bibliography_keys(bib_text)
    duplicate_bib = sorted({key for key in bib_keys if bib_keys.count(key) > 1})
    missing_citations = sorted(cited - set(bib_keys))
    unused_bib = sorted(set(bib_keys) - cited)
    checks.append(
        AuditCheck(
            "Citation/reference consistency",
            "FAIL" if duplicate_bib or missing_citations or unused_bib else "PASS",
            f"duplicates={duplicate_bib} missing={missing_citations} unused={unused_bib}" if duplicate_bib or missing_citations or unused_bib else f"{len(cited)} cited keys map one-to-one to bibliography entries",
        )
    )
    checks.append(_reference_coverage_check(cited, bib_text))

    labels, refs = label_and_reference_keys(report_text)
    duplicate_labels = sorted({key for key in labels if labels.count(key) > 1})
    missing_refs = sorted(set(refs) - set(labels))
    checks.append(
        AuditCheck(
            "LaTeX labels/references",
            "FAIL" if duplicate_labels or missing_refs else "PASS",
            f"duplicate labels={duplicate_labels} missing refs={missing_refs}" if duplicate_labels or missing_refs else f"{len(set(labels))} unique labels; every reference resolves statically",
        )
    )

    requirements = _requirements_names(project_root / "requirements.txt")
    unmapped = sorted(requirements - set(DIRECT_DEPENDENCY_CITATIONS))
    uncited_dependencies = sorted(
        name for name, key in DIRECT_DEPENDENCY_CITATIONS.items() if name in requirements and key not in cited
    )
    checks.append(
        AuditCheck(
            "Direct dependency citations",
            "FAIL" if unmapped or uncited_dependencies else "PASS",
            f"unmapped={unmapped} uncited={uncited_dependencies}" if unmapped or uncited_dependencies else f"All {len(requirements)} pinned direct dependencies are cited",
        )
    )
    known_bad_metadata = [value for value in ("10.5281/zenodo.592536", "10.5281/zenodo.44297") if value in bib_text]
    checks.append(
        AuditCheck(
            "Known software-reference metadata",
            "FAIL" if known_bad_metadata else "PASS",
            f"Misattributed historical DOI(s): {known_bad_metadata}" if known_bad_metadata else "Joblib and Pillow use version-matched release records",
        )
    )

    checks.extend(_figure_checks(report_text, figure_dir))
    checks.append(_table_consistency_check(project_root))
    checks.append(_split_contract_check(project_root))
    checks.append(_test_isolation_check(project_root, report_text))
    checks.extend(_wording_checks(report_text))
    checks.append(_traditional_rewrite_contract_check(report_text))
    checks.extend(_claim_checks(report_text, project_root))

    forbidden_claims = (
        "Validation controls all selection decisions",
        "pretraining causes",
        "Grad-CAM proves",
        "least damaging corruption at mild and moderate levels",
        "worst high-severity condition for every model",
    )
    present_forbidden = [phrase for phrase in forbidden_claims if phrase.lower() in report_text.lower()]
    checks.append(
        AuditCheck(
            "Causal/protocol wording guard",
            "FAIL" if present_forbidden else "PASS",
            f"Forbidden overclaims: {present_forbidden}" if present_forbidden else "No known causal or protocol overclaim pattern",
        )
    )

    if verify_frozen:
        frozen_failures = [row for row in verify_frozen_artifacts() if not row["passed"]]
        checks.append(
            AuditCheck(
                "Frozen Stage 1-4 evidence",
                "FAIL" if frozen_failures else "PASS",
                "; ".join(f"{row['path']} changed" for row in frozen_failures) if frozen_failures else "Every contracted evidence hash matches",
            )
        )
    else:
        checks.append(AuditCheck("Frozen Stage 1-4 evidence", "WARN", "Skipped by explicit development/test option"))
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit final CVPR report sources against frozen Stage 1-4 evidence.")
    parser.add_argument("--tex-dir", type=Path, default=DEFAULT_TEX_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Development only: downgrade the five author/zID placeholders from FAIL to WARN.",
    )
    parser.add_argument("--skip-frozen-verification", action="store_true", help="Development/tests only.")
    parser.add_argument("--reference-report", type=Path, default=DEFAULT_REFERENCE_REPORT)
    parser.add_argument("--consistency-report", type=Path, default=DEFAULT_CONSISTENCY_REPORT)
    parser.add_argument(
        "--no-write-reports",
        action="store_true",
        help="Print checks without replacing the two canonical Markdown audit reports.",
    )
    return parser


def _audit_lines(checks: Sequence[AuditCheck]) -> list[str]:
    lines = [f"[{check.status}] {check.name}: {check.detail}" for check in checks]
    failures = sum(check.status == "FAIL" for check in checks)
    warnings = sum(check.status == "WARN" for check in checks)
    passes = len(checks) - failures - warnings
    lines.append(f"[SUMMARY] checks={len(checks)} pass={passes} warn={warnings} fail={failures}")
    return lines


def _overall_status(checks: Sequence[AuditCheck]) -> str:
    if any(check.status == "FAIL" for check in checks):
        return "FAIL"
    if any(check.status == "WARN" for check in checks):
        return "PASS WITH WARNINGS"
    return "PASS"


def _markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def _render_audit_report(
    title: str,
    scoped_checks: Sequence[AuditCheck],
    all_checks: Sequence[AuditCheck],
    *,
    command: str,
) -> str:
    rows = "\n".join(
        f"| {check.status} | {_markdown_cell(check.name)} | {_markdown_cell(check.detail)} |"
        for check in scoped_checks
    )
    exact_output = "\n".join(_audit_lines(all_checks))
    return f"""# {title}

Overall result: **{_overall_status(scoped_checks)}**

This audit is generated atomically from the report sources and frozen Stage 1--4
evidence.  It performs no training or inference.

## Command

Working directory: `{PROJECT_ROOT}`

```text
$ {command}
```

## Checks

| Status | Check | Detail |
|---|---|---|
{rows}

## Exact command output

```text
{exact_output}
```
"""


def _atomic_write_text(path: Path, content: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    cli_args = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(cli_args)
    checks = audit_report(
        args.tex_dir.resolve(),
        args.figure_dir.resolve(),
        project_root=PROJECT_ROOT,
        allow_placeholders=args.allow_placeholders,
        verify_frozen=not args.skip_frozen_verification,
    )
    lines = _audit_lines(checks)
    for line in lines:
        print(line)
    command = shlex.join([sys.executable, sys.argv[0], *cli_args])
    if not args.no_write_reports:
        reference_checks = [check for check in checks if check.name in REFERENCE_CHECK_NAMES]
        consistency_checks = [check for check in checks if check.name in CONSISTENCY_CHECK_NAMES]
        _atomic_write_text(
            args.reference_report,
            _render_audit_report(
                "Report Reference Audit", reference_checks, checks, command=command
            ),
        )
        _atomic_write_text(
            args.consistency_report,
            _render_audit_report(
                "Report Consistency Audit", consistency_checks, checks, command=command
            ),
        )
    failures = sum(check.status == "FAIL" for check in checks)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
