"""Frozen Stage 1--4 evidence contract used by Stage 5 report tooling.

Stage 5 may derive report tables and copies of figures, but it must never
silently consume changed experiment evidence.  The hashes below are the final
accepted values observed after Stage 4 completed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd

from src.data.common import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FrozenArtifact:
    stage: str
    artifact_type: str
    relative_path: str
    sha256: str

    @property
    def path(self) -> Path:
        return (PROJECT_ROOT / self.relative_path).resolve()


STAGE3_FREEZE_RELATIVE_PATH = "results/metrics/deep_learning/stage3_freeze.json"
STAGE3_TEST_ACCESS_RELATIVE_PATH = (
    "results/metrics/deep_learning/stage3_test_access.json"
)
STAGE3_CONTINUITY_RELATIVE_PATH = (
    "results/metrics/deep_learning/stage3_traditional_rewrite_continuity.json"
)
STAGE3_FREEZE_SHA256 = (
    "bea68b307fb7ff5db4f066b47ff00ade462bd71609ed9735c4faff125ce2257f"
)
STAGE3_TEST_ACCESS_SHA256 = (
    "04420ab5eb9bca3f5a6e56f6015fd2ce158ba9121d0e328b076912bac2292f2b"
)
STAGE3_CONTINUITY_SHA256 = (
    "dfab994d6d351341fe96b93e504fa47c8049bf371bee7f8bc3f1b2e674462528"
)


@dataclass(frozen=True)
class LegacyStage2ContinuityBinding:
    """One exact historical Stage 2 binding retained in Stage 3's freeze."""

    location: str
    artifact_path: str
    legacy_sha256: str
    legacy_bytes: int
    current_sha256: str
    current_bytes: int
    disposition: str


LEGACY_STAGE2_CONTINUITY_BINDINGS: tuple[LegacyStage2ContinuityBinding, ...] = (
    LegacyStage2ContinuityBinding(
        location="$.stage2_artifacts.STAGE2_FINAL_ACCEPTANCE_REPORT.md",
        artifact_path="STAGE2_FINAL_ACCEPTANCE_REPORT.md",
        legacy_sha256="a2a6aa5df577b3d56da5be1a3329227250d15e978b2f3a912f573b90a014c6d9",
        legacy_bytes=30_903,
        current_sha256="dea1c7a46e8c48609906c118c6c6e8fe8ae68412025da7dd8db4a764c5890759",
        current_bytes=694,
        disposition="superseded_historical_report_notice",
    ),
    LegacyStage2ContinuityBinding(
        location="$.stage2_artifacts.STAGE2_TRADITIONAL_REPORT.md",
        artifact_path="STAGE2_TRADITIONAL_REPORT.md",
        legacy_sha256="874130190f8a9f976e84009e0b1459d268626361e3e38382c935de66e51ac9e6",
        legacy_bytes=30_583,
        current_sha256="39426c5a8c31735d61b8e0ad3a04a434b0347842f63613671fd1a6450799d1c7",
        current_bytes=690,
        disposition="superseded_historical_report_notice",
    ),
    LegacyStage2ContinuityBinding(
        location="$.stage2_artifacts.results/metrics/stage2_freeze.json",
        artifact_path="results/metrics/stage2_freeze.json",
        legacy_sha256="6f959c8b1253616e04a549d6063ce71528dc9872cd865435a7310866c8f1e92f",
        legacy_bytes=3_556,
        current_sha256="f5b908ce68b0cfb95b3047c2ba5a875fb975e56026b6337c966d5c326140562d",
        current_bytes=16_287,
        disposition="regenerated_stage2_evidence",
    ),
    LegacyStage2ContinuityBinding(
        location=(
            "$.stage2_artifacts.results/tables/"
            "stage2_feature_ablation_results.csv"
        ),
        artifact_path="results/tables/stage2_feature_ablation_results.csv",
        legacy_sha256="a919aaeca2ae28f0fd6f5a0e7f51c2a6ec0dbf4faac6069c5deb84ea9e80afb6",
        legacy_bytes=708,
        current_sha256="ba723a0a362bd50a569a12bbcb6f56578ed37f1a6f351daed793c74ad692e2ce",
        current_bytes=705,
        disposition="regenerated_stage2_evidence",
    ),
    LegacyStage2ContinuityBinding(
        location="$.stage2_artifacts.results/tables/stage2_traditional_results.csv",
        artifact_path="results/tables/stage2_traditional_results.csv",
        legacy_sha256="cfdcff1a73ef8dbf06083ce296469c08bd1a5c034902b00526d0f63241d89c56",
        legacy_bytes=907,
        current_sha256="b2a2d554975c06e28639a39d44b80e120e36dc5eb8e23bc0d75715a8ac15768c",
        current_bytes=934,
        disposition="regenerated_stage2_evidence",
    ),
)


# These values must change only through a new, explicitly accepted evidence
# freeze.  Updating a report must not update this mapping as a side effect.
FROZEN_ARTIFACTS: tuple[FrozenArtifact, ...] = (
    FrozenArtifact("stage1", "acceptance_report", "STAGE1_ACCEPTANCE_REPORT.md", "e60310de7682a7c8b7c9092671bd4ae59251a962c98f41950dc8139f9bdaed9a"),
    FrozenArtifact("stage1", "fix_report", "STAGE1_FIX_REPORT.md", "a4e489e4c8901c9e3718385d0d05f5c1e13c0c560f762d653ff8b00a9b79dfdc"),
    FrozenArtifact("stage1", "selected_classes", "data/splits/selected_classes_seed42.txt", "3c2af417a5c174ef14f2be117a00880b0a08f8dce0c24fad6aaa8ea5dcde8141"),
    FrozenArtifact("stage1", "class_mapping", "data/splits/class_mapping.json", "2d6e60104b133d2758dffd585d373b085ddb42845d0b4b2c93bf5277ef9a0a1b"),
    FrozenArtifact("stage1", "train_manifest", "data/splits/train.csv", "d8f4abccc568484f6efaff243c409b8ddcbc8fe46d2146a6922230b4428706c7"),
    FrozenArtifact("stage1", "validation_manifest", "data/splits/val.csv", "4adf287996d35d9e57f829426ff7480bbaf0b74e1c319a29d57fe74b3f0366bb"),
    FrozenArtifact("stage1", "test_manifest", "data/splits/test.csv", "45ba7673b6b0e27cab8ef5a482ea8142de54e35be05a5b1ba9bb3fe20d5b8f91"),
    FrozenArtifact("stage1", "split_summary", "data/splits/split_summary.json", "324413fb3d8d3affe611f8fd1454358cf54f2e6fcd81ef961994be0cc19f0b76"),
    FrozenArtifact("stage2", "results_table", "results/tables/stage2_traditional_results.csv", "b2a2d554975c06e28639a39d44b80e120e36dc5eb8e23bc0d75715a8ac15768c"),
    FrozenArtifact("stage2", "ablation_table", "results/tables/stage2_feature_ablation_results.csv", "ba723a0a362bd50a569a12bbcb6f56578ed37f1a6f351daed793c74ad692e2ce"),
    FrozenArtifact("stage2", "freeze", "results/metrics/stage2_freeze.json", "f5b908ce68b0cfb95b3047c2ba5a875fb975e56026b6337c966d5c326140562d"),
    FrozenArtifact("stage3", "report", "STAGE3_DEEP_LEARNING_REPORT.md", "06d7670ee4d9c627e26912119fa325dd8b7f2de4c383a2926b33f4eb8533a18c"),
    FrozenArtifact("stage3", "results_table", "results/tables/stage3_deep_learning_results.csv", "5f5434e5a571f9e5f033594cdb25031483ccd6f2d3efe8e8166bdbc5038a911d"),
    FrozenArtifact("stage3", "ablation_table", "results/tables/stage3_ablation_results.csv", "79cf56f87fef800632adbaec8d9a4ab80686ff4f89db75dc25321cd47fc35b93"),
    FrozenArtifact("stage3", "comparison_table", "results/tables/final_model_comparison.csv", "322f88ce28f9c919b09a068ee55f9c87b537e4d36c9995994386f618d3ab39d5"),
    FrozenArtifact("stage3", "freeze", STAGE3_FREEZE_RELATIVE_PATH, STAGE3_FREEZE_SHA256),
    FrozenArtifact("stage3", "test_access", STAGE3_TEST_ACCESS_RELATIVE_PATH, STAGE3_TEST_ACCESS_SHA256),
    FrozenArtifact("stage3", "traditional_rewrite_continuity", STAGE3_CONTINUITY_RELATIVE_PATH, STAGE3_CONTINUITY_SHA256),
    FrozenArtifact("stage4", "report", "STAGE4_TRADITIONAL_REWRITE_RERUN_REPORT.md", "49c4c7132b0ea1cb62d0a842badabdc2a704c91f861f4e37d8b981ea142c1381"),
    FrozenArtifact("stage4", "protocol_freeze", "results/metrics/stage4_robustness_protocol.json", "0dcbc3f7d7380367cb2a9522ee28e4dc2239307575fe57892f13556a607f5233"),
    FrozenArtifact("stage4", "robustness_results", "results/tables/robustness_results.csv", "0ddd20fc88adcf801b06e029c7ce7b9c088dc772055854288a1cfdbfd571d481"),
    FrozenArtifact("stage4", "robustness_drops", "results/tables/robustness_drop_summary.csv", "81d962333b1a114e638c1709a645f26efb68811695959cf86c00e0f892a4b018"),
    FrozenArtifact("stage4", "robustness_ranking", "results/tables/robustness_model_ranking.csv", "d2677b44bdd21c540fbb67287ad61bb6a6a5a53669bd43f11e2fb102d0e89245"),
    FrozenArtifact("stage4", "gradcam_selection", "results/tables/gradcam_selected_examples.csv", "564ff298161eaf7068e10e57137b2ade7da33edca295aacf0e985e2b1073a705"),
    FrozenArtifact("stage4", "gradcam_analysis", "results/tables/gradcam_analysis.md", "005e3155f635f5b3c7340d817833771f31aaf124302d09c78bb2b3cec7821c9a"),
    FrozenArtifact("stage4", "gradcam_selection_manifest", "results/metrics/explainability/gradcam_selection_manifest.json", "15eb43c7bc375f2e61850801b58e7076ba6684bc4bb6cbe18640ec182e05b4b7"),
    FrozenArtifact("stage4", "gradcam_manifest", "results/metrics/explainability/gradcam_manifest.json", "ac43697cf36e5dcc340db21514477e9dc812c9b2ca9e2df90e1e18734968b8fa"),
    FrozenArtifact("stage4", "summary_table", "results/tables/stage4_summary.csv", "7d7bd7b960ec43714cfbff1238f5dbc7b7b13ef00e1f0fcdf6d05a8d98741bf9"),
    FrozenArtifact("stage4", "comparison_table", "results/tables/final_model_comparison_with_robustness.csv", "ac021fd4a00b7e858a1785b955081b6f6239484778359bf19ebb7292866c3cbb"),
    FrozenArtifact("stage4", "failure_table", "results/tables/stage4_hardest_classes.csv", "a115ade2d9c3db8d1429440e9199afac300685beaf6f556787134d03ac383474"),
    FrozenArtifact("stage4", "failure_table", "results/tables/stage4_easiest_classes.csv", "0268c0b59f14518926fbd25205c8707640c29f29181f85a3a80d11d1d8f426aa"),
    FrozenArtifact("stage4", "failure_table", "results/tables/stage4_high_confidence_failures.csv", "d8b94cc1e347e354e2ee477fa4142480822eeb09acb0c6671289126a5e4a0252"),
    FrozenArtifact("stage4", "failure_table", "results/tables/stage4_low_confidence_correct.csv", "185948ce101811e9137847e6474c350549d802b8dfcfb4329d039a48cc4ca99e"),
    FrozenArtifact("stage4", "failure_table", "results/tables/stage4_confused_species_pairs.csv", "3884a6562905e790eeb9e269295a29d59b5e5312ac34abe4beef16956c1dba9c"),
)

FROZEN_BY_PATH: dict[str, FrozenArtifact] = {
    artifact.relative_path: artifact for artifact in FROZEN_ARTIFACTS
}

NESTED_BINDING_MANIFESTS: tuple[str, ...] = (
    "results/metrics/stage2_freeze.json",
    STAGE3_FREEZE_RELATIVE_PATH,
    STAGE3_TEST_ACCESS_RELATIVE_PATH,
    STAGE3_CONTINUITY_RELATIVE_PATH,
    "results/metrics/stage4_robustness_protocol.json",
    "results/metrics/explainability/gradcam_selection_manifest.json",
    "results/metrics/explainability/gradcam_manifest.json",
)

MAIN_SOURCE_COLUMNS = (
    "model_name", "method_family", "architecture", "pretrained", "augmentation",
    "freeze_backbone", "train_top1_best", "val_top1", "val_top5",
    "val_macro_precision", "val_macro_recall", "val_macro_f1", "test_top1",
    "test_top5", "test_macro_precision", "test_macro_recall", "test_macro_f1",
    "training_time_seconds", "test_time_seconds", "num_parameters",
    "trainable_parameters", "notes",
)

MAIN_SOURCE_MODELS = (
    "linear_svm", "random_forest", "sift_bovw_svm",
    "efficientnet_b0_pretrained_strong_aug", "resnet18_pretrained_basic",
    "resnet18_pretrained_frozen_head", "resnet18_pretrained_strong_aug",
    "resnet18_scratch_basic",
)

MAIN_REPORT_MODELS = (
    "linear_svm", "random_forest", "sift_bovw_svm", "resnet18_scratch_basic",
    "resnet18_pretrained_strong_aug", "efficientnet_b0_pretrained_strong_aug",
)

STAGE4_PROTOCOL_MODELS = (
    "traditional_random_forest", "resnet18_scratch_basic",
    "resnet18_pretrained_strong_aug", "efficientnet_b0_pretrained_strong_aug",
)

# Closed evidence trees created by Stages 1--4.  Stage 5 may read these files
# but may not rewrite them.  The before/after snapshot deliberately covers more
# than the accepted-hash and nested-manifest subsets above: it also binds every
# robustness cell, prediction, checkpoint, feature cache, figure, and log that
# existed when Stage 5 began.
READ_ONLY_EVIDENCE_ROOTS: tuple[str, ...] = (
    "data/splits",
    "configs",
    "results/metrics",
    "results/predictions",
    "results/models",
    "results/features",
    "results/figures",
    "results/logs",
    "results/tables",
)

READ_ONLY_REPORTS: tuple[str, ...] = (
    "DATASET_SMOKE_TEST_REPORT.md",
    "DEEP_LEARNING_SMOKE_TEST_REPORT.md",
    "STAGE1_ACCEPTANCE_REPORT.md",
    "STAGE1_FIX_REPORT.md",
    "STAGE2_FINAL_ACCEPTANCE_REPORT.md",
    "STAGE2_TRADITIONAL_REPORT.md",
    "STAGE3_DEEP_LEARNING_REPORT.md",
    "STAGE3_PRECHECK_REPORT.md",
    "STAGE4_ADVANCED_ANALYSIS_REPORT.md",
    "STAGE4_PRECHECK_REPORT.md",
    "TRADITIONAL_REWRITE_INTEGRATION_AUDIT.md",
    "STAGE4_TRADITIONAL_REWRITE_RERUN_REPORT.md",
)

STAGE5_DERIVED_EVIDENCE_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "results/metrics/stage5_frozen_hashes_before.json",
        "results/metrics/stage5_frozen_hashes_after.json",
        "results/tables/final_report_main_results.csv",
        "results/tables/final_report_ablation_results.csv",
        "results/tables/final_report_robustness_summary.csv",
        "results/tables/final_report_tables_manifest.json",
        "results/final_evidence_inventory.csv",
    }
)
STAGE5_DERIVED_EVIDENCE_PREFIXES: tuple[str, ...] = (
    "results/logs/plotting-cache/stage5/",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


_STAGE3_CONTINUITY_PURPOSE = (
    "Preserve the immutable Stage 3 freeze while explicitly reconciling its "
    "historical Stage 2 provenance with the regenerated "
    "traditional-computer-vision evidence."
)
_STAGE3_CONTINUITY_ASSERTIONS = {
    "stage3_freeze_and_test_access_unchanged": True,
    "stage3_checkpoints_and_model_evidence_unchanged": True,
    "stage3_training_semantically_independent_of_stage2_models_and_tables": True,
    "legacy_stage2_bindings_are_historical_provenance_only": True,
}
_STAGE3_CONTINUITY_RATIONALE = (
    "Stage 3 training used the immutable Stage 1 split contract and Stage 3 "
    "configurations; the preserved freeze and test-access record bind its "
    "checkpoints and model-specific evidence. Stage 3 did not train from, "
    "initialize from, or derive weights from Stage 2 models or Stage 2 result "
    "tables. The Stage 2 entries embedded in the immutable Stage 3 freeze therefore "
    "remain historical governance provenance; the current replacements above are "
    "the authoritative post-rewrite Stage 2 evidence."
)


def _expected_stage3_continuity_document() -> dict[str, Any]:
    legacy_rows: list[dict[str, Any]] = []
    for binding in LEGACY_STAGE2_CONTINUITY_BINDINGS:
        legacy_rows.append(
            {
                "origin": STAGE3_FREEZE_RELATIVE_PATH,
                "location": binding.location,
                "legacy": {
                    "path_at_stage3_freeze": binding.artifact_path,
                    "sha256_at_stage3_freeze": binding.legacy_sha256,
                    "bytes_at_stage3_freeze": binding.legacy_bytes,
                },
                "current_replacement": {
                    "path": binding.artifact_path,
                    "sha256": binding.current_sha256,
                    "bytes": binding.current_bytes,
                },
                "disposition": binding.disposition,
            }
        )
    return {
        "schema_version": 1,
        "status": "accepted_continuity",
        "purpose": _STAGE3_CONTINUITY_PURPOSE,
        "immutable_stage3_evidence": {
            "stage3_freeze": {
                "path": STAGE3_FREEZE_RELATIVE_PATH,
                "sha256": STAGE3_FREEZE_SHA256,
                "bytes": 21_726,
            },
            "stage3_test_access": {
                "path": STAGE3_TEST_ACCESS_RELATIVE_PATH,
                "sha256": STAGE3_TEST_ACCESS_SHA256,
                "bytes": 8_072,
            },
        },
        "legacy_stage2_bindings": legacy_rows,
        "assertions": dict(_STAGE3_CONTINUITY_ASSERTIONS),
        "rationale": _STAGE3_CONTINUITY_RATIONALE,
    }


def _validate_stage3_continuity_document(document: Mapping[str, Any]) -> None:
    """Require the exact, closed reconciliation schema and every assertion."""

    expected = _expected_stage3_continuity_document()
    if document != expected:
        raise ValueError(
            "Stage 3 traditional-rewrite continuity overlay content mismatch; "
            "its immutable evidence, legacy locations/hashes/paths, current "
            "replacements, assertions, and rationale must match the closed contract"
        )


def _normalize_manifest_reference(manifest_value: str | Path) -> tuple[str, Path]:
    manifest = Path(manifest_value)
    if manifest.is_absolute():
        candidate = manifest
    else:
        raw = manifest.as_posix()
        portable = PurePosixPath(raw)
        if (
            not raw
            or "\\" in raw
            or portable.is_absolute()
            or ".." in portable.parts
        ):
            raise ValueError(f"Unsafe nested manifest path: {manifest_value}")
        candidate = PROJECT_ROOT.joinpath(*portable.parts)
    if candidate.is_symlink():
        raise ValueError(f"Symlink nested manifest is forbidden: {manifest_value}")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Manifest is outside project root: {manifest_value}") from error
    return relative, resolved


def _verify_stage3_continuity_overlay(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Stage 3 traditional-rewrite continuity overlay is missing: {path}"
        )
    if path.is_symlink():
        raise ValueError("Stage 3 traditional-rewrite continuity overlay is a symlink")
    document = load_json(path)
    _validate_stage3_continuity_document(document)
    observed = sha256_file(path)
    if observed != STAGE3_CONTINUITY_SHA256:
        raise ValueError(
            "Stage 3 traditional-rewrite continuity overlay hash mismatch; "
            f"expected={STAGE3_CONTINUITY_SHA256}, observed={observed}"
        )


def verify_frozen_artifacts() -> list[dict[str, Any]]:
    """Verify every contracted artifact and return report-ready check rows."""

    rows: list[dict[str, Any]] = []
    for artifact in FROZEN_ARTIFACTS:
        exists = artifact.path.is_file()
        observed = sha256_file(artifact.path) if exists else ""
        rows.append(
            {
                "stage": artifact.stage,
                "artifact_type": artifact.artifact_type,
                "path": artifact.relative_path,
                "exists": exists,
                "expected_sha256": artifact.sha256,
                "observed_sha256": observed,
                "passed": exists and observed == artifact.sha256,
            }
        )
    return rows


def require_frozen_artifacts() -> None:
    failures = [row for row in verify_frozen_artifacts() if not row["passed"]]
    if failures:
        details = "; ".join(
            f"{row['path']} (expected {row['expected_sha256']}, observed {row['observed_sha256'] or 'MISSING'})"
            for row in failures
        )
        raise ValueError(f"Frozen evidence verification failed: {details}")


def snapshot_read_only_evidence() -> list[dict[str, Any]]:
    """Hash the complete closed set of pre-Stage-5 experiment evidence.

    Hidden placeholders and explicit Stage 5 derivatives are excluded.  Any
    symlink or path escaping the project root fails closed.
    """

    candidates: set[Path] = set()
    for relative_root in READ_ONLY_EVIDENCE_ROOTS:
        directory = PROJECT_ROOT / relative_root
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"Missing or unsafe evidence root: {relative_root}")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Symlink in read-only evidence tree: {path}")
            if path.is_file() and path.name != ".gitkeep":
                candidates.add(path)
    for relative in READ_ONLY_REPORTS:
        path = PROJECT_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe Stage 1--4 report: {relative}")
        candidates.add(path)

    rows: list[dict[str, Any]] = []
    project = PROJECT_ROOT.resolve()
    for path in sorted(candidates, key=lambda value: value.as_posix()):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(project).as_posix()
        except ValueError as error:
            raise ValueError(f"Evidence path escapes project root: {path}") from error
        if relative in STAGE5_DERIVED_EVIDENCE_EXCLUSIONS:
            continue
        if relative.startswith("results/metrics/stage5_"):
            continue
        if relative.startswith(STAGE5_DERIVED_EVIDENCE_PREFIXES):
            continue
        rows.append(
            {
                "path": relative,
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    if not rows:
        raise ValueError("Read-only Stage 1--4 evidence inventory is empty")
    return rows


def compare_read_only_evidence(
    current_rows: Iterable[Mapping[str, Any]],
    baseline_document: Mapping[str, Any],
) -> tuple[bool, str]:
    """Compare current rows with the exact closed set stored before Stage 5."""

    raw_baseline = baseline_document.get("read_only_evidence")
    if not isinstance(raw_baseline, Mapping):
        return False, "baseline snapshot has no read_only_evidence mapping"
    current = {
        str(row["path"]): {
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
        }
        for row in current_rows
    }
    baseline: dict[str, dict[str, Any]] = {}
    for path, value in raw_baseline.items():
        if not isinstance(path, str) or not isinstance(value, Mapping):
            return False, "baseline read_only_evidence has an invalid entry"
        baseline[path] = {
            "bytes": value.get("bytes"),
            "sha256": value.get("sha256"),
        }
    missing = sorted(set(baseline) - set(current))
    added = sorted(set(current) - set(baseline))
    changed = sorted(
        path
        for path in set(current) & set(baseline)
        if current[path] != baseline[path]
    )
    passed = not missing and not added and not changed
    details = (
        f"files={len(current)}, bytes={sum(value['bytes'] for value in current.values())}, "
        f"missing={missing[:5]}, added={added[:5]}, changed={changed[:5]}"
    )
    return passed, details


def _iter_nested_bindings(
    value: Any, *, origin: str, location: str = "$"
) -> Iterator[dict[str, Any]]:
    """Yield every nested object that has the artifact path/hash contract."""

    if isinstance(value, Mapping):
        has_path = "path" in value
        has_hash = "sha256" in value
        if has_path != has_hash:
            # Many documents contain descriptive ``*_path`` scalar fields; only
            # an object using the canonical key ``path`` opts into this contract.
            if has_path:
                raise ValueError(
                    f"Partial artifact binding in {origin} at {location}: "
                    "objects with 'path' must also contain 'sha256'"
                )
        if has_path and has_hash:
            yield {
                "origin": origin,
                "location": location,
                "path": value["path"],
                "sha256": value["sha256"],
                "bytes": value.get(
                    "bytes",
                    value.get("size", value.get("size_bytes", value.get("file_size_bytes"))),
                ),
            }
            return
        for key, child in value.items():
            yield from _iter_nested_bindings(
                child, origin=origin, location=f"{location}.{key}"
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_nested_bindings(
                child, origin=origin, location=f"{location}[{index}]"
            )


def _is_exact_allowlisted_stage3_legacy_binding(binding: Mapping[str, Any]) -> bool:
    """Recognize only the five immutable Stage 3-to-legacy-Stage-2 bindings."""

    if binding.get("origin") != STAGE3_FREEZE_RELATIVE_PATH:
        return False
    by_location = {
        item.location: item for item in LEGACY_STAGE2_CONTINUITY_BINDINGS
    }
    expected = by_location.get(str(binding.get("location")))
    if expected is None:
        return False
    observed = (
        binding.get("path"),
        binding.get("sha256"),
        binding.get("bytes"),
    )
    required = (
        expected.artifact_path,
        expected.legacy_sha256,
        expected.legacy_bytes,
    )
    if observed != required:
        raise ValueError(
            "Immutable Stage 3 legacy Stage 2 binding differs from the exact "
            f"continuity allowlist at {binding.get('location')}: "
            f"expected={required!r}, observed={observed!r}"
        )
    return True


def _safe_bound_path(raw: Any, *, origin: str, location: str) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"Unsafe artifact path in {origin} at {location}: {raw!r}")
    portable = PurePosixPath(raw)
    if portable.is_absolute() or raw in {"", "."} or ".." in portable.parts:
        raise ValueError(f"Unsafe artifact path in {origin} at {location}: {raw!r}")
    normalized = portable.as_posix()
    candidate = PROJECT_ROOT.joinpath(*portable.parts)
    if candidate.is_symlink():
        raise ValueError(f"Symlink artifact binding is forbidden: {normalized}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Artifact binding resolves outside project root: {normalized}") from error
    return normalized, resolved


def verify_nested_artifact_bindings(
    manifest_paths: Iterable[str | Path] = NESTED_BINDING_MANIFESTS,
) -> list[dict[str, Any]]:
    """Verify and deduplicate all nested artifact objects in frozen manifests.

    Duplicate bindings are accepted only when they agree on SHA-256 and any
    declared byte size.  Each physical file is hashed exactly once.
    """

    manifests = tuple(
        _normalize_manifest_reference(manifest_value)
        for manifest_value in manifest_paths
    )
    manifest_names = {relative for relative, _ in manifests}
    has_stage3_freeze = STAGE3_FREEZE_RELATIVE_PATH in manifest_names
    has_continuity = STAGE3_CONTINUITY_RELATIVE_PATH in manifest_names
    if has_stage3_freeze and not has_continuity:
        raise ValueError(
            "Stage 3 freeze verification requires the traditional-rewrite "
            f"continuity overlay manifest: {STAGE3_CONTINUITY_RELATIVE_PATH}"
        )
    if has_continuity:
        continuity_path = next(
            path
            for relative, path in manifests
            if relative == STAGE3_CONTINUITY_RELATIVE_PATH
        )
        _verify_stage3_continuity_overlay(continuity_path)

    collected: dict[str, dict[str, Any]] = {}
    for manifest_text, manifest_path in manifests:
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        document = load_json(manifest_path)
        for binding in _iter_nested_bindings(document, origin=manifest_text):
            if _is_exact_allowlisted_stage3_legacy_binding(binding):
                if not has_continuity:  # Defensive: required above for canonical input.
                    raise ValueError(
                        "Legacy Stage 2 binding cannot be exempted without the "
                        "verified Stage 3 continuity overlay"
                    )
                continue
            relative, resolved = _safe_bound_path(
                binding["path"], origin=binding["origin"], location=binding["location"]
            )
            digest = binding["sha256"]
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(
                    f"Invalid SHA-256 in {binding['origin']} at {binding['location']}"
                )
            size = binding["bytes"]
            if size is not None:
                if isinstance(size, bool):
                    raise ValueError(
                        f"Invalid artifact size in {binding['origin']} at {binding['location']}"
                    )
                try:
                    size = int(size)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid artifact size in {binding['origin']} at {binding['location']}"
                    ) from error
                if size < 0:
                    raise ValueError(
                        f"Invalid artifact size in {binding['origin']} at {binding['location']}"
                    )
            existing = collected.get(relative)
            provenance = f"{binding['origin']}:{binding['location']}"
            if existing is not None:
                if existing["expected_sha256"] != digest:
                    raise ValueError(f"Conflicting duplicate hashes for bound artifact: {relative}")
                if (
                    existing["expected_bytes"] is not None
                    and size is not None
                    and existing["expected_bytes"] != size
                ):
                    raise ValueError(f"Conflicting duplicate sizes for bound artifact: {relative}")
                if existing["expected_bytes"] is None and size is not None:
                    existing["expected_bytes"] = size
                existing["origins"].append(provenance)
                continue
            collected[relative] = {
                "path": relative,
                "resolved_path": resolved,
                "expected_sha256": digest,
                "expected_bytes": size,
                "origins": [provenance],
            }

    rows: list[dict[str, Any]] = []
    for relative in sorted(collected):
        binding = collected[relative]
        path = binding["resolved_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-bound artifact is missing: {relative}")
        if path.is_symlink():
            raise ValueError(f"Symlink artifact binding is forbidden: {relative}")
        observed_size = path.stat().st_size
        expected_size = binding["expected_bytes"]
        if expected_size is not None and observed_size != expected_size:
            raise ValueError(
                f"Manifest-bound artifact size mismatch: {relative}; "
                f"expected={expected_size}, observed={observed_size}"
            )
        observed_hash = sha256_file(path)
        if observed_hash != binding["expected_sha256"]:
            raise ValueError(
                f"Manifest-bound artifact hash mismatch: {relative}; "
                f"expected={binding['expected_sha256']}, observed={observed_hash}"
            )
        rows.append(
            {
                "path": relative,
                "bytes": observed_size,
                "sha256": observed_hash,
                "binding_count": len(binding["origins"]),
                "origins": list(binding["origins"]),
                "passed": True,
            }
        )
    if not rows:
        raise ValueError("Frozen manifests contain no nested artifact bindings")
    return rows


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def require_unique_labels(frame: pd.DataFrame, column: str, expected: tuple[str, ...], label: str) -> None:
    observed = frame[column].astype(str).tolist()
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError(
            f"{label} membership mismatch; expected={sorted(expected)}, observed={sorted(observed)}"
        )
