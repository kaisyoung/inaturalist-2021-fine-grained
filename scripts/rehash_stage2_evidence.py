"""Dry-run and explicitly re-anchor downstream Stage 2 evidence hashes.

An apply is accepted only with the exact confirmation token printed by a prior
dry run over the same artifact/source bytes::

    python -m scripts.rehash_stage2_evidence
    python -m scripts.rehash_stage2_evidence --apply --confirm PLAN_SHA256

When the Stage 2 rebuild has deliberately retired the derived Stage 3 comparison,
an explicit bootstrap pass may update only the prerequisite pins needed to run the
Stage 3 collector.  A normal complete pass is still mandatory afterwards::

    python -m scripts.rehash_stage2_evidence --bootstrap-stage3-comparison
    python -m scripts.rehash_stage2_evidence \
        --bootstrap-stage3-comparison --apply --confirm PLAN_SHA256
    python -m src.analysis.collect_stage3_results --force
    python -m scripts.rehash_stage2_evidence
    python -m scripts.rehash_stage2_evidence --apply --confirm PLAN_SHA256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERIVED_STAGE3_COMPARISON = "results/tables/final_model_comparison.csv"
RANDOM_FOREST_MODEL = "results/models/traditional/random_forest.joblib"
ARTIFACTS = (
    "results/metrics/stage2_selection_manifest.json",
    "results/metrics/stage2_freeze.json",
    "results/tables/stage2_traditional_results.csv",
    "results/tables/stage2_feature_ablation_results.csv",
    RANDOM_FOREST_MODEL,
    DERIVED_STAGE3_COMPARISON,
)
SOURCES = (
    "src/robustness/freeze_protocol.py",
    "src/analysis/stage5_contract.py",
    "src/deep_learning/freeze_stage3.py",
    "scripts/precheck_stage4.py",
    "src/analysis/collect_stage3_results.py",
)
EXPECTED_SOURCE_PINS = {
    "src/robustness/freeze_protocol.py": (
        "results/metrics/stage2_freeze.json",
        "results/tables/stage2_traditional_results.csv",
        "results/tables/stage2_feature_ablation_results.csv",
    ),
    "src/analysis/stage5_contract.py": (
        "results/metrics/stage2_freeze.json",
        "results/tables/stage2_traditional_results.csv",
        "results/tables/stage2_feature_ablation_results.csv",
        "results/tables/final_model_comparison.csv",
    ),
    "src/deep_learning/freeze_stage3.py": (
        "results/metrics/stage2_selection_manifest.json",
        "results/metrics/stage2_freeze.json",
        "results/tables/stage2_traditional_results.csv",
        "results/tables/stage2_feature_ablation_results.csv",
    ),
    "scripts/precheck_stage4.py": (
        "results/metrics/stage2_freeze.json",
        "results/tables/stage2_traditional_results.csv",
        "results/tables/stage2_feature_ablation_results.csv",
        RANDOM_FOREST_MODEL,
        DERIVED_STAGE3_COMPARISON,
    ),
    "src/analysis/collect_stage3_results.py": (
        "results/tables/stage2_traditional_results.csv",
    ),
}


@dataclass(frozen=True)
class PlannedEdit:
    path: Path
    original: bytes
    updated: bytes
    changes: tuple[tuple[str, str, str], ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pin_pattern(relative: str) -> re.Pattern[str]:
    return re.compile(
        r'("' + re.escape(relative) + r'"\s*[,:]\s*\(?\s*")([0-9a-f]{64})(")'
    )


def _stage_bytes(path: Path, payload: bytes, mode: int) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def plan(
    *,
    root: Path | None = None,
    artifacts: Sequence[str] = ARTIFACTS,
    sources: Sequence[str] = SOURCES,
    expected_source_pins: Mapping[str, Sequence[str]] = EXPECTED_SOURCE_PINS,
) -> tuple[list[PlannedEdit], str]:
    root = PROJECT_ROOT if root is None else root.resolve()
    if set(expected_source_pins) != set(sources):
        raise ValueError("Source/pin allowlists disagree")
    artifact_set = set(artifacts)
    if len(artifact_set) != len(artifacts):
        raise ValueError("Artifact allowlist contains duplicates")
    consumed = {
        relative for values in expected_source_pins.values() for relative in values
    }
    if consumed != artifact_set:
        raise ValueError(
            "Artifact/pin allowlists disagree: "
            f"missing consumers={sorted(artifact_set - consumed)}, "
            f"unknown pins={sorted(consumed - artifact_set)}"
        )

    missing_artifacts = [relative for relative in artifacts if not (root / relative).is_file()]
    missing_sources = [relative for relative in sources if not (root / relative).is_file()]
    if missing_artifacts or missing_sources:
        details = []
        if missing_artifacts:
            details.append("artifacts=" + ", ".join(missing_artifacts))
        if missing_sources:
            details.append("sources=" + ", ".join(missing_sources))
        raise ValueError("Missing required rehash inputs: " + "; ".join(details))

    observed = {
        relative: sha256(root / relative)
        for relative in artifacts
    }
    edits: list[PlannedEdit] = []
    source_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    pin_records: list[tuple[str, str, str, str]] = []
    for source in sources:
        path = root / source
        original_bytes = path.read_bytes()
        try:
            original = original_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Rehash source is not UTF-8: {source}") from error
        updated = original
        changes: list[tuple[str, str, str]] = []
        expected_pins = set(expected_source_pins[source])
        if len(expected_pins) != len(expected_source_pins[source]):
            raise ValueError(f"Duplicate expected pin in source allowlist: {source}")
        for relative, digest in observed.items():
            pattern = pin_pattern(relative)
            matches = list(pattern.finditer(original))
            expected_count = 1 if relative in expected_pins else 0
            if len(matches) != expected_count:
                raise ValueError(
                    f"{source} must contain exactly {expected_count} hash pin(s) "
                    f"for {relative}; found {len(matches)}"
                )
            if not matches:
                continue
            previous = matches[0].group(2)
            updated = pattern.sub(
                lambda match, value=digest: match.group(1) + value + match.group(3),
                updated,
                count=1,
            )
            pin_records.append((source, relative, previous, digest))
            if previous != digest:
                changes.append((relative, previous, digest))
        updated_bytes = updated.encode("utf-8")
        source_hashes[source] = hashlib.sha256(original_bytes).hexdigest()
        output_hashes[source] = hashlib.sha256(updated_bytes).hexdigest()
        if updated != original:
            edits.append(
                PlannedEdit(
                    path=path,
                    original=original_bytes,
                    updated=updated_bytes,
                    changes=tuple(changes),
                )
            )
    token_payload = {
        "artifacts": observed,
        "sources": source_hashes,
        "outputs": output_hashes,
        "pins": sorted(pin_records),
    }
    token = hashlib.sha256(
        json.dumps(token_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return edits, token


def bootstrap_plan(
    *,
    root: Path | None = None,
    artifacts: Sequence[str] = ARTIFACTS,
    sources: Sequence[str] = SOURCES,
    expected_source_pins: Mapping[str, Sequence[str]] = EXPECTED_SOURCE_PINS,
    derived_artifact: str = DERIVED_STAGE3_COMPARISON,
) -> tuple[list[PlannedEdit], str]:
    """Plan the explicit first pass while exactly one derived table is absent."""

    resolved_root = PROJECT_ROOT if root is None else root.resolve()
    if set(expected_source_pins) != set(sources):
        raise ValueError("Source/pin allowlists disagree")
    if tuple(artifacts).count(derived_artifact) != 1:
        raise ValueError(
            "Bootstrap derived artifact must occur exactly once in the artifact allowlist"
        )
    for source, pins in expected_source_pins.items():
        if len(set(pins)) != len(pins):
            raise ValueError(f"Duplicate expected pin in source allowlist: {source}")
    derived_path = resolved_root / derived_artifact
    if derived_path.exists():
        raise ValueError(
            "--bootstrap-stage3-comparison is allowed only while the derived "
            f"comparison is absent: {derived_path}"
        )
    core_artifacts = tuple(
        relative for relative in artifacts if relative != derived_artifact
    )
    core_pins = {
        source: tuple(
            relative
            for relative in expected_source_pins[source]
            if relative != derived_artifact
        )
        for source in sources
    }
    edits, token = plan(
        root=resolved_root,
        artifacts=core_artifacts,
        sources=sources,
        expected_source_pins=core_pins,
    )
    # The absent derived table cannot be hashed yet, but its source-pin topology
    # remains part of the fail-closed contract.  Requiring the placeholder now
    # guarantees that the mandatory normal second pass can update it exactly once.
    pattern = pin_pattern(derived_artifact)
    for source in sources:
        source_text = (resolved_root / source).read_text(encoding="utf-8")
        expected_count = 1 if derived_artifact in expected_source_pins[source] else 0
        observed_count = len(list(pattern.finditer(source_text)))
        if observed_count != expected_count:
            raise ValueError(
                f"{source} must contain exactly {expected_count} hash pin(s) "
                f"for {derived_artifact}; found {observed_count}"
            )
    return edits, token


def apply_plan(edits: Sequence[PlannedEdit]) -> None:
    """Install all source rewrites, rolling the group back on any failure."""

    candidates: dict[Path, Path] = {}
    modes: dict[Path, int] = {}
    installed: list[PlannedEdit] = []
    try:
        for edit in edits:
            current = edit.path.read_bytes()
            if current != edit.original:
                raise ValueError(f"Rehash source changed after dry-run planning: {edit.path}")
            mode = stat.S_IMODE(edit.path.stat().st_mode)
            modes[edit.path] = mode
            candidates[edit.path] = _stage_bytes(edit.path, edit.updated, mode)
        for edit in edits:
            os.replace(candidates[edit.path], edit.path)
            installed.append(edit)
    except Exception as error:
        rollback_errors: list[str] = []
        for edit in reversed(installed):
            try:
                rollback = _stage_bytes(edit.path, edit.original, modes[edit.path])
                os.replace(rollback, edit.path)
            except Exception as rollback_error:  # pragma: no cover - catastrophic I/O.
                rollback_errors.append(f"{edit.path}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                "Rehash apply failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from error
        raise
    finally:
        for candidate in candidates.values():
            candidate.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--bootstrap-stage3-comparison",
        action="store_true",
        help=(
            "first pass only: permit the derived final_model_comparison.csv to be "
            "absent so the Stage 3 collector pin can be updated"
        ),
    )
    parser.add_argument(
        "--confirm",
        help="exact plan SHA-256 printed by the immediately preceding dry run",
    )
    args = parser.parse_args(argv)
    try:
        planner = bootstrap_plan if args.bootstrap_stage3_comparison else plan
        edits, token = planner()
    except (OSError, UnicodeError, ValueError) as error:
        print(f"[fail] {error}", file=sys.stderr)
        return 1
    for edit in edits:
        print(f"[edit] {edit.path.relative_to(PROJECT_ROOT)}")
        for relative, old, new in edit.changes:
            print(f"       {relative}: {old[:12]}... -> {new[:12]}...")
    print(f"PLAN_SHA256={token}")
    if not args.apply:
        print(f"Dry run only: {sum(len(edit.changes) for edit in edits)} pin(s)")
        return 0
    if args.confirm != token:
        print(
            "[fail] --apply requires --confirm with the current dry-run PLAN_SHA256",
            file=sys.stderr,
        )
        return 1
    try:
        apply_plan(edits)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[fail] {error}", file=sys.stderr)
        return 1
    print(f"Applied {sum(len(edit.changes) for edit in edits)} pin(s)")
    if args.bootstrap_stage3_comparison:
        print(
            "BOOTSTRAP ONLY: run src.analysis.collect_stage3_results, then complete "
            "a normal dry-run/confirmed apply before any downstream precheck."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
