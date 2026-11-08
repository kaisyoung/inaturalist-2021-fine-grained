"""Dry-run and explicitly re-anchor downstream Stage 4 evidence hashes.

The command updates only the six enumerated Stage 4 pins in the Stage 5
contract. An apply is transactional and requires the exact token from a current
dry run::

    python -m scripts.rehash_stage4_evidence
    python -m scripts.rehash_stage4_evidence --apply --confirm PLAN_SHA256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from scripts.rehash_stage2_evidence import PlannedEdit, apply_plan, plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    "results/metrics/stage4_robustness_protocol.json",
    "results/tables/robustness_results.csv",
    "results/tables/robustness_drop_summary.csv",
    "results/tables/robustness_model_ranking.csv",
    "results/tables/stage4_summary.csv",
    "results/tables/final_model_comparison_with_robustness.csv",
)
SOURCES = ("src/analysis/stage5_contract.py",)
EXPECTED_SOURCE_PINS = {SOURCES[0]: ARTIFACTS}


def plan_stage4(*, root: Path | None = None) -> tuple[list[PlannedEdit], str]:
    """Return the complete Stage 4 pin rewrite plan and confirmation token."""

    return plan(
        root=PROJECT_ROOT if root is None else root,
        artifacts=ARTIFACTS,
        sources=SOURCES,
        expected_source_pins=EXPECTED_SOURCE_PINS,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--confirm",
        help="exact plan SHA-256 printed by the immediately preceding dry run",
    )
    args = parser.parse_args(argv)
    try:
        edits, token = plan_stage4()
    except (OSError, UnicodeError, ValueError) as error:
        print(f"[fail] {error}", file=sys.stderr)
        return 1

    display_root = PROJECT_ROOT.resolve()
    for edit in edits:
        print(f"[edit] {edit.path.relative_to(display_root)}")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
