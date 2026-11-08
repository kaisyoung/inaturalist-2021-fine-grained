"""Focused tests for initial and protected-replacement Stage 4 prechecks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.precheck_stage4 import (
    EXPECTED_MODEL_HASHES,
    EXPECTED_PROTOCOL_CORRUPTIONS,
    EXPECTED_PROTOCOL_MODEL_ARTIFACTS,
    render_report,
    run_precheck,
)


def _result_map(checks: list[tuple[str, str, str]]) -> dict[str, str]:
    return {label: result for result, label, _detail in checks}


def _write_protocol(root: Path) -> Path:
    path = root / "results/metrics/stage4_robustness_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "stage4_robustness",
                "status": "frozen",
                "seed": 42,
                "split": "test",
                "expected_num_images_per_run": 5000,
                "expected_run_count": 64,
                "model_order": list(EXPECTED_PROTOCOL_MODEL_ARTIFACTS),
                "models": {
                    name: {"artifact": {"path": artifact}}
                    for name, artifact in EXPECTED_PROTOCOL_MODEL_ARTIFACTS.items()
                },
                "corruptions": EXPECTED_PROTOCOL_CORRUPTIONS,
                "policy": {
                    "corruptions_applied_on_the_fly": True,
                    "model_selection_locked_before_robustness": True,
                    "no_retraining_or_weight_updates": True,
                    "no_test_tuning": True,
                },
                "stage1_artifacts": {"test.csv": {}},
                "stage2_evidence": {"stage2_freeze.json": {}},
                "stage3_evidence": {"stage3_freeze.json": {}},
            }
        ),
        encoding="utf-8",
    )
    return path


class Stage4PrecheckModeTests(unittest.TestCase):
    def test_initial_mode_requires_protocol_absence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checks, _ = run_precheck(root)
            self.assertEqual(
                _result_map(checks)["No pre-existing Stage 4 protocol"], "PASS"
            )
            _write_protocol(root)
            checks, _ = run_precheck(root)
            self.assertEqual(
                _result_map(checks)["No pre-existing Stage 4 protocol"], "FAIL"
            )

    def test_replacement_mode_requires_protocol_token_and_empty_cell_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checks, _ = run_precheck(root, replacement_rerun=True)
            results = _result_map(checks)
            self.assertEqual(
                results["Existing Stage 4 protocol replacement token"], "FAIL"
            )
            self.assertEqual(
                results["Canonical Stage 4 cell directories are empty"], "PASS"
            )

            protocol = root / "results/metrics/stage4_robustness_protocol.json"
            protocol.parent.mkdir(parents=True, exist_ok=True)
            protocol.write_text(
                '{"schema_version": 1, "stage": "stage4_robustness", "status": "frozen"}',
                encoding="utf-8",
            )
            checks, _ = run_precheck(root, replacement_rerun=True)
            self.assertEqual(
                _result_map(checks)["Existing Stage 4 protocol replacement token"],
                "FAIL",
            )

            _write_protocol(root)
            checks, _ = run_precheck(root, replacement_rerun=True)
            results = _result_map(checks)
            self.assertEqual(
                results["Existing Stage 4 protocol replacement token"], "PASS"
            )
            self.assertEqual(
                results["Canonical Stage 4 cell directories are empty"], "PASS"
            )

            metrics = root / "results/metrics/robustness"
            metrics.mkdir(parents=True)
            (metrics / "old_cell.json").write_text("{}", encoding="utf-8")
            checks, _ = run_precheck(root, replacement_rerun=True)
            self.assertEqual(
                _result_map(checks)[
                    "Canonical Stage 4 cell directories are empty"
                ],
                "FAIL",
            )

    def test_replacement_report_records_mode_and_comparator_distinction(self) -> None:
        rendered = render_report([], [], replacement_rerun=True)
        self.assertIn("--replacement-rerun --force", rendered)
        self.assertIn("2,424-D handcrafted-only", rendered)
        self.assertIn("3,704-D Stage 2 LinearSVC", rendered)

    def test_random_forest_hash_is_keyed_by_its_artifact_path(self) -> None:
        self.assertEqual(
            set(EXPECTED_MODEL_HASHES),
            {"results/models/traditional/random_forest.joblib"},
        )
        self.assertEqual(
            len(
                EXPECTED_MODEL_HASHES[
                    "results/models/traditional/random_forest.joblib"
                ]
            ),
            64,
        )


if __name__ == "__main__":
    unittest.main()
