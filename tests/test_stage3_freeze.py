"""Selection and immutability contracts for the Stage 3 freeze."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from src.deep_learning.freeze_stage3 import (
    CANONICAL_ACCESS_RECORD,
    CANONICAL_FREEZE_PATH,
    EFFICIENTNET_EXPERIMENT,
    MAX_VALIDATION_METRIC_TOLERANCE,
    SCRATCH_EXPERIMENT,
    STAGE2_FILES,
    _stage2_artifacts,
    _canonical_stage2_paths,
    _expected_selection,
    _verify_validation_agreement,
    build_parser,
    freeze,
)
from src.data.common import sha256_file


def _candidate(name: str, *, f1: float, top1: float, pretrained: bool) -> dict:
    return {
        "experiment_name": name,
        "pretrained": pretrained,
        "best_epoch": 2,
        "validation_metrics": {
            "macro_f1": f1,
            "top1_accuracy": top1,
            "top5_accuracy": top1 + 0.2,
            "loss": 1.0,
        },
    }


def _write_stage2_contract(root: Path) -> tuple[Path, Path]:
    selected = ["linear_svm", "random_forest", "sift_bovw_svm"]
    selection = {
        "schema_version": 2,
        "status": "frozen",
        "selected_models": selected,
        "allowed_test_models": selected,
        "models": {name: {} for name in selected},
        "feature_configs": {name: {} for name in selected},
        "validation_metrics": {name: {} for name in selected},
        "validation_grid": {"rows": 19},
        "stage1_artifacts": {},
    }
    selection_path = root / "stage2_selection_manifest.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    freeze = {
        **selection,
        "schema_version": 1,
        "selection_schema_version": 2,
        "compatibility_role": "stage2_freeze alias for the authoritative selection",
        "selection_manifest": {
            "path": selection_path.as_posix(),
            "sha256": sha256_file(selection_path),
        },
    }
    freeze_path = root / "stage2_freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    return selection_path, freeze_path


class Stage3FreezeTests(unittest.TestCase):
    def test_locked_selection_uses_validation_ranking(self) -> None:
        candidates = {
            SCRATCH_EXPERIMENT: _candidate(
                SCRATCH_EXPERIMENT, f1=0.1, top1=0.12, pretrained=False
            ),
            "resnet18_pretrained_basic": _candidate(
                "resnet18_pretrained_basic", f1=0.3, top1=0.31, pretrained=True
            ),
            "resnet18_pretrained_strong_aug": _candidate(
                "resnet18_pretrained_strong_aug", f1=0.32, top1=0.33, pretrained=True
            ),
            EFFICIENTNET_EXPERIMENT: _candidate(
                EFFICIENTNET_EXPERIMENT, f1=0.35, top1=0.36, pretrained=True
            ),
        }
        selected, analysis_model, decision = _expected_selection(candidates)
        self.assertEqual(
            selected,
            [
                SCRATCH_EXPERIMENT,
                "resnet18_pretrained_strong_aug",
                EFFICIENTNET_EXPERIMENT,
            ],
        )
        self.assertEqual(analysis_model, EFFICIENTNET_EXPERIMENT)
        self.assertEqual(decision["selected_pretrained_resnet"], "resnet18_pretrained_strong_aug")

    def test_required_candidates_cannot_be_omitted(self) -> None:
        candidates = {
            SCRATCH_EXPERIMENT: _candidate(
                SCRATCH_EXPERIMENT, f1=0.1, top1=0.12, pretrained=False
            )
        }
        with self.assertRaisesRegex(ValueError, "requires completed D1 and D2"):
            _expected_selection(candidates)

    def test_default_freeze_path_is_deep_learning_directory(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--selected-experiment",
                SCRATCH_EXPERIMENT,
            ]
        )
        self.assertEqual(
            args.output.as_posix(),
            "results/metrics/deep_learning/stage3_freeze.json",
        )

    def test_freeze_rejects_parallel_output_path_before_artifact_access(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args = argparse.Namespace(
                output=Path(name) / "parallel-freeze.json",
                selected_experiment=[SCRATCH_EXPERIMENT],
            )
            with self.assertRaisesRegex(ValueError, "freeze path is fixed"):
                freeze(args)

    def test_freeze_rejects_alternate_access_record_path(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            args = argparse.Namespace(
                output=CANONICAL_FREEZE_PATH,
                access_record=Path(name) / "alternate-access.json",
                selected_experiment=[SCRATCH_EXPERIMENT],
            )
            with self.assertRaisesRegex(ValueError, "test-access path is fixed"):
                freeze(args)
        self.assertTrue(CANONICAL_ACCESS_RECORD.is_absolute())

    def test_freeze_recomputes_agreement_with_checkpoint_metrics(self) -> None:
        observed = {
            "loss": 1.0,
            "top1_accuracy": 0.2,
            "top5_accuracy": 0.5,
            "overall_accuracy": 0.2,
            "macro_precision": 0.21,
            "macro_recall": 0.2,
            "macro_f1": 0.19,
        }
        persisted = dict(observed)
        checkpoint = dict(observed)
        checkpoint["macro_f1"] = 0.1
        with self.assertRaisesRegex(ValueError, "checkpoint=0.1"):
            _verify_validation_agreement(
                observed,
                persisted,
                checkpoint,
                tolerance=MAX_VALIDATION_METRIC_TOLERANCE,
                name="fixture",
            )

    def test_freeze_rejects_unbounded_validation_tolerance(self) -> None:
        observed = {
            "loss": 1.0,
            "top1_accuracy": 0.2,
            "top5_accuracy": 0.5,
            "overall_accuracy": 0.2,
            "macro_precision": 0.21,
            "macro_recall": 0.2,
            "macro_f1": 0.19,
        }
        with self.assertRaisesRegex(ValueError, "fixed maximum"):
            _verify_validation_agreement(
                observed,
                observed,
                observed,
                tolerance=MAX_VALIDATION_METRIC_TOLERANCE * 10,
                name="fixture",
            )

    def test_stage2_evidence_override_cannot_bypass_accepted_paths(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaisesRegex(ValueError, "paths are fixed"):
                _canonical_stage2_paths(
                    [Path(name) / "replacement-stage2-freeze.json"]
                )

    def test_canonical_stage2_evidence_includes_authoritative_selection(self) -> None:
        paths = _canonical_stage2_paths(None)
        self.assertEqual(
            [path.name for path in paths].count("stage2_selection_manifest.json"),
            1,
        )
        self.assertIn("results/metrics/stage2_selection_manifest.json", STAGE2_FILES)

    def test_stage2_alias_must_bind_authoritative_schema2_selection(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            selection, freeze_path = _write_stage2_contract(Path(name))
            artifacts = _stage2_artifacts([selection, freeze_path])
            self.assertEqual(len(artifacts), 2)

            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            freeze["selection_manifest"]["sha256"] = "0" * 64
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a valid alias"):
                _stage2_artifacts([selection, freeze_path])

    def test_stage2_alias_cannot_disagree_with_selection_content(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            selection, freeze_path = _write_stage2_contract(Path(name))
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            freeze["validation_grid"] = {"rows": 18}
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                _stage2_artifacts([selection, freeze_path])


if __name__ == "__main__":
    unittest.main()
