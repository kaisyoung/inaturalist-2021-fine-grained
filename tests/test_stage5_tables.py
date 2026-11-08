"""Tests for Stage 5 frozen-evidence contracts and report-table derivation."""

from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from src.analysis.build_final_report_tables import (
    ABLATION_COLUMNS,
    MAIN_COLUMNS,
    ROBUSTNESS_COLUMNS,
    _metric,
    ablation_table_tex,
    build_ablation_results,
    build_main_results,
    build_robustness_summary,
    detailed_test_table_tex,
    main_table_tex,
    robustness_table_tex,
)
from src.analysis.stage5_contract import (
    FROZEN_ARTIFACTS,
    PROJECT_ROOT,
    compare_read_only_evidence,
    require_frozen_artifacts,
    snapshot_read_only_evidence,
    verify_nested_artifact_bindings,
    verify_frozen_artifacts,
)


HAS_CANONICAL_STAGE5_EVIDENCE = (
    (PROJECT_ROOT / "STAGE4_ADVANCED_ANALYSIS_REPORT.md").is_file()
    and (
        PROJECT_ROOT
        / "results/models/deep_learning/efficientnet_b0_pretrained_strong_aug_best.pt"
    ).is_file()
)


class FrozenEvidenceTests(unittest.TestCase):
    @unittest.skipUnless(
        HAS_CANONICAL_STAGE5_EVIDENCE,
        "canonical Stage 1-4 reports/models are intentionally excluded from the code ZIP",
    )
    def test_every_contracted_hash_matches(self) -> None:
        rows = verify_frozen_artifacts()
        self.assertEqual(len(rows), len(FROZEN_ARTIFACTS))
        self.assertGreaterEqual(len(rows), 30)
        self.assertTrue(all(row["exists"] and row["passed"] for row in rows))
        require_frozen_artifacts()

    @unittest.skipUnless(
        HAS_CANONICAL_STAGE5_EVIDENCE,
        "canonical Stage 1-4 reports/models are intentionally excluded from the code ZIP",
    )
    def test_all_nested_manifest_bindings_match_and_deduplicate(self) -> None:
        rows = verify_nested_artifact_bindings()
        self.assertGreaterEqual(len(rows), 60)
        self.assertTrue(all(row["passed"] for row in rows))
        self.assertTrue(any(row["binding_count"] > 1 for row in rows))

    @unittest.skipUnless(
        HAS_CANONICAL_STAGE5_EVIDENCE,
        "canonical Stage 1-4 reports/models are intentionally excluded from the code ZIP",
    )
    def test_closed_evidence_snapshot_covers_unbound_stage4_outputs(self) -> None:
        rows = snapshot_read_only_evidence()
        by_path = {row["path"]: row for row in rows}
        self.assertTrue(
            any(path.startswith("results/metrics/robustness/") for path in by_path)
        )
        self.assertTrue(
            any(path.startswith("results/predictions/robustness/") for path in by_path)
        )
        self.assertIn(
            "results/models/deep_learning/resnet18_pretrained_basic_best.pt", by_path
        )
        self.assertIn(
            "results/models/deep_learning/resnet18_pretrained_frozen_head_best.pt",
            by_path,
        )
        self.assertIn("DATASET_SMOKE_TEST_REPORT.md", by_path)
        self.assertNotIn("results/tables/final_report_main_results.csv", by_path)
        self.assertNotIn("results/metrics/stage5_frozen_hashes_before.json", by_path)

        baseline = {
            "read_only_evidence": {
                path: {"bytes": row["bytes"], "sha256": row["sha256"]}
                for path, row in by_path.items()
            }
        }
        passed, _ = compare_read_only_evidence(rows, baseline)
        self.assertTrue(passed)
        truncated = list(rows[:-1])
        passed, details = compare_read_only_evidence(truncated, baseline)
        self.assertFalse(passed)
        self.assertIn("missing=", details)

    def test_nested_binding_rejects_unsafe_and_conflicting_entries(self) -> None:
        with tempfile.TemporaryDirectory(prefix=".stage5-bindings-", dir=PROJECT_ROOT / "results") as name:
            root = Path(name)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"frozen evidence")
            relative = artifact.relative_to(PROJECT_ROOT).as_posix()
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

            def manifest(filename: str, document: dict[str, object]) -> str:
                path = root / filename
                path.write_text(json.dumps(document), encoding="utf-8")
                return path.relative_to(PROJECT_ROOT).as_posix()

            valid = manifest(
                "valid.json",
                {
                    "first": {"path": relative, "sha256": digest, "bytes": artifact.stat().st_size},
                    "second": {"path": relative, "sha256": digest},
                },
            )
            rows = verify_nested_artifact_bindings((valid,))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["binding_count"], 2)

            absolute = manifest(
                "absolute.json", {"artifact": {"path": str(artifact), "sha256": digest}}
            )
            with self.assertRaisesRegex(ValueError, "Unsafe artifact path"):
                verify_nested_artifact_bindings((absolute,))

            traversal = manifest(
                "traversal.json", {"artifact": {"path": "../escape.bin", "sha256": digest}}
            )
            with self.assertRaisesRegex(ValueError, "Unsafe artifact path"):
                verify_nested_artifact_bindings((traversal,))

            conflict = manifest(
                "conflict.json",
                {
                    "first": {"path": relative, "sha256": digest},
                    "second": {"path": relative, "sha256": "0" * 64},
                },
            )
            with self.assertRaisesRegex(ValueError, "Conflicting duplicate hashes"):
                verify_nested_artifact_bindings((conflict,))

            wrong_size = manifest(
                "wrong_size.json",
                {"artifact": {"path": relative, "sha256": digest, "bytes": 1}},
            )
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_nested_artifact_bindings((wrong_size,))

    def test_stage5_builders_contain_no_training_or_inference_calls(self) -> None:
        for relative in (
            "src/analysis/stage5_contract.py",
            "src/analysis/build_final_report_tables.py",
            "scripts/precheck_stage5.py",
        ):
            source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
            for forbidden in (".fit(", ".backward(", "optimizer.step(", "model("):
                self.assertNotIn(forbidden, source, relative)


class FinalTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main_source = pd.read_csv(PROJECT_ROOT / "results/tables/final_model_comparison.csv")
        cls.stage2_ablation = pd.read_csv(
            PROJECT_ROOT / "results/tables/stage2_feature_ablation_results.csv"
        )
        cls.stage3_ablation = pd.read_csv(
            PROJECT_ROOT / "results/tables/stage3_ablation_results.csv"
        )
        cls.stage4_summary = pd.read_csv(PROJECT_ROOT / "results/tables/stage4_summary.csv")
        cls.ranking = pd.read_csv(
            PROJECT_ROOT / "results/tables/robustness_model_ranking.csv"
        )
        cls.drops = pd.read_csv(
            PROJECT_ROOT / "results/tables/robustness_drop_summary.csv"
        )

    def test_main_selects_six_unique_tested_methods(self) -> None:
        frame = build_main_results(self.main_source)
        self.assertEqual(tuple(frame.columns), MAIN_COLUMNS)
        self.assertEqual(len(frame), 6)
        self.assertFalse(frame["method_name"].duplicated().any())
        self.assertFalse(frame[["test_top1", "test_top5", "test_macro_f1"]].isna().any().any())
        self.assertEqual(
            int((frame["method_name"] == "Random Forest").sum()), 1
        )
        self.assertEqual(
            int((frame["method_name"] == "EfficientNet-B0 (ImageNet pretrained)").sum()),
            1,
        )
        self.assertIn("Handcrafted + BoVW (LinearSVC)", set(frame["method_name"]))
        self.assertIn("SIFT-BoVW (LinearSVC)", set(frame["method_name"]))

    def test_unavailable_values_stay_blank_not_zero(self) -> None:
        main = build_main_results(self.main_source)
        traditional = main.loc[main["method_family"] == "traditional"]
        self.assertTrue(traditional["train_top1_best"].eq("").all())
        self.assertTrue(traditional["parameters"].eq("").all())
        ablation = build_ablation_results(self.stage2_ablation, self.stage3_ablation)
        traditional_ablation = ablation.loc[ablation["method_family"] == "traditional"]
        self.assertTrue(traditional_ablation["train_top1_best"].eq("").all())
        self.assertTrue(traditional_ablation["parameters"].eq("").all())
        self.assertEqual(_metric(""), "--")
        self.assertNotEqual(_metric(""), "0")

    def test_ablation_membership_is_unique_and_validation_only(self) -> None:
        frame = build_ablation_results(self.stage2_ablation, self.stage3_ablation)
        self.assertEqual(tuple(frame.columns), ABLATION_COLUMNS)
        self.assertEqual(len(frame), 8)
        self.assertFalse(frame["experiment_name"].duplicated().any())
        self.assertNotIn("test_top1", frame.columns)
        basic = frame.loc[frame["experiment_name"] == "resnet18_pretrained_basic"].iloc[0]
        self.assertIn("pretraining_comparison", basic["study_membership"])
        self.assertIn("augmentation_reference", basic["study_membership"])
        self.assertIn("fine_tuning_reference", basic["study_membership"])

    def test_robustness_max_drop_is_independent_of_macro_f1_worst(self) -> None:
        frame = build_robustness_summary(self.stage4_summary, self.ranking, self.drops)
        self.assertEqual(tuple(frame.columns), ROBUSTNESS_COLUMNS)
        self.assertEqual(len(frame), 4)
        forest = frame.loc[frame["model_name"] == "Random Forest"].iloc[0]
        forest_drops = self.drops.loc[
            self.drops["model_name"] == "traditional_random_forest"
        ].copy()
        forest_drops["absolute_top1_drop"] = pd.to_numeric(
            forest_drops["absolute_top1_drop"], errors="raise"
        )
        maximum = float(forest_drops["absolute_top1_drop"].max())
        expected_worst = forest_drops.loc[
            forest_drops["absolute_top1_drop"].sub(maximum).abs() <= 1e-12
        ].sort_values(
            ["corruption_type", "severity"],
            ascending=[True, False],
            kind="stable",
        ).iloc[0]
        forest_summary = self.stage4_summary.loc[
            self.stage4_summary["model_name"] == "traditional_random_forest"
        ].iloc[0]
        self.assertEqual(
            forest["worst_corruption_type_by_macro_f1"],
            forest_summary["worst_corruption_type"],
        )
        self.assertEqual(
            forest["worst_top1_corruption_type"], expected_worst["corruption_type"]
        )
        self.assertAlmostEqual(
            float(forest["average_top1_drop"]),
            float(forest_drops["absolute_top1_drop"].mean()),
            places=12,
        )
        self.assertAlmostEqual(float(forest["worst_top1_drop"]), maximum, places=12)
        self.assertEqual(set(frame["robustness_rank"]), {1, 2, 3, 4})

    def test_missing_source_column_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing columns"):
            build_main_results(self.main_source.drop(columns=["test_macro_f1"]))

    def test_latex_fragments_are_complete_and_stably_labelled(self) -> None:
        main = main_table_tex(build_main_results(self.main_source))
        ablation = ablation_table_tex(
            build_ablation_results(self.stage2_ablation, self.stage3_ablation)
        )
        robustness = robustness_table_tex(
            build_robustness_summary(self.stage4_summary, self.ranking, self.drops)
        )
        detailed = detailed_test_table_tex(build_main_results(self.main_source))
        for fragment, label in (
            (main, r"\label{tab:main-results}"),
            (detailed, r"\label{tab:test-details}"),
            (ablation, r"\label{tab:ablations}"),
            (robustness, r"\label{tab:robustness}"),
        ):
            self.assertIn(r"\begin{table*}", fragment)
            self.assertIn(r"\end{table*}", fragment)
            self.assertIn(label, fragment)


if __name__ == "__main__":
    unittest.main()
