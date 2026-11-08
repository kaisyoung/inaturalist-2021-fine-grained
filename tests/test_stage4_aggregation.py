"""Focused tests for strict Stage 4 aggregation and persisted failure analysis."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.analysis.collect_robustness_results import Protocol, validate_result_matrix
from src.analysis.collect_stage4_results import _require_selection_manifest_status
from src.analysis.robustness_analysis import _relative_drop, analyse_robustness
from src.analysis.stage4_failure_analysis import build_failure_tables


def protocol() -> Protocol:
    models = {
        f"model_{index}": {"method_family": "deep", "architecture": "toy"}
        for index in range(4)
    }
    corruptions = {
        name: {"parameter": "amount", "severities": [1, 2, 3, 4]}
        for name in ("noise", "blur", "motion", "jpeg")
    }
    return Protocol(Path("protocol.json"), "0" * 64, {}, models, corruptions, {})


def robustness_frame() -> pd.DataFrame:
    rows = []
    for model_index in range(4):
        model = f"model_{model_index}"
        clean = 0.0 if model_index == 0 else 0.8 - 0.1 * model_index
        rows.append(
            dict(model_name=model, method_family="deep", architecture="toy", corruption_type="clean", severity=0, severity_value=0, severity_parameter="none", source="frozen_clean_test_metrics", top1=clean, top5=min(1, clean + .1), overall_accuracy=clean, macro_precision=clean, macro_recall=clean, macro_f1=clean, inference_time_seconds=1, num_images=5000)
        )
        for corruption_index, corruption in enumerate(("noise", "blur", "motion", "jpeg")):
            for severity in range(1, 5):
                score = max(0, clean - .01 * (corruption_index + severity))
                rows.append(
                    dict(model_name=model, method_family="deep", architecture="toy", corruption_type=corruption, severity=severity, severity_value=severity, severity_parameter="amount", source="stage4_on_the_fly_corruption", top1=score, top5=min(1, score + .1), overall_accuracy=score, macro_precision=score, macro_recall=score, macro_f1=score, inference_time_seconds=2, num_images=5000)
                )
    return pd.DataFrame(rows)


class RobustnessMatrixTests(unittest.TestCase):
    def test_complete_matrix_and_zero_clean_relative_drop(self) -> None:
        frame = robustness_frame()
        validate_result_matrix(frame, protocol())
        drops, ranking = analyse_robustness(frame, protocol())
        self.assertEqual(len(drops), 64)
        self.assertEqual(len(ranking), 4)
        self.assertTrue(drops.loc[drops.model_name == "model_0", "relative_top1_drop"].isna().all())
        self.assertTrue(np.isnan(_relative_drop(0.0, 0.0)))

    def test_missing_duplicate_and_unknown_cells_fail(self) -> None:
        base = robustness_frame()
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            validate_result_matrix(base.drop(base.index[-1]), protocol())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_result_matrix(pd.concat([base, base.iloc[[-1]]], ignore_index=True), protocol())
        changed = base.copy()
        changed.loc[changed.index[-1], "corruption_type"] = "unknown"
        with self.assertRaisesRegex(ValueError, "matrix mismatch"):
            validate_result_matrix(changed, protocol())


class FailureAnalysisTests(unittest.TestCase):
    def test_selector_terminal_status_is_selected_not_complete(self) -> None:
        _require_selection_manifest_status({"status": "selected"})
        for invalid in ("complete", "pending", None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "not selected"):
                    _require_selection_manifest_status({"status": invalid})

    def test_tables_are_built_only_from_persisted_rows(self) -> None:
        predictions = pd.DataFrame(
            {
                "image_id": range(8),
                "local_image_path": [f"test/{i}.jpg" for i in range(8)],
                "true_label": [0, 0, 1, 1, 2, 2, 3, 3],
                "predicted_label": [0, 1, 1, 1, 3, 2, 3, 2],
                "top1_score": [.9, .8, .7, .6, .95, .55, .5, .4],
            }
        )
        predictions["correct"] = predictions.true_label == predictions.predicted_label
        predictions["true_scientific_name"] = predictions.true_label.map(lambda x: f"Sci {x}")
        predictions["true_common_name"] = predictions.true_label.map(lambda x: f"Common {x}")
        predictions["predicted_scientific_name"] = predictions.predicted_label.map(lambda x: f"Sci {x}")
        predictions["predicted_common_name"] = predictions.predicted_label.map(lambda x: f"Common {x}")
        precision, recall, f1, support = precision_recall_fscore_support(
            predictions.true_label, predictions.predicted_label, labels=range(4), zero_division=0
        )
        per_class = pd.DataFrame(
            {
                "new_label": range(4), "scientific_name": [f"Sci {i}" for i in range(4)],
                "common_name": [f"Common {i}" for i in range(4)], "precision": precision,
                "recall": recall, "f1": f1, "support": support,
                "predicted_count": predictions.predicted_label.value_counts().reindex(range(4), fill_value=0).values,
                "true_positives": predictions.loc[predictions.correct].true_label.value_counts().reindex(range(4), fill_value=0).values,
            }
        )
        confusions = pd.DataFrame([{"rank": 1, "class_a_new_label": 0, "class_a_scientific_name": "Sci 0", "class_b_new_label": 1, "class_b_scientific_name": "Sci 1", "a_as_b": 1, "b_as_a": 0, "total_confusions": 1}])
        selected = predictions.iloc[[0, 1]].copy()
        selected["selection_type"] = ["correct_high_confidence", "incorrect_high_confidence"]
        selected["source_prediction_row"] = [0, 1]
        provenance = {
            "analysis_experiment": "efficientnet_b0_pretrained_strong_aug",
            "stage3_freeze_sha256": "a", "stage3_test_access_sha256": "b",
            "prediction_artifact_sha256": "c", "per_class_artifact_sha256": "d",
            "confusion_artifact_sha256": "e", "gradcam_selection_sha256": "f",
            "gradcam_selection_manifest_sha256": "g",
        }
        for key in ("analysis_experiment", "prediction_artifact_sha256", "confusion_artifact_sha256", "stage3_freeze_sha256", "stage3_test_access_sha256"):
            selected[key] = provenance[key]
        tables = build_failure_tables(predictions, per_class, confusions, selected, provenance, expected_images=8, expected_classes=4, expected_support=2)
        self.assertEqual(set(tables), {"hardest", "easiest", "high_failures", "low_correct", "confused_pairs"})
        self.assertIn("representative_image_path", tables["hardest"])
        self.assertIn("prediction_artifact_sha256", tables["confused_pairs"])


class NoTrainingTests(unittest.TestCase):
    def test_analysis_modules_have_no_training_calls(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("collect_robustness_results.py", "plot_robustness.py", "robustness_analysis.py", "stage4_failure_analysis.py", "collect_stage4_results.py"):
            source = (root / "src/analysis" / name).read_text(encoding="utf-8")
            self.assertNotIn(".fit(", source)
            self.assertNotIn(".backward(", source)


if __name__ == "__main__":
    unittest.main()
