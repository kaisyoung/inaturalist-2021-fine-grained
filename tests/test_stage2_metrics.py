"""Focused tests for Stage 2 score alignment and classification metrics."""

from __future__ import annotations

import unittest

import numpy as np

from src.analysis.metrics import (
    align_class_scores,
    compute_confusion_matrix,
    evaluate_aligned_scores,
    macro_classification_metrics,
    stable_score_order,
    top_confused_pairs,
)


class ScoreAlignmentTests(unittest.TestCase):
    def test_shuffled_and_missing_estimator_classes_are_aligned(self) -> None:
        raw = np.asarray([[0.3, 0.9], [0.8, 0.1]])
        aligned = align_class_scores(raw, [2, 0], labels=[0, 1, 2])
        np.testing.assert_allclose(aligned[:, [0, 2]], raw[:, [1, 0]])
        self.assertTrue(np.isneginf(aligned[:, 1]).all())

    def test_stable_ties_prefer_lower_label(self) -> None:
        scores = np.asarray([[0.5, 0.5, 0.2], [1.0, 1.0, 1.0]])
        np.testing.assert_array_equal(
            stable_score_order(scores), np.asarray([[0, 1, 2], [0, 1, 2]])
        )

    def test_binary_margin_is_expanded(self) -> None:
        aligned = align_class_scores(np.asarray([-2.0, 3.0]), [4, 8], labels=[4, 8])
        np.testing.assert_array_equal(aligned, [[2.0, -2.0], [-3.0, 3.0]])


class ClassificationMetricTests(unittest.TestCase):
    def test_macro_metrics_include_absent_declared_class(self) -> None:
        result = macro_classification_metrics([0, 0], [0, 0], labels=[0, 1])
        self.assertAlmostEqual(result["macro_precision"], 0.5)
        self.assertAlmostEqual(result["macro_recall"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 0.5)

    def test_confusion_orientation_and_pair_direction(self) -> None:
        matrix = compute_confusion_matrix([0, 0, 1, 2], [1, 0, 0, 1], [0, 1, 2])
        np.testing.assert_array_equal(matrix, [[1, 1, 0], [1, 0, 0], [0, 1, 0]])
        pairs = top_confused_pairs(matrix, [0, 1, 2], limit=2)
        self.assertEqual(pairs.iloc[0]["class_a_new_label"], 0)
        self.assertEqual(pairs.iloc[0]["class_b_new_label"], 1)
        self.assertEqual(pairs.iloc[0]["a_as_b"], 1)
        self.assertEqual(pairs.iloc[0]["b_as_a"], 1)

    def test_evaluation_top_five_is_inclusion_based(self) -> None:
        scores = np.asarray(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]]
        )
        metrics, predictions, _, matrix = evaluate_aligned_scores(
            [1, 4], scores, labels=range(6)
        )
        np.testing.assert_array_equal(predictions, [5, 0])
        self.assertEqual(metrics["top1_accuracy"], 0.0)
        self.assertEqual(metrics["top5_accuracy"], 1.0)
        self.assertEqual(matrix.sum(), 2)


if __name__ == "__main__":
    unittest.main()
