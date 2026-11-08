"""Reusable, deterministic classification metrics for the 500-class project.

Score matrices are always aligned to an explicit label vector before ranking.
This is important for scikit-learn estimators, whose score columns follow
``estimator.classes_`` rather than an implicit ``0..N-1`` convention.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


DEFAULT_NUM_CLASSES = 500


def _as_unique_integer_labels(labels: Iterable[int]) -> np.ndarray:
    values = np.asarray(list(labels), dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if np.unique(values).size != values.size:
        raise ValueError("labels must be unique")
    if np.any(values[1:] < values[:-1]):
        raise ValueError("labels must be sorted so score ties have a stable policy")
    return values


def align_class_scores(
    raw_scores: np.ndarray,
    estimator_classes: Sequence[int] | np.ndarray,
    labels: Iterable[int] = range(DEFAULT_NUM_CLASSES),
) -> np.ndarray:
    """Align estimator score columns to *labels*, filling absent classes with -inf.

    One-dimensional binary ``decision_function`` output is converted to the
    conventional ``[-margin, margin]`` pair. Missing estimator classes remain
    rankable only after every represented class because their score is ``-inf``.
    """

    target_labels = _as_unique_integer_labels(labels)
    classes = np.asarray(estimator_classes, dtype=np.int64)
    if classes.ndim != 1 or classes.size == 0:
        raise ValueError("estimator_classes must be a non-empty 1D sequence")
    if np.unique(classes).size != classes.size:
        raise ValueError("estimator_classes contains duplicates")

    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim == 1:
        if classes.size != 2:
            raise ValueError(
                "one-dimensional scores are supported only for binary estimators"
            )
        scores = np.column_stack((-scores, scores))
    elif scores.ndim != 2:
        raise ValueError("raw_scores must be a one- or two-dimensional array")
    if scores.shape[1] != classes.size:
        raise ValueError(
            f"score columns ({scores.shape[1]}) do not match estimator classes "
            f"({classes.size})"
        )
    if not np.isfinite(scores).all():
        raise ValueError("raw_scores contains NaN or infinite values")

    label_to_column = {int(label): index for index, label in enumerate(target_labels)}
    unknown = sorted(set(int(value) for value in classes) - set(label_to_column))
    if unknown:
        raise ValueError(f"estimator contains labels outside the target set: {unknown}")

    aligned = np.full(
        (scores.shape[0], target_labels.size), -np.inf, dtype=np.float64
    )
    for source_column, label in enumerate(classes):
        aligned[:, label_to_column[int(label)]] = scores[:, source_column]
    return aligned


def stable_score_order(scores: np.ndarray) -> np.ndarray:
    """Return descending score-column order, breaking ties by lower column index."""

    values = np.asarray(scores)
    if values.ndim != 2:
        raise ValueError("scores must be two-dimensional")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("scores contains NaN or positive infinity")
    # Stable sorting preserves the ascending input column order for equal scores.
    return np.argsort(-values, axis=1, kind="stable")


def top_k_accuracy(
    y_true: Sequence[int] | np.ndarray,
    aligned_scores: np.ndarray,
    labels: Iterable[int] = range(DEFAULT_NUM_CLASSES),
    *,
    k: int = 5,
) -> float:
    """Compute top-k inclusion accuracy from an already aligned score matrix."""

    target_labels = _as_unique_integer_labels(labels)
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(aligned_scores)
    if truth.ndim != 1 or scores.ndim != 2 or truth.size != scores.shape[0]:
        raise ValueError("y_true and aligned_scores have incompatible shapes")
    if scores.shape[1] != target_labels.size:
        raise ValueError("aligned_scores columns do not match labels")
    if not 1 <= k <= target_labels.size:
        raise ValueError(f"k must be between 1 and {target_labels.size}")
    top_labels = target_labels[stable_score_order(scores)[:, :k]]
    return float(np.mean(np.any(top_labels == truth[:, None], axis=1)))


def macro_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Iterable[int] = range(DEFAULT_NUM_CLASSES),
) -> dict[str, float]:
    """Return macro precision/recall/F1 over every declared class."""

    target_labels = _as_unique_integer_labels(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=target_labels,
        average="macro",
        zero_division=0,
    )
    return {
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def compute_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Iterable[int] = range(DEFAULT_NUM_CLASSES),
) -> np.ndarray:
    """Return an integer confusion matrix with rows=true and columns=predicted."""

    target_labels = _as_unique_integer_labels(labels)
    return confusion_matrix(y_true, y_pred, labels=target_labels).astype(
        np.int64, copy=False
    )


def per_class_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Iterable[int] = range(DEFAULT_NUM_CLASSES),
) -> pd.DataFrame:
    """Return precision, recall, F1, support, predictions, and TP per class."""

    target_labels = _as_unique_integer_labels(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=target_labels,
        average=None,
        zero_division=0,
    )
    matrix = compute_confusion_matrix(y_true, y_pred, target_labels)
    return pd.DataFrame(
        {
            "new_label": target_labels,
            "precision": precision.astype(float),
            "recall": recall.astype(float),
            "f1": f1.astype(float),
            "support": support.astype(np.int64),
            "predicted_count": matrix.sum(axis=0).astype(np.int64),
            "true_positives": np.diag(matrix).astype(np.int64),
        }
    )


def top_confused_pairs(
    matrix: np.ndarray,
    labels: Iterable[int] = range(DEFAULT_NUM_CLASSES),
    *,
    limit: int | None = 20,
    scientific_names: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Rank unordered class pairs by their two directional error counts."""

    target_labels = _as_unique_integer_labels(labels)
    values = np.asarray(matrix)
    expected = (target_labels.size, target_labels.size)
    if values.shape != expected:
        raise ValueError(f"confusion matrix must have shape {expected}")
    if np.any(values < 0) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("confusion matrix must contain non-negative integers")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive or None")

    records: list[dict[str, int | str]] = []
    names = scientific_names or {}
    for first_index in range(target_labels.size):
        for second_index in range(first_index + 1, target_labels.size):
            first_as_second = int(values[first_index, second_index])
            second_as_first = int(values[second_index, first_index])
            total = first_as_second + second_as_first
            if total == 0:
                continue
            first_label = int(target_labels[first_index])
            second_label = int(target_labels[second_index])
            records.append(
                {
                    "class_a_new_label": first_label,
                    "class_a_scientific_name": names.get(first_label, ""),
                    "class_b_new_label": second_label,
                    "class_b_scientific_name": names.get(second_label, ""),
                    "a_as_b": first_as_second,
                    "b_as_a": second_as_first,
                    "total_confusions": total,
                }
            )
    columns = [
        "rank",
        "class_a_new_label",
        "class_a_scientific_name",
        "class_b_new_label",
        "class_b_scientific_name",
        "a_as_b",
        "b_as_a",
        "total_confusions",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame.from_records(records)
    result["_strongest_direction"] = result[["a_as_b", "b_as_a"]].max(axis=1)
    result = result.sort_values(
        [
            "total_confusions",
            "_strongest_direction",
            "class_a_new_label",
            "class_b_new_label",
        ],
        ascending=[False, False, True, True],
        kind="stable",
    ).drop(columns="_strongest_direction")
    if limit is not None:
        result = result.head(limit)
    result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=np.int64))
    return result.loc[:, columns].reset_index(drop=True)


def classification_report_frame(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Iterable[int] = range(DEFAULT_NUM_CLASSES),
) -> pd.DataFrame:
    """Export scikit-learn's classification report as a tabular artifact."""

    target_labels = _as_unique_integer_labels(labels)
    report = classification_report(
        y_true,
        y_pred,
        labels=target_labels,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().rename_axis("class_or_average").reset_index()


def evaluate_aligned_scores(
    y_true: Sequence[int] | np.ndarray,
    aligned_scores: np.ndarray,
    labels: Iterable[int] = range(DEFAULT_NUM_CLASSES),
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray]:
    """Compute required aggregate metrics and ranking outputs in one pass."""

    target_labels = _as_unique_integer_labels(labels)
    truth = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(aligned_scores, dtype=np.float64)
    if truth.ndim != 1 or scores.shape != (truth.size, target_labels.size):
        raise ValueError("y_true and aligned_scores have incompatible shapes")
    unknown = sorted(set(int(value) for value in np.unique(truth)) - set(target_labels))
    if unknown:
        raise ValueError(f"y_true contains labels outside the target set: {unknown}")

    order = stable_score_order(scores)
    predicted = target_labels[order[:, 0]]
    top1 = float(accuracy_score(truth, predicted))
    metrics: dict[str, float | int] = {
        "num_samples": int(truth.size),
        "num_classes": int(target_labels.size),
        "top1_accuracy": top1,
        "top5_accuracy": top_k_accuracy(
            truth, scores, target_labels, k=min(5, target_labels.size)
        ),
        "overall_accuracy": top1,
    }
    metrics.update(macro_classification_metrics(truth, predicted, target_labels))
    matrix = compute_confusion_matrix(truth, predicted, target_labels)
    return metrics, predicted, order, matrix
