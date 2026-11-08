"""Build Stage 5 report tables and deterministic LaTeX fragments.

Only frozen Stage 1--4 CSV evidence is read.  The script performs no inference,
training, model loading, or test-set selection.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.analysis.stage5_contract import (
    FROZEN_BY_PATH,
    MAIN_REPORT_MODELS,
    MAIN_SOURCE_COLUMNS,
    MAIN_SOURCE_MODELS,
    PROJECT_ROOT,
    STAGE4_PROTOCOL_MODELS,
    require_columns,
    require_frozen_artifacts,
    require_unique_labels,
)
from src.data.common import sha256_file


MAIN_COLUMNS = (
    "method_name", "method_family", "architecture_or_classifier", "pretrained",
    "augmentation", "train_top1_best", "val_top1", "val_top5",
    "val_macro_precision", "val_macro_recall", "val_macro_f1", "test_top1",
    "test_top5", "test_macro_precision", "test_macro_recall", "test_macro_f1",
    "training_time_seconds", "test_time_seconds", "parameters", "notes",
)

ABLATION_COLUMNS = (
    "study_membership", "experiment_name", "method_family",
    "representation_or_architecture", "pretrained", "augmentation",
    "freeze_backbone", "feature_dim", "parameters", "trainable_parameters",
    "train_top1_best", "val_top1", "val_top5", "val_macro_precision",
    "val_macro_recall", "val_macro_f1", "training_time_seconds", "notes",
)

ROBUSTNESS_COLUMNS = (
    "model_name", "method_family", "architecture", "clean_test_macro_f1",
    "average_corrupted_macro_f1", "worst_corruption_type_by_macro_f1",
    "worst_severity_by_macro_f1", "worst_severity_value_by_macro_f1",
    "worst_corruption_macro_f1", "average_top1_drop", "worst_top1_drop",
    "worst_top1_corruption_type", "worst_top1_severity",
    "worst_top1_severity_value", "robustness_rank", "notes",
)

DISPLAY_NAMES = {
    "linear_svm": "Handcrafted + BoVW (LinearSVC)",
    "random_forest": "Random Forest",
    "sift_bovw_svm": "SIFT-BoVW (LinearSVC)",
    "resnet18_scratch_basic": "ResNet18 (scratch)",
    "resnet18_pretrained_strong_aug": "ResNet18 (ImageNet pretrained)",
    "efficientnet_b0_pretrained_strong_aug": "EfficientNet-B0 (ImageNet pretrained)",
    "traditional_random_forest": "Random Forest",
}


def _finite_score(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{label} must be finite and within [0,1]")
    return result


def _optional(value: Any) -> Any:
    return "" if pd.isna(value) else value


def _bool(value: Any, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"{label} must be boolean")


def build_main_results(source: pd.DataFrame) -> pd.DataFrame:
    require_columns(source, MAIN_SOURCE_COLUMNS, "final_model_comparison.csv")
    require_unique_labels(source, "model_name", MAIN_SOURCE_MODELS, "final_model_comparison.csv")
    indexed = source.set_index("model_name", verify_integrity=True)
    rows: list[dict[str, Any]] = []
    for model_name in MAIN_REPORT_MODELS:
        row = indexed.loc[model_name]
        for metric in (
            "val_top1", "val_top5", "val_macro_precision", "val_macro_recall",
            "val_macro_f1", "test_top1", "test_top5", "test_macro_precision",
            "test_macro_recall", "test_macro_f1",
        ):
            _finite_score(row[metric], f"{model_name}.{metric}")
        family = str(row["method_family"])
        training_top1 = _optional(row["train_top1_best"])
        parameters = _optional(row["num_parameters"])
        if family == "traditional" and (training_top1 != "" or parameters != ""):
            raise ValueError(f"Traditional unavailable fields must remain blank: {model_name}")
        if family == "deep_learning" and (training_top1 == "" or parameters == ""):
            raise ValueError(f"Deep-learning training/parameter fields are missing: {model_name}")
        if parameters != "":
            parameters = int(parameters)
        notes = str(row["notes"])
        if model_name == "random_forest":
            notes += (
                "; canonical Stage 4 comparator; corruption evaluation "
                "recomputes the handcrafted-only descriptor"
            )
        rows.append(
            {
                "method_name": DISPLAY_NAMES[model_name],
                "method_family": family,
                "architecture_or_classifier": row["architecture"],
                "pretrained": _bool(row["pretrained"], f"{model_name}.pretrained"),
                "augmentation": "" if family == "traditional" else row["augmentation"],
                "train_top1_best": training_top1,
                "val_top1": row["val_top1"],
                "val_top5": row["val_top5"],
                "val_macro_precision": row["val_macro_precision"],
                "val_macro_recall": row["val_macro_recall"],
                "val_macro_f1": row["val_macro_f1"],
                "test_top1": row["test_top1"],
                "test_top5": row["test_top5"],
                "test_macro_precision": row["test_macro_precision"],
                "test_macro_recall": row["test_macro_recall"],
                "test_macro_f1": row["test_macro_f1"],
                "training_time_seconds": row["training_time_seconds"],
                "test_time_seconds": row["test_time_seconds"],
                "parameters": parameters,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=MAIN_COLUMNS)


def build_ablation_results(stage2: pd.DataFrame, stage3: pd.DataFrame) -> pd.DataFrame:
    stage2_columns = (
        "experiment_name", "feature_components", "feature_dim", "val_top1",
        "val_top5", "val_macro_precision", "val_macro_recall", "val_macro_f1",
        "train_time_seconds", "val_time_seconds",
    )
    stage3_columns = (
        "model_name", "architecture", "pretrained", "augmentation",
        "freeze_backbone", "train_top1_best", "val_top1", "val_top5",
        "val_macro_precision", "val_macro_recall", "val_macro_f1",
        "training_time_seconds", "num_parameters", "trainable_parameters", "notes",
    )
    require_columns(stage2, stage2_columns, "stage2_feature_ablation_results.csv")
    require_columns(stage3, stage3_columns, "stage3_ablation_results.csv")
    stage2_names = (
        "full",
        "ablation_only_color",
        "ablation_only_texture",
        "ablation_only_hog",
    )
    stage3_names = (
        "efficientnet_b0_pretrained_strong_aug", "resnet18_pretrained_basic",
        "resnet18_pretrained_frozen_head", "resnet18_pretrained_strong_aug",
        "resnet18_scratch_basic",
    )
    require_unique_labels(stage2, "experiment_name", stage2_names, "Stage 2 ablations")
    require_unique_labels(stage3, "model_name", stage3_names, "Stage 3 ablations")
    rows: list[dict[str, Any]] = []
    s2 = stage2.set_index("experiment_name", verify_integrity=True)
    for name in stage2_names:
        row = s2.loc[name]
        rows.append(
            {
                "study_membership": (
                    "traditional_feature_ablation_reference"
                    if name == "full" else "traditional_feature_ablation"
                ),
                "experiment_name": name,
                "method_family": "traditional",
                "representation_or_architecture": row["feature_components"],
                "pretrained": False,
                "augmentation": "",
                "freeze_backbone": "",
                "feature_dim": int(row["feature_dim"]),
                "parameters": "",
                "trainable_parameters": "",
                "train_top1_best": "",
                "val_top1": row["val_top1"],
                "val_top5": row["val_top5"],
                "val_macro_precision": row["val_macro_precision"],
                "val_macro_recall": row["val_macro_recall"],
                "val_macro_f1": row["val_macro_f1"],
                "training_time_seconds": row["train_time_seconds"],
                "notes": "validation-only; no held-out test metrics",
            }
        )
    memberships = {
        "resnet18_scratch_basic": "pretraining_comparison",
        "resnet18_pretrained_basic": "pretraining_comparison;augmentation_reference;fine_tuning_reference",
        "resnet18_pretrained_strong_aug": "augmentation_ablation",
        "resnet18_pretrained_frozen_head": "fine_tuning_ablation",
    }
    s3 = stage3.set_index("model_name", verify_integrity=True)
    for name in (
        "resnet18_scratch_basic", "resnet18_pretrained_basic",
        "resnet18_pretrained_strong_aug", "resnet18_pretrained_frozen_head",
    ):
        row = s3.loc[name]
        for metric in ("val_top1", "val_top5", "val_macro_precision", "val_macro_recall", "val_macro_f1"):
            _finite_score(row[metric], f"{name}.{metric}")
        note = str(row["notes"])
        if name in {"resnet18_scratch_basic", "resnet18_pretrained_basic"}:
            note += "; pretraining comparison is not a one-factor strict ablation"
        rows.append(
            {
                "study_membership": memberships[name],
                "experiment_name": name,
                "method_family": "deep_learning",
                "representation_or_architecture": row["architecture"],
                "pretrained": _bool(row["pretrained"], f"{name}.pretrained"),
                "augmentation": row["augmentation"],
                "freeze_backbone": _bool(row["freeze_backbone"], f"{name}.freeze_backbone"),
                "feature_dim": "",
                "parameters": int(row["num_parameters"]),
                "trainable_parameters": int(row["trainable_parameters"]),
                "train_top1_best": row["train_top1_best"],
                "val_top1": row["val_top1"],
                "val_top5": row["val_top5"],
                "val_macro_precision": row["val_macro_precision"],
                "val_macro_recall": row["val_macro_recall"],
                "val_macro_f1": row["val_macro_f1"],
                "training_time_seconds": row["training_time_seconds"],
                "notes": note,
            }
        )
    return pd.DataFrame(rows, columns=ABLATION_COLUMNS)


def build_robustness_summary(
    summary: pd.DataFrame, ranking: pd.DataFrame, drops: pd.DataFrame
) -> pd.DataFrame:
    summary_columns = (
        "model_name", "method_family", "architecture", "clean_test_macro_f1",
        "average_corrupted_macro_f1", "worst_corruption_type", "worst_severity",
        "worst_corruption_macro_f1", "robustness_rank",
    )
    ranking_columns = (
        "model_name", "mean_absolute_top1_drop", "worst_severity_value",
        "robustness_rank",
    )
    drop_columns = (
        "model_name", "corruption_type", "severity", "severity_value",
        "clean_top1", "corrupted_top1", "absolute_top1_drop",
    )
    require_columns(summary, summary_columns, "stage4_summary.csv")
    require_columns(ranking, ranking_columns, "robustness_model_ranking.csv")
    require_columns(drops, drop_columns, "robustness_drop_summary.csv")
    require_unique_labels(summary, "model_name", STAGE4_PROTOCOL_MODELS, "Stage 4 summary")
    require_unique_labels(ranking, "model_name", STAGE4_PROTOCOL_MODELS, "robustness ranking")
    expected_drop_keys = {
        (model, corruption, severity)
        for model in STAGE4_PROTOCOL_MODELS
        for corruption in ("gaussian_noise", "gaussian_blur", "motion_blur", "jpeg_compression")
        for severity in range(1, 5)
    }
    observed_drop_keys = set(
        zip(
            drops["model_name"].astype(str),
            drops["corruption_type"].astype(str),
            pd.to_numeric(drops["severity"], errors="raise").astype(int),
        )
    )
    if len(drops) != 64 or observed_drop_keys != expected_drop_keys:
        raise ValueError("Robustness drop table is not the complete 4x4x4 matrix")
    s = summary.set_index("model_name", verify_integrity=True)
    r = ranking.set_index("model_name", verify_integrity=True)
    rows: list[dict[str, Any]] = []
    for model in STAGE4_PROTOCOL_MODELS:
        summary_row = s.loc[model]
        ranking_row = r.loc[model]
        group = drops.loc[drops["model_name"] == model].copy()
        group["absolute_top1_drop"] = pd.to_numeric(group["absolute_top1_drop"], errors="raise")
        maximum = float(group["absolute_top1_drop"].max())
        candidates = group.loc[np.isclose(group["absolute_top1_drop"], maximum, rtol=0, atol=1e-12)]
        worst_top1 = candidates.sort_values(
            ["corruption_type", "severity"], ascending=[True, False], kind="stable"
        ).iloc[0]
        observed_mean = float(group["absolute_top1_drop"].mean())
        if not math.isclose(
            observed_mean, float(ranking_row["mean_absolute_top1_drop"]),
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError(f"Average top-1 drop mismatch: {model}")
        macro_group = group.loc[
            (group["corruption_type"] == summary_row["worst_corruption_type"])
            & (pd.to_numeric(group["severity"], errors="raise") == int(summary_row["worst_severity"]))
        ]
        if len(macro_group) != 1:
            raise ValueError(f"Macro-F1 worst condition is absent from drop evidence: {model}")
        rows.append(
            {
                "model_name": DISPLAY_NAMES[model],
                "method_family": summary_row["method_family"],
                "architecture": summary_row["architecture"],
                "clean_test_macro_f1": summary_row["clean_test_macro_f1"],
                "average_corrupted_macro_f1": summary_row["average_corrupted_macro_f1"],
                "worst_corruption_type_by_macro_f1": summary_row["worst_corruption_type"],
                "worst_severity_by_macro_f1": int(summary_row["worst_severity"]),
                "worst_severity_value_by_macro_f1": ranking_row["worst_severity_value"],
                "worst_corruption_macro_f1": summary_row["worst_corruption_macro_f1"],
                "average_top1_drop": observed_mean,
                "worst_top1_drop": maximum,
                "worst_top1_corruption_type": worst_top1["corruption_type"],
                "worst_top1_severity": int(worst_top1["severity"]),
                "worst_top1_severity_value": worst_top1["severity_value"],
                "robustness_rank": int(summary_row["robustness_rank"]),
                "notes": "worst macro-F1 and worst top-1 conditions are selected independently",
            }
        )
    return pd.DataFrame(rows, columns=ROBUSTNESS_COLUMNS).sort_values(
        "robustness_rank", kind="stable"
    ).reset_index(drop=True)


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def _metric(value: Any, digits: int = 4) -> str:
    if value == "" or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _percent(value: Any, digits: int = 2) -> str:
    """Format a unit-interval metric as a percentage without a percent sign."""

    return f"{100.0 * _finite_score(value, 'report macro metric'):.{digits}f}"


def _rows_tex(rows: list[list[str]]) -> str:
    return "\n".join(" & ".join(row) + r" \\" for row in rows)


def result_macros_tex(main: pd.DataFrame, robustness: pd.DataFrame) -> str:
    def main_row(model_name: str) -> pd.Series:
        rows = main.loc[main["method_name"] == DISPLAY_NAMES[model_name]]
        if len(rows) != 1:
            raise ValueError(f"Report macro source row is not unique: {model_name}")
        return rows.iloc[0]

    def robustness_row(model_name: str) -> pd.Series:
        rows = robustness.loc[robustness["model_name"] == DISPLAY_NAMES[model_name]]
        if len(rows) != 1:
            raise ValueError(f"Robustness macro source row is not unique: {model_name}")
        return rows.iloc[0]

    primary = main_row("linear_svm")
    forest = main_row("random_forest")
    bovw = main_row("sift_bovw_svm")
    best = main_row("efficientnet_b0_pretrained_strong_aug")
    robust = robustness_row("efficientnet_b0_pretrained_strong_aug")
    forest_robust = robustness_row("traditional_random_forest")
    # The drops are persisted decimal metrics.  Stabilise subtraction before
    # percentage formatting so an IEEE-754 residue cannot flip a half-even
    # display boundary (for example, 0.06265 must render as 6.26, matching the
    # authoritative ``mean_corrupted_top1`` evidence column).
    best_mean_corrupted_top1 = round(
        float(best["test_top1"]) - float(robust["average_top1_drop"]), 12
    )
    best_worst_corrupted_top1 = round(
        float(best["test_top1"]) - float(robust["worst_top1_drop"]), 12
    )
    forest_mean_corrupted_top1 = round(
        float(forest["test_top1"]) - float(forest_robust["average_top1_drop"]), 12
    )
    forest_worst_corrupted_top1 = round(
        float(forest["test_top1"]) - float(forest_robust["worst_top1_drop"]), 12
    )
    return (
        "% Generated from frozen Stage 1--4 evidence. Do not edit by hand.\n"
        f"\\newcommand{{\\BestTestTopOne}}{{{_metric(best['test_top1'])}}}\n"
        f"\\newcommand{{\\BestTestTopFive}}{{{_metric(best['test_top5'])}}}\n"
        f"\\newcommand{{\\BestTestMacroFOne}}{{{_metric(best['test_macro_f1'], 6)}}}\n"
        f"\\newcommand{{\\BestMeanCorruptedMacroFOne}}{{{_metric(robust['average_corrupted_macro_f1'], 6)}}}\n"
        f"\\newcommand{{\\PrimaryTraditionalTestTopOnePercent}}{{{_percent(primary['test_top1'])}}}\n"
        f"\\newcommand{{\\PrimaryTraditionalTestTopFivePercent}}{{{_percent(primary['test_top5'])}}}\n"
        f"\\newcommand{{\\PrimaryTraditionalTestMacroFOnePercent}}{{{_percent(primary['test_macro_f1'])}}}\n"
        f"\\newcommand{{\\RandomForestTestTopOnePercent}}{{{_percent(forest['test_top1'])}}}\n"
        f"\\newcommand{{\\RandomForestTestTopFivePercent}}{{{_percent(forest['test_top5'])}}}\n"
        f"\\newcommand{{\\RandomForestTestMacroFOnePercent}}{{{_percent(forest['test_macro_f1'])}}}\n"
        f"\\newcommand{{\\BoVWOnlyTestTopOnePercent}}{{{_percent(bovw['test_top1'])}}}\n"
        f"\\newcommand{{\\BoVWOnlyTestTopFivePercent}}{{{_percent(bovw['test_top5'])}}}\n"
        f"\\newcommand{{\\BoVWOnlyTestMacroFOnePercent}}{{{_percent(bovw['test_macro_f1'])}}}\n"
        f"\\newcommand{{\\BestTestTopOnePercent}}{{{_percent(best['test_top1'])}}}\n"
        f"\\newcommand{{\\BestTestTopFivePercent}}{{{_percent(best['test_top5'])}}}\n"
        f"\\newcommand{{\\BestTestMacroFOnePercent}}{{{_percent(best['test_macro_f1'])}}}\n"
        f"\\newcommand{{\\BestMeanCorruptedTopOnePercent}}{{{_percent(best_mean_corrupted_top1)}}}\n"
        f"\\newcommand{{\\BestMeanCorruptedMacroFOnePercent}}{{{_percent(robust['average_corrupted_macro_f1'])}}}\n"
        f"\\newcommand{{\\BestWorstCorruptedTopOnePercent}}{{{_percent(best_worst_corrupted_top1)}}}\n"
        f"\\newcommand{{\\BestWorstCorruptedMacroFOnePercent}}{{{_percent(robust['worst_corruption_macro_f1'])}}}\n"
        f"\\newcommand{{\\RandomForestMeanCorruptedTopOnePercent}}{{{_percent(forest_mean_corrupted_top1)}}}\n"
        f"\\newcommand{{\\RandomForestMeanCorruptedMacroFOnePercent}}{{{_percent(forest_robust['average_corrupted_macro_f1'])}}}\n"
        f"\\newcommand{{\\RandomForestWorstCorruptedTopOnePercent}}{{{_percent(forest_worst_corrupted_top1)}}}\n"
        f"\\newcommand{{\\RandomForestWorstCorruptedMacroFOnePercent}}{{{_percent(forest_robust['worst_corruption_macro_f1'])}}}\n"
    )


def main_table_tex(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append([
            _latex_escape(row.method_name), "Yes" if row.pretrained else "No",
            _metric(row.val_top1), _metric(row.val_top5), _metric(row.val_macro_f1, 6),
            _metric(row.test_top1), _metric(row.test_top5), _metric(row.test_macro_f1, 6),
        ])
    return """% Generated from frozen evidence. Do not edit by hand.
\\begin{table*}[t]
\\centering
\\caption{Validation and held-out test results. All values are fractions and are derived from the current hash-bound evidence. The rewritten Stage~2 configuration is frozen going forward; its pre-test locking chronology cannot be independently established retrospectively.}
\\label{tab:main-results}
\\small
\\begin{tabularx}{\\textwidth}{@{}Xccccccc@{}}
\\toprule
Method & Pretr. & Val top-1 & Val top-5 & Val F1 & Test top-1 & Test top-5 & Test F1 \\\\
\\midrule
""" + _rows_tex(rows) + """
\\bottomrule
\\end{tabularx}
\\end{table*}
"""


def detailed_test_table_tex(frame: pd.DataFrame) -> str:
    rows: list[list[str]] = []
    for row in frame.itertuples(index=False):
        rows.append(
            [
                _latex_escape(row.method_name),
                _metric(row.test_macro_precision, 6),
                _metric(row.test_macro_recall, 6),
                _metric(row.test_macro_f1, 6),
                f"{float(row.training_time_seconds):.1f}",
                f"{float(row.test_time_seconds):.3f}",
            ]
        )
    return """% Generated from frozen evidence. Do not edit by hand.
\\begin{table*}[t]
\\centering
\\caption{Held-out macro metrics and measured fit/test timings. Overall accuracy equals top-1 for this single-label task; timing boundaries differ between cached-feature and end-to-end neural evaluation (Section~\\ref{sec:setup}).}
\\label{tab:test-details}
\\small
\\begin{tabularx}{\\textwidth}{@{}Xccccr@{}}
\\toprule
Method & Macro precision & Macro recall & Macro F1 & Train (s) & Test (s) \\\\
\\midrule
""" + _rows_tex(rows) + """
\\bottomrule
\\end{tabularx}
\\end{table*}
"""


def ablation_table_tex(frame: pd.DataFrame) -> str:
    study_labels = {
        "traditional_feature_ablation_reference": "Feature reference",
        "traditional_feature_ablation": "Feature ablation",
        "pretraining_comparison": "Pretraining",
        "pretraining_comparison;augmentation_reference;fine_tuning_reference": "Pretrain / aug. / tuning ref.",
        "augmentation_ablation": "Augmentation",
        "fine_tuning_ablation": "Frozen head",
    }
    experiment_labels = {
        "full": "All blocks + BoVW",
        "ablation_only_color": "Colour only",
        "ablation_only_texture": "Texture (LBP) only",
        "ablation_only_hog": "HOG only",
        "resnet18_scratch_basic": "ResNet18 scratch / basic",
        "resnet18_pretrained_basic": "ResNet18 pretrained / basic",
        "resnet18_pretrained_strong_aug": "ResNet18 pretrained / strong",
        "resnet18_pretrained_frozen_head": "ResNet18 frozen backbone",
    }
    rows = []
    for row in frame.itertuples(index=False):
        rows.append([
            _latex_escape(study_labels[str(row.study_membership)]),
            _latex_escape(experiment_labels[str(row.experiment_name)]),
            _metric(row.val_top1), _metric(row.val_top5), _metric(row.val_macro_f1, 6),
        ])
    return """% Generated from frozen validation evidence. Do not edit by hand.
\\begin{table*}[t]
\\centering
\\caption{Validation-only traditional and deep-learning ablations.}
\\label{tab:ablations}
\\small
\\begin{tabularx}{\\textwidth}{@{}lXccc@{}}
\\toprule
Study membership & Experiment & Val top-1 & Val top-5 & Val macro-F1 \\\\
\\midrule
""" + _rows_tex(rows) + """
\\bottomrule
\\end{tabularx}
\\end{table*}
"""


def robustness_table_tex(frame: pd.DataFrame) -> str:
    compact_names = {
        "EfficientNet-B0 (ImageNet pretrained)": "EfficientNet-B0 pretrained",
        "ResNet18 (ImageNet pretrained)": "ResNet18 pretrained",
        "ResNet18 (scratch)": "ResNet18 scratch",
        "Random Forest": "RF (2,424-D handcrafted)",
    }
    rows = []
    for row in frame.itertuples(index=False):
        rows.append([
            _latex_escape(compact_names[str(row.model_name)]),
            _metric(row.clean_test_macro_f1, 6),
            _metric(row.average_corrupted_macro_f1, 6),
            _metric(row.worst_corruption_macro_f1, 6), _metric(row.average_top1_drop, 6),
            _metric(row.worst_top1_drop, 6), str(int(row.robustness_rank)),
        ])
    return """% Generated from frozen Stage 4 evidence. Do not edit by hand.
\\begin{table*}[t]
\\centering
\\caption{Fixed-protocol robustness summary over 16 corrupted test conditions per model. The traditional comparator is the 2,424-D handcrafted Random Forest, not the primary 3,704-D LinearSVC.}
\\label{tab:robustness}
\\small
\\begin{tabularx}{\\textwidth}{@{}Xcccccc@{}}
\\toprule
Model & Clean F1 & Mean corrupt. F1 & Worst F1 & Mean $\\Delta$ top-1 & Max $\\Delta$ top-1 & Rank \\\\
\\midrule
""" + _rows_tex(rows) + """
\\bottomrule
\\end{tabularx}
\\end{table*}
"""


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.StringIO(newline="")
    frame.to_csv(buffer, index=False, lineterminator="\n", na_rep="")
    return buffer.getvalue().encode("utf-8")


def _install(outputs: Mapping[Path, bytes], *, force: bool) -> str:
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        if len(existing) == len(outputs) and all(
            path.is_file() and path.read_bytes() == payload
            for path, payload in outputs.items()
        ):
            return "cache_hit"
        raise FileExistsError(
            "Derived report outputs are stale or partial; use --force deliberately: "
            + ", ".join(str(path) for path in existing)
        )
    common = PROJECT_ROOT / "results"
    common.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage5_tables.", dir=common) as name:
        stage = Path(name)
        staged: dict[Path, Path] = {}
        for index, (destination, payload) in enumerate(outputs.items()):
            temporary = stage / f"{index:02d}_{destination.name}"
            temporary.write_bytes(payload)
            staged[destination] = temporary
        backups: dict[Path, Path] = {}
        installed: list[Path] = []
        try:
            for index, destination in enumerate(existing):
                backup = stage / f"backup_{index:02d}_{destination.name}"
                os.replace(destination, backup)
                backups[destination] = backup
            for destination, source in staged.items():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, destination)
                installed.append(destination)
        except Exception:
            for destination in installed:
                destination.unlink(missing_ok=True)
            for destination, backup in backups.items():
                if backup.exists():
                    os.replace(backup, destination)
            raise
    return "generated"


def build_all(
    output_dir: Path, generated_dir: Path, *, force: bool
) -> tuple[dict[str, str], str]:
    require_frozen_artifacts()
    main_source = pd.read_csv(PROJECT_ROOT / "results/tables/final_model_comparison.csv")
    s2_ablation = pd.read_csv(PROJECT_ROOT / "results/tables/stage2_feature_ablation_results.csv")
    s3_ablation = pd.read_csv(PROJECT_ROOT / "results/tables/stage3_ablation_results.csv")
    stage4_summary = pd.read_csv(PROJECT_ROOT / "results/tables/stage4_summary.csv")
    ranking = pd.read_csv(PROJECT_ROOT / "results/tables/robustness_model_ranking.csv")
    drops = pd.read_csv(PROJECT_ROOT / "results/tables/robustness_drop_summary.csv")
    main = build_main_results(main_source)
    ablation = build_ablation_results(s2_ablation, s3_ablation)
    robustness = build_robustness_summary(stage4_summary, ranking, drops)

    output_dir = output_dir.resolve()
    generated_dir = generated_dir.resolve()
    payloads: dict[Path, bytes] = {
        output_dir / "final_report_main_results.csv": _csv_bytes(main),
        output_dir / "final_report_ablation_results.csv": _csv_bytes(ablation),
        output_dir / "final_report_robustness_summary.csv": _csv_bytes(robustness),
        generated_dir / "result_macros.tex": result_macros_tex(main, robustness).encode("utf-8"),
        generated_dir / "main_results_table.tex": main_table_tex(main).encode("utf-8"),
        generated_dir / "detailed_test_table.tex": detailed_test_table_tex(main).encode("utf-8"),
        generated_dir / "ablation_results_table.tex": ablation_table_tex(ablation).encode("utf-8"),
        generated_dir / "robustness_summary_table.tex": robustness_table_tex(robustness).encode("utf-8"),
    }
    output_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(payload).hexdigest()
        for path, payload in payloads.items()
    }
    source_paths = (
        "results/tables/final_model_comparison.csv",
        "results/tables/stage2_feature_ablation_results.csv",
        "results/tables/stage3_ablation_results.csv",
        "results/tables/stage4_summary.csv",
        "results/tables/robustness_model_ranking.csv",
        "results/tables/robustness_drop_summary.csv",
    )
    manifest = {
        "schema_version": 1,
        "policy": {
            "frozen_evidence_only": True,
            "training_or_inference_performed": False,
            "unavailable_values": "empty CSV field / LaTeX --; never zero-filled",
        },
        "sources": {path: sha256_file(PROJECT_ROOT / path) for path in source_paths},
        "outputs": output_hashes,
        "row_counts": {"main_results": len(main), "ablation_results": len(ablation), "robustness_summary": len(robustness)},
    }
    manifest_path = output_dir / "final_report_tables_manifest.json"
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    payloads[manifest_path] = manifest_bytes
    status = _install(payloads, force=force)
    return (
        {
            **output_hashes,
            manifest_path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
        },
        status,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--generated-dir", type=Path, default=Path("report/cvpr_latex/generated"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        hashes, status = build_all(
            args.output_dir, args.generated_dir, force=bool(args.force)
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if status == "cache_hit":
        print(
            f"[PASS] Valid report-table cache hit for {len(hashes)} "
            "frozen-evidence artifacts."
        )
    else:
        print(
            f"[PASS] Generated {len(hashes)} frozen-evidence report artifacts "
            "without training or inference."
        )
    for path, digest in hashes.items():
        print(f"[PASS] {digest}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
