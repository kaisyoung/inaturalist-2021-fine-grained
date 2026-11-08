"""Compute fixed, reproducible robustness-drop summaries and model rankings."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.collect_robustness_results import (
    Protocol,
    load_protocol,
    validate_result_matrix,
)


DROP_COLUMNS = [
    "model_name",
    "method_family",
    "architecture",
    "corruption_type",
    "severity",
    "severity_value",
    "severity_parameter",
    "clean_top1",
    "corrupted_top1",
    "absolute_top1_drop",
    "relative_top1_drop",
    "clean_top5",
    "corrupted_top5",
    "absolute_top5_drop",
    "relative_top5_drop",
    "clean_macro_f1",
    "corrupted_macro_f1",
    "absolute_macro_f1_drop",
    "relative_macro_f1_drop",
    "is_worst_condition_for_model",
    "most_robust_model_for_corruption",
    "is_most_robust_for_corruption",
]

RANKING_COLUMNS = [
    "robustness_rank",
    "model_name",
    "method_family",
    "architecture",
    "mean_corrupted_top1",
    "mean_corrupted_top5",
    "mean_corrupted_macro_f1",
    "average_robustness_score",
    "worst_corruption_type",
    "worst_severity",
    "worst_severity_value",
    "worst_condition_top1",
    "worst_condition_macro_f1",
    "mean_absolute_top1_drop",
    "mean_relative_top1_drop",
    "mean_absolute_macro_f1_drop",
    "mean_relative_macro_f1_drop",
    "num_corrupted_conditions",
]


def _relative_drop(clean: float, corrupted: float) -> float:
    """Return relative loss; zero clean baselines are explicitly unavailable."""

    if math.isclose(clean, 0.0, rel_tol=0.0, abs_tol=0.0):
        return float("nan")
    return (clean - corrupted) / clean


def analyse_robustness(frame: pd.DataFrame, protocol: Protocol) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_result_matrix(frame, protocol)
    clean = (
        frame.loc[frame["corruption_type"] == "clean"]
        .set_index("model_name", verify_integrity=True)
        .copy()
    )
    corrupted = frame.loc[frame["corruption_type"] != "clean"].copy()
    rows: list[dict[str, Any]] = []
    for record in corrupted.to_dict(orient="records"):
        baseline = clean.loc[str(record["model_name"])]
        output = {
            key: record[key]
            for key in (
                "model_name",
                "method_family",
                "architecture",
                "corruption_type",
                "severity",
                "severity_value",
                "severity_parameter",
            )
        }
        for metric in ("top1", "top5", "macro_f1"):
            clean_value = float(baseline[metric])
            corrupt_value = float(record[metric])
            output[f"clean_{metric}"] = clean_value
            output[f"corrupted_{metric}"] = corrupt_value
            output[f"absolute_{metric}_drop"] = clean_value - corrupt_value
            output[f"relative_{metric}_drop"] = _relative_drop(clean_value, corrupt_value)
        rows.append(output)
    drops = pd.DataFrame(rows)

    # Worst condition is defined by minimum corrupted macro F1, then top-1,
    # corruption name, and severity index for deterministic tie-breaking.
    drops["is_worst_condition_for_model"] = False
    for _, indices in drops.groupby("model_name", sort=False).groups.items():
        group = drops.loc[list(indices)].sort_values(
            ["corrupted_macro_f1", "corrupted_top1", "corruption_type", "severity"],
            ascending=[True, True, True, False],
            kind="stable",
        )
        drops.loc[group.index[0], "is_worst_condition_for_model"] = True

    # Most robust for one corruption is the model with the largest mean macro
    # F1 across its four fixed severities; top-1 and model name break ties.
    corruption_winners: dict[str, str] = {}
    for corruption, group in drops.groupby("corruption_type", sort=False):
        means = (
            group.groupby("model_name", as_index=False)
            .agg(
                mean_macro_f1=("corrupted_macro_f1", "mean"),
                mean_top1=("corrupted_top1", "mean"),
            )
            .sort_values(
                ["mean_macro_f1", "mean_top1", "model_name"],
                ascending=[False, False, True],
                kind="stable",
            )
        )
        corruption_winners[str(corruption)] = str(means.iloc[0]["model_name"])
    drops["most_robust_model_for_corruption"] = drops["corruption_type"].map(
        corruption_winners
    )
    drops["is_most_robust_for_corruption"] = (
        drops["model_name"] == drops["most_robust_model_for_corruption"]
    )
    drops = drops[DROP_COLUMNS].sort_values(
        ["model_name", "corruption_type", "severity"], kind="stable"
    ).reset_index(drop=True)

    ranking_rows: list[dict[str, Any]] = []
    for model_name, group in drops.groupby("model_name", sort=False):
        worst = group.loc[group["is_worst_condition_for_model"]].iloc[0]
        baseline = clean.loc[str(model_name)]
        # Fixed average robustness score: equal-weight arithmetic mean of
        # top-1, top-5 and macro-F1 across all 16 corrupted conditions.
        fixed_score = float(
            np.mean(
                group[["corrupted_top1", "corrupted_top5", "corrupted_macro_f1"]]
                .to_numpy(dtype=float)
            )
        )
        ranking_rows.append(
            {
                "model_name": model_name,
                "method_family": baseline["method_family"],
                "architecture": baseline["architecture"],
                "mean_corrupted_top1": group["corrupted_top1"].mean(),
                "mean_corrupted_top5": group["corrupted_top5"].mean(),
                "mean_corrupted_macro_f1": group["corrupted_macro_f1"].mean(),
                "average_robustness_score": fixed_score,
                "worst_corruption_type": worst["corruption_type"],
                "worst_severity": int(worst["severity"]),
                "worst_severity_value": worst["severity_value"],
                "worst_condition_top1": worst["corrupted_top1"],
                "worst_condition_macro_f1": worst["corrupted_macro_f1"],
                "mean_absolute_top1_drop": group["absolute_top1_drop"].mean(),
                "mean_relative_top1_drop": group["relative_top1_drop"].mean(skipna=True),
                "mean_absolute_macro_f1_drop": group["absolute_macro_f1_drop"].mean(),
                "mean_relative_macro_f1_drop": group["relative_macro_f1_drop"].mean(
                    skipna=True
                ),
                "num_corrupted_conditions": len(group),
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["mean_corrupted_macro_f1", "mean_corrupted_top1", "model_name"],
        ascending=[False, False, True],
        kind="stable",
    )
    ranking.insert(0, "robustness_rank", range(1, len(ranking) + 1))
    ranking = ranking[RANKING_COLUMNS].reset_index(drop=True)
    return drops, ranking


def _atomic_table_set(outputs: dict[Path, pd.DataFrame], *, force: bool) -> None:
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Outputs already exist; use --force deliberately: "
            + ", ".join(str(path) for path in existing)
        )
    parent = next(iter(outputs)).parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage4_robustness_analysis.", dir=parent) as name:
        stage = Path(name)
        staged: dict[Path, Path] = {}
        for index, (destination, frame) in enumerate(outputs.items()):
            temporary = stage / f"{index:02d}_{destination.name}"
            frame.to_csv(temporary, index=False, lineterminator="\n")
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("results/tables/robustness_results.csv")
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/metrics/stage4_robustness_protocol.json"),
    )
    parser.add_argument(
        "--drop-output",
        type=Path,
        default=Path("results/tables/robustness_drop_summary.csv"),
    )
    parser.add_argument(
        "--ranking-output",
        type=Path,
        default=Path("results/tables/robustness_model_ranking.csv"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        if not args.input.is_file():
            raise FileNotFoundError(args.input)
        frame = pd.read_csv(args.input, keep_default_na=False)
        drops, ranking = analyse_robustness(frame, protocol)
        _atomic_table_set(
            {args.drop_output: drops, args.ranking_output: ranking},
            force=bool(args.force),
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"Wrote {len(drops)} corruption-drop rows and {len(ranking)} model rankings."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
