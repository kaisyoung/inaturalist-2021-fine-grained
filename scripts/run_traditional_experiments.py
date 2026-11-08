"""Run the exact 19-configuration Stage 2 validation grid (never test)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


@dataclass(frozen=True)
class Experiment:
    name: str
    model: str
    feature_set: str
    components: tuple[str, ...]
    alpha: float
    blockwise: bool = True

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return (
            self.model,
            self.feature_set,
            self.components,
            self.alpha,
            self.blockwise,
        )


def experiment_grid() -> tuple[Experiment, ...]:
    """Return the reviewed grid with no duplicate scientific configuration."""

    handcrafted = ("color", "texture", "hog")
    experiments = [
        Experiment("linear_svm", "svm", "combined", (*handcrafted, "bovw"), 0.3),
        Experiment("random_forest", "random_forest", "handcrafted", handcrafted, 0.3),
        Experiment("sift_bovw_svm", "svm", "bovw", ("bovw",), 0.3),
        Experiment("ablation_cb_color_bovw_a03", "svm", "combined", ("color", "bovw"), 0.3),
        Experiment("ablation_cb_color_bovw_a05", "svm", "combined", ("color", "bovw"), 0.5),
        Experiment("ablation_drop_color", "svm", "handcrafted", ("texture", "hog"), 0.5),
        Experiment("ablation_drop_texture", "svm", "handcrafted", ("color", "hog"), 0.5),
        Experiment("ablation_drop_hog", "svm", "handcrafted", ("color", "texture"), 0.5),
        Experiment("ablation_only_color", "svm", "handcrafted", ("color",), 0.5),
        Experiment("ablation_only_texture", "svm", "handcrafted", ("texture",), 0.5),
        Experiment("ablation_only_hog", "svm", "handcrafted", ("hog",), 0.5),
    ]
    experiments.extend(
        Experiment(
            f"ablation_norm_blk_a{str(alpha).replace('.', '')}",
            "svm",
            "handcrafted",
            handcrafted,
            alpha,
            True,
        )
        for alpha in (0.1, 0.2, 0.3, 0.5, 1.0)
    )
    experiments.extend(
        Experiment(
            f"ablation_norm_glb_a{str(alpha).replace('.', '')}",
            "svm",
            "handcrafted",
            handcrafted,
            alpha,
            False,
        )
        for alpha in (0.3, 0.5, 1.0)
    )
    result = tuple(experiments)
    if len(result) != 19 or len({item.name for item in result}) != 19:
        raise RuntimeError("Internal error: validation grid is not exactly 19 names")
    if len({item.fingerprint for item in result}) != 19:
        raise RuntimeError("Internal error: validation grid has duplicate configurations")
    return result


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}")


def _matches(document: dict, experiment: Experiment) -> bool:
    normalisation = document.get("normalisation") or {}
    expected_classifier = "RandomForestClassifier" if experiment.model == "random_forest" else "LinearSVC"
    return bool(
        document.get("status") == "complete"
        and document.get("split") == "val"
        and document.get("experiment_name") == experiment.name
        and document.get("classifier") == expected_classifier
        and document.get("feature_set") == experiment.feature_set
        and tuple(document.get("feature_components", ())) == experiment.components
        and abs(float(normalisation.get("alpha", -1)) - experiment.alpha) < 1e-12
        and normalisation.get("blockwise") is experiment.blockwise
    )


def run_one(experiment: Experiment, dangerous_overwrite: bool) -> str:
    metrics_path = METRICS_DIR / f"traditional_{experiment.name}_val_metrics.json"
    if metrics_path.is_file() and not dangerous_overwrite:
        document = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not _matches(document, experiment):
            raise RuntimeError(
                f"Stale/mismatched validation evidence exists for {experiment.name}; "
                "inspect it before dangerous replacement"
            )
        return f"{experiment.name}: verified cache"

    train = [
        sys.executable,
        "-m",
        "src.traditional.train_traditional",
        "--model",
        experiment.model,
        "--feature-set",
        experiment.feature_set,
        "--components",
        ",".join(experiment.components),
        "--alpha",
        str(experiment.alpha),
        "--experiment-name",
        experiment.name,
    ]
    if experiment.model == "svm":
        train.extend(("--svm-backend", "linearsvc"))
    if not experiment.blockwise:
        train.append("--global-norm")
    if dangerous_overwrite:
        train.append("--dangerously-overwrite")
    _run(train)

    evaluate = [
        sys.executable,
        "-m",
        "src.traditional.evaluate_traditional",
        "--model",
        f"results/models/traditional/{experiment.name}.joblib",
        "--split",
        "val",
    ]
    if dangerous_overwrite:
        evaluate.append("--dangerously-overwrite-validation")
    _run(evaluate)
    return f"{experiment.name}: complete"


def validation_table(grid: Sequence[Experiment]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment in grid:
        path = METRICS_DIR / f"traditional_{experiment.name}_val_metrics.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        if not _matches(document, experiment):
            raise ValueError(f"Validation evidence does not match grid: {experiment.name}")
        metrics = document["metrics"]
        rows.append(
            {
                "experiment": experiment.name,
                "classifier": document["classifier"],
                "feature_set": experiment.feature_set,
                "components": "+".join(experiment.components),
                "alpha": experiment.alpha,
                "blockwise": experiment.blockwise,
                "dim": document["feature_dim"],
                "top1": metrics["top1_accuracy"],
                "top5": metrics["top5_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "train_s": document["training_time_seconds"],
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--dangerously-overwrite-existing",
        action="store_true",
        help="replace existing model and validation output sets",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results/tables/traditional_validation_grid.csv",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    grid = experiment_grid()
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for message in executor.map(
            lambda item: run_one(item, args.dangerously_overwrite_existing), grid
        ):
            print(message, flush=True)
    frame = validation_table(grid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False, lineterminator="\n")
    print(f"WROTE {args.out} ({len(frame)} unique configurations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
