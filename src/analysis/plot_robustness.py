"""Create honest-scale robustness curves and a model summary figure."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_MPL_CACHE = Path(tempfile.gettempdir()) / "comp9517-matplotlib"
_XDG_CACHE = Path(tempfile.gettempdir()) / "comp9517-cache"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
_XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.collect_robustness_results import load_protocol, validate_result_matrix


METRIC_LABELS = {
    "top1": "Top-1 accuracy",
    "macro_f1": "Macro F1",
    "top5": "Top-5 accuracy",
}


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _curve_figure(frame: pd.DataFrame, protocol: Any, metric: str) -> plt.Figure:
    corruptions = list(protocol.corruptions)
    models = list(protocol.models)
    colors = plt.get_cmap("tab10")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    clean = frame.loc[frame["corruption_type"] == "clean"].set_index("model_name")
    for axis, corruption in zip(axes.flat, corruptions):
        subset = frame.loc[frame["corruption_type"] == corruption]
        definition = protocol.corruptions[corruption]
        for index, model in enumerate(models):
            model_rows = subset.loc[subset["model_name"] == model].sort_values("severity")
            x = [0, *model_rows["severity"].astype(int).tolist()]
            y = [float(clean.loc[model, metric]), *model_rows[metric].astype(float).tolist()]
            axis.plot(x, y, marker="o", linewidth=2, label=model, color=colors(index))
        labels = ["0\nclean"] + [
            f"{index}\n{value:g}"
            for index, value in enumerate(definition["severities"], start=1)
        ]
        axis.set_xticks(range(5), labels=labels)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(f"Severity index / {definition['parameter']}")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.set_title(corruption.replace("_", " ").title())
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle(f"Stage 4 robustness: {METRIC_LABELS[metric]}")
    figure.tight_layout(rect=(0, 0.09, 1, 0.96))
    return figure


def _summary_figure(frame: pd.DataFrame, protocol: Any) -> plt.Figure:
    corrupted = frame.loc[frame["corruption_type"] != "clean"]
    summary = (
        corrupted.groupby("model_name", as_index=False)
        .agg(mean_top1=("top1", "mean"), mean_macro_f1=("macro_f1", "mean"))
        .set_index("model_name")
        .loc[list(protocol.models)]
        .reset_index()
    )
    positions = np.arange(len(summary), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(positions - width / 2, summary["mean_top1"], width, label="Mean top-1")
    axis.bar(
        positions + width / 2,
        summary["mean_macro_f1"],
        width,
        label="Mean macro F1",
    )
    axis.set_xticks(positions, labels=summary["model_name"], rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean score over 16 corrupted conditions")
    axis.set_title("Stage 4 average corrupted performance")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def create_plots(
    frame: pd.DataFrame,
    protocol: Any,
    output_dir: Path,
    *,
    force: bool,
) -> list[Path]:
    validate_result_matrix(frame, protocol)
    outputs = [
        output_dir / "robustness_top1_curves.png",
        output_dir / "robustness_macro_f1_curves.png",
        output_dir / "robustness_top5_curves.png",
        output_dir / "robustness_summary_bar.png",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Figures already exist; use --force deliberately: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".robustness_plots.", dir=output_dir) as name:
        stage = Path(name)
        staged = [stage / path.name for path in outputs]
        for metric, path in zip(("top1", "macro_f1", "top5"), staged[:3]):
            _save(_curve_figure(frame, protocol, metric), path)
        _save(_summary_figure(frame, protocol), staged[3])
        backups: dict[Path, Path] = {}
        installed: list[Path] = []
        try:
            for index, destination in enumerate(existing):
                backup = stage / f"backup_{index:02d}_{destination.name}"
                os.replace(destination, backup)
                backups[destination] = backup
            for destination, source in zip(outputs, staged):
                os.replace(source, destination)
                installed.append(destination)
        except Exception:
            for destination in installed:
                destination.unlink(missing_ok=True)
            for destination, backup in backups.items():
                if backup.exists():
                    os.replace(backup, destination)
            raise
    return outputs


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
        "--output-dir", type=Path, default=Path("results/figures/robustness")
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
        outputs = create_plots(frame, protocol, args.output_dir, force=bool(args.force))
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {len(outputs)} robustness figures to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
