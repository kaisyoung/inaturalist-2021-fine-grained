"""Deterministically select frozen Stage 3 test examples for Grad-CAM.

Selection is analysis-only: it reads the already-persisted prediction and
confusion artifacts for the frozen EfficientNet analysis model.  It never
loads a checkpoint or performs inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from src.data.common import is_safe_relative_posix_path, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FREEZE = Path("results/metrics/deep_learning/stage3_freeze.json")
DEFAULT_ACCESS = Path("results/metrics/deep_learning/stage3_test_access.json")
DEFAULT_OUTPUT = Path("results/tables/gradcam_selected_examples.csv")
DEFAULT_MANIFEST = Path(
    "results/metrics/explainability/gradcam_selection_manifest.json"
)
DEFAULT_FIGURE_DIR = Path("results/figures/gradcam")
DEFAULT_IMAGE_ROOT = Path("data/processed/images")

SELECTION_COLUMNS = [
    "selection_rank",
    "image_id",
    "local_image_path",
    "true_label",
    "predicted_label",
    "top1_score",
    "correct",
    "selection_type",
    "scientific_name",
    "common_name",
    "predicted_scientific_name",
    "predicted_common_name",
    "confused_pair_id",
    "confused_pair_rank",
    "pair_direction",
    "source_prediction_row",
    "analysis_experiment",
    "prediction_artifact_sha256",
    "confusion_artifact_sha256",
    "stage3_freeze_sha256",
    "stage3_test_access_sha256",
    "test_manifest_sha256",
]

_PREDICTION_COLUMNS = {
    "image_id",
    "local_image_path",
    "true_label",
    "predicted_label",
    "correct",
    "top1_score",
    "true_scientific_name",
    "true_common_name",
    "predicted_scientific_name",
    "predicted_common_name",
}
_PAIR_COLUMNS = {
    "rank",
    "class_a_new_label",
    "class_b_new_label",
    "a_as_b",
    "b_as_a",
    "total_confusions",
}
_SELECTION_TYPES = (
    "correct_high_confidence",
    "incorrect_high_confidence",
    "confused_pair",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": _portable(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_bound_artifact(entry: Any, label: str) -> Path:
    if not isinstance(entry, Mapping):
        raise ValueError(f"Missing or malformed artifact binding: {label}")
    path = _resolve(str(entry.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"Bound artifact is missing ({label}): {path}")
    observed_hash = sha256_file(path)
    if observed_hash != entry.get("sha256"):
        raise ValueError(f"Bound artifact hash changed: {label}")
    if int(entry.get("bytes", -1)) != path.stat().st_size:
        raise ValueError(f"Bound artifact size changed: {label}")
    return path


def validate_frozen_selection_sources(
    *,
    freeze_path: Path,
    access_path: Path,
    predictions_path: Path | None = None,
    pairs_path: Path | None = None,
) -> dict[str, Any]:
    """Validate and return hash-bound frozen analysis inputs."""

    freeze_path = freeze_path.resolve()
    access_path = access_path.resolve()
    if not freeze_path.is_file() or not access_path.is_file():
        raise FileNotFoundError("Stage 3 freeze and completed test-access record are required")
    freeze_hash = sha256_file(freeze_path)
    access_hash = sha256_file(access_path)
    freeze = _read_json(freeze_path)
    access = _read_json(access_path)
    if freeze.get("schema_version") != 1 or freeze.get("status") != "frozen":
        raise ValueError("Stage 3 freeze manifest is not valid and frozen")
    analysis_model = freeze.get("analysis_model")
    models = freeze.get("models")
    selected = freeze.get("selected_experiments")
    if (
        not isinstance(analysis_model, str)
        or not isinstance(models, Mapping)
        or analysis_model not in models
        or not isinstance(selected, list)
        or analysis_model not in selected
    ):
        raise ValueError("Stage 3 freeze has no valid selected analysis model")
    model_entry = models[analysis_model]
    if (
        not isinstance(model_entry, Mapping)
        or model_entry.get("model_name") != "efficientnet_b0"
        or not bool(model_entry.get("pretrained"))
    ):
        raise ValueError(
            "Canonical Stage 4 Grad-CAM requires the frozen pretrained "
            "EfficientNet-B0 analysis model"
        )
    if (
        access.get("schema_version") != 1
        or access.get("status") != "completed"
        or access.get("outcome") != "complete"
        or access.get("freeze_manifest_sha256") != freeze_hash
        or access.get("analysis_model") != analysis_model
        or access.get("selected_experiments") != selected
    ):
        raise ValueError("Stage 3 test-access record is incomplete or not bound to the freeze")
    attempts = access.get("attempts")
    attempt = attempts.get(analysis_model) if isinstance(attempts, Mapping) else None
    if not isinstance(attempt, Mapping) or attempt.get("status") != "complete":
        raise ValueError("Frozen analysis model has no completed Stage 3 test attempt")
    artifacts = attempt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Frozen analysis-model test artifacts are missing")
    bound_predictions = _verify_bound_artifact(
        artifacts.get("predictions"), f"{analysis_model}.test_predictions"
    )
    bound_pairs = _verify_bound_artifact(
        artifacts.get("pairs"), f"{analysis_model}.test_top_confusions"
    )
    if predictions_path is not None and predictions_path.resolve() != bound_predictions:
        raise ValueError("Explicit predictions path is not the frozen analysis artifact")
    if pairs_path is not None and pairs_path.resolve() != bound_pairs:
        raise ValueError("Explicit confusion-pairs path is not the frozen analysis artifact")

    stage1 = freeze.get("stage1_artifacts")
    test_entry = stage1.get("test.csv") if isinstance(stage1, Mapping) else None
    test_path = _verify_bound_artifact(test_entry, "Stage1.test.csv")
    return {
        "freeze": freeze,
        "freeze_path": freeze_path,
        "freeze_sha256": freeze_hash,
        "access": access,
        "access_path": access_path,
        "access_sha256": access_hash,
        "analysis_model": analysis_model,
        "model_entry": model_entry,
        "predictions_path": bound_predictions,
        "pairs_path": bound_pairs,
        "test_path": test_path,
        "test_sha256": str(test_entry["sha256"]),
    }


def _coerce_bool(series: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"Column {name!r} contains invalid booleans")
    return mapped.astype(bool)


def _validated_predictions(path: Path, test_path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
        manifest = pd.read_csv(test_path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read frozen prediction inputs: {error}") from error
    missing = sorted(_PREDICTION_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Frozen predictions are missing columns: {missing}")
    if len(frame) != 5000 or len(manifest) != 5000:
        raise ValueError("Frozen test predictions and manifest must each contain 5,000 rows")
    frame = frame.copy()
    frame["source_prediction_row"] = np.arange(len(frame), dtype=np.int64)
    for column in ("image_id", "true_label", "predicted_label"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"Frozen prediction column {column!r} is not integral")
        frame[column] = values.astype(np.int64)
    frame["top1_score"] = pd.to_numeric(frame["top1_score"], errors="coerce")
    if (
        not np.isfinite(frame["top1_score"].to_numpy(dtype=float)).all()
        or not frame["top1_score"].between(0.0, 1.0).all()
    ):
        raise ValueError("Frozen prediction top1_score must be finite and in [0,1]")
    frame["correct"] = _coerce_bool(frame["correct"], "correct")
    if not (frame["correct"] == (frame["true_label"] == frame["predicted_label"])).all():
        raise ValueError("Frozen prediction correctness flags are inconsistent")
    if frame["image_id"].duplicated().any() or frame["local_image_path"].duplicated().any():
        raise ValueError("Frozen predictions contain duplicate image identifiers or paths")
    if not frame["local_image_path"].astype(str).map(is_safe_relative_posix_path).all():
        raise ValueError("Frozen predictions contain unsafe local_image_path values")
    for prediction_column, manifest_column in (
        ("image_id", "image_id"),
        ("local_image_path", "local_image_path"),
        ("true_label", "new_label"),
    ):
        left = frame[prediction_column].astype(str).to_numpy()
        right = manifest[manifest_column].astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(
                f"Frozen predictions are not in test-manifest order: {prediction_column}"
            )
    return frame


def _validated_pairs(path: Path, predictions: pd.DataFrame) -> pd.DataFrame:
    try:
        pairs = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read frozen confusion pairs: {error}") from error
    missing = sorted(_PAIR_COLUMNS - set(pairs.columns))
    if missing:
        raise ValueError(f"Frozen confusion table is missing columns: {missing}")
    pairs = pairs.copy()
    for column in _PAIR_COLUMNS:
        values = pd.to_numeric(pairs[column], errors="coerce")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"Confusion column {column!r} is not integral")
        pairs[column] = values.astype(np.int64)
    pairs = pairs.sort_values(
        ["rank", "class_a_new_label", "class_b_new_label"], kind="mergesort"
    ).reset_index(drop=True)
    if pairs.empty or pairs["rank"].duplicated().any():
        raise ValueError("Frozen confusion table must have unique ranked rows")
    for row in pairs.itertuples(index=False):
        a = int(row.class_a_new_label)
        b = int(row.class_b_new_label)
        observed_ab = int(
            ((predictions["true_label"] == a) & (predictions["predicted_label"] == b)).sum()
        )
        observed_ba = int(
            ((predictions["true_label"] == b) & (predictions["predicted_label"] == a)).sum()
        )
        if (
            observed_ab != int(row.a_as_b)
            or observed_ba != int(row.b_as_a)
            or observed_ab + observed_ba != int(row.total_confusions)
        ):
            raise ValueError(f"Confusion-pair counts disagree with predictions at rank {row.rank}")
    return pairs


def _ranked(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["top1_score", "image_id", "source_prediction_row"],
        ascending=[False, True, True],
        kind="mergesort",
    )


def select_examples(
    predictions: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    correct_count: int = 12,
    incorrect_count: int = 12,
    pair_count: int = 5,
) -> pd.DataFrame:
    """Return deterministic unique correct, incorrect, and pair examples."""

    for name, value in {
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "pair_count": pair_count,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    required = _PREDICTION_COLUMNS | {"source_prediction_row"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing selection columns: {missing}")
    missing_pairs = sorted(_PAIR_COLUMNS - set(pairs.columns))
    if missing_pairs:
        raise ValueError(f"Pairs are missing selection columns: {missing_pairs}")

    pair_records: list[dict[str, Any]] = []
    pair_pool_ids: set[int] = set()
    used_ids: set[int] = set()
    for pair in pairs.sort_values(
        ["rank", "class_a_new_label", "class_b_new_label"], kind="mergesort"
    ).head(pair_count).itertuples(index=False):
        a, b, rank = (
            int(pair.class_a_new_label),
            int(pair.class_b_new_label),
            int(pair.rank),
        )
        pair_id = f"pair_{rank:03d}_{a:03d}_{b:03d}"
        for source, destination in ((a, b), (b, a)):
            direction_pool = predictions.loc[
                (predictions["true_label"] == source)
                & (predictions["predicted_label"] == destination)
            ]
            pair_pool_ids.update(int(value) for value in direction_pool["image_id"])
            candidates = direction_pool.loc[
                ~direction_pool["image_id"].astype(int).isin(used_ids)
            ]
            if candidates.empty:
                continue
            chosen = _ranked(candidates).iloc[0].to_dict()
            image_id = int(chosen["image_id"])
            used_ids.add(image_id)
            chosen.update(
                {
                    "selection_type": "confused_pair",
                    "confused_pair_id": pair_id,
                    "confused_pair_rank": rank,
                    "pair_direction": f"{source}_to_{destination}",
                }
            )
            pair_records.append(chosen)

    correct_candidates = predictions.loc[predictions["correct"]].copy()
    correct = _ranked(correct_candidates).head(correct_count)
    if len(correct) != correct_count:
        raise ValueError(f"Only {len(correct)} correct examples are available")
    used_ids.update(int(value) for value in correct["image_id"])

    incorrect_candidates = predictions.loc[
        (~predictions["correct"])
        & (~predictions["image_id"].astype(int).isin(used_ids | pair_pool_ids))
    ].copy()
    incorrect = _ranked(incorrect_candidates).head(incorrect_count)
    if len(incorrect) != incorrect_count:
        raise ValueError(f"Only {len(incorrect)} non-pair incorrect examples are available")
    used_ids.update(int(value) for value in incorrect["image_id"])

    records: list[dict[str, Any]] = []
    for selection_type, selected in (
        ("correct_high_confidence", correct),
        ("incorrect_high_confidence", incorrect),
    ):
        for row in selected.to_dict(orient="records"):
            row.update(
                {
                    "selection_type": selection_type,
                    "confused_pair_id": "",
                    "confused_pair_rank": "",
                    "pair_direction": "",
                }
            )
            records.append(row)
    records.extend(pair_records)
    selected = pd.DataFrame.from_records(records)
    if selected["image_id"].duplicated().any():
        raise RuntimeError("Selection unexpectedly contains duplicate images")
    selected.insert(0, "selection_rank", np.arange(1, len(selected) + 1))
    return selected


def _finalize_selection(
    selected: pd.DataFrame,
    source: Mapping[str, Any],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "selection_rank": selected["selection_rank"].astype(int),
            "image_id": selected["image_id"].astype(int),
            "local_image_path": selected["local_image_path"].astype(str),
            "true_label": selected["true_label"].astype(int),
            "predicted_label": selected["predicted_label"].astype(int),
            "top1_score": selected["top1_score"].astype(float),
            "correct": selected["correct"].astype(bool),
            "selection_type": selected["selection_type"].astype(str),
            "scientific_name": selected["true_scientific_name"].fillna("").astype(str),
            "common_name": selected["true_common_name"].fillna("").astype(str),
            "predicted_scientific_name": selected["predicted_scientific_name"].fillna("").astype(str),
            "predicted_common_name": selected["predicted_common_name"].fillna("").astype(str),
            "confused_pair_id": selected["confused_pair_id"].fillna("").astype(str),
            "confused_pair_rank": selected["confused_pair_rank"],
            "pair_direction": selected["pair_direction"].fillna("").astype(str),
            "source_prediction_row": selected["source_prediction_row"].astype(int),
            "analysis_experiment": str(source["analysis_model"]),
            "prediction_artifact_sha256": sha256_file(source["predictions_path"]),
            "confusion_artifact_sha256": sha256_file(source["pairs_path"]),
            "stage3_freeze_sha256": str(source["freeze_sha256"]),
            "stage3_test_access_sha256": str(source["access_sha256"]),
            "test_manifest_sha256": str(source["test_sha256"]),
        }
    )
    frame["confused_pair_rank"] = frame["confused_pair_rank"].map(
        lambda value: "" if value == "" or pd.isna(value) else int(value)
    )
    if list(frame.columns) != SELECTION_COLUMNS:
        raise RuntimeError("Internal Grad-CAM selection schema error")
    return frame


def _safe_image_path(image_root: Path, relative: str) -> Path:
    if not is_safe_relative_posix_path(relative):
        raise ValueError(f"Unsafe selected local image path: {relative!r}")
    candidate = image_root.joinpath(*PurePosixPath(relative).parts).resolve()
    try:
        candidate.relative_to(image_root)
    except ValueError as error:
        raise ValueError(f"Selected image escapes image root: {relative!r}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"Selected image is missing: {candidate}")
    return candidate


def _selection_preview_grid(
    rows: pd.DataFrame,
    *,
    image_root: Path,
    output: Path,
) -> None:
    """Write a compact original-image preview; canonical CAM grids come later."""

    columns = 4
    cell_width, cell_height = 260, 235
    image_height = 180
    row_count = max(1, math.ceil(len(rows) / columns))
    canvas = Image.new("RGB", (columns * cell_width, row_count * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows.itertuples(index=False)):
        path = _safe_image_path(image_root, str(row.local_image_path))
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                image.thumbnail((cell_width - 12, image_height), Image.Resampling.LANCZOS)
                image.load()
        except Exception as error:
            raise ValueError(f"Could not decode selected image {path}: {error}") from error
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        canvas.paste(
            image,
            (x + (cell_width - image.width) // 2, y + (image_height - image.height) // 2),
        )
        label = (
            f"id {int(row.image_id)} | true {int(row.true_label)} -> "
            f"pred {int(row.predicted_label)}\np={float(row.top1_score):.3f}"
        )
        draw.multiline_text((x + 6, y + image_height + 4), label, fill="black", spacing=2)
    canvas.save(output, format="PNG", optimize=False)


def _atomic_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _install_new(staged: Mapping[Path, Path]) -> None:
    conflicts = [target for target in staged if target.exists()]
    if conflicts:
        raise FileExistsError(
            "Grad-CAM selection refuses existing/conflicting outputs: "
            + ", ".join(str(path) for path in conflicts)
        )
    installed: list[Path] = []
    try:
        for target, temporary in staged.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, target)
            installed.append(target)
    except Exception:
        for path in installed:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def run_selection(args: argparse.Namespace) -> dict[str, Any]:
    output = _resolve(args.output)
    manifest_path = _resolve(args.manifest)
    figure_dir = _resolve(args.figure_dir)
    image_root = _resolve(args.image_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root is missing: {image_root}")
    source = validate_frozen_selection_sources(
        freeze_path=_resolve(args.freeze_manifest),
        access_path=_resolve(args.test_access),
        predictions_path=_resolve(args.predictions) if args.predictions else None,
        pairs_path=_resolve(args.confusion_pairs) if args.confusion_pairs else None,
    )
    predictions = _validated_predictions(source["predictions_path"], source["test_path"])
    pairs = _validated_pairs(source["pairs_path"], predictions)
    selected = select_examples(
        predictions,
        pairs,
        correct_count=int(args.correct_count),
        incorrect_count=int(args.incorrect_count),
        pair_count=int(args.pair_count),
    )
    finalized = _finalize_selection(selected, source)

    preview_paths = {
        selection_type: figure_dir / f"selection_{selection_type}_grid.png"
        for selection_type in _SELECTION_TYPES
    }
    all_targets = [output, manifest_path, *preview_paths.values()]
    existing = [path for path in all_targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Grad-CAM selection refuses existing/conflicting outputs: "
            + ", ".join(str(path) for path in existing)
        )

    staged: dict[Path, Path] = {}
    try:
        csv_text = finalized.to_csv(index=False, lineterminator="\n")
        staged[output] = _atomic_text(output, csv_text)
        for selection_type, target in preview_paths.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, name = tempfile.mkstemp(
                prefix=f".{target.stem}.", suffix=".png", dir=target.parent
            )
            os.close(handle)
            temporary = Path(name)
            _selection_preview_grid(
                finalized.loc[finalized["selection_type"] == selection_type],
                image_root=image_root,
                output=temporary,
            )
            staged[target] = temporary

        # Bind the bytes staged for the selected CSV and previews before
        # writing the manifest.  Paths are final portable destinations.
        manifest = {
            "schema_version": 1,
            "status": "selected",
            "selection_policy": {
                "correct_high_confidence": int(args.correct_count),
                "incorrect_high_confidence_excluding_pair_examples": int(args.incorrect_count),
                "top_confused_pairs": int(args.pair_count),
                "representatives_per_available_direction": 1,
                "ranking": "top1_score descending, image_id ascending",
                "unique_images": True,
            },
            "analysis_experiment": source["analysis_model"],
            "inputs": {
                "stage3_freeze": _artifact(source["freeze_path"]),
                "stage3_test_access": _artifact(source["access_path"]),
                "test_manifest": _artifact(source["test_path"]),
                "predictions": _artifact(source["predictions_path"]),
                "confusion_pairs": _artifact(source["pairs_path"]),
            },
            "outputs": {
                "selected_examples": {
                    "path": _portable(output),
                    "bytes": staged[output].stat().st_size,
                    "sha256": sha256_file(staged[output]),
                },
                "selection_preview_grids": {
                    key: {
                        "path": _portable(path),
                        "bytes": staged[path].stat().st_size,
                        "sha256": sha256_file(staged[path]),
                    }
                    for key, path in preview_paths.items()
                },
            },
            "counts": {
                key: int((finalized["selection_type"] == key).sum())
                for key in _SELECTION_TYPES
            },
            "total_selected": int(len(finalized)),
            "columns": SELECTION_COLUMNS,
            # No timestamp: the same frozen inputs produce byte-identical CSV
            # and manifest artifacts.
        }
        manifest_text = json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"
        staged[manifest_path] = _atomic_text(manifest_path, manifest_text)
        _install_new(staged)
    except Exception:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise

    print(
        f"[PASS] selected {len(finalized)} unique Grad-CAM examples for "
        f"{source['analysis_model']}"
    )
    print(f"[PASS] selected examples: {_portable(output)}")
    print(f"[PASS] selection manifest: {_portable(manifest_path)}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select deterministic examples from the frozen Stage 3 analysis model."
    )
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--test-access", type=Path, default=DEFAULT_ACCESS)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optional path; must equal the prediction artifact bound by test access.",
    )
    parser.add_argument(
        "--confusion-pairs",
        type=Path,
        help="Optional path; must equal the confusion table bound by test access.",
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--correct-count", type=int, default=12)
    parser.add_argument("--incorrect-count", type=int, default=12)
    parser.add_argument("--pair-count", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_selection(args)
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MANIFEST",
    "DEFAULT_OUTPUT",
    "SELECTION_COLUMNS",
    "build_parser",
    "main",
    "run_selection",
    "select_examples",
    "validate_frozen_selection_sources",
]
