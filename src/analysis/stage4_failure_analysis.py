"""Build advanced Stage 4 failure tables from frozen persisted evidence only.

No model is loaded and no inference is performed.  The inputs are the frozen
EfficientNet test predictions/per-class/confusion artifacts and the hash-bound
Grad-CAM selection table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.common import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_MODEL = "efficientnet_b0_pretrained_strong_aug"
OUTPUT_NAMES = {
    "hardest": "stage4_hardest_classes.csv",
    "easiest": "stage4_easiest_classes.csv",
    "high_failures": "stage4_high_confidence_failures.csv",
    "low_correct": "stage4_low_confidence_correct.csv",
    "confused_pairs": "stage4_confused_species_pairs.csv",
}
PROVENANCE_COLUMNS = [
    "analysis_experiment",
    "stage3_freeze_sha256",
    "stage3_test_access_sha256",
    "prediction_artifact_sha256",
    "per_class_artifact_sha256",
    "confusion_artifact_sha256",
    "gradcam_selection_sha256",
    "gradcam_selection_manifest_sha256",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _resolve(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _artifact(entry: Any, *, label: str) -> Path:
    if not isinstance(entry, Mapping):
        raise ValueError(f"Malformed artifact binding: {label}")
    path = _resolve(entry.get("path", ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"Artifact hash mismatch: {label}")
    if entry.get("bytes") is not None and path.stat().st_size != int(entry["bytes"]):
        raise ValueError(f"Artifact byte-size mismatch: {label}")
    return path


def _find_bound_artifact(document: Mapping[str, Any], expected_path: Path) -> Mapping[str, Any] | None:
    for value in document.values():
        if isinstance(value, Mapping):
            if "path" in value and "sha256" in value:
                try:
                    if _resolve(value["path"]) == expected_path.resolve():
                        return value
                except (OSError, ValueError):
                    pass
            nested = _find_bound_artifact(value, expected_path)
            if nested is not None:
                return nested
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    nested = _find_bound_artifact(item, expected_path)
                    if nested is not None:
                        return nested
    return None


def _bool_series(series: pd.Series, label: str) -> pd.Series:
    mapped = series.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if mapped.isna().any():
        raise ValueError(f"{label} contains invalid booleans")
    return mapped.astype(bool)


def _verify_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    freeze_path = Path(args.freeze_manifest).resolve()
    access_path = Path(args.test_access).resolve()
    freeze = _load_json(freeze_path)
    access = _load_json(access_path)
    freeze_hash = sha256_file(freeze_path)
    access_hash = sha256_file(access_path)
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status") != "frozen"
        or freeze.get("analysis_model") != ANALYSIS_MODEL
        or ANALYSIS_MODEL not in freeze.get("selected_experiments", [])
    ):
        raise ValueError("Invalid Stage 3 freeze or analysis-model designation")
    attempts = access.get("attempts")
    if (
        access.get("status") != "completed"
        or access.get("outcome") != "complete"
        or access.get("freeze_manifest_sha256") != freeze_hash
        or not isinstance(attempts, Mapping)
        or ANALYSIS_MODEL not in attempts
        or attempts[ANALYSIS_MODEL].get("status") != "complete"
    ):
        raise ValueError("Stage 3 frozen-set test access is incomplete or mismatched")
    attempt = attempts[ANALYSIS_MODEL]
    metrics_entry = attempt.get("artifacts", {}).get("metrics")
    metrics_path = _artifact(metrics_entry, label="stage3.analysis_model.metrics")
    metrics_doc = _load_json(metrics_path)
    metric_artifacts = metrics_doc.get("artifacts")
    if (
        metrics_doc.get("status") != "complete"
        or metrics_doc.get("split") != "test"
        or metrics_doc.get("experiment_name") != ANALYSIS_MODEL
        or not isinstance(metric_artifacts, Mapping)
    ):
        raise ValueError("Invalid frozen EfficientNet test metrics")
    artifacts: dict[str, Path] = {}
    for key in ("predictions", "per_class", "pairs"):
        if attempt.get("artifacts", {}).get(key) != metric_artifacts.get(key):
            raise ValueError(f"Stage 3 access/metrics artifact mismatch: {key}")
        artifacts[key] = _artifact(metric_artifacts.get(key), label=f"stage3.{key}")

    selection_path = Path(args.gradcam_selection).resolve()
    selection_manifest_path = Path(args.gradcam_selection_manifest).resolve()
    selection_manifest = _load_json(selection_manifest_path)
    binding = _find_bound_artifact(selection_manifest, selection_path)
    if binding is None or _artifact(binding, label="gradcam.selection") != selection_path:
        raise ValueError("Grad-CAM selection manifest does not bind the selected CSV")

    predictions = pd.read_csv(artifacts["predictions"], keep_default_na=False)
    per_class = pd.read_csv(artifacts["per_class"], keep_default_na=False)
    confusions = pd.read_csv(artifacts["pairs"], keep_default_na=False)
    selected = pd.read_csv(selection_path, keep_default_na=False)
    provenance = {
        "analysis_experiment": ANALYSIS_MODEL,
        "stage3_freeze_sha256": freeze_hash,
        "stage3_test_access_sha256": access_hash,
        "prediction_artifact_sha256": sha256_file(artifacts["predictions"]),
        "per_class_artifact_sha256": sha256_file(artifacts["per_class"]),
        "confusion_artifact_sha256": sha256_file(artifacts["pairs"]),
        "gradcam_selection_sha256": sha256_file(selection_path),
        "gradcam_selection_manifest_sha256": sha256_file(selection_manifest_path),
    }
    return predictions, per_class, confusions, selected, provenance


def build_failure_tables(
    predictions: pd.DataFrame,
    per_class: pd.DataFrame,
    confusions: pd.DataFrame,
    selected: pd.DataFrame,
    provenance: Mapping[str, str],
    *,
    expected_images: int = 5_000,
    expected_classes: int = 500,
    expected_support: int = 10,
) -> dict[str, pd.DataFrame]:
    prediction_required = {
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
    per_class_required = {
        "new_label",
        "scientific_name",
        "common_name",
        "precision",
        "recall",
        "f1",
        "support",
        "predicted_count",
        "true_positives",
    }
    confusion_required = {
        "rank",
        "class_a_new_label",
        "class_a_scientific_name",
        "class_b_new_label",
        "class_b_scientific_name",
        "a_as_b",
        "b_as_a",
        "total_confusions",
    }
    selected_required = {
        "image_id",
        "local_image_path",
        "true_label",
        "predicted_label",
        "top1_score",
        "correct",
        "selection_type",
        "source_prediction_row",
        "analysis_experiment",
        "prediction_artifact_sha256",
        "confusion_artifact_sha256",
        "stage3_freeze_sha256",
        "stage3_test_access_sha256",
    }
    for label, frame, required in (
        ("predictions", predictions, prediction_required),
        ("per-class", per_class, per_class_required),
        ("confusions", confusions, confusion_required),
        ("Grad-CAM selection", selected, selected_required),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} table is missing columns: {missing}")
    if len(predictions) != expected_images or len(per_class) != expected_classes:
        raise ValueError("Frozen prediction/per-class counts are invalid")

    predictions = predictions.copy()
    per_class = per_class.copy()
    confusions = confusions.copy()
    selected = selected.copy()
    for column in ("image_id", "true_label", "predicted_label"):
        predictions[column] = pd.to_numeric(predictions[column], errors="raise").astype("int64")
    predictions["correct"] = _bool_series(predictions["correct"], "predictions.correct")
    predictions["top1_score"] = pd.to_numeric(predictions["top1_score"], errors="raise")
    if (
        predictions["image_id"].duplicated().any()
        or predictions["local_image_path"].duplicated().any()
        or not predictions["true_label"].between(0, expected_classes - 1).all()
        or not predictions["predicted_label"].between(0, expected_classes - 1).all()
        or not np.isfinite(predictions["top1_score"]).all()
        or not predictions["top1_score"].between(0.0, 1.0).all()
        or not (
            predictions["correct"].to_numpy()
            == (predictions["true_label"].to_numpy() == predictions["predicted_label"].to_numpy())
        ).all()
    ):
        raise ValueError("Frozen prediction table violates identity/score invariants")

    for column in ("new_label", "support", "predicted_count", "true_positives"):
        per_class[column] = pd.to_numeric(per_class[column], errors="raise").astype("int64")
    for column in ("precision", "recall", "f1"):
        per_class[column] = pd.to_numeric(per_class[column], errors="raise")
        if not np.isfinite(per_class[column]).all() or not per_class[column].between(0, 1).all():
            raise ValueError(f"Invalid per-class {column}")
    if (
        set(per_class["new_label"]) != set(range(expected_classes))
        or per_class["new_label"].duplicated().any()
        or not (per_class["support"] == expected_support).all()
    ):
        raise ValueError("Per-class labels/support do not match the frozen contract")
    actual_support = predictions.groupby("true_label").size().reindex(range(expected_classes), fill_value=0)
    actual_predicted = predictions.groupby("predicted_label").size().reindex(range(expected_classes), fill_value=0)
    actual_tp = (
        predictions.loc[predictions["correct"]].groupby("true_label").size().reindex(range(expected_classes), fill_value=0)
    )
    ordered_class = per_class.set_index("new_label").loc[range(expected_classes)]
    expected_recall = np.divide(
        actual_tp.to_numpy(dtype=float),
        actual_support.to_numpy(dtype=float),
        out=np.zeros(expected_classes, dtype=float),
        where=actual_support.to_numpy() != 0,
    )
    expected_precision = np.divide(
        actual_tp.to_numpy(dtype=float),
        actual_predicted.to_numpy(dtype=float),
        out=np.zeros(expected_classes, dtype=float),
        where=actual_predicted.to_numpy() != 0,
    )
    expected_f1 = np.divide(
        2 * expected_precision * expected_recall,
        expected_precision + expected_recall,
        out=np.zeros(expected_classes, dtype=float),
        where=(expected_precision + expected_recall) != 0,
    )
    if not (
        np.array_equal(ordered_class["support"].to_numpy(), actual_support.to_numpy())
        and np.array_equal(ordered_class["predicted_count"].to_numpy(), actual_predicted.to_numpy())
        and np.array_equal(ordered_class["true_positives"].to_numpy(), actual_tp.to_numpy())
        and np.allclose(
            ordered_class["recall"].to_numpy(), expected_recall,
            rtol=1e-12,
            atol=1e-12,
        )
        and np.allclose(ordered_class["precision"].to_numpy(), expected_precision, rtol=1e-12, atol=1e-12)
        and np.allclose(ordered_class["f1"].to_numpy(), expected_f1, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("Per-class table disagrees with persisted predictions")

    # Verify selected examples are an exact, unique subset of persisted predictions.
    selected["image_id"] = pd.to_numeric(selected["image_id"], errors="raise").astype("int64")
    selected["source_prediction_row"] = pd.to_numeric(
        selected["source_prediction_row"], errors="raise"
    ).astype("int64")
    selected["correct"] = _bool_series(selected["correct"], "selected.correct")
    if selected.empty or selected["image_id"].duplicated().any():
        raise ValueError("Grad-CAM selection must be nonempty with unique image IDs")
    for key in (
        "analysis_experiment",
        "prediction_artifact_sha256",
        "confusion_artifact_sha256",
        "stage3_freeze_sha256",
        "stage3_test_access_sha256",
    ):
        if set(selected[key].astype(str)) != {str(provenance[key])}:
            raise ValueError(f"Grad-CAM selection provenance mismatch: {key}")
    for row in selected.itertuples(index=False):
        index = int(row.source_prediction_row)
        if not 0 <= index < len(predictions):
            raise ValueError("Grad-CAM source_prediction_row is out of range")
        source = predictions.iloc[index]
        if (
            int(row.image_id) != int(source["image_id"])
            or str(row.local_image_path) != str(source["local_image_path"])
            or int(row.true_label) != int(source["true_label"])
            or int(row.predicted_label) != int(source["predicted_label"])
            or bool(row.correct) != bool(source["correct"])
            or not math.isclose(float(row.top1_score), float(source["top1_score"]), rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError("Grad-CAM selection row disagrees with source prediction")

    selection_lookup = selected.set_index("image_id")
    class_lookup = per_class.set_index("new_label")

    def representative(label: int) -> str:
        candidates = predictions.loc[predictions["true_label"] == label].sort_values(
            ["correct", "top1_score", "image_id"],
            ascending=[False, False, True],
            kind="stable",
        )
        return str(candidates.iloc[0]["local_image_path"])

    classes = per_class.copy()
    classes["class_top1_accuracy"] = classes["recall"]
    classes["representative_image_path"] = [
        representative(int(label)) for label in classes["new_label"]
    ]
    class_columns = [
        "new_label",
        "scientific_name",
        "common_name",
        "support",
        "class_top1_accuracy",
        "f1",
        "precision",
        "recall",
        "predicted_count",
        "true_positives",
        "representative_image_path",
    ]
    hardest = classes.sort_values(
        ["class_top1_accuracy", "f1", "new_label"],
        ascending=[True, True, True],
        kind="stable",
    ).head(25)[class_columns]
    easiest = classes.sort_values(
        ["class_top1_accuracy", "f1", "new_label"],
        ascending=[False, False, True],
        kind="stable",
    ).head(25)[class_columns]

    def enrich_examples(source: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for record in source.to_dict(orient="records"):
            truth = class_lookup.loc[int(record["true_label"])]
            selected_row = (
                selection_lookup.loc[int(record["image_id"])]
                if int(record["image_id"]) in selection_lookup.index
                else None
            )
            rows.append(
                {
                    **record,
                    "true_class_support": int(truth["support"]),
                    "true_class_top1_accuracy": float(truth["recall"]),
                    "true_class_f1": float(truth["f1"]),
                    "true_class_predicted_count": int(truth["predicted_count"]),
                    "representative_image_path": record["local_image_path"],
                    "gradcam_selected": selected_row is not None,
                    "gradcam_selection_type": (
                        str(selected_row["selection_type"]) if selected_row is not None else ""
                    ),
                }
            )
        return pd.DataFrame(rows)

    high_failures = enrich_examples(
        predictions.loc[~predictions["correct"]]
        .sort_values(["top1_score", "image_id"], ascending=[False, True], kind="stable")
        .head(50)
    )
    low_correct = enrich_examples(
        predictions.loc[predictions["correct"]]
        .sort_values(["top1_score", "image_id"], ascending=[True, True], kind="stable")
        .head(50)
    )

    for column in ("rank", "class_a_new_label", "class_b_new_label", "a_as_b", "b_as_a", "total_confusions"):
        confusions[column] = pd.to_numeric(confusions[column], errors="raise").astype("int64")
    if (
        confusions["rank"].tolist() != list(range(1, len(confusions) + 1))
        or not (confusions["class_a_new_label"] < confusions["class_b_new_label"]).all()
        or not (confusions["a_as_b"] + confusions["b_as_a"] == confusions["total_confusions"]).all()
    ):
        raise ValueError("Persisted confusion-pair table violates directional invariants")
    pair_rows: list[dict[str, Any]] = []
    for pair in confusions.head(50).to_dict(orient="records"):
        a = int(pair["class_a_new_label"])
        b = int(pair["class_b_new_label"])
        ab = predictions.loc[(predictions["true_label"] == a) & (predictions["predicted_label"] == b)]
        ba = predictions.loc[(predictions["true_label"] == b) & (predictions["predicted_label"] == a)]
        if len(ab) != int(pair["a_as_b"]) or len(ba) != int(pair["b_as_a"]):
            raise ValueError("Confusion-pair counts disagree with persisted predictions")
        def pair_path(rows: pd.DataFrame) -> str:
            if rows.empty:
                return ""
            return str(rows.sort_values(["top1_score", "image_id"], ascending=[False, True], kind="stable").iloc[0]["local_image_path"])
        class_a = class_lookup.loc[a]
        class_b = class_lookup.loc[b]
        pair_rows.append(
            {
                **pair,
                "class_a_common_name": class_a["common_name"],
                "class_a_support": int(class_a["support"]),
                "class_a_top1_accuracy": float(class_a["recall"]),
                "class_a_f1": float(class_a["f1"]),
                "class_a_predicted_count": int(class_a["predicted_count"]),
                "class_b_common_name": class_b["common_name"],
                "class_b_support": int(class_b["support"]),
                "class_b_top1_accuracy": float(class_b["recall"]),
                "class_b_f1": float(class_b["f1"]),
                "class_b_predicted_count": int(class_b["predicted_count"]),
                "representative_a_as_b_path": pair_path(ab),
                "representative_b_as_a_path": pair_path(ba),
            }
        )
    confused_pairs = pd.DataFrame(pair_rows)

    tables = {
        "hardest": hardest.reset_index(drop=True),
        "easiest": easiest.reset_index(drop=True),
        "high_failures": high_failures.reset_index(drop=True),
        "low_correct": low_correct.reset_index(drop=True),
        "confused_pairs": confused_pairs.reset_index(drop=True),
    }
    for name, table in tables.items():
        for key in PROVENANCE_COLUMNS:
            table[key] = provenance[key]
        tables[name] = table
    return tables


def _install_tables(tables: Mapping[Path, pd.DataFrame], *, force: bool) -> None:
    existing = [path for path in tables if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Failure-analysis outputs already exist; use --force deliberately: "
            + ", ".join(str(path) for path in existing)
        )
    parent = next(iter(tables)).parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage4_failure_analysis.", dir=parent) as name:
        stage = Path(name)
        staged: dict[Path, Path] = {}
        for index, (destination, frame) in enumerate(tables.items()):
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
        "--freeze-manifest",
        type=Path,
        default=Path("results/metrics/deep_learning/stage3_freeze.json"),
    )
    parser.add_argument(
        "--test-access",
        type=Path,
        default=Path("results/metrics/deep_learning/stage3_test_access.json"),
    )
    parser.add_argument(
        "--gradcam-selection",
        type=Path,
        default=Path("results/tables/gradcam_selected_examples.csv"),
    )
    parser.add_argument(
        "--gradcam-selection-manifest",
        type=Path,
        default=Path("results/metrics/explainability/gradcam_selection_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        inputs = _verify_inputs(args)
        tables = build_failure_tables(*inputs)
        destinations = {
            args.output_dir / OUTPUT_NAMES[name]: frame for name, frame in tables.items()
        }
        _install_tables(destinations, force=bool(args.force))
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {len(tables)} Stage 4 failure-analysis tables to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
