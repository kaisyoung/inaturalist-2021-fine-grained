"""Create deterministic Stage 3 error-analysis tables from saved predictions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import numpy as np

from src.data.common import sha256_file
from src.deep_learning.train_utils import atomic_write_bytes


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def _analysis_model(freeze: dict[str, Any]) -> str:
    value = freeze.get("analysis_model")
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        for key in ("experiment_name", "name", "model_name"):
            if value.get(key):
                return str(value[key])
    raise ValueError("freeze manifest does not declare an analysis_model")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_bytes(
        path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    )


def _resolve_artifact(entry: Any, *, label: str) -> Path:
    if not isinstance(entry, Mapping):
        raise ValueError(f"malformed metric artifact entry: {label}")
    path = Path(str(entry.get("path", "")))
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"metric artifact hash mismatch: {label}")
    if entry.get("bytes") is not None and path.stat().st_size != int(entry["bytes"]):
        raise ValueError(f"metric artifact size mismatch: {label}")
    return path


def _verified_sources(
    args: argparse.Namespace, split: str
) -> tuple[str, str, dict[str, Path]]:
    freeze_path = Path(args.freeze_manifest)
    metrics_dir = Path(getattr(args, "metrics_dir", "results/metrics/deep_learning"))
    freeze: dict[str, Any] | None = None
    freeze_hash = ""
    test_attempt: Mapping[str, Any] | None = None
    if freeze_path.is_file():
        freeze = _load_json(freeze_path)
        if freeze.get("schema_version") != 1 or freeze.get("status") != "frozen":
            raise ValueError("invalid Stage 3 freeze manifest")
        freeze_hash = sha256_file(freeze_path)

    if split == "test":
        if args.experiment:
            raise ValueError("test error analysis cannot override the frozen analysis model")
        if freeze is None:
            raise FileNotFoundError(freeze_path)
        experiment = _analysis_model(freeze)
        selected = freeze.get("selected_experiments")
        if not isinstance(selected, list) or experiment not in selected:
            raise ValueError("frozen analysis model is not in the selected test set")
        access_arg = getattr(args, "access_record", None)
        access_path = (
            Path(access_arg)
            if access_arg is not None
            else metrics_dir / "stage3_test_access.json"
        )
        if not access_path.is_file():
            raise FileNotFoundError(access_path)
        access = _load_json(access_path)
        attempts = access.get("attempts")
        if (
            access.get("status") != "completed"
            or access.get("outcome") != "complete"
            or access.get("freeze_manifest_sha256") != freeze_hash
            or list(access.get("selected_experiments", [])) != selected
            or not isinstance(attempts, Mapping)
            or set(attempts) != set(selected)
            or any(
                not isinstance(attempts[name], Mapping)
                or attempts[name].get("status") != "complete"
                for name in selected
            )
        ):
            raise ValueError("Stage 3 frozen-set test access is incomplete or mismatched")
        test_attempt = attempts[experiment]
    else:
        if args.experiment:
            experiment = str(args.experiment)
        else:
            if freeze is None:
                raise FileNotFoundError(freeze_path)
            experiment = _analysis_model(freeze)

    metrics_path = metrics_dir / f"{experiment}_{split}_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    document = _load_json(metrics_path)
    provenance = document.get("provenance")
    if (
        document.get("status") != "complete"
        or document.get("split") != split
        or document.get("experiment_name") != experiment
        or not isinstance(provenance, Mapping)
    ):
        raise ValueError(f"invalid independent {split} metrics for {experiment}")
    if split == "val":
        recheck = document.get("validation_recheck")
        if (
            not isinstance(recheck, Mapping)
            or recheck.get("agrees_with_stored_best") is not True
            or float(recheck.get("tolerance", 1.0)) > 1e-5
        ):
            raise ValueError("validation metrics did not pass the locked checkpoint recheck")
        if freeze is not None:
            frozen_entry = freeze.get("models", {}).get(experiment, {})
            frozen_metrics = frozen_entry.get("independent_validation_metrics", {})
            if frozen_metrics.get("sha256") != sha256_file(metrics_path):
                raise ValueError("validation metrics no longer match the frozen artifact")
    elif provenance.get("freeze_manifest_sha256") != freeze_hash:
        raise ValueError("test metrics do not match the frozen model set")

    artifact_entries = document.get("artifacts")
    required = {"predictions", "confusion", "per_class", "pairs"}
    if not isinstance(artifact_entries, Mapping) or not required <= set(artifact_entries):
        raise ValueError("metrics document does not bind all required analysis artifacts")
    artifacts = {
        key: _resolve_artifact(entry, label=f"{experiment}.{split}.{key}")
        for key, entry in artifact_entries.items()
    }
    if split == "test":
        assert test_attempt is not None
        access_artifacts = test_attempt.get("artifacts")
        if (
            document.get("metrics") != test_attempt.get("metrics")
            or not isinstance(access_artifacts, Mapping)
            or set(access_artifacts) != {"metrics", *artifact_entries.keys()}
        ):
            raise ValueError("test metrics/artifacts do not match the completed access record")
        metrics_binding = access_artifacts["metrics"]
        if _resolve_artifact(metrics_binding, label=f"{experiment}.test.metrics") != metrics_path.resolve():
            raise ValueError("test metrics path does not match the completed access record")
        for key, entry in artifact_entries.items():
            if access_artifacts.get(key) != entry:
                raise ValueError(
                    f"test artifact disagrees with the completed access record: {key}"
                )
    expected = {
        "predictions": Path(args.predictions_dir)
        / f"{experiment}_{split}_predictions.csv",
        "per_class": Path(args.tables_dir)
        / f"{experiment}_{split}_per_class_metrics.csv",
        "pairs": Path(args.tables_dir) / f"{experiment}_{split}_top_confusions.csv",
    }
    for key, path in expected.items():
        if artifacts[key] != path.resolve():
            raise ValueError(f"{key} path disagrees with the metrics provenance")
    return experiment, freeze_hash, artifacts


def _install_table_set(
    staged: Mapping[Path, Path], *, force: bool, backup_dir: Path
) -> None:
    existing = [destination for destination in staged if destination.exists()]
    if existing and not force:
        raise FileExistsError(
            "analysis outputs already exist: " + ", ".join(str(path) for path in existing)
        )
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for index, destination in enumerate(existing):
            backup = backup_dir / f"backup_{index:02d}_{destination.name}"
            os.replace(destination, backup)
            backups[destination] = backup
        for destination, source in staged.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            installed.append(destination)
    except Exception:
        for path in installed:
            path.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise


def analyse(args: argparse.Namespace) -> dict[str, Any]:
    split = str(args.split)
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    experiment, freeze_hash, artifacts = _verified_sources(args, split)
    prediction_path = artifacts["predictions"]
    per_class_path = artifacts["per_class"]
    top_confusions_path = artifacts["pairs"]
    predictions = pd.read_csv(prediction_path, keep_default_na=False)
    per_class = pd.read_csv(per_class_path, keep_default_na=False)
    confusions = pd.read_csv(top_confusions_path, keep_default_na=False)
    prediction_required = {
        "image_id",
        "local_image_path",
        "true_label",
        "predicted_label",
        "correct",
        "top1_score",
        "score_type",
    }
    missing = sorted(prediction_required - set(predictions.columns))
    if missing:
        raise ValueError(f"prediction table is missing columns: {missing}")
    per_class_required = {"new_label", "precision", "recall", "f1", "support"}
    missing = sorted(per_class_required - set(per_class.columns))
    if missing:
        raise ValueError(f"per-class table is missing columns: {missing}")
    if len(predictions) != 5000:
        raise ValueError(f"expected 5000 {split} predictions, got {len(predictions)}")
    if len(per_class) != 500:
        raise ValueError(f"expected 500 per-class rows, got {len(per_class)}")
    predictions["correct"] = predictions["correct"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if predictions["correct"].isna().any():
        raise ValueError("prediction correct column contains invalid values")
    predictions["top1_score"] = pd.to_numeric(predictions["top1_score"], errors="raise")
    predictions["image_id"] = pd.to_numeric(predictions["image_id"], errors="raise")
    predictions["true_label"] = pd.to_numeric(predictions["true_label"], errors="raise")
    predictions["predicted_label"] = pd.to_numeric(
        predictions["predicted_label"], errors="raise"
    )
    for column in ("image_id", "true_label", "predicted_label"):
        if not bool((predictions[column] % 1 == 0).all()):
            raise ValueError(f"prediction {column} values must be integers")
        predictions[column] = predictions[column].astype("int64")
    if (
        predictions["image_id"].duplicated().any()
        or predictions["local_image_path"].duplicated().any()
    ):
        raise ValueError("prediction identifiers/paths must be unique")
    if not (
        predictions["true_label"].between(0, 499).all()
        and predictions["predicted_label"].between(0, 499).all()
    ):
        raise ValueError("prediction labels must remain within 0..499")
    if not np.isfinite(predictions["top1_score"]).all() or not predictions[
        "top1_score"
    ].between(0.0, 1.0).all():
        raise ValueError("prediction confidence values must be finite probabilities")
    if set(predictions["score_type"].astype(str)) != {"softmax_probability"}:
        raise ValueError("prediction score_type must be softmax_probability")
    if not (
        predictions["correct"].to_numpy()
        == (
            predictions["true_label"].to_numpy()
            == predictions["predicted_label"].to_numpy()
        )
    ).all():
        raise ValueError("prediction correctness disagrees with true/predicted labels")

    manifest_path = Path(getattr(args, "split_dir", "data/splits")) / f"{split}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    manifest_required = {"image_id", "new_label", "local_image_path"}
    if len(manifest) != 5000 or not manifest_required <= set(manifest.columns):
        raise ValueError(f"invalid frozen {split} manifest for error analysis")
    manifest_ids = pd.to_numeric(manifest["image_id"], errors="raise").astype("int64")
    manifest_labels = pd.to_numeric(manifest["new_label"], errors="raise").astype("int64")
    if not (
        predictions["image_id"].reset_index(drop=True).equals(manifest_ids.reset_index(drop=True))
        and predictions["true_label"].reset_index(drop=True).equals(
            manifest_labels.reset_index(drop=True)
        )
        and predictions["local_image_path"].astype(str).reset_index(drop=True).equals(
            manifest["local_image_path"].astype(str).reset_index(drop=True)
        )
    ):
        raise ValueError("predictions are not in the frozen manifest order")

    per_class["new_label"] = pd.to_numeric(per_class["new_label"], errors="raise")
    per_class["support"] = pd.to_numeric(per_class["support"], errors="raise")
    if (
        set(per_class["new_label"].astype(int)) != set(range(500))
        or per_class["new_label"].duplicated().any()
        or not (per_class["support"] == 10).all()
    ):
        raise ValueError("per-class table must contain labels 0..499 with support 10")
    for column in ("precision", "recall", "f1"):
        per_class[column] = pd.to_numeric(per_class[column], errors="raise")
        if not np.isfinite(per_class[column]).all() or not per_class[column].between(
            0.0, 1.0
        ).all():
            raise ValueError(f"per-class {column} values must be finite and within [0, 1]")

    confusion_required = {
        "rank",
        "class_a_new_label",
        "class_b_new_label",
        "a_as_b",
        "b_as_a",
        "total_confusions",
    }
    if not confusion_required <= set(confusions.columns):
        raise ValueError("top-confusions table has an invalid schema")
    for column in confusion_required:
        confusions[column] = pd.to_numeric(confusions[column], errors="raise")
        if not bool((confusions[column] % 1 == 0).all()):
            raise ValueError(f"top-confusions {column} values must be integers")
        confusions[column] = confusions[column].astype("int64")
    if (
        not confusions["class_a_new_label"].between(0, 499).all()
        or not confusions["class_b_new_label"].between(0, 499).all()
        or not (confusions["class_a_new_label"] < confusions["class_b_new_label"]).all()
        or not (confusions[["a_as_b", "b_as_a", "total_confusions"]] >= 0).all().all()
        or not (
            confusions["a_as_b"] + confusions["b_as_a"]
            == confusions["total_confusions"]
        ).all()
        or confusions["rank"].tolist() != list(range(1, len(confusions) + 1))
    ):
        raise ValueError("top-confusions rows violate directional-count invariants")

    provenance = {
        "analysis_experiment": experiment,
        "analysis_split": split,
        "prediction_artifact_sha256": sha256_file(prediction_path),
        "stage3_freeze_sha256": freeze_hash,
    }
    easiest = per_class.sort_values(
        ["recall", "f1", "new_label"],
        ascending=[False, False, True],
        kind="stable",
    ).head(25)
    hardest = per_class.sort_values(
        ["recall", "f1", "new_label"],
        ascending=[True, True, True],
        kind="stable",
    ).head(25)
    successes = predictions.loc[predictions["correct"]].sort_values(
        ["top1_score", "image_id"], ascending=[False, True], kind="stable"
    ).head(50)
    failures = predictions.loc[~predictions["correct"]].sort_values(
        ["top1_score", "image_id"], ascending=[False, True], kind="stable"
    ).head(50)
    confusions = confusions.head(50).copy()
    tables = {
        "top_confusions": confusions,
        "hardest_classes": hardest,
        "easiest_classes": easiest,
        "failure_examples": failures,
        "success_examples": successes,
    }
    prefix = "best_model" if split == "test" else "best_model_val"
    output_dir = Path(args.output_dir)
    destinations = {
        suffix: output_dir / f"{prefix}_{suffix}.csv" for suffix in tables
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{prefix}_analysis.", dir=output_dir.parent
    ) as temporary:
        stage_root = Path(temporary)
        staged: dict[Path, Path] = {}
        for index, (suffix, frame) in enumerate(tables.items()):
            enriched = frame.copy()
            for key, value in provenance.items():
                enriched[key] = value
            staged_path = stage_root / f"{index:02d}_{destinations[suffix].name}"
            _write_frame(staged_path, enriched)
            staged[destinations[suffix]] = staged_path
        _install_table_set(staged, force=bool(args.force), backup_dir=stage_root)
    outputs = {
        path.as_posix(): sha256_file(path) for path in destinations.values()
    }
    return {
        "experiment_name": experiment,
        "split": split,
        "prediction_sha256": provenance["prediction_artifact_sha256"],
        "outputs": outputs,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--experiment")
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=Path("results/metrics/deep_learning/stage3_freeze.json"),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("results/predictions/deep_learning"),
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=Path("results/tables/deep_learning")
    )
    parser.add_argument(
        "--metrics-dir", type=Path, default=Path("results/metrics/deep_learning")
    )
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--access-record", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/tables/deep_learning")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = analyse(_parse_args(argv))
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"Created {len(result['outputs'])} {result['split']} error-analysis tables "
        f"for {result['experiment_name']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
