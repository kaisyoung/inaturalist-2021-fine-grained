"""Generate Grad-CAM figures for the frozen Stage 3 analysis model.

Canonical mode creates a predicted-class CAM for every selected example and an
additional true-class CAM for each incorrect/confused-pair example.  All model,
config, prediction, selection, and Stage 1/2/3 bindings are validated before
an image is processed.  Outputs are staged and installed as one transaction;
existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data.common import is_safe_relative_posix_path, sha256_file
from src.deep_learning.evaluate_deep import (
    _model_for_context,
    _select_device,
    _validate_checkpoint_context,
    _verify_freeze,
)
from src.explainability.gradcam import (
    GradCAM,
    MappedHeatmap,
    map_crop_heatmap_to_original,
    prepare_stage3_eval_image,
)
from src.explainability.select_gradcam_examples import (
    DEFAULT_ACCESS,
    DEFAULT_FREEZE,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_MANIFEST as DEFAULT_SELECTION_MANIFEST,
    DEFAULT_OUTPUT as DEFAULT_SELECTED_EXAMPLES,
    PROJECT_ROOT,
    SELECTION_COLUMNS,
    _read_json,
    _validated_predictions,
    validate_frozen_selection_sources,
)


DEFAULT_OUTPUT_DIR = Path("results/figures/gradcam")
DEFAULT_RUN_MANIFEST = Path("results/metrics/explainability/gradcam_manifest.json")
_SAFE_SELECTION_TYPE = re.compile(r"[a-z][a-z0-9_]*\Z")
_SUMMARY_FILES = {
    "correct_high_confidence": "correct_high_confidence_grid.png",
    "incorrect_high_confidence": "incorrect_high_confidence_grid.png",
    "confused_pair": "confused_pairs_grid.png",
}


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


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": _portable(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_manifest_artifact(entry: Any, path: Path, label: str) -> None:
    if (
        not isinstance(entry, Mapping)
        or _resolve(str(entry.get("path", ""))) != path.resolve()
        or entry.get("sha256") != sha256_file(path)
        or int(entry.get("bytes", -1)) != path.stat().st_size
    ):
        raise ValueError(f"Grad-CAM selection binding changed: {label}")


def _load_selected_examples(
    *,
    selected_path: Path,
    selection_manifest_path: Path,
    source: Mapping[str, Any],
    max_examples: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not selected_path.is_file() or not selection_manifest_path.is_file():
        raise FileNotFoundError(
            "Grad-CAM selected CSV and its selection manifest are required"
        )
    manifest = _read_json(selection_manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("status") != "selected":
        raise ValueError("Grad-CAM selection manifest is invalid")
    if manifest.get("analysis_experiment") != source["analysis_model"]:
        raise ValueError("Grad-CAM selection used a different analysis model")
    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ValueError("Grad-CAM selection manifest lacks provenance bindings")
    for key, path in (
        ("stage3_freeze", source["freeze_path"]),
        ("stage3_test_access", source["access_path"]),
        ("test_manifest", source["test_path"]),
        ("predictions", source["predictions_path"]),
        ("confusion_pairs", source["pairs_path"]),
    ):
        _verify_manifest_artifact(inputs.get(key), path, key)
    _verify_manifest_artifact(
        outputs.get("selected_examples"), selected_path, "selected_examples"
    )
    try:
        selected = pd.read_csv(selected_path, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read Grad-CAM selected examples: {error}") from error
    if list(selected.columns) != SELECTION_COLUMNS or selected.empty:
        raise ValueError("Grad-CAM selected examples have an invalid schema or are empty")
    if int(manifest.get("total_selected", -1)) != len(selected):
        raise ValueError("Grad-CAM selected row count differs from its manifest")

    integer_columns = (
        "selection_rank",
        "image_id",
        "true_label",
        "predicted_label",
        "source_prediction_row",
    )
    for column in integer_columns:
        values = pd.to_numeric(selected[column], errors="coerce")
        if values.isna().any() or (values % 1 != 0).any():
            raise ValueError(f"Selected column {column!r} is not integral")
        selected[column] = values.astype(np.int64)
    selected["top1_score"] = pd.to_numeric(selected["top1_score"], errors="coerce")
    if (
        not np.isfinite(selected["top1_score"].to_numpy(dtype=float)).all()
        or not selected["top1_score"].between(0.0, 1.0).all()
    ):
        raise ValueError("Selected top1 scores must be finite probabilities")
    selected["correct"] = selected["correct"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if selected["correct"].isna().any():
        raise ValueError("Selected correctness values are invalid")
    if not (
        selected["correct"]
        == (selected["true_label"] == selected["predicted_label"])
    ).all():
        raise ValueError("Selected correctness values disagree with labels")
    if (
        selected["image_id"].duplicated().any()
        or selected["local_image_path"].duplicated().any()
        or selected["selection_rank"].tolist() != list(range(1, len(selected) + 1))
    ):
        raise ValueError("Selected examples are not unique and consecutively ranked")

    provenance = {
        "analysis_experiment": source["analysis_model"],
        "prediction_artifact_sha256": sha256_file(source["predictions_path"]),
        "confusion_artifact_sha256": sha256_file(source["pairs_path"]),
        "stage3_freeze_sha256": source["freeze_sha256"],
        "stage3_test_access_sha256": source["access_sha256"],
        "test_manifest_sha256": source["test_sha256"],
    }
    for column, expected in provenance.items():
        if set(selected[column].astype(str)) != {str(expected)}:
            raise ValueError(f"Selected per-row provenance mismatch: {column}")

    predictions = _validated_predictions(source["predictions_path"], source["test_path"])
    for row in selected.itertuples(index=False):
        source_row = int(row.source_prediction_row)
        if source_row < 0 or source_row >= len(predictions):
            raise ValueError("Selected source_prediction_row is outside predictions")
        prediction = predictions.iloc[source_row]
        comparisons = {
            "image_id": (int(row.image_id), int(prediction["image_id"])),
            "local_image_path": (str(row.local_image_path), str(prediction["local_image_path"])),
            "true_label": (int(row.true_label), int(prediction["true_label"])),
            "predicted_label": (int(row.predicted_label), int(prediction["predicted_label"])),
            "correct": (bool(row.correct), bool(prediction["correct"])),
        }
        for label, (observed, expected) in comparisons.items():
            if observed != expected:
                raise ValueError(f"Selected row no longer matches predictions: {label}")
        if not math.isclose(
            float(row.top1_score),
            float(prediction["top1_score"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Selected confidence no longer matches predictions")
        if not _SAFE_SELECTION_TYPE.fullmatch(str(row.selection_type)):
            raise ValueError(f"Unsafe selection type: {row.selection_type!r}")

    if max_examples is not None:
        if isinstance(max_examples, bool) or max_examples <= 0:
            raise ValueError("max_examples must be positive")
        selected = selected.head(max_examples).copy()
    return selected, manifest


def load_frozen_analysis_model(
    *,
    source: Mapping[str, Any],
    split_dir: Path,
    checkpoint_path: Path | None,
    model_config_path: Path | None,
    source_config_path: Path | None,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Strictly revalidate Stage 3 and load only its frozen analysis model."""

    freeze_path = source["freeze_path"]
    freeze = _verify_freeze(
        freeze_path, split_dir=split_dir, verify_test_manifest=True
    )
    if freeze != source["freeze"]:
        raise ValueError("Stage 3 freeze changed between validation steps")
    name = source["analysis_model"]
    entry = freeze["models"][name]
    bound_checkpoint = _resolve(entry["checkpoint"]["path"])
    bound_resolved = _resolve(entry["resolved_config"]["path"])
    bound_source = _resolve(entry["source_config"]["path"])
    if checkpoint_path is not None and checkpoint_path.resolve() != bound_checkpoint:
        raise ValueError("Explicit checkpoint is not the frozen analysis checkpoint")
    if model_config_path is not None and model_config_path.resolve() != bound_resolved:
        raise ValueError("Explicit model config is not the frozen resolved config")
    if source_config_path is not None and source_config_path.resolve() != bound_source:
        raise ValueError("Explicit source config is not the frozen source config")
    context = _validate_checkpoint_context(
        bound_checkpoint,
        split_dir=split_dir,
        metrics_dir=bound_resolved.parent,
        resolved_config_path=bound_resolved,
        source_config_path=bound_source,
        best_metrics_path=_resolve(entry["stored_best_metrics"]["path"]),
        expected_name=name,
        frozen_entry=entry,
        verify_test_manifest=True,
    )
    if (
        context["checkpoint_sha256"] != entry["checkpoint"]["sha256"]
        or context["resolved_config_sha256"] != entry["resolved_config"]["sha256"]
        or context["source_config_sha256"] != entry["source_config"]["sha256"]
    ):
        raise ValueError("Frozen checkpoint/config hash validation failed")
    model = _model_for_context(context, device)
    model.eval()
    return model, context


def _safe_image_path(image_root: Path, relative: str) -> Path:
    if not is_safe_relative_posix_path(relative):
        raise ValueError(f"Unsafe local image path: {relative!r}")
    path = image_root.joinpath(*PurePosixPath(relative).parts).resolve()
    try:
        path.relative_to(image_root)
    except ValueError as error:
        raise ValueError(f"Image path escapes image root: {relative!r}") from error
    if not path.is_file():
        raise FileNotFoundError(f"Selected Grad-CAM image is missing: {path}")
    return path


def _colourize(heatmap: np.ndarray) -> np.ndarray:
    if heatmap.ndim != 2 or not np.isfinite(heatmap).all():
        raise ValueError("Heatmap must be a finite 2D array")
    return np.asarray(plt.colormaps["turbo"](np.clip(heatmap, 0.0, 1.0))[..., :3])


def _overlay(
    original: Image.Image,
    mapped: MappedHeatmap,
    *,
    opacity: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    if source.shape[:2] != mapped.heatmap.shape:
        raise ValueError("Mapped heatmap does not match original image geometry")
    colour = _colourize(mapped.heatmap)
    alpha = opacity * mapped.crop_mask * (0.25 + 0.75 * mapped.heatmap)
    composite = source * (1.0 - alpha[..., None]) + colour * alpha[..., None]
    # This RGBA heat layer is exactly transparent outside the model crop.
    rgba_layer = np.concatenate([colour, alpha[..., None]], axis=2)
    return np.clip(composite, 0.0, 1.0), rgba_layer


def _figure_title(row: Any, target_mode: str, target_label: int, target_score: float) -> str:
    correctness = "correct" if bool(row.correct) else "incorrect"
    return (
        f"image {int(row.image_id)} | {row.selection_type} | {correctness}\n"
        f"true {int(row.true_label)}: {row.scientific_name} | "
        f"pred {int(row.predicted_label)}: {row.predicted_scientific_name} "
        f"(p={float(row.top1_score):.3f}) | "
        f"CAM target={target_mode}:{target_label} (p={target_score:.3f})"
    )


def render_gradcam_figure(
    *,
    prepared: Any,
    crop_heatmap: np.ndarray,
    row: Any,
    target_mode: str,
    target_label: int,
    target_score: float,
    output: Path,
) -> np.ndarray:
    """Render original, model crop, crop CAM, and aligned full overlay."""

    mapped = map_crop_heatmap_to_original(crop_heatmap, prepared.geometry)
    composite, rgba_layer = _overlay(prepared.original_rgb, mapped)
    if not np.all(rgba_layer[..., 3][mapped.crop_mask == 0] == 0):
        raise RuntimeError("Grad-CAM layer leaked outside the evaluation crop")

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.8), constrained_layout=True)
    axes[0].imshow(prepared.original_rgb)
    axes[0].set_title("EXIF-oriented original")
    axes[1].imshow(prepared.model_crop_rgb)
    axes[1].set_title("Stage 3 model crop (224×224)")
    axes[2].imshow(crop_heatmap, cmap="turbo", vmin=0.0, vmax=1.0)
    axes[2].set_title("Grad-CAM heatmap")
    axes[3].imshow(composite)
    axes[3].set_title("Aligned full-image overlay")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        _figure_title(row, target_mode, target_label, target_score), fontsize=10
    )
    figure.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "COMP9517 Stage 4 Grad-CAM"},
    )
    plt.close(figure)
    return composite


def _summary_grid(
    records: list[dict[str, Any]],
    *,
    selection_type: str,
    output: Path,
) -> None:
    selected = [item for item in records if item["selection_type"] == selection_type]
    if not selected:
        # A max-samples smoke run may omit a category.  No misleading blank
        # canonical grid is emitted; the run manifest records this omission.
        return
    columns = 4
    rows = math.ceil(len(selected) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.4 * columns, 3.2 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, item in zip(axes.flat, selected):
        axis.imshow(item["overlay"])
        axis.set_title(
            f"id {item['image_id']} | {item['true_label']}→{item['predicted_label']}\n"
            f"p={item['top1_score']:.3f}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    figure.suptitle(selection_type.replace("_", " ").title())
    figure.savefig(
        output,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "COMP9517 Stage 4 Grad-CAM"},
    )
    plt.close(figure)


def _targets_for_row(row: Any, mode: str) -> list[tuple[str, int]]:
    if mode == "predicted":
        return [("predicted", int(row.predicted_label))]
    if mode == "true":
        return [("true", int(row.true_label))]
    if mode != "canonical":
        raise ValueError("target_mode must be canonical, predicted, or true")
    targets = [("predicted", int(row.predicted_label))]
    if not bool(row.correct) or str(row.selection_type) == "confused_pair":
        targets.append(("true", int(row.true_label)))
    return targets


def _relative_figure_path(row: Any, target_mode: str, target_label: int) -> Path:
    selection_type = str(row.selection_type)
    if not _SAFE_SELECTION_TYPE.fullmatch(selection_type):
        raise ValueError(f"Unsafe selection type for output path: {selection_type!r}")
    name = (
        f"{int(row.image_id)}__row-{int(row.source_prediction_row):04d}"
        f"__target-{target_mode}-label-{target_label:03d}_gradcam.png"
    )
    return Path(selection_type) / name


def _planned_outputs(
    selected: pd.DataFrame,
    *,
    target_mode: str,
    output_dir: Path,
    manifest_path: Path,
) -> list[Path]:
    planned = [manifest_path]
    for row in selected.itertuples(index=False):
        for mode, label in _targets_for_row(row, target_mode):
            planned.append(output_dir / _relative_figure_path(row, mode, label))
    present_types = set(selected["selection_type"].astype(str))
    planned.extend(
        output_dir / filename
        for selection_type, filename in _SUMMARY_FILES.items()
        if selection_type in present_types
    )
    if len(planned) != len(set(planned)):
        raise ValueError("Grad-CAM output naming collision detected")
    return planned


def _install_transaction(staged: Mapping[Path, Path]) -> None:
    conflicts = [path for path in staged if path.exists()]
    if conflicts:
        raise FileExistsError(
            "Grad-CAM refuses existing/conflicting outputs: "
            + ", ".join(str(path) for path in conflicts)
        )
    installed: list[Path] = []
    try:
        for final, temporary in staged.items():
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, final)
            installed.append(final)
    except Exception:
        for path in reversed(installed):
            path.unlink(missing_ok=True)
        raise


def run_gradcam(args: argparse.Namespace) -> dict[str, Any]:
    freeze_path = _resolve(args.freeze_manifest)
    access_path = _resolve(args.test_access)
    split_dir = _resolve(args.split_dir)
    image_root = _resolve(args.image_root)
    selected_path = _resolve(args.selected_examples)
    selection_manifest_path = _resolve(args.selection_manifest)
    output_dir = _resolve(args.output_dir)
    manifest_path = _resolve(args.manifest)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root is missing: {image_root}")
    source = validate_frozen_selection_sources(
        freeze_path=freeze_path,
        access_path=access_path,
    )
    selected, selection_manifest = _load_selected_examples(
        selected_path=selected_path,
        selection_manifest_path=selection_manifest_path,
        source=source,
        max_examples=args.max_examples,
    )
    planned = _planned_outputs(
        selected,
        target_mode=args.target_mode,
        output_dir=output_dir,
        manifest_path=manifest_path,
    )
    existing = [path for path in planned if path.exists()]
    if existing:
        raise FileExistsError(
            "Grad-CAM refuses existing/conflicting outputs: "
            + ", ".join(str(path) for path in existing)
        )

    device = _select_device(args.device)
    model, context = load_frozen_analysis_model(
        source=source,
        split_dir=split_dir,
        checkpoint_path=_resolve(args.checkpoint) if args.checkpoint else None,
        model_config_path=_resolve(args.model_config) if args.model_config else None,
        source_config_path=_resolve(args.source_config) if args.source_config else None,
        device=device,
    )
    checkpoint = context["checkpoint"]
    image_size = int(checkpoint["image_size"])

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged: dict[Path, Path] = {}
    figure_entries: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix=".stage4_gradcam.", dir=output_dir.parent
    ) as temporary_name:
        stage_root = Path(temporary_name)
        with GradCAM(model, args.target_layer) as generator:
            for row in selected.itertuples(index=False):
                path = _safe_image_path(image_root, str(row.local_image_path))
                try:
                    with Image.open(path) as opened:
                        prepared = prepare_stage3_eval_image(
                            opened, image_size=image_size
                        )
                except Exception as error:
                    raise ValueError(f"Could not decode selected image {path}: {error}") from error
                input_tensor = prepared.tensor.unsqueeze(0).to(
                    device=device, dtype=next(model.parameters()).dtype
                )
                first_overlay: np.ndarray | None = None
                for requested_mode, expected_target in _targets_for_row(
                    row, args.target_mode
                ):
                    result = generator.generate(
                        input_tensor,
                        target_mode=requested_mode,
                        true_labels=int(row.true_label)
                        if requested_mode == "true"
                        else None,
                    )
                    predicted = int(result.predicted_labels[0].item())
                    target = int(result.target_labels[0].item())
                    if predicted != int(row.predicted_label):
                        raise ValueError(
                            f"Frozen prediction mismatch for image {row.image_id}: "
                            f"persisted={row.predicted_label}, reloaded={predicted}"
                        )
                    probability = float(result.probabilities[0, predicted].item())
                    if not math.isclose(
                        probability,
                        float(row.top1_score),
                        rel_tol=1e-4,
                        abs_tol=1e-4,
                    ):
                        raise ValueError(
                            f"Frozen confidence mismatch for image {row.image_id}: "
                            f"persisted={row.top1_score}, reloaded={probability}"
                        )
                    if target != expected_target:
                        raise RuntimeError("Grad-CAM targeted an unexpected class")
                    target_score = float(result.probabilities[0, target].item())
                    relative = _relative_figure_path(row, requested_mode, target)
                    staged_path = stage_root / relative
                    staged_path.parent.mkdir(parents=True, exist_ok=True)
                    crop_heatmap = result.heatmaps[0].detach().cpu().numpy()
                    overlay = render_gradcam_figure(
                        prepared=prepared,
                        crop_heatmap=crop_heatmap,
                        row=row,
                        target_mode=requested_mode,
                        target_label=target,
                        target_score=target_score,
                        output=staged_path,
                    )
                    final_path = output_dir / relative
                    staged[final_path] = staged_path
                    figure_entries.append(
                        {
                            "selection_rank": int(row.selection_rank),
                            "source_prediction_row": int(row.source_prediction_row),
                            "image_id": int(row.image_id),
                            "selection_type": str(row.selection_type),
                            "target_mode": requested_mode,
                            "target_label": target,
                            "target_score": target_score,
                            "path": _portable(final_path),
                            "bytes": staged_path.stat().st_size,
                            "sha256": sha256_file(staged_path),
                        }
                    )
                    if first_overlay is None:
                        first_overlay = overlay
                if first_overlay is None:
                    raise RuntimeError("No Grad-CAM target was generated")
                summary_records.append(
                    {
                        "selection_type": str(row.selection_type),
                        "image_id": int(row.image_id),
                        "true_label": int(row.true_label),
                        "predicted_label": int(row.predicted_label),
                        "top1_score": float(row.top1_score),
                        "overlay": first_overlay,
                    }
                )

        summary_entries: dict[str, dict[str, Any]] = {}
        for selection_type, filename in _SUMMARY_FILES.items():
            if selection_type not in set(selected["selection_type"].astype(str)):
                continue
            staged_path = stage_root / filename
            _summary_grid(
                summary_records,
                selection_type=selection_type,
                output=staged_path,
            )
            final_path = output_dir / filename
            staged[final_path] = staged_path
            summary_entries[selection_type] = {
                "path": _portable(final_path),
                "bytes": staged_path.stat().st_size,
                "sha256": sha256_file(staged_path),
            }

        run_manifest = {
            "schema_version": 1,
            "status": "complete",
            "method": {
                "name": "Grad-CAM",
                "implementation": "manual PyTorch activation and gradient hooks",
                "channel_weighting": "spatial mean of target-score gradients",
                "combination": "ReLU(weighted channel sum)",
                "normalization": "per-image finite min-max to [0,1]",
                "target_layer": generator.target_layer_name,
                "target_policy": args.target_mode,
                "canonical_policy": (
                    "predicted CAM for every row plus true CAM for every incorrect/"
                    "confused-pair row"
                ),
            },
            "analysis_experiment": source["analysis_model"],
            "device": str(device),
            "model_eval_mode": not bool(model.training),
            "parameter_updates": False,
            "preprocessing": {
                "contract": "Stage 3 eval: EXIF RGB, bicubic short-edge 256, center crop 224, ImageNet normalization",
                "image_size": image_size,
                "full_image_mapping": "inverse recorded resize/crop geometry; outside-crop alpha=0",
            },
            "inputs": {
                "stage3_freeze": _artifact(source["freeze_path"]),
                "stage3_test_access": _artifact(source["access_path"]),
                "test_manifest": _artifact(source["test_path"]),
                "checkpoint": _artifact(context["checkpoint_path"]),
                "resolved_config": _artifact(context["resolved_config_path"]),
                "source_config": _artifact(context["source_config_path"]),
                "predictions": _artifact(source["predictions_path"]),
                "confusion_pairs": _artifact(source["pairs_path"]),
                "selected_examples": _artifact(selected_path),
                "selection_manifest": _artifact(selection_manifest_path),
            },
            "selection_manifest_sha256": sha256_file(selection_manifest_path),
            "source_selection_total": int(selection_manifest["total_selected"]),
            "processed_examples": int(len(selected)),
            "max_examples": args.max_examples,
            "cam_counts": {
                "predicted": sum(
                    item["target_mode"] == "predicted" for item in figure_entries
                ),
                "true": sum(item["target_mode"] == "true" for item in figure_entries),
                "total": len(figure_entries),
            },
            "figures": figure_entries,
            "summary_grids": summary_entries,
            # Deliberately omit a timestamp: all provenance comes from hashes.
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        handle, manifest_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.", dir=manifest_path.parent
        )
        manifest_temporary = Path(manifest_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(
                    run_manifest,
                    stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            staged[manifest_path] = manifest_temporary
            _install_transaction(staged)
        except Exception:
            manifest_temporary.unlink(missing_ok=True)
            raise

    print(
        f"[PASS] generated {len(figure_entries)} Grad-CAM figures for "
        f"{len(selected)} unique images on {device}"
    )
    print(f"[PASS] target layer: {run_manifest['method']['target_layer']}")
    print(f"[PASS] Grad-CAM manifest: {_portable(manifest_path)}")
    return run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM for the frozen Stage 3 EfficientNet analysis model."
    )
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--test-access", type=Path, default=DEFAULT_ACCESS)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--selected-examples", type=Path, default=DEFAULT_SELECTED_EXAMPLES)
    parser.add_argument(
        "--selection-manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional explicit checkpoint; it must equal the frozen analysis checkpoint.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="Optional resolved config; it must equal the frozen config binding.",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        help="Optional source YAML; it must equal the frozen source-config binding.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument(
        "--target-mode",
        choices=("canonical", "predicted", "true"),
        default="canonical",
        help=(
            "canonical: predicted for all plus true for errors; predicted/true: "
            "one requested target for every row"
        ),
    )
    parser.add_argument(
        "--target-layer",
        default="auto",
        help="auto, or a safe registered-module dotted path such as features.8",
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_gradcam(args)
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_RUN_MANIFEST",
    "build_parser",
    "load_frozen_analysis_model",
    "main",
    "render_gradcam_figure",
    "run_gradcam",
]
