"""Create a deterministic 500-class iNaturalist-2021 Mini data split."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from .common import (
    CSV_COLUMNS,
    LOCAL_IMAGE_PATH_COLUMN,
    NEW_LABEL_COLUMN,
    NUM_CLASSES,
    RNG_DESCRIPTION,
    SCHEMA_VERSION,
    TAXONOMY_FIELDS,
    TEST_IMAGES_PER_CLASS,
    TRAIN_IMAGES_PER_CLASS,
    TRAIN_MINI_IMAGES_PER_CLASS,
    VAL_IMAGES_PER_CLASS,
    deterministic_json_bytes,
    deterministic_record_positions,
    is_safe_relative_posix_path,
    select_class_ids,
    sha256_bytes,
    sha256_file,
)


@dataclass(frozen=True)
class ImageRecord:
    """The annotation fields needed to construct one split row."""

    image_id: int
    image_path: str
    original_category_id: int


@dataclass
class CocoDataset:
    """Validated COCO-style categories and image records grouped by category."""

    categories: dict[int, dict[str, Any]]
    records_by_category: dict[int, list[ImageRecord]]
    image_count: int


def _require_integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return value


def load_coco_dataset(path: Path, expected_prefix: str) -> CocoDataset:
    """Load and rigorously validate one iNaturalist COCO-style annotation file."""

    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    required_sections = ("images", "annotations", "categories")
    for section in required_sections:
        if section not in payload or not isinstance(payload[section], list):
            raise ValueError(f"{path}: {section!r} must be present and be a list")

    categories: dict[int, dict[str, Any]] = {}
    for category in payload["categories"]:
        if not isinstance(category, dict):
            raise ValueError(f"{path}: every category must be an object")
        category_id = _require_integer(category.get("id"), "category.id")
        if category_id in categories:
            raise ValueError(f"{path}: duplicate category id {category_id}")
        name = category.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}: category {category_id} has no scientific name")
        for field in ("image_dir_name", *TAXONOMY_FIELDS):
            value = category.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{path}: category {category_id} has invalid metadata field "
                    f"{field!r}: {value!r}"
                )
        categories[category_id] = category

    images: dict[int, dict[str, Any]] = {}
    image_paths: set[str] = set()
    for image in payload["images"]:
        if not isinstance(image, dict):
            raise ValueError(f"{path}: every image must be an object")
        image_id = _require_integer(image.get("id"), "image.id")
        image_path = image.get("file_name")
        if not isinstance(image_path, str) or not is_safe_relative_posix_path(image_path):
            raise ValueError(f"{path}: image {image_id} has unsafe file_name {image_path!r}")
        if not image_path.startswith(f"{expected_prefix}/"):
            raise ValueError(
                f"{path}: image {image_id} path {image_path!r} does not start with "
                f"{expected_prefix!r}"
            )
        if image_id in images:
            raise ValueError(f"{path}: duplicate image id {image_id}")
        if image_path in image_paths:
            raise ValueError(f"{path}: duplicate image path {image_path!r}")
        images[image_id] = image
        image_paths.add(image_path)

    annotations_by_image: dict[int, dict[str, Any]] = {}
    annotation_ids: set[int] = set()
    for annotation in payload["annotations"]:
        if not isinstance(annotation, dict):
            raise ValueError(f"{path}: every annotation must be an object")
        annotation_id = _require_integer(annotation.get("id"), "annotation.id")
        image_id = _require_integer(annotation.get("image_id"), "annotation.image_id")
        category_id = _require_integer(
            annotation.get("category_id"), "annotation.category_id"
        )
        if annotation_id in annotation_ids:
            raise ValueError(f"{path}: duplicate annotation id {annotation_id}")
        if image_id in annotations_by_image:
            raise ValueError(f"{path}: image {image_id} has multiple annotations")
        if image_id not in images:
            raise ValueError(f"{path}: annotation references unknown image {image_id}")
        if category_id not in categories:
            raise ValueError(f"{path}: annotation references unknown category {category_id}")
        annotation_ids.add(annotation_id)
        annotations_by_image[image_id] = annotation

    if set(images) != set(annotations_by_image):
        missing_annotations = set(images).difference(annotations_by_image)
        extra_annotations = set(annotations_by_image).difference(images)
        raise ValueError(
            f"{path}: image/annotation IDs do not match "
            f"({len(missing_annotations)} missing, {len(extra_annotations)} extra)"
        )

    records_by_category: defaultdict[int, list[ImageRecord]] = defaultdict(list)
    for image_id, image in images.items():
        category_id = int(annotations_by_image[image_id]["category_id"])
        image_path = str(image["file_name"])
        path_parts = PurePosixPath(image_path).parts
        expected_directory = str(categories[category_id]["image_dir_name"])
        if len(path_parts) < 3 or path_parts[1] != expected_directory:
            raise ValueError(
                f"{path}: image {image_id} directory does not match category "
                f"{category_id}: {image_path!r}"
            )
        records_by_category[category_id].append(
            ImageRecord(
                image_id=image_id,
                image_path=image_path,
                original_category_id=category_id,
            )
        )

    return CocoDataset(
        categories=categories,
        records_by_category=dict(records_by_category),
        image_count=len(images),
    )


def _row(
    record: ImageRecord,
    *,
    split: str,
    source_split: str,
    new_label: int,
    category: dict[str, Any],
) -> dict[str, Any]:
    local_image_path = PurePosixPath(
        split, PurePosixPath(record.image_path).name
    ).as_posix()
    return {
        "split": split,
        "source_split": source_split,
        "image_path": record.image_path,
        LOCAL_IMAGE_PATH_COLUMN: local_image_path,
        "image_id": record.image_id,
        NEW_LABEL_COLUMN: new_label,
        "original_category_id": record.original_category_id,
        "scientific_name": str(category["name"]),
        "common_name": str(category.get("common_name") or ""),
    }


def _class_mapping_entry(
    new_label: int, category_id: int, category: dict[str, Any]
) -> dict[str, Any]:
    return {
        "new_label": new_label,
        "original_category_id": category_id,
        "scientific_name": str(category["name"]),
        "common_name": str(category.get("common_name") or ""),
        "image_dir_name": str(category.get("image_dir_name") or ""),
        "taxonomy": {
            field: str(category.get(field) or "") for field in TAXONOMY_FIELDS
        },
    }


def _make_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=CSV_COLUMNS)
    frame = frame.sort_values(
        [NEW_LABEL_COLUMN, "image_id", "image_path"], kind="mergesort"
    ).reset_index(drop=True)
    frame["image_id"] = frame["image_id"].astype("int64")
    frame[NEW_LABEL_COLUMN] = frame[NEW_LABEL_COLUMN].astype("int64")
    frame["original_category_id"] = frame["original_category_id"].astype("int64")
    return frame


def _validate_generated_frames(frames: dict[str, pd.DataFrame]) -> None:
    """Fail before writing if any generated split invariant is violated."""

    expected_rows = {
        "train": NUM_CLASSES * TRAIN_IMAGES_PER_CLASS,
        "val": NUM_CLASSES * VAL_IMAGES_PER_CLASS,
        "test": NUM_CLASSES * TEST_IMAGES_PER_CLASS,
    }
    expected_per_class = {
        "train": TRAIN_IMAGES_PER_CLASS,
        "val": VAL_IMAGES_PER_CLASS,
        "test": TEST_IMAGES_PER_CLASS,
    }
    expected_labels = set(range(NUM_CLASSES))

    for split, frame in frames.items():
        if list(frame.columns) != CSV_COLUMNS:
            raise ValueError(f"{split}: generated CSV columns are inconsistent")
        if len(frame) != expected_rows[split]:
            raise ValueError(
                f"{split}: generated {len(frame)} rows, expected {expected_rows[split]}"
            )
        if set(frame[NEW_LABEL_COLUMN].tolist()) != expected_labels:
            raise ValueError(f"{split}: labels are not exactly 0..{NUM_CLASSES - 1}")
        counts = frame.groupby(NEW_LABEL_COLUMN, sort=True).size()
        if not bool((counts == expected_per_class[split]).all()):
            raise ValueError(f"{split}: per-class counts are incorrect")
        expected_local_paths = frame["image_path"].map(
            lambda image_path: PurePosixPath(
                split, PurePosixPath(str(image_path)).name
            ).as_posix()
        )
        if not frame[LOCAL_IMAGE_PATH_COLUMN].equals(expected_local_paths):
            raise ValueError(f"{split}: generated local image paths are inconsistent")
        if frame["image_path"].duplicated().any():
            raise ValueError(f"{split}: duplicate image paths were generated")
        if frame[LOCAL_IMAGE_PATH_COLUMN].duplicated().any():
            raise ValueError(f"{split}: duplicate local image paths were generated")
        if frame["image_id"].duplicated().any():
            raise ValueError(f"{split}: duplicate image IDs were generated")

    combined = pd.concat(frames.values(), ignore_index=True)
    if combined["image_path"].duplicated().any():
        raise ValueError("Generated image paths overlap across splits")
    if combined[LOCAL_IMAGE_PATH_COLUMN].duplicated().any():
        raise ValueError("Generated local image paths overlap across splits")
    if combined["image_id"].duplicated().any():
        raise ValueError("Generated image IDs overlap across splits")


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _preflight_output_targets(
    output_dir: Path, artifact_names: Iterable[str], *, overwrite: bool
) -> None:
    """Reject unsafe or unintended output replacements before expensive work."""

    output_dir.mkdir(parents=True, exist_ok=True)
    non_files: list[str] = []
    existing_files: list[str] = []
    for name in artifact_names:
        target = output_dir / name
        if target.exists() and not target.is_file():
            non_files.append(str(target))
        elif target.is_file():
            existing_files.append(str(target))
    if non_files:
        raise FileExistsError(
            f"Output targets are not regular files: {', '.join(non_files)}"
        )
    if existing_files and not overwrite:
        raise FileExistsError(
            f"Output artifacts already exist: {', '.join(existing_files)}. "
            "Use --overwrite to replace them."
        )


def _write_artifacts(
    output_dir: Path, artifacts: dict[str, bytes], *, overwrite: bool
) -> None:
    """Preflight and atomically replace each prepared artifact."""

    _preflight_output_targets(output_dir, artifacts, overwrite=overwrite)

    temporary_paths: dict[str, Path] = {}
    try:
        for name, content in artifacts.items():
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_dir,
                prefix=f".{name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_paths[name] = Path(temporary.name)

        for name in artifacts:
            os.replace(temporary_paths[name], output_dir / name)
            del temporary_paths[name]
    finally:
        for temporary_path in temporary_paths.values():
            temporary_path.unlink(missing_ok=True)


def prepare_subset(
    train_json: Path,
    val_json: Path,
    output_dir: Path,
    *,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Prepare, validate, and write the requested deterministic subset."""

    selected_name = f"selected_classes_seed{seed}.txt"
    expected_artifact_names = [
        selected_name,
        "class_mapping.json",
        "train.csv",
        "val.csv",
        "test.csv",
        "split_summary.json",
    ]
    other_seed_files = sorted(
        path
        for path in output_dir.glob("selected_classes_seed*.txt")
        if path.name != selected_name
    )
    if other_seed_files:
        raise FileExistsError(
            "Output directory contains selected classes for another seed: "
            f"{', '.join(str(path) for path in other_seed_files)}. "
            "Use a separate output directory for each seed."
        )
    _preflight_output_targets(
        output_dir, expected_artifact_names, overwrite=overwrite
    )

    print(f"Loading training annotations: {train_json}")
    train_data = load_coco_dataset(train_json, expected_prefix="train_mini")
    print(f"Loading official validation annotations: {val_json}")
    official_val_data = load_coco_dataset(val_json, expected_prefix="val")

    if set(train_data.categories) != set(official_val_data.categories):
        raise ValueError("Train and official-validation category ID sets do not match")
    metadata_mismatches = [
        category_id
        for category_id in sorted(train_data.categories)
        if train_data.categories[category_id] != official_val_data.categories[category_id]
    ]
    if metadata_mismatches:
        raise ValueError(
            "Category metadata differs between sources for IDs: "
            f"{metadata_mismatches[:10]}"
        )

    eligible_ids = sorted(
        category_id
        for category_id in train_data.categories
        if len(train_data.records_by_category.get(category_id, []))
        >= TRAIN_MINI_IMAGES_PER_CLASS
        and len(official_val_data.records_by_category.get(category_id, []))
        >= TEST_IMAGES_PER_CLASS
    )
    if len(eligible_ids) < NUM_CLASSES:
        raise ValueError(
            f"Only {len(eligible_ids)} classes are eligible; {NUM_CLASSES} are required"
        )

    selected_ids = select_class_ids(eligible_ids, seed)

    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    mapping_entries: list[dict[str, Any]] = []
    for new_label, category_id in enumerate(
        tqdm(selected_ids, desc="Preparing selected classes", unit="class")
    ):
        category = train_data.categories[category_id]
        mapping_entries.append(
            _class_mapping_entry(new_label, category_id, category)
        )

        train_pool = sorted(
            train_data.records_by_category[category_id],
            key=lambda record: (record.image_id, record.image_path),
        )
        train_positions = deterministic_record_positions(
            len(train_pool),
            TRAIN_MINI_IMAGES_PER_CLASS,
            seed=seed,
            category_id=category_id,
            source_tag=0,
        )
        chosen_train_mini = [
            train_pool[index] for index in train_positions
        ]
        train_records = chosen_train_mini[:TRAIN_IMAGES_PER_CLASS]
        validation_records = chosen_train_mini[TRAIN_IMAGES_PER_CLASS:]

        test_pool = sorted(
            official_val_data.records_by_category[category_id],
            key=lambda record: (record.image_id, record.image_path),
        )
        test_positions = deterministic_record_positions(
            len(test_pool),
            TEST_IMAGES_PER_CLASS,
            seed=seed,
            category_id=category_id,
            source_tag=1,
        )
        test_records = [test_pool[index] for index in test_positions]

        rows["train"].extend(
            _row(
                record,
                split="train",
                source_split="train_mini",
                new_label=new_label,
                category=category,
            )
            for record in train_records
        )
        rows["val"].extend(
            _row(
                record,
                split="val",
                source_split="train_mini",
                new_label=new_label,
                category=category,
            )
            for record in validation_records
        )
        rows["test"].extend(
            _row(
                record,
                split="test",
                source_split="val",
                new_label=new_label,
                category=category,
            )
            for record in test_records
        )

    frames = {split: _make_dataframe(split_rows) for split, split_rows in rows.items()}
    _validate_generated_frames(frames)

    selected_content = ("\n".join(str(value) for value in selected_ids) + "\n").encode(
        "utf-8"
    )
    mapping = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "num_classes": NUM_CLASSES,
        "classes": mapping_entries,
    }

    artifacts: dict[str, bytes] = {
        selected_name: selected_content,
        "class_mapping.json": deterministic_json_bytes(mapping),
        "train.csv": _csv_bytes(frames["train"]),
        "val.csv": _csv_bytes(frames["val"]),
        "test.csv": _csv_bytes(frames["test"]),
    }
    artifact_metadata = {
        name: {"bytes": len(content), "sha256": sha256_bytes(content)}
        for name, content in artifacts.items()
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "rng": RNG_DESCRIPTION,
        "configuration": {
            "num_classes": NUM_CLASSES,
            "train_mini_images_selected_per_class": TRAIN_MINI_IMAGES_PER_CLASS,
            "train_images_per_class": TRAIN_IMAGES_PER_CLASS,
            "validation_images_per_class": VAL_IMAGES_PER_CLASS,
            "test_images_per_class": TEST_IMAGES_PER_CLASS,
        },
        "counts": {
            "eligible_classes": len(eligible_ids),
            "selected_classes": NUM_CLASSES,
            "source_train_images": train_data.image_count,
            "source_official_validation_images": official_val_data.image_count,
            "train_rows": len(frames["train"]),
            "validation_rows": len(frames["val"]),
            "test_rows": len(frames["test"]),
        },
        "csv_columns": CSV_COLUMNS,
        "source_annotations": {
            "train_mini": {
                "file_name": train_json.name,
                "bytes": train_json.stat().st_size,
                "sha256": sha256_file(train_json),
            },
            "official_validation": {
                "file_name": val_json.name,
                "bytes": val_json.stat().st_size,
                "sha256": sha256_file(val_json),
            },
        },
        "artifacts": artifact_metadata,
    }
    artifacts["split_summary.json"] = deterministic_json_bytes(summary)
    _write_artifacts(output_dir, artifacts, overwrite=overwrite)

    return {
        "eligible_classes": len(eligible_ids),
        "selected_classes": len(selected_ids),
        "train_rows": len(frames["train"]),
        "validation_rows": len(frames["val"]),
        "test_rows": len(frames["test"]),
        "output_dir": str(output_dir),
    }


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select exactly 500 iNaturalist species and create deterministic "
            "40-train/10-validation/10-test manifests."
        )
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path("data/raw/train_mini.json"),
        help="Extracted train_mini annotation JSON.",
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        default=Path("data/raw/val.json"),
        help="Extracted official-validation annotation JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory for split artifacts (default: data/splits).",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_integer,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing split artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare_subset(
            args.train_json.expanduser(),
            args.val_json.expanduser(),
            args.output_dir.expanduser(),
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError, TypeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1

    print("\nSubset preparation complete:")
    print(f"  Eligible classes: {result['eligible_classes']}")
    print(f"  Selected classes: {result['selected_classes']}")
    print(f"  Train rows: {result['train_rows']}")
    print(f"  Validation rows: {result['validation_rows']}")
    print(f"  Test rows: {result['test_rows']}")
    print(f"  Output directory: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
