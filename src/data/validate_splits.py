"""Validate split counts, mappings, provenance, and leakage safeguards."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
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
    deterministic_record_positions,
    is_safe_relative_posix_path,
    select_class_ids,
    sha256_file,
)


@dataclass
class ValidationReport:
    """Collect and print validation outcomes."""

    passed: int = 0
    failed: int = 0
    skipped: int = 0

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        if condition:
            self.passed += 1
            suffix = f" - {detail}" if detail else ""
            print(f"[PASS] {name}{suffix}")
            return True
        self.failed += 1
        suffix = f" - {detail}" if detail else ""
        print(f"[FAIL] {name}{suffix}")
        return False

    def fail(self, name: str, detail: str) -> None:
        self.check(name, False, detail)

    def skip(self, name: str, detail: str) -> None:
        self.skipped += 1
        print(f"[SKIP] {name} - {detail}")

    def print_summary(self) -> None:
        total = self.passed + self.failed + self.skipped
        status = "VALID" if self.failed == 0 else "INVALID"
        print("\n" + "=" * 72)
        print(f"Validation result: {status}")
        print(
            f"Checks: {total} total, {self.passed} passed, "
            f"{self.failed} failed, {self.skipped} skipped"
        )
        print("=" * 72)


@dataclass
class SourceSnapshot:
    """Validated source metadata retained for provenance and RNG reconstruction."""

    referenced_records: set[tuple[int, str, int]]
    categories: dict[int, dict[str, Any]]
    records_by_category: dict[int, list[tuple[int, str]]]
    diagnostics: dict[str, int]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_csv(path: Path, report: ValidationReport) -> pd.DataFrame | None:
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (
        OSError,
        ValueError,
        UnicodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        report.fail(f"Read {path.name}", str(error))
        return None

    columns_ok = list(frame.columns) == CSV_COLUMNS
    report.check(
        f"{path.name} column schema",
        columns_ok,
        f"expected {CSV_COLUMNS}, got {list(frame.columns)}",
    )
    if not columns_ok:
        return None

    for column in ("image_id", NEW_LABEL_COLUMN, "original_category_id"):
        try:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            limits = np.iinfo(np.int64)
            integral = (
                numeric.notna()
                & np.isfinite(numeric)
                & (numeric % 1 == 0)
                & (numeric >= limits.min)
                & (numeric <= limits.max)
            )
        except (TypeError, ValueError, OverflowError) as error:
            report.fail(f"{path.name} {column} values are integers", str(error))
            return None
        if not report.check(
            f"{path.name} {column} values are integers",
            bool(integral.all()),
            f"invalid rows: {int((~integral).sum())}",
        ):
            return None
        try:
            frame[column] = numeric.astype("int64")
        except (TypeError, ValueError, OverflowError) as error:
            report.fail(f"{path.name} {column} conversion", str(error))
            return None

    required_nonempty = [
        "split",
        "source_split",
        "image_path",
        LOCAL_IMAGE_PATH_COLUMN,
        "scientific_name",
    ]
    empty_count = int(
        sum((frame[column].str.strip() == "").sum() for column in required_nonempty)
    )
    report.check(
        f"{path.name} required values are non-empty",
        empty_count == 0,
        f"empty required values: {empty_count}",
    )
    return frame


def _first_values(values: Iterable[Any], limit: int = 3) -> str:
    sample = list(values)[:limit]
    return ", ".join(repr(value) for value in sample)


def _is_exact_integer(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _validate_split_structure(
    frames: dict[str, pd.DataFrame], report: ValidationReport
) -> None:
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
    source_by_split = {"train": "train_mini", "val": "train_mini", "test": "val"}
    prefix_by_split = {"train": "train_mini", "val": "train_mini", "test": "val"}
    expected_labels = set(range(NUM_CLASSES))

    for split, frame in frames.items():
        report.check(
            f"{split} row count",
            len(frame) == expected_rows[split],
            f"actual {len(frame)}, expected {expected_rows[split]}",
        )
        labels = set(int(value) for value in frame[NEW_LABEL_COLUMN].tolist())
        report.check(
            f"{split} new labels are contiguous 0..{NUM_CLASSES - 1}",
            labels == expected_labels,
            f"observed {len(labels)} unique labels",
        )
        counts = frame.groupby(NEW_LABEL_COLUMN, sort=True).size().reindex(
            range(NUM_CLASSES), fill_value=0
        )
        unexpected_labels = labels.difference(expected_labels)
        counts_ok = (
            bool((counts == expected_per_class[split]).all())
            and not unexpected_labels
        )
        bad_counts = counts[counts != expected_per_class[split]]
        report.check(
            f"{split} has exactly {expected_per_class[split]} images per class",
            counts_ok,
            (
                f"bad labels/counts: {bad_counts.head(3).to_dict()}, "
                f"unexpected labels: {sorted(unexpected_labels)[:3]}"
            ),
        )

        split_values = set(frame["split"].tolist())
        source_values = set(frame["source_split"].tolist())
        report.check(
            f"{split} split/source provenance",
            split_values == {split} and source_values == {source_by_split[split]},
            f"split={sorted(split_values)}, source_split={sorted(source_values)}",
        )

        safe_paths = frame["image_path"].map(is_safe_relative_posix_path)
        report.check(
            f"{split} paths are safe relative POSIX paths",
            bool(safe_paths.all()),
            f"unsafe paths: {int((~safe_paths).sum())}",
        )
        correct_prefix = frame["image_path"].map(
            lambda value: PurePosixPath(value).parts[:1] == (prefix_by_split[split],)
            if is_safe_relative_posix_path(value)
            else False
        )
        report.check(
            f"{split} paths match their source",
            bool(correct_prefix.all()),
            f"incorrect prefixes: {int((~correct_prefix).sum())}",
        )

        safe_local_paths = frame[LOCAL_IMAGE_PATH_COLUMN].map(
            is_safe_relative_posix_path
        )
        report.check(
            f"{split} local paths are safe relative POSIX paths",
            bool(safe_local_paths.all()),
            f"unsafe paths: {int((~safe_local_paths).sum())}",
        )
        expected_local_paths = frame["image_path"].map(
            lambda value: PurePosixPath(
                split, PurePosixPath(str(value)).name
            ).as_posix()
        )
        correct_local_paths = frame[LOCAL_IMAGE_PATH_COLUMN] == expected_local_paths
        report.check(
            f"{split} local paths match split and UUID filename",
            bool(correct_local_paths.all()),
            f"incorrect local paths: {int((~correct_local_paths).sum())}",
        )

        normalized_paths = frame["image_path"].map(
            lambda value: str(PurePosixPath(value))
        )
        normalized_local_paths = frame[LOCAL_IMAGE_PATH_COLUMN].map(
            lambda value: str(PurePosixPath(value))
        )
        duplicate_paths = normalized_paths[normalized_paths.duplicated(keep=False)]
        duplicate_local_paths = normalized_local_paths[
            normalized_local_paths.duplicated(keep=False)
        ]
        duplicate_ids = frame.loc[frame["image_id"].duplicated(keep=False), "image_id"]
        basenames = frame["image_path"].map(lambda value: PurePosixPath(value).name)
        duplicate_basenames = basenames[basenames.duplicated(keep=False)]
        report.check(
            f"{split} contains no duplicate paths",
            duplicate_paths.empty,
            f"examples: {_first_values(duplicate_paths.unique())}",
        )
        report.check(
            f"{split} contains no duplicate local paths",
            duplicate_local_paths.empty,
            f"examples: {_first_values(duplicate_local_paths.unique())}",
        )
        report.check(
            f"{split} contains no duplicate image IDs",
            duplicate_ids.empty,
            f"examples: {_first_values(duplicate_ids.unique())}",
        )
        report.check(
            f"{split} contains no duplicate UUID filenames",
            duplicate_basenames.empty,
            f"examples: {_first_values(duplicate_basenames.unique())}",
        )

    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in split_pairs:
        left_frame, right_frame = frames[left], frames[right]
        identifier_sets = {
            "paths": (
                set(
                    left_frame["image_path"].map(
                        lambda value: str(PurePosixPath(value))
                    )
                ),
                set(
                    right_frame["image_path"].map(
                        lambda value: str(PurePosixPath(value))
                    )
                ),
            ),
            "local paths": (
                set(
                    left_frame[LOCAL_IMAGE_PATH_COLUMN].map(
                        lambda value: str(PurePosixPath(value))
                    )
                ),
                set(
                    right_frame[LOCAL_IMAGE_PATH_COLUMN].map(
                        lambda value: str(PurePosixPath(value))
                    )
                ),
            ),
            "image IDs": (
                set(int(value) for value in left_frame["image_id"]),
                set(int(value) for value in right_frame["image_id"]),
            ),
            "UUID filenames": (
                set(left_frame["image_path"].map(lambda value: PurePosixPath(value).name)),
                set(right_frame["image_path"].map(lambda value: PurePosixPath(value).name)),
            ),
        }
        for identifier, (left_values, right_values) in identifier_sets.items():
            overlap = left_values.intersection(right_values)
            report.check(
                f"{left}/{right} have disjoint {identifier}",
                not overlap,
                f"overlap {len(overlap)}; examples: {_first_values(sorted(overlap))}",
            )


def _validate_mapping_and_selection(
    frames: dict[str, pd.DataFrame],
    mapping: Any,
    selected_ids: list[int],
    seed: int,
    report: ValidationReport,
) -> None:
    mapping_is_object = isinstance(mapping, dict)
    report.check("class_mapping.json is a JSON object", mapping_is_object)
    if not mapping_is_object:
        return

    report.check(
        "class mapping schema version",
        _is_exact_integer(mapping.get("schema_version"), SCHEMA_VERSION),
        f"actual {mapping.get('schema_version')!r}, expected {SCHEMA_VERSION}",
    )
    report.check(
        "class mapping seed",
        _is_exact_integer(mapping.get("seed"), seed),
        f"actual {mapping.get('seed')!r}, expected {seed}",
    )
    report.check(
        "class mapping declares 500 classes",
        _is_exact_integer(mapping.get("num_classes"), NUM_CLASSES),
        f"actual {mapping.get('num_classes')!r}",
    )
    classes = mapping.get("classes")
    classes_ok = isinstance(classes, list) and len(classes) == NUM_CLASSES
    report.check(
        "class mapping contains exactly 500 classes",
        classes_ok,
        f"actual {len(classes) if isinstance(classes, list) else 'not a list'}",
    )
    if not classes_ok:
        return

    well_formed = all(
        isinstance(entry, dict)
        and isinstance(entry.get("new_label"), int)
        and isinstance(entry.get("original_category_id"), int)
        and isinstance(entry.get("scientific_name"), str)
        and isinstance(entry.get("common_name"), str)
        and isinstance(entry.get("image_dir_name"), str)
        and isinstance(entry.get("taxonomy"), dict)
        and set(entry["taxonomy"]) == set(TAXONOMY_FIELDS)
        for entry in classes
    )
    report.check("class mapping entries are well formed", well_formed)
    if not well_formed:
        return

    mapping_labels = [int(entry["new_label"]) for entry in classes]
    mapping_original_ids = [int(entry["original_category_id"]) for entry in classes]
    report.check(
        "class mapping order is new-label order 0..499",
        mapping_labels == list(range(NUM_CLASSES)),
    )
    report.check(
        "class mapping original IDs are unique and sorted",
        len(set(mapping_original_ids)) == NUM_CLASSES
        and mapping_original_ids == sorted(mapping_original_ids),
    )
    report.check(
        "selected class file matches mapping order",
        selected_ids == mapping_original_ids,
        f"selected {len(selected_ids)}, mapped {len(mapping_original_ids)}",
    )

    expected_metadata = {
        int(entry["new_label"]): (
            int(entry["original_category_id"]),
            str(entry["scientific_name"]),
            str(entry["common_name"]),
        )
        for entry in classes
    }
    combined = pd.concat(frames.values(), ignore_index=True)
    observed_rows = combined[
        [NEW_LABEL_COLUMN, "original_category_id", "scientific_name", "common_name"]
    ].drop_duplicates()
    observed_metadata = {
        int(row.new_label): (
            int(row.original_category_id),
            str(row.scientific_name),
            str(row.common_name),
        )
        for row in observed_rows.itertuples(index=False)
    }
    report.check(
        "CSV new labels and names match class mapping",
        len(observed_rows) == NUM_CLASSES and observed_metadata == expected_metadata,
        f"observed unique metadata rows: {len(observed_rows)}",
    )


def _validate_summary(
    split_dir: Path,
    summary: Any,
    selected_name: str,
    seed: int,
    report: ValidationReport,
) -> None:
    if not report.check("split_summary.json is a JSON object", isinstance(summary, dict)):
        return
    report.check(
        "summary schema version",
        _is_exact_integer(summary.get("schema_version"), SCHEMA_VERSION),
        f"actual {summary.get('schema_version')!r}",
    )
    report.check(
        "summary seed",
        _is_exact_integer(summary.get("seed"), seed),
        f"actual {summary.get('seed')!r}, expected {seed}",
    )
    report.check(
        "summary RNG method",
        summary.get("rng") == RNG_DESCRIPTION,
        f"actual {summary.get('rng')!r}",
    )
    raw_configuration = summary.get("configuration")
    configuration = raw_configuration if isinstance(raw_configuration, dict) else {}
    expected_configuration = {
        "num_classes": NUM_CLASSES,
        "train_mini_images_selected_per_class": TRAIN_MINI_IMAGES_PER_CLASS,
        "train_images_per_class": TRAIN_IMAGES_PER_CLASS,
        "validation_images_per_class": VAL_IMAGES_PER_CLASS,
        "test_images_per_class": TEST_IMAGES_PER_CLASS,
    }
    configuration_ok = set(configuration) == set(expected_configuration) and all(
        _is_exact_integer(configuration.get(key), value)
        for key, value in expected_configuration.items()
    )
    report.check("summary configuration", configuration_ok, f"actual {configuration}")
    report.check(
        "summary CSV column schema",
        summary.get("csv_columns") == CSV_COLUMNS,
    )

    raw_counts = summary.get("counts")
    counts = raw_counts if isinstance(raw_counts, dict) else {}
    expected_counts = {
        "selected_classes": NUM_CLASSES,
        "train_rows": NUM_CLASSES * TRAIN_IMAGES_PER_CLASS,
        "validation_rows": NUM_CLASSES * VAL_IMAGES_PER_CLASS,
        "test_rows": NUM_CLASSES * TEST_IMAGES_PER_CLASS,
    }
    counts_ok = all(
        _is_exact_integer(counts.get(key), value)
        for key, value in expected_counts.items()
    )
    report.check("summary required counts", counts_ok, f"actual {counts}")

    artifacts = summary.get("artifacts")
    expected_artifacts = {
        selected_name,
        "class_mapping.json",
        "train.csv",
        "val.csv",
        "test.csv",
    }
    artifact_names_ok = isinstance(artifacts, dict) and set(artifacts) == expected_artifacts
    report.check(
        "summary artifact inventory",
        artifact_names_ok,
        f"actual {sorted(artifacts) if isinstance(artifacts, dict) else 'not an object'}",
    )
    if artifact_names_ok:
        for name in sorted(expected_artifacts):
            path = split_dir / name
            metadata = artifacts[name]
            metadata_ok = (
                isinstance(metadata, dict)
                and metadata.get("bytes") == path.stat().st_size
                and metadata.get("sha256") == sha256_file(path)
            )
            report.check(
                f"{name} size and SHA-256 match summary",
                metadata_ok,
            )


def _load_source_snapshot(path: Path, needed_ids: set[int]) -> SourceSnapshot:
    """Load source truth and retain enough information to replay split generation."""

    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    for section in ("images", "annotations", "categories"):
        if section not in payload or not isinstance(payload[section], list):
            raise ValueError(f"{path}: missing list section {section!r}")

    categories: dict[int, dict[str, Any]] = {}
    duplicate_categories = 0
    for category in payload["categories"]:
        if not isinstance(category, dict):
            raise ValueError(f"{path}: every category must be an object")
        category_id = category.get("id")
        if not isinstance(category_id, int) or isinstance(category_id, bool):
            raise ValueError(f"{path}: invalid category ID {category_id!r}")
        if category_id in categories:
            duplicate_categories += 1
        else:
            categories[category_id] = category

    images: dict[int, str] = {}
    all_paths: set[str] = set()
    duplicate_image_ids = 0
    duplicate_paths = 0
    unsafe_image_paths = 0
    for image in payload["images"]:
        if not isinstance(image, dict):
            raise ValueError(f"{path}: every image must be an object")
        image_id = image.get("id")
        image_path = image.get("file_name")
        if not isinstance(image_id, int) or isinstance(image_id, bool):
            raise ValueError(f"{path}: invalid image ID {image_id!r}")
        if not isinstance(image_path, str):
            raise ValueError(f"{path}: invalid image path for ID {image_id}")
        if image_id in images:
            duplicate_image_ids += 1
        else:
            images[image_id] = image_path
        if image_path in all_paths:
            duplicate_paths += 1
        all_paths.add(image_path)
        if not is_safe_relative_posix_path(image_path):
            unsafe_image_paths += 1

    referenced_records: set[tuple[int, str, int]] = set()
    records_by_category: defaultdict[int, list[tuple[int, str]]] = defaultdict(list)
    all_annotation_ids: set[int] = set()
    all_annotated_image_ids: set[int] = set()
    duplicate_annotation_ids = 0
    duplicate_annotated_images = 0
    unknown_annotation_categories = 0
    unknown_annotation_images = 0
    image_directory_mismatches = 0
    for annotation in payload["annotations"]:
        if not isinstance(annotation, dict):
            raise ValueError(f"{path}: every annotation must be an object")
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (annotation_id, image_id, category_id)
        ):
            raise ValueError(f"{path}: invalid annotation identifiers")
        if annotation_id in all_annotation_ids:
            duplicate_annotation_ids += 1
        first_image_annotation = image_id not in all_annotated_image_ids
        if not first_image_annotation:
            duplicate_annotated_images += 1
        if category_id not in categories:
            unknown_annotation_categories += 1
        if image_id not in images:
            unknown_annotation_images += 1

        all_annotation_ids.add(annotation_id)
        all_annotated_image_ids.add(image_id)
        if (
            first_image_annotation
            and image_id in images
            and category_id in categories
        ):
            image_path = images[image_id]
            records_by_category[category_id].append((image_id, image_path))
            if image_id in needed_ids:
                referenced_records.add((image_id, image_path, category_id))
            path_parts = PurePosixPath(image_path).parts
            expected_directory = categories[category_id].get("image_dir_name")
            if len(path_parts) < 3 or path_parts[1] != expected_directory:
                image_directory_mismatches += 1

    diagnostics = {
        "category_count": len(categories),
        "image_count": len(images),
        "annotation_count": len(all_annotation_ids),
        "duplicate_category_ids": duplicate_categories,
        "duplicate_image_ids": duplicate_image_ids,
        "duplicate_paths": duplicate_paths,
        "unsafe_image_paths": unsafe_image_paths,
        "duplicate_annotation_ids": duplicate_annotation_ids,
        "duplicate_annotated_images": duplicate_annotated_images,
        "unknown_annotation_categories": unknown_annotation_categories,
        "unknown_annotation_images": unknown_annotation_images,
        "image_directory_mismatches": image_directory_mismatches,
        "image_annotation_id_symmetric_difference": len(
            set(images).symmetric_difference(all_annotated_image_ids)
        ),
        "referenced_records_found": len(referenced_records),
    }
    return SourceSnapshot(
        referenced_records=referenced_records,
        categories=categories,
        records_by_category=dict(records_by_category),
        diagnostics=diagnostics,
    )


def _validate_annotation_provenance(
    frames: dict[str, pd.DataFrame],
    mapping: Any,
    selected_ids: list[int],
    seed: int,
    train_json: Path,
    val_json: Path,
    summary: Any,
    report: ValidationReport,
) -> None:
    raw_sources = (
        summary.get("source_annotations") if isinstance(summary, dict) else None
    )
    sources = raw_sources if isinstance(raw_sources, dict) else {}
    report.check(
        "summary source annotation inventory",
        set(sources) == {"train_mini", "official_validation"},
    )
    source_specs = {
        "train_mini": (train_json, sources.get("train_mini", {})),
        "official_validation": (val_json, sources.get("official_validation", {})),
    }
    for source_name, (path, metadata) in source_specs.items():
        metadata_ok = (
            isinstance(metadata, dict)
            and metadata.get("file_name") == path.name
            and metadata.get("bytes") == path.stat().st_size
            and metadata.get("sha256") == sha256_file(path)
        )
        report.check(f"{source_name} annotation provenance hash", metadata_ok)

    train_rows = pd.concat([frames["train"], frames["val"]], ignore_index=True)
    source_requests = {
        "train_mini": (train_json, train_rows),
        "official_validation": (val_json, frames["test"]),
    }
    snapshots: dict[str, SourceSnapshot] = {}
    zero_diagnostic_keys = (
        "duplicate_category_ids",
        "duplicate_image_ids",
        "duplicate_paths",
        "unsafe_image_paths",
        "duplicate_annotation_ids",
        "duplicate_annotated_images",
        "unknown_annotation_categories",
        "unknown_annotation_images",
        "image_directory_mismatches",
        "image_annotation_id_symmetric_difference",
    )
    for source_name, (path, frame) in source_requests.items():
        needed_ids = set(int(value) for value in frame["image_id"])
        try:
            snapshot = _load_source_snapshot(path, needed_ids)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            report.fail(f"Read {source_name} annotation source", str(error))
            continue
        snapshots[source_name] = snapshot
        source_unique = all(
            snapshot.diagnostics[key] == 0 for key in zero_diagnostic_keys
        )
        report.check(
            f"{source_name} source IDs, paths, and directories are one-to-one",
            source_unique,
            f"diagnostics {snapshot.diagnostics}",
        )
        expected_records = set(
            zip(
                (int(value) for value in frame["image_id"]),
                (str(value) for value in frame["image_path"]),
                (int(value) for value in frame["original_category_id"]),
            )
        )
        report.check(
            f"{source_name} CSV rows match source annotations",
            expected_records == snapshot.referenced_records
            and len(snapshot.referenced_records) == len(frame),
            (
                f"CSV records {len(expected_records)}, source matches "
                f"{len(snapshot.referenced_records)}, rows {len(frame)}"
            ),
        )

    if set(snapshots) != {"train_mini", "official_validation"}:
        report.skip("deterministic split reconstruction", "source loading failed")
        return

    train_snapshot = snapshots["train_mini"]
    val_snapshot = snapshots["official_validation"]
    train_categories = train_snapshot.categories
    val_categories = val_snapshot.categories
    raw_mapping_classes = mapping.get("classes") if isinstance(mapping, dict) else None
    mapping_classes = raw_mapping_classes if isinstance(raw_mapping_classes, list) else []
    mapped_category_ids = [
        entry["original_category_id"]
        for entry in mapping_classes
        if isinstance(entry, dict)
        and isinstance(entry.get("original_category_id"), int)
        and not isinstance(entry.get("original_category_id"), bool)
    ]
    selected_metadata_matches = all(
        category_id in train_categories
        and category_id in val_categories
        and train_categories[category_id] == val_categories[category_id]
        for category_id in mapped_category_ids
    )
    report.check(
        "selected category metadata matches across annotation sources",
        len(mapped_category_ids) == NUM_CLASSES and selected_metadata_matches,
    )

    def entry_matches_source(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        category_id = entry.get("original_category_id")
        if (
            not isinstance(category_id, int)
            or isinstance(category_id, bool)
            or category_id not in train_categories
        ):
            return False
        category = train_categories[category_id]
        return (
            entry.get("scientific_name") == str(category.get("name") or "")
            and entry.get("common_name") == str(category.get("common_name") or "")
            and entry.get("image_dir_name")
            == str(category.get("image_dir_name") or "")
            and entry.get("taxonomy")
            == {
                field: str(category.get(field) or "") for field in TAXONOMY_FIELDS
            }
        )

    mapping_matches_source = len(mapping_classes) == NUM_CLASSES and all(
        entry_matches_source(entry) for entry in mapping_classes
    )
    report.check("class mapping metadata matches source categories", mapping_matches_source)

    eligible_ids = sorted(
        category_id
        for category_id in set(train_categories).intersection(val_categories)
        if len(train_snapshot.records_by_category.get(category_id, []))
        >= TRAIN_MINI_IMAGES_PER_CLASS
        and len(val_snapshot.records_by_category.get(category_id, []))
        >= TEST_IMAGES_PER_CLASS
    )
    expected_selected_ids = select_class_ids(eligible_ids, seed)
    report.check(
        "selected class IDs exactly reproduce the declared seed",
        selected_ids == expected_selected_ids,
        f"eligible {len(eligible_ids)}; differing IDs "
        f"{len(set(selected_ids).symmetric_difference(expected_selected_ids))}",
    )

    raw_counts = summary.get("counts") if isinstance(summary, dict) else None
    summary_counts = raw_counts if isinstance(raw_counts, dict) else {}
    source_counts_ok = (
        _is_exact_integer(summary_counts.get("eligible_classes"), len(eligible_ids))
        and _is_exact_integer(
            summary_counts.get("source_train_images"),
            train_snapshot.diagnostics["image_count"],
        )
        and _is_exact_integer(
            summary_counts.get("source_official_validation_images"),
            val_snapshot.diagnostics["image_count"],
        )
    )
    report.check(
        "summary eligibility and source-image counts",
        source_counts_ok,
        f"actual {summary_counts}",
    )

    expected_assignments: dict[str, set[tuple[int, int, str, int]]] = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    try:
        for new_label, category_id in enumerate(expected_selected_ids):
            train_pool = sorted(
                train_snapshot.records_by_category[category_id],
                key=lambda record: (record[0], record[1]),
            )
            train_positions = deterministic_record_positions(
                len(train_pool),
                TRAIN_MINI_IMAGES_PER_CLASS,
                seed=seed,
                category_id=category_id,
                source_tag=0,
            )
            chosen_train = [train_pool[index] for index in train_positions]
            for image_id, image_path in chosen_train[:TRAIN_IMAGES_PER_CLASS]:
                expected_assignments["train"].add(
                    (new_label, image_id, image_path, category_id)
                )
            for image_id, image_path in chosen_train[TRAIN_IMAGES_PER_CLASS:]:
                expected_assignments["val"].add(
                    (new_label, image_id, image_path, category_id)
                )

            test_pool = sorted(
                val_snapshot.records_by_category[category_id],
                key=lambda record: (record[0], record[1]),
            )
            test_positions = deterministic_record_positions(
                len(test_pool),
                TEST_IMAGES_PER_CLASS,
                seed=seed,
                category_id=category_id,
                source_tag=1,
            )
            for index in test_positions:
                image_id, image_path = test_pool[index]
                expected_assignments["test"].add(
                    (new_label, image_id, image_path, category_id)
                )
    except (KeyError, ValueError) as error:
        report.fail("Reconstruct deterministic assignments", str(error))
        return

    for split, frame in frames.items():
        actual_assignments = set(
            zip(
                (int(value) for value in frame[NEW_LABEL_COLUMN]),
                (int(value) for value in frame["image_id"]),
                (str(value) for value in frame["image_path"]),
                (int(value) for value in frame["original_category_id"]),
            )
        )
        expected = expected_assignments[split]
        report.check(
            f"{split} assignments exactly reproduce the declared seed",
            actual_assignments == expected and len(actual_assignments) == len(frame),
            f"expected {len(expected)}, actual {len(actual_assignments)}",
        )


def _validate_image_files(
    frames: dict[str, pd.DataFrame],
    image_root: Path,
    *,
    hash_files: bool,
    report: ValidationReport,
) -> None:
    image_root_resolved = image_root.resolve()
    files: list[tuple[str, str, Path]] = []
    unsafe_join = 0
    for split, frame in frames.items():
        for local_image_path in frame[LOCAL_IMAGE_PATH_COLUMN]:
            pure_path = PurePosixPath(str(local_image_path))
            full_path = image_root.joinpath(*pure_path.parts)
            resolved_path = full_path.resolve(strict=False)
            if (
                resolved_path != image_root_resolved
                and image_root_resolved not in resolved_path.parents
            ):
                unsafe_join += 1
            files.append((split, str(local_image_path), full_path))

    missing = [
        (split, image_path)
        for split, image_path, path in files
        if not path.is_file()
    ]
    report.check(
        "all referenced image files exist",
        not missing and unsafe_join == 0,
        (
            f"missing {len(missing)}, unsafe joins {unsafe_join}; "
            f"examples: {_first_values(missing)}"
        ),
    )
    if not hash_files:
        report.skip("image-content hash leakage check", "--hash-files was not supplied")
        return
    if missing or unsafe_join:
        report.skip("image-content hash leakage check", "image existence check failed")
        return

    digest_records: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    try:
        for split, image_path, full_path in tqdm(
            files, desc="Hashing images", unit="image"
        ):
            digest_records[sha256_file(full_path)].append((split, image_path))
    except OSError as error:
        report.fail("read image files for content hashing", str(error))
        return
    duplicates = {
        digest: records for digest, records in digest_records.items() if len(records) > 1
    }
    report.check(
        "no byte-identical images occur within or across splits",
        not duplicates,
        f"duplicate hashes: {len(duplicates)}",
    )


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the deterministic 500-class iNaturalist split and detect "
            "metadata-level data leakage."
        )
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing generated split artifacts.",
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
        "--image-root",
        type=Path,
        default=Path("data/processed/images"),
        help="Root against which CSV local_image_path values are resolved.",
    )
    parser.add_argument(
        "--seed",
        type=_nonnegative_integer,
        default=42,
        help="Seed identifying selected_classes_seed<seed>.txt (default: 42).",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Require every image referenced by the CSV files to exist.",
    )
    parser.add_argument(
        "--hash-files",
        action="store_true",
        help="Hash existing images to detect byte-identical content; implies --check-files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    split_dir = args.split_dir.expanduser()
    train_json = args.train_json.expanduser()
    val_json = args.val_json.expanduser()
    selected_name = f"selected_classes_seed{args.seed}.txt"
    required_paths = {
        "selected classes": split_dir / selected_name,
        "class mapping": split_dir / "class_mapping.json",
        "train CSV": split_dir / "train.csv",
        "validation CSV": split_dir / "val.csv",
        "test CSV": split_dir / "test.csv",
        "split summary": split_dir / "split_summary.json",
        "train annotations": train_json,
        "official-validation annotations": val_json,
    }

    print("iNaturalist split validation report")
    print("=" * 72)
    report = ValidationReport()
    all_required_exist = True
    for description, path in required_paths.items():
        exists = path.is_file()
        all_required_exist &= exists
        report.check(f"Required {description} exists", exists, str(path))
    if not all_required_exist:
        report.skip("remaining validation", "one or more required files are missing")
        report.print_summary()
        return 1

    frames: dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        frame = _load_csv(split_dir / f"{split}.csv", report)
        if frame is not None:
            frames[split] = frame
    if set(frames) != {"train", "val", "test"}:
        report.skip("remaining split validation", "one or more CSV files are invalid")
        report.print_summary()
        return 1

    try:
        mapping = _load_json(split_dir / "class_mapping.json")
        summary = _load_json(split_dir / "split_summary.json")
        selected_lines = (split_dir / selected_name).read_text(encoding="utf-8").splitlines()
        selected_ids = [int(value) for value in selected_lines if value.strip()]
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        report.fail("Read mapping, summary, and selected classes", str(error))
        report.print_summary()
        return 1

    report.check(
        "selected class file contains exactly 500 unique IDs",
        len(selected_ids) == NUM_CLASSES and len(set(selected_ids)) == NUM_CLASSES,
        f"rows {len(selected_ids)}, unique {len(set(selected_ids))}",
    )

    try:
        _validate_split_structure(frames, report)
        _validate_mapping_and_selection(
            frames, mapping, selected_ids, args.seed, report
        )
        _validate_summary(split_dir, summary, selected_name, args.seed, report)
        _validate_annotation_provenance(
            frames,
            mapping,
            selected_ids,
            args.seed,
            train_json,
            val_json,
            summary,
            report,
        )

        if args.check_files or args.hash_files:
            _validate_image_files(
                frames,
                args.image_root.expanduser(),
                hash_files=args.hash_files,
                report=report,
            )
        else:
            report.skip("image file existence check", "--check-files was not supplied")
            report.skip(
                "image-content hash leakage check", "--hash-files was not supplied"
            )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
    ) as error:
        report.fail(
            "Complete validation without an internal exception",
            f"{type(error).__name__}: {error}",
        )

    report.print_summary()
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
