"""Shared constants and small helpers for the dataset pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


SCHEMA_VERSION = 2
NUM_CLASSES = 500
TRAIN_IMAGES_PER_CLASS = 40
VAL_IMAGES_PER_CLASS = 10
TEST_IMAGES_PER_CLASS = 10
TRAIN_MINI_IMAGES_PER_CLASS = TRAIN_IMAGES_PER_CLASS + VAL_IMAGES_PER_CLASS

NEW_LABEL_COLUMN = "new_label"
LOCAL_IMAGE_PATH_COLUMN = "local_image_path"

CSV_COLUMNS = [
    "split",
    "source_split",
    "image_path",
    LOCAL_IMAGE_PATH_COLUMN,
    "image_id",
    NEW_LABEL_COLUMN,
    "original_category_id",
    "scientific_name",
    "common_name",
]

TAXONOMY_FIELDS = [
    "supercategory",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "specific_epithet",
]

RNG_DESCRIPTION = {
    "algorithm": "NumPy PCG64",
    "class_selection": "PCG64(seed), choice without replacement from sorted eligible IDs",
    "train_per_class": "PCG64(SeedSequence([seed, original_category_id, 0]))",
    "test_per_class": "PCG64(SeedSequence([seed, original_category_id, 1]))",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 digest of an in-memory artifact."""

    return hashlib.sha256(content).hexdigest()


def deterministic_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically and terminate it with one newline."""

    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return f"{text}\n".encode("utf-8")


def select_class_ids(eligible_ids: list[int], seed: int) -> list[int]:
    """Select and sort the deterministic project class IDs."""

    if len(eligible_ids) < NUM_CLASSES:
        raise ValueError(
            f"Only {len(eligible_ids)} classes are eligible; {NUM_CLASSES} are required"
        )
    generator = np.random.Generator(np.random.PCG64(seed))
    return sorted(
        int(value)
        for value in generator.choice(
            np.asarray(sorted(eligible_ids), dtype=np.int64),
            size=NUM_CLASSES,
            replace=False,
        ).tolist()
    )


def deterministic_record_positions(
    pool_size: int,
    take: int,
    *,
    seed: int,
    category_id: int,
    source_tag: int,
) -> list[int]:
    """Return deterministic record positions for one category and source."""

    if pool_size < take:
        raise ValueError(f"Cannot select {take} records from a pool of {pool_size}")
    seed_sequence = np.random.SeedSequence([seed, category_id, source_tag])
    generator = np.random.Generator(np.random.PCG64(seed_sequence))
    return [int(index) for index in generator.permutation(pool_size)[:take]]


def is_safe_relative_posix_path(value: str) -> bool:
    """Return whether *value* is a portable, non-traversing relative POSIX path."""

    if not value or "\\" in value:
        return False
    candidate = PurePosixPath(value)
    return (
        not candidate.is_absolute()
        and bool(candidate.parts)
        and str(candidate) not in {"", "."}
        and ".." not in candidate.parts
    )
