"""Focused tests for the optional SIFT Bag-of-Visual-Words implementation."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.data.common import CSV_COLUMNS, sha256_file
from src.traditional.bovw import (
    BoVWConfig,
    FittedVocabulary,
    SiftResult,
    deterministic_balanced_indices,
    encode_descriptors,
    evaluate_pilot_gate,
    extract_sift_descriptors,
    fit_vocabulary,
    load_grayscale,
    project_bovw_runtime,
)
from src.traditional.extract_bovw_features import (
    STATUS_NAME,
    build_split_artifact_metadata,
    extract_bovw,
    load_manifests,
    main as bovw_main,
)


class _VocabularyFixture:
    n_clusters = 3

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        return np.asarray([0, 2, 2, 1], dtype=np.int64)[: len(descriptors)]


class _SoftKMeansFixture:
    n_clusters = 256

    cluster_centers_ = np.vstack(
        [
            np.zeros((1, 128), dtype=np.float32),
            np.ones((1, 128), dtype=np.float32),
            np.full((1, 128), 2.0, dtype=np.float32),
            np.full((253, 128), 100.0, dtype=np.float32),
        ]
    )

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        raise AssertionError("production soft assignment must use direct top-k search")

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        return np.zeros(len(descriptors), dtype=np.int64)


class BoVWTests(unittest.TestCase):
    def _write_stage1_fixture(self, root: Path) -> Path:
        split_dir = root / "splits"
        split_dir.mkdir()
        specifications = {
            "train": (40, "train_mini", 0),
            "val": (10, "train_mini", 100_000),
            "test": (10, "val", 200_000),
        }
        for split, (per_class, source_split, offset) in specifications.items():
            labels = np.repeat(np.arange(500, dtype=np.int64), per_class)
            image_ids = np.arange(len(labels), dtype=np.int64) + offset
            names = [f"{value:032x}.jpg" for value in image_ids]
            frame = pd.DataFrame(
                {
                    "split": split,
                    "source_split": source_split,
                    "image_path": [
                        f"{source_split}/{label}/{name}"
                        for label, name in zip(labels, names)
                    ],
                    "local_image_path": [f"{split}/{name}" for name in names],
                    "image_id": image_ids,
                    "new_label": labels,
                    "original_category_id": labels + 1_000,
                    "scientific_name": [f"Species {label}" for label in labels],
                    "common_name": [f"Common {label}" for label in labels],
                },
                columns=CSV_COLUMNS,
            )
            frame.to_csv(split_dir / f"{split}.csv", index=False)
        (split_dir / "selected_classes_seed42.txt").write_text(
            "".join(f"{value}\n" for value in range(500)), encoding="utf-8"
        )
        (split_dir / "class_mapping.json").write_text("{}\n", encoding="utf-8")
        artifact_names = (
            "selected_classes_seed42.txt",
            "class_mapping.json",
            "train.csv",
            "val.csv",
            "test.csv",
        )
        summary = {
            "schema_version": 2,
            "seed": 42,
            "csv_columns": CSV_COLUMNS,
            "artifacts": {
                name: {
                    "bytes": (split_dir / name).stat().st_size,
                    "sha256": sha256_file(split_dir / name),
                }
                for name in artifact_names
            },
        }
        (split_dir / "split_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return split_dir

    def test_grayscale_loading_preserves_aspect_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgba.png"
            pixels = np.zeros((10, 20, 4), dtype=np.uint8)
            pixels[..., 0] = 200
            pixels[..., 3] = 128
            Image.fromarray(pixels, mode="RGBA").save(path)
            grayscale = load_grayscale(path, max_side=32)
            self.assertEqual(grayscale.shape, (16, 32))
            self.assertEqual(grayscale.dtype, np.uint8)

    def test_featureless_image_has_zero_bovw_histogram(self) -> None:
        blank = np.zeros((64, 64), dtype=np.uint8)
        descriptors = extract_sift_descriptors(blank, nfeatures=20)
        self.assertIsNone(descriptors)
        encoded = encode_descriptors(descriptors, _VocabularyFixture())
        np.testing.assert_array_equal(encoded, np.zeros(3, dtype=np.float32))

    def test_legacy_array_encoding_returns_raw_counts(self) -> None:
        descriptors = np.ones((4, 128), dtype=np.float32)
        encoded = encode_descriptors(descriptors, _VocabularyFixture())
        np.testing.assert_allclose(encoded, [1.0, 1.0, 2.0])
        self.assertEqual(encoded.dtype, np.float32)

    def test_vocabulary_fit_is_deterministic(self) -> None:
        descriptors = np.vstack(
            [np.zeros((8, 128), dtype=np.float32), np.ones((8, 128), dtype=np.float32)]
        )
        old_cpu_limit = os.environ.get("LOKY_MAX_CPU_COUNT")
        os.environ["LOKY_MAX_CPU_COUNT"] = "1"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                first = fit_vocabulary(
                    descriptors, vocabulary_size=2, batch_size=8, n_init=1, seed=42
                )
                second = fit_vocabulary(
                    descriptors, vocabulary_size=2, batch_size=8, n_init=1, seed=42
                )
        finally:
            if old_cpu_limit is None:
                os.environ.pop("LOKY_MAX_CPU_COUNT", None)
            else:
                os.environ["LOKY_MAX_CPU_COUNT"] = old_cpu_limit
        np.testing.assert_allclose(first.cluster_centers_, second.cluster_centers_)

    def test_soft_spatial_encoding_has_exact_1280_width(self) -> None:
        generator = np.random.default_rng(7)
        descriptors = generator.normal(size=(4, 128)).astype(np.float32)
        config = BoVWConfig(vocabulary_size=256, soft_assign_k=3, pyramid_grid=2)
        vocabulary = FittedVocabulary(
            kmeans=_SoftKMeansFixture(), sigma=1.0, config=config
        )
        sift = SiftResult(
            descriptors=descriptors,
            points=np.asarray([[1, 1], [9, 1], [1, 9], [9, 9]], dtype=np.float32),
            height=10,
            width=10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            encoded = encode_descriptors(sift, vocabulary)
        self.assertEqual(encoded.shape, (1280,))
        self.assertTrue(np.all(encoded >= 0))
        # Each descriptor contributes once globally and once to a 2x2 cell.
        self.assertAlmostEqual(float(encoded.sum()), 8.0, places=5)

    def test_balanced_selection_is_deterministic_and_train_local(self) -> None:
        labels = np.repeat(np.arange(3), 5)
        first = deterministic_balanced_indices(
            labels, 2, seed=42, source_tag=4
        )
        second = deterministic_balanced_indices(
            labels, 2, seed=42, source_tag=4
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(set(labels[first]), {0, 1, 2})
        self.assertTrue(np.all(first < len(labels)))

    def test_gate_requires_every_condition(self) -> None:
        passed = evaluate_pilot_gate(
            decode_failures=0,
            classes_with_descriptors=500,
            expected_classes=500,
            descriptor_count=256,
            vocabulary_size=256,
            available_disk_bytes=6 * 1024**3,
            peak_memory_bytes=2 * 1024**3,
            projected_seconds=100.0,
            benchmark_succeeded=True,
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["status"], "PASS")

        rejected = evaluate_pilot_gate(
            decode_failures=1,
            classes_with_descriptors=499,
            expected_classes=500,
            descriptor_count=255,
            vocabulary_size=256,
            available_disk_bytes=1,
            peak_memory_bytes=13 * 1024**3,
            projected_seconds=10_801.0,
            benchmark_succeeded=False,
        )
        self.assertFalse(rejected["passed"])
        self.assertEqual(rejected["status"], "NOT RUN")
        self.assertFalse(any(rejected["checks"].values()))

    def test_runtime_projection_counts_all_32000_sift_passes(self) -> None:
        projection = project_bovw_runtime(
            pilot_extraction_seconds=10.0,
            pilot_encoding_seconds=5.0,
            pilot_clustering_seconds=2.0,
            pilot_images=1_000,
            pilot_vocabulary_descriptors=100_000,
            total_encoding_images=30_000,
            vocabulary_training_images=2_000,
            vocabulary_descriptors_per_image=100,
            safety_factor=1.25,
        )
        self.assertEqual(projection["full_sift_image_passes"], 32_000)
        self.assertEqual(projection["projected_extraction_seconds"], 320.0)
        self.assertEqual(projection["projected_encoding_seconds"], 150.0)
        self.assertEqual(projection["projected_clustering_seconds"], 4.0)
        self.assertEqual(projection["projected_seconds"], 592.5)

    def test_bovw_artifact_metadata_matches_evaluator_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "val.csv"
            feature = root / "bovw_val_features.npz"
            manifest.write_text("manifest\n", encoding="utf-8")
            feature.write_bytes(b"features")
            entry = build_split_artifact_metadata(
                manifest,
                feature,
                rows=5_000,
                statistics={"encoding_seconds": 1.5},
            )
            self.assertEqual(entry["manifest"]["file_name"], "val.csv")
            self.assertEqual(entry["manifest"]["rows"], 5_000)
            self.assertEqual(
                entry["feature_file"]["file_name"], "bovw_val_features.npz"
            )
            self.assertEqual(entry["feature_file"]["sha256"], sha256_file(feature))
            self.assertEqual(entry["output_sha256"], entry["feature_file"]["sha256"])

    def test_bovw_manifests_require_frozen_provenance_and_source_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_dir = self._write_stage1_fixture(root)
            frames, paths, stage1 = load_manifests(split_dir, root / "images")
            self.assertEqual([len(frames[name]) for name in ("train", "val")], [20_000, 5_000])
            self.assertNotIn("test", frames)
            self.assertEqual(len(paths["train"]), 20_000)
            self.assertEqual(stage1["seed"], 42)

            val_path = split_dir / "val.csv"
            val_frame = pd.read_csv(val_path, keep_default_na=False)
            val_frame["source_split"] = "val"
            val_frame.to_csv(val_path, index=False)
            with self.assertRaisesRegex(ValueError, "incorrect source_split"):
                load_manifests(split_dir, root / "images")

    def test_fabricated_selection_cannot_unlock_bovw_test_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text(
                json.dumps({"schema_version": 2, "status": "frozen"}),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                max_side=256,
                sift_nfeatures=300,
                vocabulary_size=256,
                vocabulary_images_per_class=4,
                pilot_images_per_class=2,
                descriptors_per_image=100,
                batch_size=4096,
                n_init=3,
                soft_assign_k=3,
                pyramid_grid=2,
                no_spatial_pyramid=False,
                seed=42,
                splits="test",
                selection_manifest=selection,
                split_dir=root / "splits",
                image_root=root / "images",
                output_dir=root / "bovw",
                force=False,
            )
            from unittest.mock import patch

            with patch(
                "src.traditional.extract_bovw_features.load_manifests"
            ) as load_test_manifest:
                with self.assertRaises(ValueError):
                    extract_bovw(args)
            load_test_manifest.assert_not_called()

    def test_canonical_status_is_preserved_without_force_and_failed_with_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "bovw"
            output_dir.mkdir()
            status_path = output_dir / STATUS_NAME
            status_path.write_text(
                json.dumps({"schema_version": 1, "status": "COMPLETE"}),
                encoding="utf-8",
            )
            arguments = [
                "--output-dir",
                str(output_dir),
                "--split-dir",
                str(root / "missing-splits"),
                "--image-root",
                str(root / "images"),
                "--pilot-only",
            ]
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bovw_main(arguments)
            self.assertEqual(
                json.loads(status_path.read_text(encoding="utf-8"))["status"],
                "COMPLETE",
            )

            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    bovw_main([*arguments, "--force"])
            failure = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "FAILED")
            self.assertEqual(failure["error_type"], "ValueError")
            self.assertIn("Missing Stage 1 manifest", failure["error"])


if __name__ == "__main__":
    unittest.main()
