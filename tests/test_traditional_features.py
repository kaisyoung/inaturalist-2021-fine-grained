"""Focused unit tests for Stage 2 handcrafted features and cache safety."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

from src.data.common import CSV_COLUMNS, sha256_file
from src.traditional.extract_features import (
    CACHE_SCHEMA_VERSION,
    NPZ_FIELDS,
    FeatureCacheError,
    _atomic_save_npz,
    _cache_is_valid,
    _extract_matrix,
    extract_all_splits,
    _implementation_fingerprint,
    _npz_payload,
    _package_versions,
    _resolve_image_paths,
    _validate_npz_contract,
)
from src.traditional.features import (
    ImageFeatureError,
    extract_features,
    extract_features_from_array,
    feature_configuration,
    load_image_rgb,
    validate_image_size,
)


class FeatureExtractionTests(unittest.TestCase):
    """Test deterministic preprocessing and exact descriptor contracts."""

    def test_default_feature_dimension_and_component_order(self) -> None:
        rows, columns = np.indices((128, 128))
        image = np.stack(
            (
                (rows * 2) % 256,
                (columns * 2) % 256,
                (rows + columns) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)

        feature = extract_features_from_array(image)
        config = feature_configuration(128)

        self.assertEqual(feature.dtype, np.float32)
        self.assertEqual(feature.shape, (2424,))
        self.assertEqual(config["feature_dim"], 2424)
        self.assertEqual(config["concatenation_order"], ["color", "texture", "hog"])
        self.assertEqual(config["components"]["color"]["dimension"], 520)
        self.assertEqual(config["components"]["texture"]["dimension"], 140)
        self.assertEqual(config["components"]["hog"]["dimension"], 1764)
        self.assertTrue(np.all(feature >= 0))
        self.assertTrue(np.isfinite(feature).all())

    def test_grayscale_and_rgba_are_converted_to_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            grayscale_path = root / "gray.png"
            transparent_path = root / "transparent.png"
            Image.new("L", (41, 53), color=80).save(grayscale_path)
            Image.new("RGBA", (37, 29), color=(255, 0, 0, 0)).save(transparent_path)

            grayscale = load_image_rgb(grayscale_path, 128)
            transparent = load_image_rgb(transparent_path, 128)

        self.assertEqual(grayscale.shape, (128, 128, 3))
        self.assertEqual(grayscale.dtype, np.uint8)
        np.testing.assert_array_equal(grayscale[..., 0], grayscale[..., 1])
        np.testing.assert_array_equal(grayscale[..., 1], grayscale[..., 2])
        # Fully transparent pixels are composited over white, not arbitrary RGB.
        self.assertTrue(np.all(transparent == 255))

    def test_image_size_rejects_hog_border_discard(self) -> None:
        self.assertEqual(validate_image_size(128), 128)
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            validate_image_size(127)
        with self.assertRaisesRegex(ValueError, "at least 32"):
            validate_image_size(16)

    def test_file_extraction_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jpg"
            generator = np.random.default_rng(42)
            pixels = generator.integers(0, 256, size=(47, 59, 3), dtype=np.uint8)
            Image.fromarray(pixels, mode="RGB").save(path)
            first = extract_features(path, 128)
            second = extract_features(path, 128)
        np.testing.assert_array_equal(first, second)

    def test_corrupt_image_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jpg"
            path.write_bytes(b"not a JPEG")
            with self.assertRaises(ImageFeatureError):
                extract_features(path, 128)


class FeatureCacheTests(unittest.TestCase):
    """Test ordered process extraction and non-pickled cache provenance."""

    def test_serial_and_parallel_extraction_match_with_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, colour in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
                path = root / f"{index}.png"
                Image.new("RGB", (40 + index, 35 + index), color=colour).save(path)
                paths.append(path)

            serial = _extract_matrix(
                paths, image_size=32, workers=1, description="serial test"
            )
            try:
                parallel = _extract_matrix(
                    paths, image_size=32, workers=2, description="parallel test"
                )
            except PermissionError:
                self.skipTest("process semaphores are unavailable in this sandbox")

        for serial_value, parallel_value in zip(serial, parallel, strict=True):
            np.testing.assert_array_equal(serial_value, parallel_value)
        self.assertEqual(serial[0].shape, (3, 696))
        self.assertEqual(len(set(serial[3].tolist())), 3)

    def test_unsafe_local_path_is_rejected(self) -> None:
        frame = pd.DataFrame({"local_image_path": ["../escape.jpg"]})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FeatureCacheError, "unsafe"):
                _resolve_image_paths(frame, Path(directory), "train")

    def test_fabricated_selection_cannot_unlock_test_manifest_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text(
                json.dumps({"schema_version": 2, "status": "frozen"}),
                encoding="utf-8",
            )
            with patch("src.traditional.extract_features._read_manifest") as read_manifest:
                with self.assertRaises(FeatureCacheError):
                    extract_all_splits(
                        split_dir=root / "splits",
                        image_root=root / "images",
                        output_dir=root / "features",
                        image_size=32,
                        workers=1,
                        splits=("test",),
                        selection_manifest=selection,
                    )
            read_manifest.assert_not_called()

    def test_implementation_change_invalidates_cache_before_artifact_reuse(self) -> None:
        self.assertEqual(CACHE_SCHEMA_VERSION, 3)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            stage1_hash = "b" * 64
            config = {
                **feature_configuration(32),
                "schema_version": CACHE_SCHEMA_VERSION,
                "stage1": {"split_summary_sha256": stage1_hash},
                "package_versions": _package_versions(),
                "implementation_fingerprint": {
                    "algorithm": "SHA-256",
                    "combined_sha256": "0" * 64,
                    "files": {},
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")
            valid, reason = _cache_is_valid(
                config_path,
                {},
                image_size=32,
                manifest_hashes={},
                manifest_rows={},
                stage1_summary_hash=stage1_hash,
                source_paths={},
            )

        self.assertFalse(valid)
        self.assertIn("implementation changed", reason)
        fingerprint = _implementation_fingerprint()
        self.assertEqual(fingerprint["algorithm"], "SHA-256")
        self.assertEqual(len(fingerprint["combined_sha256"]), 64)
        self.assertEqual(
            set(fingerprint["files"]),
            {
                "src/traditional/extract_features.py",
                "src/traditional/features.py",
            },
        )

    def test_atomic_npz_failure_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifact.npz"
            original = b"previous valid cache"
            path.write_bytes(original)

            with patch(
                "src.traditional.extract_features.np.savez_compressed",
                side_effect=RuntimeError("simulated write failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                    _atomic_save_npz(path, {"values": np.arange(3)})

            self.assertEqual(path.read_bytes(), original)
            self.assertEqual([candidate.name for candidate in root.iterdir()], [path.name])

    def test_npz_contract_loads_without_pickle_and_detects_corruption(self) -> None:
        rows = 500
        frame = pd.DataFrame(
            {
                "split": ["train"] * rows,
                "source_split": ["train_mini"] * rows,
                "image_path": [f"train_mini/class/{index}.jpg" for index in range(rows)],
                "local_image_path": [f"train/{index}.jpg" for index in range(rows)],
                "image_id": np.arange(rows, dtype=np.int64),
                "new_label": np.arange(rows, dtype=np.int64),
                "original_category_id": np.arange(1000, 1000 + rows, dtype=np.int64),
                "scientific_name": [f"Species {index}" for index in range(rows)],
                "common_name": [f"Common {index}" for index in range(rows)],
            },
            columns=CSV_COLUMNS,
        )
        manifest_hash = "a" * 64
        features = np.zeros((rows, 102), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.bin"
            source_path.write_bytes(b"original")
            source_status = source_path.stat()
            source_sizes = np.full(rows, source_status.st_size, dtype=np.int64)
            source_mtime = np.full(rows, source_status.st_mtime_ns, dtype=np.int64)
            source_hashes = np.asarray([sha256_file(source_path)] * rows)
            payload = _npz_payload(
                frame,
                features,
                manifest_path=Path("train.csv"),
                manifest_sha256=manifest_hash,
                source_sizes=source_sizes,
                source_mtime_ns=source_mtime,
                source_sha256=source_hashes,
            )
            path = root / "features.npz"
            _atomic_save_npz(path, payload)
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(tuple(archive.files), NPZ_FIELDS)
                self.assertTrue(all(archive[name].dtype != object for name in NPZ_FIELDS))
            valid, reason = _validate_npz_contract(
                path,
                expected_split="train",
                expected_rows=rows,
                expected_dimension=102,
                expected_manifest_hash=manifest_hash,
                source_paths=[source_path] * rows,
            )
            self.assertTrue(valid, reason)

            # A metadata-only touch triggers rehashing but remains valid.
            os.utime(
                source_path,
                ns=(source_status.st_atime_ns, source_status.st_mtime_ns + 1_000_000),
            )
            valid, reason = _validate_npz_contract(
                path,
                expected_split="train",
                expected_rows=rows,
                expected_dimension=102,
                expected_manifest_hash=manifest_hash,
                source_paths=[source_path] * rows,
            )
            self.assertTrue(valid, reason)

            corrupted = dict(payload)
            corrupted["features"] = features.copy()
            corrupted["features"][0, 0] = np.nan
            _atomic_save_npz(path, corrupted)
            valid, reason = _validate_npz_contract(
                path,
                expected_split="train",
                expected_rows=rows,
                expected_dimension=102,
                expected_manifest_hash=manifest_hash,
            )
            self.assertFalse(valid)
            self.assertIn("non-finite", reason)

            _atomic_save_npz(path, payload)
            source_path.write_bytes(b"modified")
            valid, reason = _validate_npz_contract(
                path,
                expected_split="train",
                expected_rows=rows,
                expected_dimension=102,
                expected_manifest_hash=manifest_hash,
                source_paths=[source_path] * rows,
            )
            self.assertFalse(valid)
            self.assertIn("source image changed", reason)


if __name__ == "__main__":
    unittest.main()
