"""Focused tests for Stage 2 traditional model training contracts."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from src.data.common import sha256_file
from src.traditional.train_traditional import (
    CANONICAL_LINEARSVC_PARAMETERS,
    CANONICAL_RANDOM_FOREST_PARAMETERS,
    balanced_pilot_indices,
    estimator_parameters,
    load_training_features,
    make_linearsvc_pipeline,
    make_random_forest,
    parse_components,
    projected_linearsvc_seconds,
    resolve_feature_selection,
    train_and_save,
    train_linear_svm,
    validate_canonical_estimator_parameters,
    validate_canonical_training_request,
)


class TraditionalModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "feature_type": "rgb_hsv_lbp_hog",
            "feature_dim": 10,
            "image_size": [128, 128],
            "concatenation_order": ["rgb_histogram", "hsv_histogram", "lbp", "hog"],
            "components": {
                "rgb_histogram": {"start": 0, "stop": 2, "dimension": 2},
                "hsv_histogram": {"start": 2, "stop": 4, "dimension": 2},
                "lbp": {"start": 4, "stop": 6, "dimension": 2},
                "hog": {"start": 6, "stop": 10, "dimension": 4},
            },
        }

    def _write_production_fixture(
        self, root: Path
    ) -> tuple[Path, Path, Path, np.ndarray, np.ndarray]:
        split_dir = root / "splits"
        split_dir.mkdir()
        manifest_path = split_dir / "train.csv"
        manifest_path.write_text("frozen Stage 1 train manifest\n", encoding="utf-8")
        manifest_hash = sha256_file(manifest_path)
        labels = np.repeat(np.arange(500, dtype=np.int64), 40)
        features = np.stack(
            [labels % 7, labels % 11, labels % 13, labels % 17], axis=1
        ).astype(np.float32)
        feature_path = root / "traditional_train_features.npz"
        np.savez_compressed(
            feature_path,
            features=features,
            new_labels=labels,
            manifest_sha256=np.asarray(manifest_hash),
            manifest_file_name=np.asarray("train.csv"),
        )
        config = {
            "feature_type": "fixture",
            "feature_dim": 4,
            "image_size": [8, 8],
            "concatenation_order": ["fixture"],
            "components": {"fixture": {"start": 0, "stop": 4}},
            "split_artifacts": {
                "train": {
                    "manifest": {
                        "file_name": "train.csv",
                        "sha256": manifest_hash,
                        "rows": 20_000,
                    },
                    "feature_file": {
                        "file_name": feature_path.name,
                        "sha256": sha256_file(feature_path),
                    },
                }
            },
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path, feature_path, split_dir, features, labels

    def test_component_aliases_and_order(self) -> None:
        selection = resolve_feature_selection(self.config, "color,hog", 10)
        self.assertEqual(
            selection.names, ("rgb_histogram", "hsv_histogram", "hog")
        )
        np.testing.assert_array_equal(
            selection.indices, np.asarray([0, 1, 2, 3, 6, 7, 8, 9])
        )
        self.assertEqual(parse_components(" lbp, hog "), ("lbp", "hog"))

    def test_component_overlap_and_unknown_are_rejected(self) -> None:
        bad = dict(self.config)
        bad["components"] = {
            "a": {"start": 0, "stop": 3},
            "b": {"start": 2, "stop": 4},
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            resolve_feature_selection(bad, "a,b", 10)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_feature_selection(self.config, "sift", 10)

    def test_balanced_pilot_is_manifest_ordered(self) -> None:
        labels = np.asarray([2, 1, 2, 0, 1, 0, 2, 1, 0])
        np.testing.assert_array_equal(
            balanced_pilot_indices(labels, 2), np.asarray([3, 5, 1, 4, 0, 2])
        )

    def test_runtime_projection_has_two_x_safety_factor(self) -> None:
        self.assertEqual(projected_linearsvc_seconds(10.0, 2_500, 20_000), 160.0)

    def test_explicit_sgd_produces_decision_scores(self) -> None:
        generator = np.random.default_rng(42)
        features = generator.gamma(0.5, 1.0, size=(60, 8)).astype(np.float32)
        labels = np.repeat(np.arange(3), 20)
        estimator, elapsed, gate, backend = train_linear_svm(
            features, labels, seed=42, backend="sgd"
        )
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(backend, "sgd")
        self.assertEqual(gate["status"], "BYPASSED_EXPLICIT_SGD")
        self.assertEqual(estimator.decision_function(features[:2]).shape, (2, 3))

    def test_random_forest_contract(self) -> None:
        forest = make_random_forest(42)
        classifier = forest.named_steps["classifier"]
        self.assertEqual(classifier.n_estimators, 300)
        self.assertEqual(classifier.max_features, "sqrt")
        self.assertEqual(classifier.min_samples_leaf, 2)
        self.assertIsNone(classifier.max_leaf_nodes)
        self.assertEqual(classifier.random_state, 42)
        self.assertEqual(forest.named_steps["normalise"].alpha, 0.3)

    def test_canonical_names_require_exact_classifier_knobs(self) -> None:
        svm = argparse.Namespace(
            model="svm", svm_backend="linearsvc", seed=42, C=1.0,
            n_estimators=300,
        )
        self.assertIsNotNone(
            validate_canonical_training_request(
                svm,
                experiment_name="linear_svm",
                feature_set="combined",
                alpha=0.3,
                blockwise=True,
            )
        )
        svm.C = 2.0
        with self.assertRaisesRegex(ValueError, "C=1"):
            validate_canonical_training_request(
                svm,
                experiment_name="linear_svm",
                feature_set="combined",
                alpha=0.3,
                blockwise=True,
            )

        forest = argparse.Namespace(
            model="random_forest", seed=42, C=1.0, n_estimators=301
        )
        with self.assertRaisesRegex(ValueError, "300-tree"):
            validate_canonical_training_request(
                forest,
                experiment_name="random_forest",
                feature_set="handcrafted",
                alpha=0.3,
                blockwise=True,
            )

    def test_canonical_parameter_maps_cover_all_classifier_defaults(self) -> None:
        for estimator, expected in (
            (make_linearsvc_pipeline(42), CANONICAL_LINEARSVC_PARAMETERS),
            (make_random_forest(42), CANONICAL_RANDOM_FOREST_PARAMETERS),
        ):
            parameters = estimator_parameters(estimator)
            validate_canonical_estimator_parameters(
                parameters, expected, context="test estimator"
            )
            parameters.pop(next(iter(expected)))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_canonical_estimator_parameters(
                    parameters, expected, context="test estimator"
                )

    def test_loader_accepts_nested_feature_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "traditional_train_features.npz"
            features = np.arange(30, dtype=np.float32).reshape(3, 10)
            np.savez_compressed(
                feature_path,
                features=features,
                new_labels=np.asarray([0, 1, 2]),
                manifest_sha256=np.asarray("abc"),
                manifest_file_name=np.asarray("train.csv"),
            )
            config = {
                **self.config,
                "split_artifacts": {
                    "train": {
                        "feature_file": {
                            "file_name": feature_path.name,
                            "sha256": sha256_file(feature_path),
                        }
                    }
                },
            }
            config_path = root / "traditional_feature_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            selected, labels, selection, _, resolved, provenance = (
                load_training_features(
                    config_path, "lbp", require_production_contract=False
                )
            )
            np.testing.assert_array_equal(selected, features[:, 4:6])
            np.testing.assert_array_equal(labels, [0, 1, 2])
            self.assertEqual(selection.names, ("lbp",))
            self.assertEqual(resolved, feature_path.resolve())
            self.assertEqual(provenance["manifest_sha256"], "abc")

    def test_saved_bundle_matches_evaluator_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, feature_path, split_dir, _features, _labels = (
                self._write_production_fixture(root)
            )
            args = argparse.Namespace(
                feature_config=config_path,
                split_dir=split_dir,
                components="full",
                model="svm",
                svm_backend="sgd",
                seed=42,
                experiment_name="fixture_svm",
                output_dir=root / "models",
                force=False,
            )
            model_path, metadata_path, metadata = train_and_save(args)
            bundle = joblib.load(model_path)
            self.assertEqual(bundle["schema_version"], 2)
            self.assertEqual(bundle["experiment_name"], "fixture_svm")
            self.assertEqual(bundle["model_type"], "linear_svm")
            self.assertEqual(bundle["classes"], list(range(500)))
            self.assertEqual(bundle["feature_components"], ["fixture"])
            self.assertEqual(len(bundle["feature_indices"]), 4)
            self.assertEqual(bundle["feature_set"], "handcrafted")
            self.assertTrue(bundle["feature_bindings"])
            self.assertIn("classifier__loss", metadata["estimator_parameters"])
            self.assertIn("estimator_diagnostics", metadata)
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(metadata["model_sha256"], sha256_file(model_path))

    def test_production_loader_rejects_truncated_training_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, feature_path, split_dir, features, labels = (
                self._write_production_fixture(root)
            )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            np.savez_compressed(
                feature_path,
                features=features[:500],
                new_labels=labels[::40],
                manifest_sha256=np.asarray(
                    config["split_artifacts"]["train"]["manifest"]["sha256"]
                ),
                manifest_file_name=np.asarray("train.csv"),
            )
            config["split_artifacts"]["train"]["feature_file"]["sha256"] = (
                sha256_file(feature_path)
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "20,000|20000"):
                load_training_features(config_path, "full", split_dir=split_dir)

    def test_production_loader_rejects_changed_current_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, _feature_path, split_dir, _features, _labels = (
                self._write_production_fixture(root)
            )
            (split_dir / "train.csv").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Current frozen training manifest hash"):
                load_training_features(config_path, "full", split_dir=split_dir)

    def test_trainer_rejects_names_the_evaluator_cannot_load(self) -> None:
        for invalid_name in ("Baseline_A", "_baseline", "-baseline"):
            with self.subTest(invalid_name=invalid_name):
                args = argparse.Namespace(
                    feature_config=Path("unused.json"),
                    split_dir=Path("unused"),
                    components="full",
                    model="svm",
                    svm_backend="sgd",
                    seed=42,
                    experiment_name=invalid_name,
                    output_dir=Path("unused"),
                    force=False,
                )
                with self.assertRaisesRegex(ValueError, "lowercase"):
                    train_and_save(args)


if __name__ == "__main__":
    unittest.main()
