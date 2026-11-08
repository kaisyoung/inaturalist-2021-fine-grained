"""Safety-focused tests for Stage 2 selection and evaluation."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.run_traditional_experiments import experiment_grid
from src.data.common import sha256_file
from src.traditional.evaluate_traditional import (
    _prediction_frame,
    _prevalidate_selection,
    evaluate,
)
from src.traditional.freeze_stage2 import (
    ALLOWED_TEST_MODELS,
    CANONICAL_MODEL_SPECS,
    EXPECTED_GRID_SPECS,
    SELECTION_SCHEMA_VERSION,
    STAGE1_FILES,
    _validate_validation_grid,
    freeze,
)
from src.traditional.train_traditional import (
    estimator_parameters,
    make_linearsvc_pipeline,
    make_random_forest,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SelectionGateTests(unittest.TestCase):
    def _selection_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        split_dir = root / "splits"
        split_dir.mkdir()
        stage1 = {}
        for name in STAGE1_FILES:
            path = split_dir / name
            path.write_bytes(f"stage1:{name}".encode())
            stage1[name] = {"sha256": sha256_file(path)}
        grid = root / "traditional_validation_grid.csv"
        grid.write_text("experiment\n19 reviewed rows\n", encoding="utf-8")
        models = {}
        feature_configs = {}
        validation_metrics = {}
        linear_model: Path | None = None
        for name, spec in CANONICAL_MODEL_SPECS.items():
            model = root / f"{name}.joblib"
            model.write_bytes(f"model:{name}".encode())
            if name == "linear_svm":
                linear_model = model
            config = root / f"{name}.feature.json"
            config.write_text("{}", encoding="utf-8")
            cache = root / f"{name}.train.npz"
            cache.write_bytes(f"cache:{name}".encode())
            binding = {
                "role": spec["feature_set"],
                "feature_config_path": str(config),
                "feature_config_sha256": sha256_file(config),
                "train_feature_path": str(cache),
                "train_feature_sha256": sha256_file(cache),
                "train_manifest_file_name": "train.csv",
                "train_manifest_sha256": sha256_file(split_dir / "train.csv"),
            }
            if name != "random_forest":
                vocabulary = root / f"{name}.vocabulary.joblib"
                vocabulary.write_bytes(f"vocabulary:{name}".encode())
                binding.update(
                    {
                        "vocabulary_path": str(vocabulary),
                        "vocabulary_sha256": sha256_file(vocabulary),
                    }
                )
            normalisation = {"alpha": 0.3, "blockwise": True, "blocks": []}
            models[name] = {
                **spec,
                "path": str(model),
                "sha256": sha256_file(model),
                "feature_components": list(spec["feature_components"]),
                "feature_config_path": str(config),
                "feature_config_sha256": sha256_file(config),
                "feature_bindings": [binding],
                "normalisation": normalisation,
            }
            feature_configs[name] = {
                "path": str(config),
                "sha256": sha256_file(config),
            }
            validation = root / f"traditional_{name}_val_metrics.json"
            validation.write_text("{}", encoding="utf-8")
            validation_metrics[name] = {
                "path": str(validation),
                "sha256": sha256_file(validation),
            }
        assert linear_model is not None
        selection = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "status": "frozen",
            "selected_models": sorted(ALLOWED_TEST_MODELS),
            "allowed_test_models": sorted(ALLOWED_TEST_MODELS),
            "stage1_artifacts": stage1,
            "models": models,
            "feature_configs": feature_configs,
            "validation_grid": {
                "path": str(grid),
                "sha256": sha256_file(grid),
                "rows": 19,
            },
            "validation_metrics": validation_metrics,
        }
        selection_path = root / "stage2_selection_manifest.json"
        _write_json(selection_path, selection)
        return selection_path, linear_model, split_dir

    def test_selection_is_checked_before_model_or_test_cache_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.joblib"
            model.write_bytes(b"model")
            args = argparse.Namespace(
                split="test",
                include_test=True,
                model=model,
                selection_manifest=root / "missing-selection.json",
                split_dir=root / "splits",
            )
            with patch("src.traditional.evaluate_traditional._load_bundle") as load_model:
                with patch("src.traditional.evaluate_traditional._load_feature_set") as load_cache:
                    with self.assertRaisesRegex(ValueError, "selection|Selection|JSON"):
                        evaluate(args)
            load_model.assert_not_called()
            load_cache.assert_not_called()

    def test_selection_binds_model_stage1_and_validation_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection, model, split_dir = self._selection_fixture(Path(directory))
            document, digest, name = _prevalidate_selection(
                selection, model_path=model, split_dir=split_dir
            )
            self.assertEqual(name, "linear_svm")
            self.assertEqual(digest, sha256_file(selection))
            self.assertEqual(document["status"], "frozen")
            model.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Selected model"):
                _prevalidate_selection(selection, model_path=model, split_dir=split_dir)

    def test_selection_rejects_each_tampered_evidence_layer(self) -> None:
        targets = (
            "stage1",
            "validation_grid",
            "validation_metrics",
            "feature_config",
            "train_cache",
            "vocabulary",
        )
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                selection_path, model, split_dir = self._selection_fixture(Path(directory))
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                entry = selection["models"]["linear_svm"]
                binding = entry["feature_bindings"][0]
                paths = {
                    "stage1": split_dir / "train.csv",
                    "validation_grid": Path(selection["validation_grid"]["path"]),
                    "validation_metrics": Path(
                        selection["validation_metrics"]["linear_svm"]["path"]
                    ),
                    "feature_config": Path(binding["feature_config_path"]),
                    "train_cache": Path(binding["train_feature_path"]),
                    "vocabulary": Path(binding["vocabulary_path"]),
                }
                paths[target].write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    _prevalidate_selection(
                        selection_path, model_path=model, split_dir=split_dir
                    )

    def test_selection_fails_closed_on_missing_contract_fields(self) -> None:
        mutations = {
            "allowed_set": lambda document: document.pop("allowed_test_models"),
            "feature_configs": lambda document: document["feature_configs"].pop(
                "random_forest"
            ),
            "canonical_dimension": lambda document: document["models"]["linear_svm"].update(
                {"feature_dim": 2424}
            ),
            "component_order": lambda document: document["models"]["linear_svm"].update(
                {"feature_components": ["bovw", "hog", "texture", "color"]}
            ),
            "normalisation": lambda document: document["models"]["linear_svm"][
                "normalisation"
            ].update({"alpha": 0.5}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                selection_path, model, split_dir = self._selection_fixture(Path(directory))
                document = json.loads(selection_path.read_text(encoding="utf-8"))
                mutate(document)
                _write_json(selection_path, document)
                with self.assertRaises(ValueError):
                    _prevalidate_selection(
                        selection_path, model_path=model, split_dir=split_dir
                    )


class FreezeTests(unittest.TestCase):
    def test_dual_manifests_are_created_once_and_are_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_dir = root / "splits"
            split_dir.mkdir()
            for name in STAGE1_FILES:
                (split_dir / name).write_bytes(name.encode())
            metrics_dir = root / "metrics"
            predictions_dir = root / "predictions"
            tables_dir = root / "tables"
            figures_dir = root / "figures"
            feature_dir = root / "features"
            for path in (metrics_dir, predictions_dir, tables_dir, figures_dir, feature_dir):
                path.mkdir()
            bundles = {}
            model_paths = []
            for name in sorted(ALLOWED_TEST_MODELS):
                spec = CANONICAL_MODEL_SPECS[name]
                model_path = root / f"{name}.joblib"
                model_path.write_bytes(name.encode())
                model_paths.append(model_path)
                config = root / f"{name}.config.json"
                config.write_text("{}", encoding="utf-8")
                binding = {"role": spec["feature_set"]}
                estimator = (
                    make_random_forest(42)
                    if name == "random_forest"
                    else make_linearsvc_pipeline(42)
                )
                parameters = estimator_parameters(estimator)
                metadata = {
                    **spec,
                    "feature_components": list(spec["feature_components"]),
                    "normalisation": {"alpha": 0.3, "blockwise": True},
                    "estimator_diagnostics": {"converged": True},
                    "estimator_parameters": parameters,
                    "svm_backend": "linearsvc" if name != "random_forest" else None,
                }
                bundles[name] = {
                    "experiment_name": name,
                    "model_type": spec["model_type"],
                    "feature_config_path": str(config),
                    "feature_config_sha256": sha256_file(config),
                    "feature_bindings": [binding],
                    "estimator": estimator,
                    "metadata": metadata,
                }
                _write_json(
                    model_path.with_suffix(".metadata.json"),
                    {
                        "model_sha256": sha256_file(model_path),
                        "feature_bindings": [binding],
                        "estimator_parameters": parameters,
                    },
                )
                _write_json(metrics_dir / f"traditional_{name}_val_metrics.json", {})

            selection = root / "stage2_selection_manifest.json"
            compatibility = root / "stage2_freeze.json"
            args = argparse.Namespace(
                model=model_paths,
                split_dir=split_dir,
                feature_dir=feature_dir,
                metrics_dir=metrics_dir,
                predictions_dir=predictions_dir,
                tables_dir=tables_dir,
                figures_dir=figures_dir,
                validation_grid=root / "grid.csv",
                output=selection,
                compatibility_output=compatibility,
                dangerously_replace_existing_selection=False,
                expected_existing_selection_sha256=None,
            )

            def fake_load(path):
                return bundles[Path(path).stem]

            with patch("src.traditional.freeze_stage2.joblib.load", side_effect=fake_load):
                with patch("src.traditional.freeze_stage2._validate_feature_binding"):
                    with patch("src.traditional.freeze_stage2._validate_validation_metrics"):
                        with patch(
                            "src.traditional.freeze_stage2._validate_validation_grid",
                            return_value={"rows": 19, "sha256": "a" * 64},
                        ):
                            bundles["linear_svm"]["metadata"][
                                "estimator_parameters"
                            ]["classifier__C"] = 2.0
                            with self.assertRaisesRegex(ValueError, "parameter"):
                                freeze(args)
                            bundles["linear_svm"]["metadata"][
                                "estimator_parameters"
                            ]["classifier__C"] = 1.0

                            forest = bundles["random_forest"]["estimator"]
                            forest.set_params(classifier__n_estimators=301)
                            with self.assertRaisesRegex(ValueError, "parameter"):
                                freeze(args)
                            forest.set_params(classifier__n_estimators=300)

                            document = freeze(args)
                            self.assertTrue(selection.is_file())
                            self.assertTrue(compatibility.is_file())
                            self.assertEqual(document["selected_models"], sorted(ALLOWED_TEST_MODELS))
                            compatibility_document = json.loads(
                                compatibility.read_text(encoding="utf-8")
                            )
                            self.assertEqual(compatibility_document["schema_version"], 1)
                            self.assertEqual(
                                compatibility_document["selection_manifest"]["sha256"],
                                sha256_file(selection),
                            )
                            with self.assertRaisesRegex(FileExistsError, "immutable"):
                                freeze(args)


class PredictionAndGridTests(unittest.TestCase):
    def test_predictions_persist_enough_scores_to_recompute_top5(self) -> None:
        scores = np.tile(np.arange(500, dtype=np.float64), (2, 1))
        scores[0, 3] = 1000
        scores[1, 4] = 1000
        order = np.argsort(-scores, axis=1, kind="stable")
        predicted = order[:, 0]
        arrays = {
            "new_labels": np.asarray([3, 1]),
            "splits": np.asarray(["val", "val"]),
            "image_ids": np.asarray([10, 11]),
            "image_paths": np.asarray(["a", "b"]),
            "local_image_paths": np.asarray(["val/a", "val/b"]),
            "scientific_names": np.asarray(["A", "B"]),
        }
        frame = _prediction_frame(
            arrays, scores, predicted, order, score_type="decision_function"
        )
        for index, row in frame.iterrows():
            labels = json.loads(row["top5_new_labels"])
            values = json.loads(row["top5_scores"])
            self.assertEqual(len(labels), 5)
            self.assertEqual(len(values), 5)
            expected = int(arrays["new_labels"][index]) in labels
            self.assertEqual(bool(row["is_top5_correct"]), expected)

    def test_experiment_grid_is_exactly_19_unique_and_canonical_alpha_point_three(self) -> None:
        grid = experiment_grid()
        self.assertEqual(len(grid), 19)
        self.assertEqual(len({item.name for item in grid}), 19)
        self.assertEqual(len({item.fingerprint for item in grid}), 19)
        canonical = {item.name: item for item in grid if item.name in ALLOWED_TEST_MODELS}
        self.assertEqual(set(canonical), set(ALLOWED_TEST_MODELS))
        self.assertTrue(all(item.alpha == 0.3 for item in canonical.values()))

    def test_validation_grid_rejects_a_tampered_scientific_cell(self) -> None:
        rows = []
        for name, spec in EXPECTED_GRID_SPECS.items():
            classifier, feature_set, components, alpha, blockwise, dimension = spec
            rows.append(
                {
                    "experiment": name,
                    "classifier": classifier,
                    "feature_set": feature_set,
                    "components": components,
                    "alpha": alpha,
                    "blockwise": blockwise,
                    "dim": dimension,
                    "top1": 0.1,
                    "top5": 0.2,
                    "macro_f1": 0.08,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grid.csv"
            import pandas as pd

            frame = pd.DataFrame(rows)
            frame.to_csv(path, index=False)
            self.assertEqual(_validate_validation_grid(path)["rows"], 19)
            frame.loc[frame["experiment"] == "linear_svm", "dim"] = 2424
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "reviewed spec"):
                _validate_validation_grid(path)

    def test_validation_grid_rejects_non_boolean_and_fractional_dimension(self) -> None:
        rows = []
        for name, spec in EXPECTED_GRID_SPECS.items():
            classifier, feature_set, components, alpha, blockwise, dimension = spec
            rows.append(
                {
                    "experiment": name,
                    "classifier": classifier,
                    "feature_set": feature_set,
                    "components": components,
                    "alpha": alpha,
                    "blockwise": blockwise,
                    "dim": dimension,
                    "top1": 0.1,
                    "top5": 0.2,
                    "macro_f1": 0.08,
                }
            )
        import pandas as pd

        for field, malformed, message in (
            ("blockwise", "garbage", "true or false"),
            ("dim", 3704.5, "integer"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                frame = pd.DataFrame(rows)
                frame.loc[frame["experiment"] == "linear_svm", field] = malformed
                path = Path(directory) / "grid.csv"
                frame.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    _validate_validation_grid(path)


if __name__ == "__main__":
    unittest.main()
