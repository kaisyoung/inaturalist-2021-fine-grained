"""Independent schema-v2 and selection-provenance tests for the Stage 2 collector."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis.collect_stage2_results import (
    ABLATION_COMPONENTS,
    ABLATION_DIMENSIONS,
    _validate_main_experiments,
    _validate_result_provenance,
    _validate_test_cache_sidecar,
    collect,
)
from src.data.common import sha256_file
from src.traditional.freeze_stage2 import (
    ALLOWED_TEST_MODELS,
    CANONICAL_MODEL_SPECS,
    SELECTION_SCHEMA_VERSION,
    STAGE1_FILES,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _metric_block(value: float) -> dict:
    return {
        "num_samples": 5000,
        "num_classes": 500,
        "top1_accuracy": value,
        "top5_accuracy": value + 0.1,
        "overall_accuracy": value,
        "macro_precision": value,
        "macro_recall": value,
        "macro_f1": value,
        "inference_time_seconds": 0.5,
        "seconds_per_image": 0.0001,
    }


def _sidecar_fixture(root: Path, role: str) -> dict:
    root.mkdir(parents=True)
    selected = root / "stage2_selection_manifest.json"
    selected.write_text("{}", encoding="utf-8")
    method = root / f"{role}_feature_config.json"
    method.write_bytes(b"method")
    cache_path = root / f"{role}_test_features.npz"
    cache_path.write_bytes(b"selected cache")
    test_manifest = root / "test.csv"
    test_manifest.write_text("image_id\n1\n", encoding="utf-8")
    train_manifest = root / "train.csv"
    train_manifest.write_text("image_id\n1\n", encoding="utf-8")
    selection_hash = sha256_file(selected)
    binding = {
        "role": role,
        "feature_config_path": str(method),
        "feature_config_sha256": sha256_file(method),
    }
    cache = {
        "role": role,
        "feature_file_path": str(cache_path),
        "feature_file_sha256": sha256_file(cache_path),
    }
    sidecar = {
        "selection_manifest": {
            "path": str(selected),
            "sha256": selection_hash,
        },
        "method_config": {
            "path": str(method),
            "sha256": sha256_file(method),
        },
        "split_artifacts": {
            "test": {
                "manifest": {
                    "file_name": "test.csv",
                    "sha256": sha256_file(test_manifest),
                    "rows": 1,
                },
                "feature_file": {
                    "file_name": cache_path.name,
                    "sha256": sha256_file(cache_path),
                    "bytes": cache_path.stat().st_size,
                },
            }
        },
    }
    if role == "bovw":
        vocabulary = root / "bovw_vocabulary.joblib"
        vocabulary.write_bytes(b"selected vocabulary")
        vocabulary_hash = sha256_file(vocabulary)
        train_hash = sha256_file(train_manifest)
        binding.update(
            {
                "vocabulary_path": str(vocabulary),
                "vocabulary_sha256": vocabulary_hash,
                "vocabulary_training_manifest_sha256": train_hash,
            }
        )
        cache.update(
            {
                "vocabulary_path": str(vocabulary),
                "vocabulary_sha256": vocabulary_hash,
            }
        )
        sidecar["vocabulary"] = {
            "output_file": vocabulary.name,
            "output_sha256": vocabulary_hash,
            "training_manifest_sha256": train_hash,
        }
    return {
        "cache": cache,
        "sidecar": sidecar,
        "sidecar_path": (root / f"{role}_test_feature_config.json").resolve(),
        "binding": binding,
        "selected_path": selected.resolve(),
        "selection_hash": selection_hash,
        "test_manifest": test_manifest,
        "train_manifest": train_manifest,
    }


class MainExperimentTests(unittest.TestCase):
    def _completed(self) -> dict[str, dict]:
        completed = {}
        for name, spec in CANONICAL_MODEL_SPECS.items():
            document = {
                **spec,
                "feature_components": list(spec["feature_components"]),
            }
            if name != "random_forest":
                document["svm_backend"] = "linearsvc"
            completed[name] = document
        return completed

    def test_exact_three_scientific_roles_are_accepted(self) -> None:
        _validate_main_experiments(self._completed())

    def test_missing_or_wrong_dimension_is_rejected(self) -> None:
        completed = self._completed()
        completed.pop("sift_bovw_svm")
        with self.assertRaisesRegex(ValueError, "exactly"):
            _validate_main_experiments(completed)
        completed = self._completed()
        completed["linear_svm"]["feature_dim"] = 2424
        with self.assertRaisesRegex(ValueError, "feature_dim"):
            _validate_main_experiments(completed)

    def test_unknown_completed_model_is_rejected(self) -> None:
        completed = self._completed()
        completed["mystery"] = completed["linear_svm"]
        with self.assertRaisesRegex(ValueError, "exactly"):
            _validate_main_experiments(completed)

    def test_rewritten_ablation_names_and_dimensions_are_exact(self) -> None:
        self.assertEqual(
            ABLATION_COMPONENTS,
            {
                "ablation_only_color": ("color",),
                "ablation_only_texture": ("texture",),
                "ablation_only_hog": ("hog",),
            },
        )
        self.assertEqual(
            ABLATION_DIMENSIONS,
            {
                "ablation_only_color": 520,
                "ablation_only_texture": 140,
                "ablation_only_hog": 1764,
            },
        )


class SchemaDispatchTests(unittest.TestCase):
    def _args_and_documents(
        self, root: Path, *, schema_version: int, include_unknown: bool = False
    ) -> argparse.Namespace:
        metrics_dir = root / "metrics"
        metrics_dir.mkdir()
        for name, spec in CANONICAL_MODEL_SPECS.items():
            document = {
                "schema_version": schema_version,
                "status": "complete",
                "model_name": name,
                "model_type": spec["model_type"],
                "classifier": spec["classifier"],
                "feature_set": spec["feature_set"],
                "feature_type": spec["feature_set"],
                "feature_components": list(spec["feature_components"]),
                "feature_dim": spec["feature_dim"],
                "image_size": [128, 128],
                "training_time_seconds": 1.0,
                "validation": _metric_block(0.1),
                "test": _metric_block(0.11),
            }
            if name != "random_forest":
                document["svm_backend"] = "linearsvc"
            _write(metrics_dir / f"traditional_{name}_metrics.json", document)
        if include_unknown:
            unknown = {
                "schema_version": schema_version,
                "status": "complete",
                "model_name": "mystery",
                "model_type": "linear_svm",
                "classifier": "LinearSVC",
                "feature_set": "handcrafted",
                "feature_components": ["color"],
                "feature_dim": 520,
                "image_size": [128, 128],
                "training_time_seconds": 1.0,
                "validation": _metric_block(0.1),
                "test": _metric_block(0.11),
            }
            _write(metrics_dir / "traditional_mystery_metrics.json", unknown)
        return argparse.Namespace(
            metrics_dir=metrics_dir,
            split_dir=root / "splits",
            selection_manifest=root / "selection.json",
            output=root / "result.csv",
            ablation_output=root / "ablation.csv",
            ablation_figure=root / "ablation.png",
            figure=root / "comparison.png",
            force=True,
        )

    def _collect_with_external_checks_mocked(self, args: argparse.Namespace):
        def fake_plot(_frame, path):
            path.write_bytes(b"plot")

        with patch("src.analysis.collect_stage2_results._validate_result_provenance"):
            with patch(
                "src.analysis.collect_stage2_results._collect_ablation_frame",
                return_value=None,
            ):
                with patch(
                    "src.analysis.collect_stage2_results.plot_model_comparison",
                    side_effect=fake_plot,
                ):
                    return collect(args)

    def test_schema_v2_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = self._collect_with_external_checks_mocked(
                self._args_and_documents(Path(directory), schema_version=2)
            )
            self.assertEqual(len(frame), 3)

    def test_schema_v1_and_v4_are_rejected(self) -> None:
        for schema in (1, 4):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as directory:
                args = self._args_and_documents(Path(directory), schema_version=schema)
                with self.assertRaisesRegex(ValueError, "Unsupported metrics schema"):
                    self._collect_with_external_checks_mocked(args)

    def test_unknown_schema_v2_experiment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._args_and_documents(
                Path(directory), schema_version=2, include_unknown=True
            )
            with self.assertRaisesRegex(ValueError, "Unknown completed"):
                self._collect_with_external_checks_mocked(args)


class ProvenanceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict, Path, argparse.Namespace, Path]:
        split_dir = root / "splits"
        split_dir.mkdir()
        stage1 = {}
        for name in STAGE1_FILES:
            path = split_dir / name
            path.write_bytes(f"stage1:{name}".encode())
            stage1[name] = {"sha256": sha256_file(path)}

        model = root / "random_forest.joblib"
        method_config = root / "traditional_feature_config.json"
        train_cache = root / "traditional_train_features.npz"
        test_cache = root / "traditional_test_features.npz"
        grid = root / "traditional_validation_grid.csv"
        for path in (model, method_config, train_cache, test_cache, grid):
            path.write_bytes(path.name.encode())

        binding = {
            "role": "handcrafted",
            "feature_config_path": str(method_config),
            "feature_config_sha256": sha256_file(method_config),
            "train_feature_path": str(train_cache),
            "train_feature_sha256": sha256_file(train_cache),
            "train_manifest_file_name": "train.csv",
            "train_manifest_sha256": sha256_file(split_dir / "train.csv"),
        }
        normalisation = {
            "alpha": 0.3,
            "blockwise": True,
            "blocks": [[0, 520], [520, 660], [660, 2424]],
        }
        validation_metrics = _metric_block(0.1)
        validation = {
            "schema_version": 2,
            "status": "complete",
            "split": "val",
            "experiment_name": "random_forest",
            "model_name": "random_forest",
            "metrics": validation_metrics,
            "provenance": {
                "model_sha256": sha256_file(model),
                "feature_config_sha256": sha256_file(method_config),
                "feature_bindings": [binding],
                "split_manifest": "val.csv",
                "split_manifest_sha256": sha256_file(split_dir / "val.csv"),
            },
        }
        validation_path = root / "traditional_random_forest_val_metrics.json"
        _write(validation_path, validation)

        models = {}
        feature_configs = {}
        selected_validation = {}
        for name, spec in CANONICAL_MODEL_SPECS.items():
            if name == "random_forest":
                entry = {
                    **spec,
                    "path": str(model),
                    "sha256": sha256_file(model),
                    "feature_components": list(spec["feature_components"]),
                    "feature_config_path": str(method_config),
                    "feature_config_sha256": sha256_file(method_config),
                    "feature_bindings": [binding],
                    "normalisation": normalisation,
                }
                feature_configs[name] = {
                    "path": str(method_config),
                    "sha256": sha256_file(method_config),
                }
                selected_validation[name] = {
                    "path": str(validation_path),
                    "sha256": sha256_file(validation_path),
                }
            else:
                entry = {
                    **spec,
                    "feature_components": list(spec["feature_components"]),
                }
                feature_configs[name] = {}
                selected_validation[name] = {}
            models[name] = entry
        selection = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "status": "frozen",
            "selected_models": sorted(ALLOWED_TEST_MODELS),
            "allowed_test_models": sorted(ALLOWED_TEST_MODELS),
            "stage1_artifacts": stage1,
            "validation_grid": {
                "path": str(grid),
                "sha256": sha256_file(grid),
                "rows": 19,
            },
            "models": models,
            "feature_configs": feature_configs,
            "validation_metrics": selected_validation,
        }
        selection_path = root / "stage2_selection_manifest.json"
        _write(selection_path, selection)
        selection_hash = sha256_file(selection_path)

        test_sidecar = root / "traditional_test_feature_config.json"
        _write(
            test_sidecar,
            {
                "selection_manifest": {
                    "path": str(selection_path),
                    "sha256": selection_hash,
                },
                "method_config": {
                    "path": str(method_config),
                    "sha256": sha256_file(method_config),
                },
                "split_artifacts": {
                    "test": {
                        "manifest": {
                            "file_name": "test.csv",
                            "sha256": sha256_file(split_dir / "test.csv"),
                            "rows": 0,
                        },
                        "feature_file": {
                            "file_name": test_cache.name,
                            "sha256": sha256_file(test_cache),
                            "bytes": test_cache.stat().st_size,
                        },
                    }
                },
            },
        )
        provenance = {
            "selection_manifest_path": str(selection_path),
            "selection_manifest_sha256": selection_hash,
            "model_path": str(model),
            "model_sha256": sha256_file(model),
            "feature_config_path": str(method_config),
            "feature_config_sha256": sha256_file(method_config),
            "feature_bindings": [binding],
            "feature_caches": [
                {
                    "role": "handcrafted",
                    "config_path": str(test_sidecar),
                    "config_sha256": sha256_file(test_sidecar),
                    "feature_file_path": str(test_cache),
                    "feature_file_sha256": sha256_file(test_cache),
                }
            ],
            "split_manifest": "test.csv",
            "split_manifest_sha256": sha256_file(split_dir / "test.csv"),
        }
        document = {
            "schema_version": 2,
            "status": "complete",
            "model_name": "random_forest",
            "model_type": "random_forest",
            "classifier": "RandomForestClassifier",
            "feature_set": "handcrafted",
            "feature_components": ["color", "texture", "hog"],
            "feature_dim": 2424,
            "normalisation": normalisation,
            "validation": validation_metrics,
            "validation_metrics_path": str(validation_path),
            "validation_metrics_sha256": sha256_file(validation_path),
            "provenance": provenance,
        }
        args = argparse.Namespace(
            selection_manifest=selection_path,
            split_dir=split_dir,
        )
        metrics_path = root / "traditional_random_forest_metrics.json"
        return document, metrics_path, args, train_cache

    def test_schema_v2_selection_chain_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document, metrics_path, args, _ = self._fixture(Path(directory))
            _validate_result_provenance(document, metrics_path, args)

    def test_tampered_train_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document, metrics_path, args, train_cache = self._fixture(Path(directory))
            train_cache.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "provenance check failed"):
                _validate_result_provenance(document, metrics_path, args)

    def test_metrics_cannot_reference_a_different_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document, metrics_path, args, _ = self._fixture(Path(directory))
            other = Path(directory) / "other_selection.json"
            other.write_text("{}", encoding="utf-8")
            document["provenance"]["selection_manifest_path"] = str(other)
            document["provenance"]["selection_manifest_sha256"] = sha256_file(other)
            with self.assertRaisesRegex(ValueError, "non-authoritative"):
                _validate_result_provenance(document, metrics_path, args)

    def test_handcrafted_unrelated_cache_with_matching_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            document, metrics_path, args, _ = self._fixture(Path(directory))
            rogue = Path(directory) / "rogue_test_features.npz"
            rogue.write_bytes(b"self-consistent but unrelated")
            cache = document["provenance"]["feature_caches"][0]
            cache["feature_file_path"] = str(rogue)
            cache["feature_file_sha256"] = sha256_file(rogue)
            with self.assertRaisesRegex(ValueError, "differs from split_artifacts"):
                _validate_result_provenance(document, metrics_path, args)


class TestCacheSidecarTests(unittest.TestCase):
    def test_combined_roles_reject_unrelated_self_consistent_cache_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for role in ("handcrafted", "bovw"):
                with self.subTest(role=role):
                    fixture = _sidecar_fixture(Path(directory) / role, role)
                    rogue = Path(directory) / role / "rogue.npz"
                    rogue.write_bytes(b"unrelated but internally hashed")
                    fixture["cache"]["feature_file_path"] = str(rogue)
                    fixture["cache"]["feature_file_sha256"] = sha256_file(rogue)
                    with self.assertRaisesRegex(
                        ValueError, "differs from split_artifacts"
                    ):
                        _validate_test_cache_sidecar(**fixture)

    def test_sidecar_manifest_rows_and_hash_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _sidecar_fixture(Path(directory) / "handcrafted", "handcrafted")
            manifest = fixture["sidecar"]["split_artifacts"]["test"]["manifest"]
            manifest["rows"] = 2
            with self.assertRaisesRegex(ValueError, "manifest metadata"):
                _validate_test_cache_sidecar(**fixture)

    def test_bovw_vocabulary_must_equal_selected_training_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _sidecar_fixture(Path(directory) / "bovw", "bovw")
            rogue = Path(directory) / "bovw" / "rogue_vocabulary.joblib"
            rogue.write_bytes(b"different vocabulary")
            rogue_hash = sha256_file(rogue)
            fixture["cache"].update(
                {"vocabulary_path": str(rogue), "vocabulary_sha256": rogue_hash}
            )
            fixture["sidecar"]["vocabulary"].update(
                {"output_file": rogue.name, "output_sha256": rogue_hash}
            )
            with self.assertRaisesRegex(ValueError, "non-selected vocabulary"):
                _validate_test_cache_sidecar(**fixture)


if __name__ == "__main__":
    unittest.main()
