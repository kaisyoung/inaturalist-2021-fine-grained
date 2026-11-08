"""Synthetic leakage, ranking, and prediction contracts for Stage 3 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.common import sha256_file
from src.deep_learning.evaluate_deep import (
    MAX_VALIDATION_METRIC_TOLERANCE,
    _access_artifact_bindings,
    _compare_validation,
    _install_staged,
    _model_for_context,
    _output_paths,
    _require_canonical_test_gate,
    _require_unchanged_freeze,
    _rollback_installation,
    _run_inference,
    _stage1_hashes,
    _verify_freeze,
    evaluate_frozen_test_set,
)
from src.deep_learning.freeze_stage3 import STAGE1_FILES


class _TwoExampleDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return (
            torch.full((3, 8, 8), float(index)),
            index,
            {
                "split": "val",
                "image_id": 100 + index,
                "image_path": f"train_mini/class/{index}.jpg",
                "local_image_path": f"val/{index}.jpg",
                "original_category_id": 1000 + index,
                "scientific_name": f"Species {index}",
                "common_name": f"Common {index}",
            },
        )


class _FixedClassifier(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(len(images), 500)
        logits[0, 0] = 4
        logits[1, 1] = 4
        return logits


class DeepEvaluationTests(unittest.TestCase):
    def test_test_acknowledgement_precedes_freeze_or_manifest_access(self) -> None:
        args = argparse.Namespace(include_test=False, checkpoint=None)
        with self.assertRaisesRegex(ValueError, "--include-test"):
            evaluate_frozen_test_set(args)

    def test_validation_stage1_binding_does_not_hash_test_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            artifacts = {}
            for file_name in STAGE1_FILES:
                if file_name == "split_summary.json":
                    continue
                path = root / file_name
                path.write_text(file_name + "\n", encoding="utf-8")
                artifacts[file_name] = {"sha256": sha256_file(path)}
            (root / "split_summary.json").write_text(
                json.dumps({"schema_version": 2, "artifacts": artifacts}),
                encoding="utf-8",
            )
            import src.deep_learning.evaluate_deep as module

            original = module.sha256_file

            def guarded(path: Path) -> str:
                if Path(path).name == "test.csv":
                    raise AssertionError("validation opened test.csv")
                return original(Path(path))

            with mock.patch.object(module, "sha256_file", side_effect=guarded):
                result = _stage1_hashes(root, verify_test_manifest=False)
            self.assertEqual(result["test.csv"]["sha256"], artifacts["test.csv"]["sha256"])

    def test_prediction_schema_and_all_class_metrics(self) -> None:
        loader = DataLoader(_TwoExampleDataset(), batch_size=2, shuffle=False)
        names = {
            label: {
                "scientific_name": f"Species {label}",
                "common_name": f"Common {label}",
                "original_category_id": 1000 + label,
            }
            for label in range(500)
        }
        with mock.patch(
            "src.deep_learning.evaluate_deep._model_for_context",
            return_value=_FixedClassifier(),
        ):
            result = _run_inference(
                {"checkpoint": {}},
                loader=loader,
                device=torch.device("cpu"),
                split="val",
                names=names,
            )
        self.assertEqual(result["metrics"]["num_classes"], 500)
        self.assertEqual(result["metrics"]["top1_accuracy"], 1.0)
        self.assertEqual(result["confusion"].shape, (500, 500))
        for column in ("true_label", "predicted_label", "correct", "top1_score"):
            self.assertIn(column, result["predictions"].columns)

    def test_frozen_model_construction_preserves_pretrained_contract(self) -> None:
        dummy = nn.Linear(2, 2)
        context = {
            "checkpoint": {
                "model_name": "resnet18",
                "pretrained": True,
                "freeze_backbone": True,
                "model_state_dict": dummy.state_dict(),
            },
            "resolved_config": {
                "model_metadata": {
                    "weights_enum": "fixture",
                    "weight_file_name": "fixture.pth",
                    "weight_file_sha256": "abc",
                }
            },
        }
        with mock.patch(
            "src.deep_learning.models.create_model",
            return_value=(
                dummy,
                {
                    "weights_enum": "fixture",
                    "weight_file_name": "fixture.pth",
                    "weight_file_sha256": "abc",
                },
            ),
        ) as factory:
            _model_for_context(context, torch.device("cpu"))
        self.assertTrue(factory.call_args.kwargs["pretrained"])
        self.assertTrue(factory.call_args.kwargs["freeze_backbone"])

    def test_output_names_are_split_qualified(self) -> None:
        args = argparse.Namespace(
            metrics_dir=Path("m"),
            predictions_dir=Path("p"),
            tables_dir=Path("t"),
            figures_dir=Path("f"),
        )
        outputs = _output_paths("experiment", "test", args)
        self.assertEqual(outputs["metrics"], Path("m/experiment_test_metrics.json"))
        self.assertEqual(outputs["predictions"], Path("p/experiment_test_predictions.csv"))

    def test_validation_tolerance_is_capped_and_checkpoint_metrics_are_checked(self) -> None:
        metrics = {
            "loss": 1.0,
            "top1_accuracy": 0.2,
            "top5_accuracy": 0.5,
            "overall_accuracy": 0.2,
            "macro_precision": 0.21,
            "macro_recall": 0.2,
            "macro_f1": 0.19,
        }
        context = {
            "checkpoint": {"best_epoch": 2, "best_metrics": dict(metrics)},
            "best_metrics": {"metrics": dict(metrics)},
        }
        with self.assertRaisesRegex(ValueError, "tolerance"):
            _compare_validation(
                context, metrics, MAX_VALIDATION_METRIC_TOLERANCE * 10
            )
        context["checkpoint"]["best_metrics"]["macro_f1"] = 0.18
        with self.assertRaisesRegex(ValueError, "macro_f1"):
            _compare_validation(
                context, metrics, MAX_VALIDATION_METRIC_TOLERANCE
            )

    def test_dangerous_replacement_rolls_back_complete_staged_set(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            staged_sets = []
            finals: list[Path] = []
            staged_metrics: Path | None = None
            for model_index in range(2):
                model = root / f"model-{model_index}"
                model.mkdir()
                outputs = {
                    "predictions": root / f"model-{model_index}-predictions.csv",
                    "metrics": root / f"model-{model_index}-metrics.json",
                }
                staged = {
                    "predictions": model / "predictions.csv",
                    "metrics": model / "metrics.json",
                }
                for path in outputs.values():
                    path.write_text(f"old:{path.name}", encoding="utf-8")
                    finals.append(path)
                for path in staged.values():
                    path.write_text(f"new:{path.name}", encoding="utf-8")
                staged_sets.append((staged, outputs))
                staged_metrics = staged["metrics"]

            assert staged_metrics is not None
            real_replace = os.replace

            def fail_last_install(source: str | os.PathLike, destination: str | os.PathLike):
                if Path(source) == staged_metrics:
                    raise OSError("synthetic final-model install failure")
                return real_replace(source, destination)

            with mock.patch(
                "src.deep_learning.evaluate_deep.os.replace",
                side_effect=fail_last_install,
            ):
                with self.assertRaisesRegex(OSError, "synthetic"):
                    _install_staged(staged_sets, allow_replace=True)
            for path in finals:
                self.assertEqual(path.read_text(encoding="utf-8"), f"old:{path.name}")
            self.assertEqual(list(root.glob(".*.stage3-backup.*")), [])

    def test_deferred_commit_can_restore_prior_complete_set(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            final = root / "model-test-metrics.json"
            staged = root / "staged-metrics.json"
            final.write_text("old", encoding="utf-8")
            staged.write_text("new", encoding="utf-8")
            transaction = _install_staged(
                [({"metrics": staged}, {"metrics": final})],
                allow_replace=True,
                defer_commit=True,
            )
            self.assertEqual(final.read_text(encoding="utf-8"), "new")
            _rollback_installation(transaction)
            self.assertEqual(final.read_text(encoding="utf-8"), "old")

    def test_freeze_rehash_detects_mid_run_change(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "freeze.json"
            path.write_text('{"status":"frozen"}\n', encoding="utf-8")
            expected = sha256_file(path)
            path.write_text('{"status":"changed"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed during test"):
                _require_unchanged_freeze(path, expected)

    def test_completed_access_binding_includes_metrics_and_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            staged_metrics = root / "staged-metrics.json"
            staged_metrics.write_text("{}\n", encoding="utf-8")
            final_metrics = root / "final-metrics.json"
            prediction = root / "predictions.csv"
            prediction.write_text("image_id\n1\n", encoding="utf-8")
            prediction_entry = {
                "path": str(prediction),
                "bytes": prediction.stat().st_size,
                "sha256": sha256_file(prediction),
            }
            bindings = _access_artifact_bindings(
                {"metrics": staged_metrics},
                {"metrics": final_metrics},
                {"artifacts": {"predictions": prediction_entry}},
            )
            self.assertEqual(set(bindings), {"metrics", "predictions"})
            self.assertEqual(bindings["metrics"]["sha256"], sha256_file(staged_metrics))
            self.assertEqual(bindings["predictions"], prediction_entry)

    def test_test_gate_rejects_alternate_freeze_path(self) -> None:
        args = argparse.Namespace(freeze_manifest=Path("alternate-freeze.json"))
        with self.assertRaisesRegex(ValueError, "freeze path is fixed"):
            _require_canonical_test_gate(args)

    def test_freeze_verification_rejects_changed_bound_model_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            split_dir = root / "splits"
            split_dir.mkdir()
            stage1 = {}
            for file_name in STAGE1_FILES:
                path = split_dir / file_name
                path.write_text(file_name + "\n", encoding="utf-8")
                stage1[file_name] = {"sha256": sha256_file(path)}
            stage2_path = root / "stage2.json"
            stage2_path.write_text("stage2\n", encoding="utf-8")
            bound = root / "bound.bin"
            bound.write_bytes(b"bound")
            entry = {
                "path": str(bound),
                "bytes": bound.stat().st_size,
                "sha256": sha256_file(bound),
            }
            scratch = "resnet18_scratch_basic"
            basic = "resnet18_pretrained_basic"
            strong = "resnet18_pretrained_strong_aug"
            candidate_metrics = {
                "macro_f1": 0.2,
                "top1_accuracy": 0.2,
                "top5_accuracy": 0.5,
                "loss": 1.0,
            }
            candidates = {
                scratch: {
                    "experiment_name": scratch,
                    "pretrained": False,
                    "best_epoch": 1,
                    "validation_metrics": dict(candidate_metrics),
                },
                basic: {
                    "experiment_name": basic,
                    "pretrained": True,
                    "best_epoch": 1,
                    "validation_metrics": dict(candidate_metrics),
                },
                strong: {
                    "experiment_name": strong,
                    "pretrained": True,
                    "best_epoch": 1,
                    "validation_metrics": dict(candidate_metrics),
                },
            }
            selected = [scratch, basic]
            freeze = {
                "schema_version": 1,
                "status": "frozen",
                "selected_experiments": selected,
                "analysis_model": basic,
                "completed_candidates": candidates,
                "stage1_artifacts": stage1,
                "stage2_artifacts": {
                    "stage2": {
                        "path": str(stage2_path),
                        "bytes": stage2_path.stat().st_size,
                        "sha256": sha256_file(stage2_path),
                    }
                },
                "models": {
                    experiment: {
                        "experiment_name": experiment,
                        **{
                            key: dict(entry)
                            for key in (
                                "checkpoint",
                                "resolved_config",
                                "source_config",
                                "training_log",
                                "stored_best_metrics",
                                "independent_validation_metrics",
                            )
                        },
                        "validation_artifacts": {"predictions": dict(entry)},
                    }
                    for experiment in selected
                },
            }
            freeze_path = root / "freeze.json"
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            _verify_freeze(
                freeze_path, split_dir=split_dir, verify_test_manifest=True
            )
            bound.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                _verify_freeze(
                    freeze_path, split_dir=split_dir, verify_test_manifest=True
                )


if __name__ == "__main__":
    unittest.main()
