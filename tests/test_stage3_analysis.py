"""Synthetic contracts for Stage 3 aggregation and saved-output error analysis."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.analysis.collect_stage3_results import COLUMNS, collect
from src.analysis.deep_error_analysis import analyse
from src.data.common import sha256_file


class Stage3AnalysisTests(unittest.TestCase):
    def test_collector_preserves_three_stage2_rows_and_blanks_unrun_test(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            metrics = root / "metrics"
            logs = root / "logs"
            metrics.mkdir()
            logs.mkdir()
            experiment = "resnet18_scratch_basic"
            val_block = {
                "num_samples": 5000,
                "num_classes": 500,
                "loss": 4.0,
                "top1_accuracy": 0.2,
                "top5_accuracy": 0.5,
                "overall_accuracy": 0.2,
                "macro_precision": 0.21,
                "macro_recall": 0.2,
                "macro_f1": 0.19,
            }
            resolved_path = metrics / f"{experiment}_resolved_config.json"
            resolved_path.write_text(
                json.dumps(
                    {
                        "model_name": "resnet18",
                        "pretrained": False,
                        "augmentation": "basic_train",
                        "freeze_backbone": False,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_path = root / f"{experiment}_best.pt"
            checkpoint_path.write_bytes(b"checkpoint fixture")
            source_path = root / "source.yaml"
            source_path.write_text("experiment_name: fixture\n", encoding="utf-8")
            best_path = metrics / f"{experiment}_best_val_metrics.json"
            best_path.write_text(
                json.dumps(
                    {
                        "best_epoch": 1,
                        "training_time_seconds": 12.5,
                        "parameter_counts": {"total": 100, "trainable": 100},
                        "metrics": val_block,
                        "checkpoint_path": checkpoint_path.as_posix(),
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                        "resolved_config_path": resolved_path.as_posix(),
                        "resolved_config_sha256": sha256_file(resolved_path),
                    }
                ),
                encoding="utf-8",
            )
            (metrics / f"{experiment}_status.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "resolved_config_sha256": sha256_file(resolved_path),
                    }
                ),
                encoding="utf-8",
            )
            (logs / f"{experiment}_train_log.jsonl").write_text(
                json.dumps({"epoch": 1, "train_top1": 0.3}) + "\n",
                encoding="utf-8",
            )
            artifact_root = root / "validation_artifacts"
            artifact_root.mkdir()
            artifact_paths = {
                "predictions": artifact_root / "predictions.csv",
                "confusion": artifact_root / "confusion.npz",
                "per_class": artifact_root / "per_class.csv",
                "pairs": artifact_root / "pairs.csv",
                "confusion_figure": artifact_root / "confusion.png",
                "pairs_figure": artifact_root / "pairs.png",
            }
            for key, path in artifact_paths.items():
                path.write_bytes(f"{key} fixture".encode("utf-8"))
            (metrics / f"{experiment}_val_metrics.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "split": "val",
                        "experiment_name": experiment,
                        "metrics": val_block,
                        "validation_recheck": {
                            "agrees_with_stored_best": True,
                            "tolerance": 1e-6,
                        },
                        "provenance": {
                            "checkpoint_path": checkpoint_path.as_posix(),
                            "checkpoint_sha256": sha256_file(checkpoint_path),
                            "resolved_config_path": resolved_path.as_posix(),
                            "resolved_config_sha256": sha256_file(resolved_path),
                            "source_config_path": source_path.as_posix(),
                            "source_config_sha256": sha256_file(source_path),
                            "stored_best_metrics_path": best_path.as_posix(),
                            "stored_best_metrics_sha256": sha256_file(best_path),
                            "split_manifest": "val.csv",
                            "split_manifest_sha256": "4adf287996d35d9e57f829426ff7480bbaf0b74e1c319a29d57fe74b3f0366bb",
                        },
                        "artifacts": {
                            key: {
                                "path": path.as_posix(),
                                "bytes": path.stat().st_size,
                                "sha256": sha256_file(path),
                            }
                            for key, path in artifact_paths.items()
                        },
                    }
                ),
                encoding="utf-8",
            )
            stage2 = root / "stage2.csv"
            records = []
            for model, feature_type, classifier, feature_dim, image_size in (
                (
                    "linear_svm",
                    "color_texture_hog_sift_bovw",
                    "LinearSVC",
                    3704,
                    "128x128",
                ),
                (
                    "random_forest",
                    "color_texture_hog",
                    "RandomForestClassifier",
                    2424,
                    "128x128",
                ),
                (
                    "sift_bovw_svm",
                    "sift_bovw",
                    "LinearSVC",
                    1280,
                    "max side 256",
                ),
            ):
                records.append(
                    {
                        "model_name": model,
                        "feature_type": feature_type,
                        "classifier": classifier,
                        "val_top1": 0.1,
                        "val_top5": 0.2,
                        "val_macro_precision": 0.1,
                        "val_macro_recall": 0.1,
                        "val_macro_f1": 0.1,
                        "test_top1": 0.1,
                        "test_top5": 0.2,
                        "test_macro_precision": 0.1,
                        "test_macro_recall": 0.1,
                        "test_macro_f1": 0.1,
                        "train_time_seconds": 1.0,
                        "test_time_seconds": 0.1,
                        "feature_dim": feature_dim,
                        "image_size": image_size,
                    }
                )
            pd.DataFrame(records).to_csv(stage2, index=False)
            args = argparse.Namespace(
                metrics_dir=metrics,
                log_dir=logs,
                stage2_results=stage2,
                output_dir=root / "tables",
                figure=root / "figures/comparison.png",
                force=False,
            )
            with patch(
                "src.analysis.collect_stage3_results.ACCEPTED_STAGE2_RESULTS_SHA256",
                sha256_file(stage2),
            ):
                result = collect(args)
            self.assertEqual(result["deep_rows"], 1)
            final = pd.read_csv(root / "tables/final_model_comparison.csv", keep_default_na=False)
            self.assertEqual(list(final.columns), COLUMNS)
            self.assertEqual(len(final), 4)
            deep = final.loc[final["model_name"] == experiment].iloc[0]
            self.assertEqual(deep["test_top1"], "")

            stale = pd.DataFrame(records)
            stale.loc[stale["model_name"] == "linear_svm", "classifier"] = (
                "SGDClassifier"
            )
            stale.to_csv(stage2, index=False)
            args.force = True
            with patch(
                "src.analysis.collect_stage3_results.ACCEPTED_STAGE2_RESULTS_SHA256",
                sha256_file(stage2),
            ):
                with self.assertRaisesRegex(ValueError, "rewritten contract"):
                    collect(args)
            pd.DataFrame(records).to_csv(stage2, index=False)

            (metrics / f"{experiment}_test_metrics.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "split": "test",
                        "experiment_name": experiment,
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "src.analysis.collect_stage3_results.ACCEPTED_STAGE2_RESULTS_SHA256",
                sha256_file(stage2),
            ):
                with self.assertRaisesRegex(ValueError, "without the canonical freeze"):
                    collect(args)

    def test_error_analysis_is_deterministic_and_prediction_only(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            predictions_dir = root / "predictions"
            tables_dir = root / "tables"
            metrics_dir = root / "metrics"
            split_dir = root / "splits"
            output_dir = root / "output"
            predictions_dir.mkdir()
            tables_dir.mkdir()
            metrics_dir.mkdir()
            split_dir.mkdir()
            experiment = "resnet18_pretrained_basic"
            rows = []
            for index in range(5000):
                correct = index % 2 == 0
                rows.append(
                    {
                        "image_id": 10_000 + index,
                        "local_image_path": f"val/{index}.jpg",
                        "true_label": index % 500,
                        "predicted_label": index % 500 if correct else (index + 1) % 500,
                        "correct": correct,
                        "top1_score": (index % 100) / 100.0,
                        "score_type": "softmax_probability",
                    }
                )
            pd.DataFrame(rows).to_csv(
                predictions_dir / f"{experiment}_val_predictions.csv", index=False
            )
            pd.DataFrame(
                {
                    "image_id": [row["image_id"] for row in rows],
                    "new_label": [row["true_label"] for row in rows],
                    "local_image_path": [row["local_image_path"] for row in rows],
                }
            ).to_csv(split_dir / "val.csv", index=False)
            per_class = pd.DataFrame(
                {
                    "new_label": np.arange(500),
                    "precision": np.linspace(0, 1, 500),
                    "recall": np.linspace(0, 1, 500),
                    "f1": np.linspace(0, 1, 500),
                    "support": np.full(500, 10),
                }
            )
            per_class.to_csv(
                tables_dir / f"{experiment}_val_per_class_metrics.csv", index=False
            )
            pd.DataFrame(
                {
                    "rank": [1],
                    "class_a_new_label": [0],
                    "class_b_new_label": [1],
                    "a_as_b": [6],
                    "b_as_a": [4],
                    "total_confusions": [10],
                }
            ).to_csv(tables_dir / f"{experiment}_val_top_confusions.csv", index=False)
            confusion_path = metrics_dir / f"{experiment}_val_confusion.npz"
            confusion_path.write_bytes(b"fixture-confusion")
            prediction_path = predictions_dir / f"{experiment}_val_predictions.csv"
            per_class_path = tables_dir / f"{experiment}_val_per_class_metrics.csv"
            pairs_path = tables_dir / f"{experiment}_val_top_confusions.csv"
            artifact_paths = {
                "predictions": prediction_path,
                "confusion": confusion_path,
                "per_class": per_class_path,
                "pairs": pairs_path,
            }
            (metrics_dir / f"{experiment}_val_metrics.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "split": "val",
                        "experiment_name": experiment,
                        "provenance": {"checkpoint_sha256": "fixture"},
                        "validation_recheck": {
                            "agrees_with_stored_best": True,
                            "tolerance": 1e-6,
                        },
                        "artifacts": {
                            key: {
                                "path": path.as_posix(),
                                "bytes": path.stat().st_size,
                                "sha256": sha256_file(path),
                            }
                            for key, path in artifact_paths.items()
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                split="val",
                experiment=experiment,
                freeze_manifest=root / "absent-freeze.json",
                predictions_dir=predictions_dir,
                tables_dir=tables_dir,
                metrics_dir=metrics_dir,
                split_dir=split_dir,
                access_record=None,
                output_dir=output_dir,
                force=False,
            )
            result = analyse(args)
            self.assertEqual(len(result["outputs"]), 5)
            hardest = pd.read_csv(output_dir / "best_model_val_hardest_classes.csv")
            easiest = pd.read_csv(output_dir / "best_model_val_easiest_classes.csv")
            self.assertEqual(int(hardest.iloc[0]["new_label"]), 0)
            self.assertEqual(int(easiest.iloc[0]["new_label"]), 499)
            failures = pd.read_csv(output_dir / "best_model_val_failure_examples.csv")
            self.assertTrue((failures["correct"] == False).all())  # noqa: E712
            with prediction_path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            args.force = True
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                analyse(args)

    def test_test_error_analysis_rejects_experiment_override(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            args = argparse.Namespace(
                split="test",
                experiment="resnet18_pretrained_basic",
                freeze_manifest=root / "stage3_freeze.json",
                predictions_dir=root / "predictions",
                tables_dir=root / "tables",
                metrics_dir=root / "metrics",
                split_dir=root / "splits",
                access_record=None,
                output_dir=root / "output",
                force=False,
            )
            with self.assertRaisesRegex(ValueError, "cannot override"):
                analyse(args)


if __name__ == "__main__":
    unittest.main()
