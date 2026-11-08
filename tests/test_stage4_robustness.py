from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.analysis.metrics import evaluate_aligned_scores
from src.data.common import deterministic_json_bytes
from src.data.common import sha256_file
from src.robustness.evaluate_robustness import (
    PREDICTION_COLUMNS,
    TRADITIONAL_FEATURE_DIM,
    RobustnessEvaluationError,
    _load_traditional_runtime,
    _validate_args,
    _output_paths,
    _prediction_frame,
    build_parser,
)
from scripts.smoke_test_robustness import main as smoke_main
from src.robustness.freeze_protocol import (
    EXPECTED_MODEL_NAMES,
    PROJECT_ROOT,
    ProtocolError,
    build_protocol_document,
    freeze_protocol,
    validate_protocol_document,
)


REAL_EVIDENCE_SENTINELS = (
    Path("results/models/traditional/random_forest.joblib"),
    Path("results/models/deep_learning/resnet18_scratch_basic_best.pt"),
    Path("results/models/deep_learning/resnet18_pretrained_strong_aug_best.pt"),
    Path("results/models/deep_learning/efficientnet_b0_pretrained_strong_aug_best.pt"),
    Path("results/predictions/traditional_random_forest_test_predictions.csv"),
    Path(
        "results/predictions/deep_learning/"
        "efficientnet_b0_pretrained_strong_aug_test_predictions.csv"
    ),
)


class Stage4RealEvidenceProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        missing = [
            relative.as_posix()
            for relative in REAL_EVIDENCE_SENTINELS
            if not (PROJECT_ROOT / relative).is_file()
        ]
        if missing:
            raise unittest.SkipTest(
                "full-workspace Stage 4 evidence is intentionally absent from the "
                "source-only code ZIP; missing: " + ", ".join(missing)
            )
        cls.document = build_protocol_document(PROJECT_ROOT)

    def test_real_frozen_evidence_builds_a_valid_64_cell_protocol(self) -> None:
        validate_protocol_document(self.document, PROJECT_ROOT)
        self.assertEqual(self.document["expected_run_count"], 64)
        self.assertEqual(self.document["model_order"], list(EXPECTED_MODEL_NAMES))
        self.assertTrue(self.document["policy"]["no_retraining_or_weight_updates"])

    def test_sorted_json_round_trip_does_not_change_protocol_semantics(self) -> None:
        restored = json.loads(deterministic_json_bytes(self.document))
        validate_protocol_document(restored, PROJECT_ROOT)

    def test_noncanonical_protocol_is_idempotent_and_requires_current_hash_to_replace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            first, first_hash = freeze_protocol(path, project_root=PROJECT_ROOT)
            second, second_hash = freeze_protocol(path, project_root=PROJECT_ROOT)
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first, second)
            with self.assertRaises(ProtocolError):
                freeze_protocol(
                    path,
                    project_root=PROJECT_ROOT,
                    dangerous_replace=True,
                    current_protocol_sha256="0" * 64,
                )


class Stage4ProtocolTests(unittest.TestCase):
    def test_traditional_runtime_requires_the_full_2424_dimensional_binding(self) -> None:
        self.assertEqual(TRADITIONAL_FEATURE_DIM, 2424)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "random_forest.joblib"
            metadata = root / "random_forest.metadata.json"
            config = root / "traditional_feature_config.json"
            metadata.write_text("{}", encoding="utf-8")
            config.write_text("{}", encoding="utf-8")
            estimator = RandomForestClassifier()
            estimator.classes_ = np.arange(500, dtype=np.int64)
            bundle = {
                "experiment_name": "random_forest",
                "model_type": "random_forest",
                "estimator": estimator,
                "feature_indices": np.arange(2424, dtype=np.int64),
            }
            joblib.dump(bundle, model)

            def artifact(path: Path) -> dict[str, object]:
                return {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }

            entry = {
                "artifact": artifact(model),
                "model_metadata": artifact(metadata),
                "feature_config": artifact(config),
            }
            runtime = _load_traditional_runtime(root, "traditional_random_forest", entry)
            self.assertEqual(len(runtime["classes"]), 500)

            bundle["feature_indices"] = np.arange(1830, dtype=np.int64)
            joblib.dump(bundle, model)
            entry["artifact"] = artifact(model)
            with self.assertRaisesRegex(
                RobustnessEvaluationError, "2424-D handcrafted"
            ):
                _load_traditional_runtime(root, "traditional_random_forest", entry)

    def test_smoke_outputs_are_isolated_from_canonical_outputs(self) -> None:
        metrics, predictions = _output_paths(
            PROJECT_ROOT,
            "traditional_random_forest",
            "gaussian_noise",
            1,
            max_samples=3,
            output_dir=None,
        )
        self.assertIn("robustness_smoke", metrics.as_posix())
        self.assertIn("robustness_smoke", predictions.as_posix())
        self.assertIn("smoke_n3", metrics.name)

    def test_cli_requires_explicit_test_acknowledgement(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model-name",
                "traditional_random_forest",
                "--corruption",
                "gaussian_noise",
                "--severity",
                "1",
            ]
        )
        self.assertFalse(args.include_test)

    def test_zero_workers_is_rejected_before_any_protocol_or_test_access(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--protocol-run",
                "--include-test",
                "--num-workers",
                "0",
            ]
        )
        with self.assertRaises(SystemExit):
            _validate_args(args, parser)
        with self.assertRaises(SystemExit):
            smoke_main(["--include-test", "--num-workers", "0"])

    def test_prediction_export_uses_stable_lower_label_tie_break(self) -> None:
        frame = pd.DataFrame(
            {
                "manifest_index": [0],
                "image_id": [123],
                "local_image_path": ["test/example.jpg"],
                "new_label": [5],
                "scientific_name": ["example"],
                "common_name": [""],
            }
        )
        scores = np.zeros((1, 500), dtype=np.float64)
        _, predicted, order, _ = evaluate_aligned_scores(
            np.array([5], dtype=np.int64), scores, np.arange(500)
        )
        exported = _prediction_frame(
            frame,
            scores,
            predicted,
            order,
            score_type="toy_scores",
            corruption_type="gaussian_noise",
            severity=1,
            root=PROJECT_ROOT,
        )
        self.assertEqual(list(exported.columns), list(PREDICTION_COLUMNS))
        self.assertEqual(int(exported.loc[0, "predicted_label"]), 0)
        self.assertEqual(json.loads(exported.loc[0, "top5_labels"]), [0, 1, 2, 3, 4])

    def test_evaluator_contains_no_training_operation(self) -> None:
        path = PROJECT_ROOT / "src/robustness/evaluate_robustness.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        forbidden_attributes = {"backward", "step", "zero_grad", "train"}
        used = {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
        }
        self.assertTrue(forbidden_attributes.isdisjoint(used))


if __name__ == "__main__":
    unittest.main()
