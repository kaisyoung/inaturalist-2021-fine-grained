"""Synthetic tests for Stage 4 Grad-CAM and deterministic example selection."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn

from src.deep_learning.transforms import get_transform
from src.explainability.gradcam import (
    GradCAM,
    map_crop_heatmap_to_original,
    prepare_stage3_eval_image,
    resolve_target_layer,
)
from src.explainability.select_gradcam_examples import select_examples


class _ToyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 3)
        with torch.no_grad():
            self.features[0].weight.copy_(
                torch.linspace(-0.2, 0.3, self.features[0].weight.numel()).reshape_as(
                    self.features[0].weight
                )
            )
            self.classifier.weight.copy_(
                torch.tensor(
                    [
                        [0.5, 0.2, -0.1, 0.3],
                        [-0.2, 0.4, 0.6, 0.1],
                        [0.1, -0.3, 0.2, 0.7],
                    ]
                )
            )
            self.classifier.bias.copy_(torch.tensor([0.1, -0.2, 0.05]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(self.pool(features).flatten(1))


def _prediction_row(
    image_id: int,
    true_label: int,
    predicted_label: int,
    score: float,
) -> dict[str, object]:
    return {
        "image_id": image_id,
        "local_image_path": f"test/{image_id}.jpg",
        "true_label": true_label,
        "predicted_label": predicted_label,
        "correct": true_label == predicted_label,
        "top1_score": score,
        "true_scientific_name": f"Species {true_label}",
        "true_common_name": f"Common {true_label}",
        "predicted_scientific_name": f"Species {predicted_label}",
        "predicted_common_name": f"Common {predicted_label}",
    }


class GradCAMCoreTests(unittest.TestCase):
    def test_predicted_and_true_targets_return_finite_input_size_cams(self) -> None:
        torch.manual_seed(7)
        model = _ToyCNN()
        inputs = torch.rand(2, 3, 17, 19)
        with GradCAM(model) as generator:
            predicted = generator.generate(inputs, target_mode="predicted")
            true = generator.generate(inputs, target_mode="true", true_labels=[0, 1])
        self.assertEqual(tuple(predicted.heatmaps.shape), (2, 17, 19))
        self.assertEqual(tuple(true.heatmaps.shape), (2, 17, 19))
        self.assertEqual(true.target_labels.tolist(), [0, 1])
        self.assertTrue(torch.isfinite(predicted.heatmaps).all())
        self.assertTrue(torch.isfinite(true.heatmaps).all())
        self.assertGreaterEqual(float(predicted.heatmaps.min()), 0.0)
        self.assertLessEqual(float(predicted.heatmaps.max()), 1.0)

    def test_gradcam_does_not_mutate_weights_or_existing_gradients(self) -> None:
        torch.manual_seed(9)
        model = _ToyCNN()
        model.train()
        weight_before = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
        }
        gradient_before: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            parameter.grad = torch.full_like(parameter, 0.125)
            gradient_before[name] = parameter.grad.detach().clone()
        with GradCAM(model) as generator:
            generator.generate(torch.rand(1, 3, 12, 12))
        self.assertFalse(model.training)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter.detach(), weight_before[name]))
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.equal(parameter.grad, gradient_before[name]))

    def test_auto_and_safe_explicit_target_layer_resolution(self) -> None:
        model = _ToyCNN()
        automatic, name = resolve_target_layer(model)
        explicit, explicit_name = resolve_target_layer(model, "features.0")
        self.assertIs(automatic, model.features[-1])
        self.assertEqual(name, "features.1")
        self.assertIs(explicit, model.features[0])
        self.assertEqual(explicit_name, "features.0")
        for unsafe in ("features.-1", "features.__dict__", ".features", "features."):
            with self.subTest(path=unsafe), self.assertRaises(ValueError):
                resolve_target_layer(model, unsafe)

    def test_stage3_preprocessing_matches_eval_transform_for_rgba(self) -> None:
        values = np.zeros((101, 173, 4), dtype=np.uint8)
        values[..., 0] = np.arange(173, dtype=np.uint8)[None, :]
        values[..., 1] = np.arange(101, dtype=np.uint8)[:, None]
        values[..., 2] = 127
        values[..., 3] = 80
        image = Image.fromarray(values, mode="RGBA")
        prepared = prepare_stage3_eval_image(image)
        oriented_rgb = image.convert("RGB")
        expected = get_transform("eval", 224)(oriented_rgb)
        self.assertEqual(prepared.original_rgb.mode, "RGB")
        self.assertEqual(prepared.model_crop_rgb.size, (224, 224))
        self.assertTrue(torch.equal(prepared.tensor, expected))

    def test_exif_orientation_is_applied_before_geometry(self) -> None:
        image = Image.new("RGB", (40, 20), "red")
        exif = image.getexif()
        exif[274] = 6  # rotate 90 degrees clockwise for display
        image.info["exif"] = exif.tobytes()
        prepared = prepare_stage3_eval_image(image)
        self.assertEqual(
            (prepared.geometry.original_width, prepared.geometry.original_height),
            (20, 40),
        )

    def test_crop_heatmap_maps_to_full_image_with_transparent_outside(self) -> None:
        image = Image.new("RGB", (400, 200), "white")
        prepared = prepare_stage3_eval_image(image)
        self.assertEqual(
            (prepared.geometry.resized_width, prepared.geometry.resized_height),
            (512, 256),
        )
        self.assertEqual(
            (prepared.geometry.crop_left, prepared.geometry.crop_top), (144, 16)
        )
        mapped = map_crop_heatmap_to_original(
            np.ones((224, 224), dtype=np.float32), prepared.geometry
        )
        self.assertEqual(mapped.heatmap.shape, (200, 400))
        self.assertEqual(mapped.crop_mask.shape, (200, 400))
        self.assertEqual(float(mapped.crop_mask[0, 0]), 0.0)
        self.assertEqual(float(mapped.heatmap[0, 0]), 0.0)
        self.assertEqual(float(mapped.crop_mask[100, 200]), 1.0)
        self.assertGreater(float(mapped.heatmap[100, 200]), 0.99)


class GradCAMSelectionTests(unittest.TestCase):
    def _fixture(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        records: list[dict[str, object]] = []
        # Correct ties verify lower-image-id tie breaking.
        for image_id in range(20, 30):
            records.append(_prediction_row(image_id, 0, 0, 0.90))
        # Two directional representatives for the top confused pair.
        records.extend(
            [
                _prediction_row(101, 1, 2, 0.96),
                _prediction_row(100, 1, 2, 0.96),
                _prediction_row(102, 2, 1, 0.95),
            ]
        )
        # Generic high-confidence errors; pair examples must be excluded.
        for offset, image_id in enumerate(range(200, 215)):
            records.append(_prediction_row(image_id, 3, 4, 0.94 - offset * 0.01))
        predictions = pd.DataFrame.from_records(records)
        predictions["source_prediction_row"] = np.arange(len(predictions))
        pairs = pd.DataFrame.from_records(
            [
                {
                    "rank": 1,
                    "class_a_new_label": 1,
                    "class_b_new_label": 2,
                    "a_as_b": 2,
                    "b_as_a": 1,
                    "total_confusions": 3,
                }
            ]
        )
        return predictions, pairs

    def test_selection_is_deterministic_unique_and_excludes_pair_errors(self) -> None:
        predictions, pairs = self._fixture()
        first = select_examples(
            predictions, pairs, correct_count=3, incorrect_count=4, pair_count=1
        )
        shuffled = predictions.sample(frac=1.0, random_state=123).reset_index(drop=True)
        # source_prediction_row is an immutable persisted-row identity, so a
        # DataFrame shuffle cannot change the result.
        second = select_examples(
            shuffled, pairs, correct_count=3, incorrect_count=4, pair_count=1
        )
        pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
        self.assertFalse(first["image_id"].duplicated().any())
        correct = first.loc[first["selection_type"] == "correct_high_confidence"]
        self.assertEqual(correct["image_id"].tolist(), [20, 21, 22])
        confused = first.loc[first["selection_type"] == "confused_pair"]
        self.assertEqual(confused["image_id"].tolist(), [100, 102])
        generic_errors = first.loc[
            first["selection_type"] == "incorrect_high_confidence", "image_id"
        ].tolist()
        self.assertTrue(set(generic_errors).isdisjoint({100, 101, 102}))


if __name__ == "__main__":
    unittest.main()
