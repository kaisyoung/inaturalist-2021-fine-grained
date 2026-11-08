"""Stage 3 transform and model-factory contract tests without network access."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18 as torchvision_resnet18

from src.deep_learning.models import count_parameters, create_model, set_frozen_backbone_eval
from src.deep_learning.transforms import get_transform


class DeepModelTests(unittest.TestCase):
    def test_transform_presets_produce_expected_shape(self) -> None:
        image = Image.new("RGB", (320, 240), color=(20, 40, 60))
        torch.manual_seed(42)
        for preset in ("eval", "basic_train", "strong_train"):
            tensor = get_transform(preset, 224)(image)
            self.assertEqual(tuple(tensor.shape), (3, 224, 224))
            self.assertTrue(torch.isfinite(tensor).all())

    def test_scratch_factories_return_batch_by_500_logits(self) -> None:
        inputs = torch.randn(2, 3, 64, 64)
        for architecture in ("resnet18", "efficientnet_b0"):
            model, metadata = create_model(
                architecture, pretrained=False, freeze_backbone=False, num_classes=500
            )
            model.eval()
            with torch.inference_mode():
                logits = model(inputs)
            self.assertEqual(tuple(logits.shape), (2, 500))
            self.assertEqual(metadata["num_classes"], 500)
            self.assertEqual(count_parameters(model)["total"], metadata["total_parameters"])

    def test_pretrained_factory_is_mocked_and_records_weight_hash(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            old_hub = torch.hub.get_dir()
            torch.hub.set_dir(name)
            checkpoint = Path(name) / "checkpoints/resnet18-f37072fd.pth"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"fixture pretrained bytes")
            try:
                with mock.patch(
                    "src.deep_learning.models.resnet18",
                    side_effect=lambda weights: torchvision_resnet18(weights=None),
                ):
                    model, metadata = create_model(
                        "resnet18", pretrained=True, freeze_backbone=True, num_classes=500
                    )
                self.assertTrue(metadata["weight_file_sha256"])
                self.assertEqual(count_parameters(model)["trainable"], 256_500)
                model.train()
                set_frozen_backbone_eval(model)
                batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
                self.assertTrue(batch_norms)
                self.assertTrue(all(not module.training for module in batch_norms))
                self.assertTrue(model.fc.training)
            finally:
                torch.hub.set_dir(old_hub)

    def test_freezing_random_backbone_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires pretrained"):
            create_model("resnet18", pretrained=False, freeze_backbone=True)


if __name__ == "__main__":
    unittest.main()

