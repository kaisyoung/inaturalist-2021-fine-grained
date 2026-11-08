"""Stage 3 manifest dataset and image-mode contract tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from src.data.common import CSV_COLUMNS
from src.deep_learning.dataset import (
    DatasetContract,
    DatasetImageError,
    DatasetManifestError,
    ManifestImageDataset,
)


class DeepDatasetTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[pd.DataFrame, Path]:
        image_root = root / "images"
        (image_root / "train").mkdir(parents=True, exist_ok=True)
        modes = ["L", "RGBA", "RGB", "RGB"]
        names = ["gray.png", "alpha.png", "oriented.jpg", "rgb.jpg"]
        for index, (mode, name) in enumerate(zip(modes, names)):
            size = (2, 4) if name == "oriented.jpg" else (8 + index, 7 + index)
            color = 100 if mode == "L" else (10, 20, 30, 120) if mode == "RGBA" else (10, 20, 30)
            image = Image.new(mode, size, color=color)
            if name == "oriented.jpg":
                exif = Image.Exif()
                exif[274] = 6
                image.save(image_root / "train" / name, exif=exif)
            else:
                image.save(image_root / "train" / name)
        rows = []
        for index, name in enumerate(names):
            label = index // 2
            rows.append(
                {
                    "split": "train",
                    "source_split": "train_mini",
                    "image_path": f"train_mini/class_{label}/{name}",
                    "local_image_path": f"train/{name}",
                    "image_id": 100 + index,
                    "new_label": label,
                    "original_category_id": 1000 + label,
                    "scientific_name": f"Species {label}",
                    "common_name": f"Common {label}",
                }
            )
        return pd.DataFrame(rows, columns=CSV_COLUMNS), image_root

    def test_grayscale_rgba_and_exif_are_rgb_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            frame, image_root = self._fixture(Path(name))
            dataset = ManifestImageDataset(
                frame,
                image_root,
                ToTensor(),
                "train",
                strict=True,
                contract=DatasetContract(
                    num_classes=2,
                    train_images_per_class=2,
                    val_images_per_class=1,
                    test_images_per_class=1,
                ),
            )
            self.assertEqual(len(dataset), 4)
            for index in (0, 1):
                tensor, label, metadata = dataset[index]
                self.assertEqual(tensor.shape[0], 3)
                self.assertEqual(label, 0)
                self.assertEqual(metadata["image_id"], 100 + index)
            oriented, _, _ = dataset[2]
            self.assertEqual(tuple(oriented.shape), (3, 2, 4))

    def test_unsafe_path_and_wrong_source_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            frame, image_root = self._fixture(Path(name))
            frame.loc[0, "local_image_path"] = "../escape.jpg"
            with self.assertRaisesRegex(DatasetManifestError, "unsafe"):
                ManifestImageDataset(frame, image_root, ToTensor(), "train", strict=False)
            frame, image_root = self._fixture(Path(name))
            frame.loc[0, "source_split"] = "val"
            with self.assertRaisesRegex(DatasetManifestError, "source_split"):
                ManifestImageDataset(frame, image_root, ToTensor(), "train", strict=False)

    def test_corrupt_image_raises_contextual_decode_error(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            frame, image_root = self._fixture(Path(name))
            (image_root / "train/gray.png").write_bytes(b"not an image")
            dataset = ManifestImageDataset(frame, image_root, ToTensor(), "train", strict=False)
            with self.assertRaisesRegex(DatasetImageError, "image_id=100"):
                dataset[0]

    def test_transform_must_return_three_channel_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            frame, image_root = self._fixture(Path(name))
            dataset = ManifestImageDataset(
                frame,
                image_root,
                lambda image: torch.zeros(1, 4, 4),
                "train",
                strict=False,
            )
            with self.assertRaisesRegex(DatasetImageError, "expected"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
