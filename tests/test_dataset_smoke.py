"""Focused tests for the Stage 2 dataset smoke-test contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts.smoke_test_dataset import (
    SmokeContract,
    SmokeTestError,
    _decode_rgb_image,
    run_smoke_test,
)
from src.data.common import CSV_COLUMNS


class DatasetSmokeTests(unittest.TestCase):
    """Exercise production behavior with an injectable small-size contract."""

    @staticmethod
    def _write_manifests(
        split_dir: Path,
        *,
        rows_per_split: int,
        suffix: str = ".png",
    ) -> None:
        split_dir.mkdir(parents=True, exist_ok=True)
        for split_index, split in enumerate(("train", "val", "test")):
            rows: list[dict[str, object]] = []
            for index in range(rows_per_split):
                source_split = "val" if split == "test" else "train_mini"
                rows.append(
                    {
                        "split": split,
                        "source_split": source_split,
                        "image_path": f"{source_split}/class/{split}-{index}{suffix}",
                        "local_image_path": f"{split}/{index}{suffix}",
                        "image_id": split_index * 100 + index,
                        "new_label": 0,
                        "original_category_id": 1000,
                        "scientific_name": "Species testensis",
                        "common_name": "Test species",
                    }
                )
            pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(
                split_dir / f"{split}.csv", index=False
            )

    @staticmethod
    def _make_rgb_images(image_root: Path, rows_per_split: int) -> None:
        for split in ("train", "val", "test"):
            directory = image_root / split
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(rows_per_split):
                Image.new(
                    "RGB",
                    (24 + index, 20 + index),
                    color=(20 * index, 80, 160),
                ).save(directory / f"{index}.png")

    def test_small_contract_fully_decodes_grayscale_and_rgba(self) -> None:
        contract = SmokeContract(
            num_classes=1,
            train_images_per_class=1,
            val_images_per_class=1,
            test_images_per_class=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_dir = root / "splits"
            image_root = root / "images"
            self._write_manifests(split_dir, rows_per_split=1)
            for split in ("train", "val", "test"):
                (image_root / split).mkdir(parents=True, exist_ok=True)
            Image.new("L", (31, 27), color=90).save(image_root / "train/0.png")
            Image.new("RGBA", (29, 23), color=(255, 0, 0, 0)).save(
                image_root / "val/0.png"
            )
            Image.new("RGB", (25, 21), color=(0, 100, 200)).save(
                image_root / "test/0.png"
            )
            figure_path = root / "sample.png"
            report_path = root / "report.md"

            run_smoke_test(
                split_dir=split_dir,
                image_root=image_root,
                figure_path=figure_path,
                report_path=report_path,
                seed=42,
                samples_per_split=1,
                contract=contract,
            )

            self.assertTrue(figure_path.is_file())
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("**Status: PASS**", report)
            self.assertIn("All 3 images were fully decoded", report)
            self.assertIn("L: 1", report)
            self.assertIn("RGBA: 1", report)

            _, _, _, grayscale = _decode_rgb_image(
                image_root / "train/0.png", retain_pixels=True
            )
            _, _, _, transparent = _decode_rgb_image(
                image_root / "val/0.png", retain_pixels=True
            )
            assert grayscale is not None and transparent is not None
            np.testing.assert_array_equal(grayscale[..., 0], grayscale[..., 1])
            np.testing.assert_array_equal(grayscale[..., 1], grayscale[..., 2])
            self.assertTrue(np.all(transparent == 255))

    def test_corrupt_non_sampled_image_is_fatal(self) -> None:
        contract = SmokeContract(
            num_classes=1,
            train_images_per_class=4,
            val_images_per_class=4,
            test_images_per_class=4,
        )
        # PCG64 seed 42 samples train position 0 when selecting one of four;
        # the corrupt train position 3 is therefore deliberately non-sampled.
        generator = np.random.Generator(np.random.PCG64(42))
        self.assertEqual(int(generator.choice(4, size=1, replace=False)[0]), 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_dir = root / "splits"
            image_root = root / "images"
            self._write_manifests(split_dir, rows_per_split=4)
            self._make_rgb_images(image_root, rows_per_split=4)
            corrupt_path = image_root / "train/3.png"
            corrupt_path.write_bytes(b"not a valid image")
            figure_path = root / "sample.png"
            report_path = root / "report.md"

            with self.assertRaisesRegex(
                SmokeTestError, r"Could not fully decode image .*train/3\.png"
            ):
                run_smoke_test(
                    split_dir=split_dir,
                    image_root=image_root,
                    figure_path=figure_path,
                    report_path=report_path,
                    seed=42,
                    samples_per_split=1,
                    contract=contract,
                )

            self.assertFalse(figure_path.exists())
            self.assertFalse(report_path.exists())


if __name__ == "__main__":
    unittest.main()
