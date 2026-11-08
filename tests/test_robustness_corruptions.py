from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from src.robustness.corruptions import (
    CORRUPTION_SEVERITIES,
    CorruptionError,
    apply_corruption,
    corruption_spec,
    deterministic_motion_angle,
    motion_kernel,
    prepare_rgb,
    protocol_corruption_matrix,
)


def patterned_image(width: int = 47, height: int = 35) -> Image.Image:
    y, x = np.indices((height, width))
    array = np.stack(
        (
            (x * 7 + y * 3) % 256,
            (x * 2 + y * 11) % 256,
            (x * 13 + y * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(array)


class CorruptionTests(unittest.TestCase):
    def test_all_fixed_corruptions_preserve_rgb_size_and_uint8_range(self) -> None:
        image = patterned_image()
        original = np.asarray(image).copy()
        for corruption_type in CORRUPTION_SEVERITIES:
            for severity in range(1, 5):
                with self.subTest(corruption=corruption_type, severity=severity):
                    result = apply_corruption(
                        image, corruption_type, severity, seed=42, image_key=123
                    )
                    pixels = np.asarray(result)
                    self.assertEqual(result.mode, "RGB")
                    self.assertEqual(result.size, image.size)
                    self.assertEqual(pixels.dtype, np.uint8)
                    self.assertEqual(pixels.shape, (image.height, image.width, 3))
                    self.assertGreaterEqual(int(pixels.min()), 0)
                    self.assertLessEqual(int(pixels.max()), 255)
        np.testing.assert_array_equal(np.asarray(image), original)

    def test_each_operation_is_deterministic(self) -> None:
        image = patterned_image()
        for corruption_type in CORRUPTION_SEVERITIES:
            for severity in range(1, 5):
                first = np.asarray(
                    apply_corruption(
                        image, corruption_type, severity, seed=42, image_key="same"
                    )
                )
                second = np.asarray(
                    apply_corruption(
                        image, corruption_type, severity, seed=42, image_key="same"
                    )
                )
                np.testing.assert_array_equal(first, second)

    def test_noise_uses_one_base_realization_across_severity(self) -> None:
        image = Image.fromarray(np.full((96, 96, 3), 128, dtype=np.uint8))
        weak = np.asarray(
            apply_corruption(image, "gaussian_noise", 1, seed=42, image_key=7),
            dtype=np.float64,
        ) - 128.0
        medium = np.asarray(
            apply_corruption(image, "gaussian_noise", 2, seed=42, image_key=7),
            dtype=np.float64,
        ) - 128.0
        usable = (np.abs(weak) >= 2) & (np.abs(medium) < 125)
        correlation = float(np.corrcoef(weak[usable], medium[usable])[0, 1])
        self.assertGreater(correlation, 0.995)
        self.assertGreater(float(np.mean(np.sign(weak[usable]) == np.sign(medium[usable]))), 0.995)
        other = np.asarray(
            apply_corruption(image, "gaussian_noise", 1, seed=42, image_key=8)
        )
        self.assertFalse(np.array_equal(other, weak + 128.0))

    def test_motion_angle_is_invariant_across_length_and_keyed_per_image(self) -> None:
        image = patterned_image()
        angle = deterministic_motion_angle(image, seed=42, image_key=100)
        self.assertGreaterEqual(angle, 0.0)
        self.assertLess(angle, 180.0)
        self.assertEqual(angle, deterministic_motion_angle(image, seed=42, image_key=100))
        self.assertNotEqual(angle, deterministic_motion_angle(image, seed=42, image_key=101))
        for length in (5, 9, 15, 25):
            kernel = motion_kernel(length, angle)
            self.assertEqual(kernel.shape, (length, length))
            self.assertTrue(np.isfinite(kernel).all())
            self.assertAlmostEqual(float(kernel.sum()), 1.0, places=6)

    def test_severity_changes_nonuniform_images(self) -> None:
        image = patterned_image(101, 79)
        for corruption_type in CORRUPTION_SEVERITIES:
            weak = np.asarray(
                apply_corruption(image, corruption_type, 1, seed=42, image_key=9)
            )
            strong = np.asarray(
                apply_corruption(image, corruption_type, 4, seed=42, image_key=9)
            )
            with self.subTest(corruption=corruption_type):
                self.assertFalse(np.array_equal(weak, strong))

    def test_grayscale_rgba_and_exif_are_prepared_before_corruption(self) -> None:
        grayscale = Image.fromarray(np.arange(120, dtype=np.uint8).reshape(10, 12))
        self.assertEqual(apply_corruption(grayscale, "jpeg", 1).mode, "RGB")
        rgba = Image.new("RGBA", (12, 10), (100, 20, 30, 0))
        prepared = prepare_rgb(rgba)
        self.assertEqual(prepared.getpixel((0, 0)), (255, 255, 255))

        source = patterned_image(12, 10)
        exif = source.getexif()
        exif[274] = 6
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG", exif=exif)
        buffer.seek(0)
        with Image.open(buffer) as opened:
            oriented = prepare_rgb(opened)
        self.assertEqual(oriented.size, (10, 12))

    def test_matrix_and_severity_contract_are_exact(self) -> None:
        self.assertEqual(
            protocol_corruption_matrix(),
            {
                "gaussian_noise": {"parameter": "sigma", "severities": [0.02, 0.05, 0.1, 0.2]},
                "gaussian_blur": {"parameter": "kernel_size", "severities": [3, 5, 9, 15]},
                "motion_blur": {"parameter": "kernel_length", "severities": [5, 9, 15, 25]},
                "jpeg_compression": {"parameter": "quality", "severities": [80, 50, 30, 10]},
            },
        )
        self.assertEqual(corruption_spec("noise", 4).value, 0.2)
        with self.assertRaises(CorruptionError):
            corruption_spec("gaussian_noise", 0)
        with self.assertRaises(CorruptionError):
            corruption_spec("brightness", 1)


if __name__ == "__main__":
    unittest.main()

