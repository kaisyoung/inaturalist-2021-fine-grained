"""Focused mathematical and persistence tests for Stage 2 normalisation."""

from __future__ import annotations

import io
import unittest

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.traditional.features import feature_configuration
from src.traditional.normalize import PowerNormalizer, blocks_from_config, power_normalize


class PowerNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(42)
        self.features = generator.gamma(0.3, 1.0, size=(8, 12))
        self.blocks = [(0, 4), (4, 12)]

    def test_alpha_half_is_the_hellinger_unit_map(self) -> None:
        mapped = power_normalize(self.features, alpha=0.5, l2=False)
        np.testing.assert_allclose(np.linalg.norm(mapped, axis=1), 1.0, atol=1e-12)

    def test_blockwise_normalises_each_block_and_alpha_defaults_to_point_three(self) -> None:
        normaliser = PowerNormalizer(blocks=self.blocks).fit(self.features)
        self.assertEqual(normaliser.alpha, 0.3)
        transformed = normaliser.transform(self.features)
        for start, stop in self.blocks:
            np.testing.assert_allclose(
                np.linalg.norm(transformed[:, start:stop], axis=1), 1.0, atol=1e-10
            )

    def test_negative_input_and_non_tiling_blocks_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PowerNormalizer().fit(-self.features)
        with self.assertRaises(ValueError):
            PowerNormalizer(blocks=[(0, 4), (5, 12)]).fit(self.features)

    def test_pipeline_pickle_preserves_raw_feature_predictions(self) -> None:
        labels = np.asarray([0, 1] * 4)
        pipeline = Pipeline(
            [
                ("normalise", PowerNormalizer(blocks=self.blocks)),
                ("classifier", RandomForestClassifier(n_estimators=5, random_state=42)),
            ]
        ).fit(self.features, labels)
        buffer = io.BytesIO()
        joblib.dump(pipeline, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        np.testing.assert_array_equal(
            restored.predict(self.features), pipeline.predict(self.features)
        )

    def test_feature_config_spans_are_exact(self) -> None:
        config = feature_configuration()
        spans = blocks_from_config(config)
        self.assertEqual(spans, [(0, 520), (520, 660), (660, 2424)])


if __name__ == "__main__":
    unittest.main()
