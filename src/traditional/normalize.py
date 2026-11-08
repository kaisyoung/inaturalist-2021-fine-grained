"""Fitted block-aware power normalisation for raw Stage 2 histograms."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


EPS = 1e-12


def power_normalize(X: np.ndarray, alpha: float, l2: bool = True) -> np.ndarray:
    """Apply row-wise L1, element-wise power, and optional L2 normalisation."""

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("power_normalize expects a finite non-negative matrix")
    values = values / np.maximum(values.sum(axis=1, keepdims=True), EPS)
    values = values**alpha
    if l2:
        values = values / np.maximum(
            np.linalg.norm(values, axis=1, keepdims=True), EPS
        )
    return values


class PowerNormalizer(TransformerMixin, BaseEstimator):
    """Normalise each configured feature block, or their concatenation."""

    def __init__(
        self,
        blocks: Sequence[tuple[int, int]] | None = None,
        alpha: float = 0.3,
        blockwise: bool = True,
        weights: Sequence[float] | None = None,
    ):
        self.blocks = blocks
        self.alpha = alpha
        self.blockwise = blockwise
        self.weights = weights

    def _spans(self, n_features: int) -> list[tuple[int, int]]:
        if not self.blocks:
            return [(0, n_features)]
        spans = [(int(start), int(stop)) for start, stop in self.blocks]
        if spans[0][0] != 0 or spans[-1][1] != n_features:
            raise ValueError(
                f"blocks must tile 0..{n_features}; got "
                f"{spans[0][0]}..{spans[-1][1]}"
            )
        for (start, stop), (next_start, next_stop) in zip(spans, spans[1:]):
            if start < 0 or stop <= start or next_stop <= next_start:
                raise ValueError("block spans must have positive widths")
            if next_start != stop:
                raise ValueError("blocks must be contiguous and non-overlapping")
        if spans[-1][1] <= spans[-1][0]:
            raise ValueError("block spans must have positive widths")
        return spans

    def fit(self, X, y=None):
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        values = check_array(X, dtype=np.float64, ensure_all_finite=True)
        if (values < 0).any():
            raise ValueError("PowerNormalizer expects non-negative features")
        self.n_features_in_ = int(values.shape[1])
        self.spans_ = self._spans(self.n_features_in_)
        if self.weights is not None:
            if len(self.weights) != len(self.spans_):
                raise ValueError("weights must have one entry per block")
            if any(not np.isfinite(weight) for weight in self.weights):
                raise ValueError("weights must be finite")
        return self

    def transform(self, X) -> np.ndarray:
        check_is_fitted(self, "spans_")
        values = check_array(X, dtype=np.float64, ensure_all_finite=True)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"expected {self.n_features_in_} features, got {values.shape[1]}"
            )
        if (values < 0).any():
            raise ValueError("PowerNormalizer expects non-negative features")

        if self.blockwise and len(self.spans_) > 1:
            parts: list[np.ndarray] = []
            for index, (start, stop) in enumerate(self.spans_):
                block = power_normalize(values[:, start:stop], self.alpha, l2=True)
                if self.weights is not None:
                    block = block * float(self.weights[index])
                parts.append(block)
            return np.hstack(parts)

        stacked = np.hstack(
            [
                values[:, start:stop]
                / np.maximum(
                    values[:, start:stop].sum(axis=1, keepdims=True), EPS
                )
                for start, stop in self.spans_
            ]
        )
        return power_normalize(stacked, self.alpha, l2=True)


def blocks_from_config(
    feature_config: dict, components: Sequence[str] | None = None
) -> list[tuple[int, int]]:
    """Read and rebase block spans from a feature configuration."""

    offsets = feature_config.get("component_offsets")
    if not isinstance(offsets, dict):
        components_block = feature_config.get("components")
        if not isinstance(components_block, dict):
            raise ValueError("feature config has no component offsets")
        offsets = {
            name: [entry["start"], entry["stop"]]
            for name, entry in components_block.items()
            if isinstance(entry, dict) and "start" in entry and "stop" in entry
        }
    order = feature_config.get("concatenation_order", list(offsets))
    chosen = list(components) if components else list(order)
    spans: list[tuple[int, int]] = []
    cursor = 0
    for name in order:
        if name not in chosen:
            continue
        start, stop = offsets[name]
        width = int(stop) - int(start)
        if width <= 0:
            raise ValueError(f"component {name!r} has an invalid width")
        spans.append((cursor, cursor + width))
        cursor += width
    if not spans:
        raise ValueError(f"no known components among {chosen}")
    return spans
