"""Deterministic handcrafted features for the rewritten Stage 2 baselines.

The raw descriptor has three non-negative blocks in a fixed order: a spatial
colour histogram, a multi-scale spatial LBP texture histogram, and HOG.  The
blocks are intentionally not normalised here; the fitted estimator pipeline
owns normalisation so cached features can be reused safely across ablations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern


DEFAULT_IMAGE_SIZE = 128
PYRAMID_GRID = 2
COLOR_H_BINS = 12
COLOR_S_BINS = 8
COLOR_V_BINS = 8
LBP_SCALES = ((8, 1), (16, 2))
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"

BLOCK_ORDER = ("color", "texture", "hog")
COMPONENT_ALIASES = {
    "colour": "color",
    "rgb": "color",
    "hsv": "color",
    "rgb_histogram": "color",
    "hsv_histogram": "color",
    "lbp": "texture",
}


class ImageFeatureError(RuntimeError):
    """Raised when an image cannot be decoded or converted into features."""


def validate_image_size(image_size: int) -> int:
    """Validate a square resize dimension against the configured HOG."""

    if isinstance(image_size, bool) or not isinstance(image_size, (int, np.integer)):
        raise TypeError("image_size must be an integer")
    size = int(image_size)
    cell = HOG_PIXELS_PER_CELL[0]
    minimum = cell * HOG_CELLS_PER_BLOCK[0]
    if size < minimum:
        raise ValueError(f"image_size must be at least {minimum} for this HOG")
    if size % cell:
        raise ValueError(f"image_size must be divisible by {cell}")
    if size % PYRAMID_GRID:
        raise ValueError(f"image_size must be divisible by {PYRAMID_GRID}")
    return size


def load_image_rgb(
    path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """Load *path*, apply EXIF orientation, flatten alpha, and resize."""

    size = validate_image_size(image_size)
    image_path = Path(path)
    try:
        with Image.open(image_path) as opened:
            oriented = ImageOps.exif_transpose(opened)
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                rgba = oriented.convert("RGBA")
                canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                oriented = Image.alpha_composite(canvas, rgba)
            rgb = oriented.convert("RGB")
            resized = rgb.resize((size, size), resample=Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.uint8).copy()
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise ImageFeatureError(f"Could not decode image {image_path}: {error}") from error

    if array.shape != (size, size, 3):
        raise ImageFeatureError(
            f"Decoded image {image_path} has shape {array.shape}, "
            f"expected {(size, size, 3)}"
        )
    return array


def _validate_rgb_array(rgb_image: np.ndarray) -> None:
    if not isinstance(rgb_image, np.ndarray):
        raise TypeError("rgb_image must be a NumPy array")
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB array, got shape {rgb_image.shape}")
    if rgb_image.size == 0:
        raise ValueError("RGB image must not be empty")
    if rgb_image.dtype != np.uint8:
        raise TypeError(f"Expected uint8 RGB pixels, got {rgb_image.dtype}")


def _spatial_regions(array: np.ndarray) -> Iterator[np.ndarray]:
    """Yield the whole image followed by the four 2x2 cells."""

    yield array
    height, width = array.shape[:2]
    for row in range(PYRAMID_GRID):
        for column in range(PYRAMID_GRID):
            yield array[
                row * height // PYRAMID_GRID : (row + 1) * height // PYRAMID_GRID,
                column * width // PYRAMID_GRID : (column + 1) * width // PYRAMID_GRID,
            ]


def extract_color(rgb_image: np.ndarray) -> np.ndarray:
    """Return joint hue-saturation and marginal-value spatial histograms."""

    _validate_rgb_array(rgb_image)
    hsv = rgb2hsv(rgb_image)
    parts: list[np.ndarray] = []
    for region in _spatial_regions(hsv):
        joint, _, _ = np.histogram2d(
            region[..., 0].ravel(),
            region[..., 1].ravel(),
            bins=(COLOR_H_BINS, COLOR_S_BINS),
            range=((0.0, 1.0), (0.0, 1.0)),
        )
        value, _ = np.histogram(
            region[..., 2], bins=COLOR_V_BINS, range=(0.0, 1.0)
        )
        parts.extend((joint.ravel(), value.astype(np.float64)))
    return np.concatenate(parts)


def extract_texture(rgb_image: np.ndarray) -> np.ndarray:
    """Return multi-scale uniform-LBP histograms over the spatial pyramid."""

    _validate_rgb_array(rgb_image)
    grayscale = np.clip(
        np.rint(rgb2gray(rgb_image) * 255.0), 0, 255
    ).astype(np.uint8)
    parts: list[np.ndarray] = []
    for points, radius in LBP_SCALES:
        codes = local_binary_pattern(
            grayscale, P=points, R=radius, method="uniform"
        )
        bins = np.arange(points + 3)
        for region in _spatial_regions(codes):
            parts.append(np.histogram(region, bins=bins)[0].astype(np.float64))
    return np.concatenate(parts)


def extract_hog(rgb_image: np.ndarray) -> np.ndarray:
    """Return the global HOG descriptor (already spatially structured)."""

    _validate_rgb_array(rgb_image)
    descriptor = hog(
        rgb2gray(rgb_image),
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        transform_sqrt=True,
        feature_vector=True,
        channel_axis=None,
    )
    return np.asarray(descriptor, dtype=np.float64)


BLOCK_EXTRACTORS = {
    "color": extract_color,
    "texture": extract_texture,
    "hog": extract_hog,
}


def extract_blocks(rgb_image: np.ndarray) -> dict[str, np.ndarray]:
    """Return each raw block separately in canonical order."""

    return {name: BLOCK_EXTRACTORS[name](rgb_image) for name in BLOCK_ORDER}


def extract_features_from_array(rgb_image: np.ndarray) -> np.ndarray:
    """Return the 2,424-dimensional unnormalised descriptor as float32."""

    feature = np.concatenate(
        [BLOCK_EXTRACTORS[name](rgb_image) for name in BLOCK_ORDER]
    ).astype(np.float32, copy=False)
    if not np.isfinite(feature).all():
        raise ImageFeatureError("Feature vector contains non-finite values")
    return feature


def extract_features(
    path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE
) -> np.ndarray:
    """Load an image and return its deterministic handcrafted descriptor."""

    try:
        return extract_features_from_array(load_image_rgb(path, image_size))
    except ImageFeatureError:
        raise
    except Exception as error:
        raise ImageFeatureError(f"Feature extraction failed for {path}: {error}") from error


def resolve_component(name: str) -> str:
    """Map a user-facing component name onto a canonical block name."""

    key = str(name).strip().lower()
    canonical = COMPONENT_ALIASES.get(key, key)
    if canonical not in BLOCK_ORDER:
        raise ValueError(
            f"Unknown feature component {name!r}; available: {', '.join(BLOCK_ORDER)}"
        )
    return canonical


def feature_configuration(image_size: int = DEFAULT_IMAGE_SIZE) -> dict[str, Any]:
    """Return the serialisable feature contract with exact block offsets."""

    size = validate_image_size(image_size)
    regions = 1 + PYRAMID_GRID**2
    cells = size // HOG_PIXELS_PER_CELL[0]
    blocks_per_side = cells - HOG_CELLS_PER_BLOCK[0] + 1
    dimensions = {
        "color": (COLOR_H_BINS * COLOR_S_BINS + COLOR_V_BINS) * regions,
        "texture": sum(points + 2 for points, _ in LBP_SCALES) * regions,
        "hog": blocks_per_side**2
        * HOG_CELLS_PER_BLOCK[0]**2
        * HOG_ORIENTATIONS,
    }
    parameters = {
        "color": {
            "h_bins": COLOR_H_BINS,
            "s_bins": COLOR_S_BINS,
            "v_bins": COLOR_V_BINS,
            "joint": "hue-saturation",
            "pyramid_regions": regions,
        },
        "texture": {
            "method": "uniform",
            "scales": [list(scale) for scale in LBP_SCALES],
            "pyramid_regions": regions,
        },
        "hog": {
            "orientations": HOG_ORIENTATIONS,
            "pixels_per_cell": list(HOG_PIXELS_PER_CELL),
            "cells_per_block": list(HOG_CELLS_PER_BLOCK),
            "block_norm": HOG_BLOCK_NORM,
            "transform_sqrt": True,
        },
    }
    components: dict[str, dict[str, Any]] = {}
    start = 0
    for name in BLOCK_ORDER:
        stop = start + dimensions[name]
        components[name] = {
            "start": start,
            "stop": stop,
            "dimension": dimensions[name],
            "parameters": parameters[name],
        }
        start = stop
    return {
        "feature_type": "color_texture_hog",
        "feature_dim": start,
        "image_size": [size, size],
        "resize": {
            "mode": "direct_square",
            "resampling": "Pillow BILINEAR",
            "exif_transpose": True,
            "colour_mode": "RGB",
        },
        "concatenation_order": list(BLOCK_ORDER),
        "components": components,
        "component_offsets": {
            name: [components[name]["start"], components[name]["stop"]]
            for name in BLOCK_ORDER
        },
        "normalisation": "applied by the model pipeline, not at extraction time",
    }


def feature_dimension(image_size: int = DEFAULT_IMAGE_SIZE) -> int:
    """Return the concatenated descriptor dimension for *image_size*."""

    return int(feature_configuration(image_size)["feature_dim"])
