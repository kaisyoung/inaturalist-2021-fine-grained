"""Deterministic, on-the-fly image corruptions for Stage 4 robustness.

The corruption matrix is deliberately fixed before held-out test evaluation.
Every public operation accepts and returns a detached RGB :class:`PIL.Image`
whose pixels are uint8.  Gaussian-noise samples and motion-blur angles are
derived from ``(seed, image_key)`` without including severity, so one image
uses the same random realization at every severity.
"""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from typing import Final, Hashable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps


DEFAULT_SEED: Final[int] = 42
CORRUPTION_SEVERITIES: Final[dict[str, tuple[int | float, ...]]] = {
    "gaussian_noise": (0.02, 0.05, 0.10, 0.20),
    "gaussian_blur": (3, 5, 9, 15),
    "motion_blur": (5, 9, 15, 25),
    "jpeg_compression": (80, 50, 30, 10),
}
CORRUPTION_PARAMETERS: Final[dict[str, str]] = {
    "gaussian_noise": "sigma",
    "gaussian_blur": "kernel_size",
    "motion_blur": "kernel_length",
    "jpeg_compression": "quality",
}
CORRUPTION_ALIASES: Final[dict[str, str]] = {
    "noise": "gaussian_noise",
    "gaussian_noise": "gaussian_noise",
    "blur": "gaussian_blur",
    "gaussian_blur": "gaussian_blur",
    "motion": "motion_blur",
    "motion_blur": "motion_blur",
    "jpeg": "jpeg_compression",
    "jpeg_compression": "jpeg_compression",
}


class CorruptionError(ValueError):
    """Raised when a corruption request violates the frozen Stage 4 matrix."""


@dataclass(frozen=True)
class CorruptionSpec:
    """One exact cell in the frozen corruption/severity matrix."""

    corruption_type: str
    severity_index: int
    parameter: str
    value: int | float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "type": self.corruption_type,
            "severity_index": self.severity_index,
            "parameter": self.parameter,
            "parameter_value": self.value,
        }


def normalise_corruption_type(value: str) -> str:
    """Return the canonical corruption name, rejecting unknown aliases."""

    if not isinstance(value, str):
        raise TypeError("corruption_type must be a string")
    try:
        return CORRUPTION_ALIASES[value.strip().lower().replace("-", "_")]
    except KeyError as error:
        raise CorruptionError(
            f"Unknown corruption {value!r}; expected one of "
            f"{sorted(CORRUPTION_SEVERITIES)}"
        ) from error


def corruption_spec(corruption_type: str, severity: int | float) -> CorruptionSpec:
    """Resolve a severity index (1--4) or an exact parameter value.

    Integer values 1--4 are treated as severity indices.  This makes the CLI
    unambiguous for the kernel/quality corruptions; callers that already have
    an exact value should use :func:`corruption_spec_from_value`.
    """

    canonical = normalise_corruption_type(corruption_type)
    if isinstance(severity, bool) or not isinstance(severity, (int, float, np.number)):
        raise TypeError("severity must be a numeric severity index")
    numeric = float(severity)
    if not math.isfinite(numeric) or not numeric.is_integer() or not 1 <= numeric <= 4:
        raise CorruptionError("severity must be an integer index from 1 to 4")
    index = int(numeric)
    return CorruptionSpec(
        corruption_type=canonical,
        severity_index=index,
        parameter=CORRUPTION_PARAMETERS[canonical],
        value=CORRUPTION_SEVERITIES[canonical][index - 1],
    )


def corruption_spec_from_value(
    corruption_type: str, parameter_value: int | float
) -> CorruptionSpec:
    """Resolve one exact parameter value from the immutable matrix."""

    canonical = normalise_corruption_type(corruption_type)
    values = CORRUPTION_SEVERITIES[canonical]
    for index, expected in enumerate(values, start=1):
        if type(expected) is int:
            matches = (
                not isinstance(parameter_value, bool)
                and isinstance(parameter_value, (int, np.integer))
                and int(parameter_value) == expected
            )
        else:
            matches = (
                not isinstance(parameter_value, bool)
                and isinstance(parameter_value, (int, float, np.number))
                and float(parameter_value) == float(expected)
            )
        if matches:
            return CorruptionSpec(
                canonical, index, CORRUPTION_PARAMETERS[canonical], expected
            )
    raise CorruptionError(
        f"{canonical} parameter must be one of {list(values)}, got {parameter_value!r}"
    )


def prepare_rgb(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation, alpha-safe RGB conversion, and detach pixels."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    oriented = ImageOps.exif_transpose(image)
    if "A" in oriented.getbands() or "transparency" in oriented.info:
        rgba = oriented.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(white, rgba).convert("RGB")
    else:
        rgb = oriented.convert("RGB")
    return rgb.copy()


def _image_key_bytes(image: Image.Image, image_key: Hashable | None) -> bytes:
    if image_key is not None:
        return str(image_key).encode("utf-8", errors="strict")
    array = np.asarray(image, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.digest()


def _seed_sequence(
    seed: int,
    namespace: str,
    image: Image.Image,
    image_key: Hashable | None,
) -> np.random.SeedSequence:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    digest = hashlib.sha256()
    digest.update(str(int(seed)).encode("ascii"))
    digest.update(b"\0")
    digest.update(namespace.encode("ascii"))
    digest.update(b"\0")
    digest.update(_image_key_bytes(image, image_key))
    words = np.frombuffer(digest.digest(), dtype="<u4").astype(np.uint32).tolist()
    return np.random.SeedSequence(words)


def deterministic_motion_angle(
    image: Image.Image,
    *,
    seed: int = DEFAULT_SEED,
    image_key: Hashable | None = None,
) -> float:
    """Return the per-image motion angle in ``[0, 180)`` degrees."""

    rgb = prepare_rgb(image)
    generator = np.random.Generator(
        np.random.PCG64(_seed_sequence(seed, "motion_blur_angle", rgb, image_key))
    )
    return float(generator.uniform(0.0, 180.0))


def _gaussian_noise(
    image: Image.Image,
    sigma: float,
    *,
    seed: int,
    image_key: Hashable | None,
) -> Image.Image:
    pixels = np.asarray(image, dtype=np.uint8)
    generator = np.random.Generator(
        np.random.PCG64(_seed_sequence(seed, "gaussian_noise", image, image_key))
    )
    # Draw once from N(0,1); only the multiplier changes with severity.
    standard_noise = generator.standard_normal(pixels.shape, dtype=np.float32)
    corrupted = pixels.astype(np.float32) / 255.0 + standard_noise * float(sigma)
    result = np.clip(np.rint(corrupted * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")


def _gaussian_blur(image: Image.Image, kernel_size: int) -> Image.Image:
    pixels = np.asarray(image, dtype=np.uint8)
    result = cv2.GaussianBlur(
        pixels,
        (int(kernel_size), int(kernel_size)),
        sigmaX=0.0,
        sigmaY=0.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    return Image.fromarray(np.asarray(result, dtype=np.uint8), mode="RGB")


def motion_kernel(length: int, angle_degrees: float) -> np.ndarray:
    """Build a finite L1-normalized line-spread kernel."""

    if isinstance(length, bool) or not isinstance(length, (int, np.integer)):
        raise TypeError("motion kernel length must be an integer")
    length = int(length)
    if length < 3 or length % 2 == 0:
        raise CorruptionError("motion kernel length must be an odd integer >= 3")
    if not math.isfinite(float(angle_degrees)):
        raise CorruptionError("motion angle must be finite")

    kernel = np.zeros((length, length), dtype=np.float32)
    centre = (length - 1) / 2.0
    radius = centre
    radians = math.radians(float(angle_degrees))
    dx, dy = radius * math.cos(radians), radius * math.sin(radians)
    start = (int(round(centre - dx)), int(round(centre - dy)))
    end = (int(round(centre + dx)), int(round(centre + dy)))
    cv2.line(kernel, start, end, color=1.0, thickness=1, lineType=cv2.LINE_8)
    total = float(kernel.sum())
    if total <= 0.0:
        raise RuntimeError("motion kernel construction produced no support")
    kernel /= total
    return kernel


def _motion_blur(
    image: Image.Image,
    length: int,
    *,
    seed: int,
    image_key: Hashable | None,
) -> Image.Image:
    angle = deterministic_motion_angle(image, seed=seed, image_key=image_key)
    kernel = motion_kernel(length, angle)
    pixels = np.asarray(image, dtype=np.uint8)
    result = cv2.filter2D(
        pixels, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101
    )
    return Image.fromarray(np.asarray(result, dtype=np.uint8), mode="RGB")


def _jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=int(quality),
        subsampling=2,
        optimize=False,
        progressive=False,
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB").copy()


def apply_corruption(
    image: Image.Image,
    corruption_type: str,
    severity: int,
    *,
    seed: int = DEFAULT_SEED,
    image_key: Hashable | None = None,
) -> Image.Image:
    """Apply one fixed severity index to an image without mutating the input."""

    spec = corruption_spec(corruption_type, severity)
    rgb = prepare_rgb(image)
    if spec.corruption_type == "gaussian_noise":
        result = _gaussian_noise(
            rgb, float(spec.value), seed=seed, image_key=image_key
        )
    elif spec.corruption_type == "gaussian_blur":
        result = _gaussian_blur(rgb, int(spec.value))
    elif spec.corruption_type == "motion_blur":
        result = _motion_blur(
            rgb, int(spec.value), seed=seed, image_key=image_key
        )
    elif spec.corruption_type == "jpeg_compression":
        result = _jpeg_compression(rgb, int(spec.value))
    else:  # pragma: no cover - corruption_spec makes this unreachable.
        raise AssertionError(f"unhandled corruption {spec.corruption_type}")

    if result.mode != "RGB":
        result = result.convert("RGB")
    array = np.asarray(result)
    if array.dtype != np.uint8 or array.shape != (rgb.height, rgb.width, 3):
        raise RuntimeError(
            f"corruption returned invalid pixels: shape={array.shape}, dtype={array.dtype}"
        )
    return result.copy()


class CorruptionTransform:
    """Pickle-safe PIL transform for composition before model preprocessing."""

    def __init__(
        self,
        corruption_type: str,
        severity: int,
        *,
        seed: int = DEFAULT_SEED,
        image_key: Hashable | None = None,
    ) -> None:
        self.spec = corruption_spec(corruption_type, severity)
        self.seed = int(seed)
        self.image_key = image_key

    def __call__(self, image: Image.Image) -> Image.Image:
        return apply_corruption(
            image,
            self.spec.corruption_type,
            self.spec.severity_index,
            seed=self.seed,
            image_key=self.image_key,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(corruption_type={self.spec.corruption_type!r}, "
            f"severity={self.spec.severity_index}, seed={self.seed})"
        )


def protocol_corruption_matrix() -> dict[str, dict[str, object]]:
    """Return a fresh JSON-compatible copy of the exact protocol matrix."""

    return {
        name: {
            "parameter": CORRUPTION_PARAMETERS[name],
            "severities": list(values),
        }
        for name, values in CORRUPTION_SEVERITIES.items()
    }


__all__ = [
    "CORRUPTION_PARAMETERS",
    "CORRUPTION_SEVERITIES",
    "DEFAULT_SEED",
    "CorruptionError",
    "CorruptionSpec",
    "CorruptionTransform",
    "apply_corruption",
    "corruption_spec",
    "corruption_spec_from_value",
    "deterministic_motion_angle",
    "motion_kernel",
    "normalise_corruption_type",
    "prepare_rgb",
    "protocol_corruption_matrix",
]
