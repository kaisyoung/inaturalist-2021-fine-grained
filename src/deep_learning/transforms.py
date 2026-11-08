"""Deterministic transform preset definitions for Stage 3.

The training presets are stochastic by design, but every operation draws from
PyTorch's seeded RNG.  Re-seeding PyTorch and data-loader workers therefore
reproduces the same augmentation stream.
"""

from __future__ import annotations

from typing import Final

from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)
DEFAULT_IMAGE_SIZE: Final[int] = 224
DEFAULT_EVAL_RESIZE_SIZE: Final[int] = 256

_PRESET_ALIASES = {
    "eval": "eval",
    "basic": "basic_train",
    "basic_train": "basic_train",
    "strong": "strong_train",
    "strong_train": "strong_train",
}


def _validate_image_size(image_size: int) -> int:
    if isinstance(image_size, bool) or not isinstance(image_size, int):
        raise TypeError("image_size must be an integer")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return image_size


def _eval_resize_size(image_size: int) -> int:
    # This is exactly 256 for the canonical 224-pixel configuration while
    # preserving the same crop ratio for miniature transform tests.
    return max(image_size, round(image_size * DEFAULT_EVAL_RESIZE_SIZE / DEFAULT_IMAGE_SIZE))


def get_transform(
    preset: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """Build one immutable Stage 3 transform preset.

    ``basic``/``strong`` are accepted aliases for the canonical YAML values
    ``basic_train``/``strong_train``.  No preset contains vertical flipping or
    an aggressive crop below the approved scale range.
    """

    size = _validate_image_size(image_size)
    if not isinstance(preset, str):
        raise TypeError("preset must be a string")
    try:
        canonical = _PRESET_ALIASES[preset.strip().lower()]
    except KeyError as error:
        raise ValueError(
            f"Unknown transform preset {preset!r}; expected one of "
            f"{sorted(_PRESET_ALIASES)}"
        ) from error

    normalise = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if canonical == "eval":
        return transforms.Compose(
            [
                transforms.Resize(
                    _eval_resize_size(size),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalise,
            ]
        )

    scale = (0.7, 1.0) if canonical == "basic_train" else (0.5, 1.0)
    operations: list[object] = [
        transforms.RandomResizedCrop(
            size,
            scale=scale,
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if canonical == "strong_train":
        operations.extend(
            [
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.RandomAffine(
                    degrees=15,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    shear=(-5.0, 5.0),
                    interpolation=InterpolationMode.BICUBIC,
                    fill=0,
                ),
            ]
        )
    operations.extend([transforms.ToTensor(), normalise])
    if canonical == "strong_train":
        operations.append(
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.20),
                ratio=(0.3, 3.3),
                value=0,
                inplace=False,
            )
        )
    return transforms.Compose(operations)


# Concise alias used by some configuration-oriented callers.
build_transform = get_transform


__all__ = [
    "DEFAULT_EVAL_RESIZE_SIZE",
    "DEFAULT_IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_transform",
    "get_transform",
]
