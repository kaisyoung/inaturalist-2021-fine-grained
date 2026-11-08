"""Manual, dependency-free Grad-CAM primitives for frozen Stage 3 CNNs.

This module deliberately contains no model-loading or artifact-writing logic.
It implements the mathematical Grad-CAM operation, the exact Stage 3
evaluation preprocessing geometry, and mapping a crop-space heatmap back to
the EXIF-oriented source image.  The command-line orchestration lives in
``run_gradcam.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor, nn
import torch.nn.functional as functional
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from src.deep_learning.transforms import (
    DEFAULT_EVAL_RESIZE_SIZE,
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


_SAFE_LAYER_TOKEN = re.compile(r"(?:[A-Za-z][A-Za-z0-9_]*|[0-9]+)\Z")


class GradCAMError(RuntimeError):
    """Raised when a CAM cannot be computed safely."""


@dataclass(frozen=True)
class EvalGeometry:
    """Geometry of the Stage 3 resize-and-centre-crop evaluation transform."""

    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    crop_left: int
    crop_top: int
    crop_width: int
    crop_height: int

    def __post_init__(self) -> None:
        values = (
            self.original_width,
            self.original_height,
            self.resized_width,
            self.resized_height,
            self.crop_width,
            self.crop_height,
        )
        if any(isinstance(value, bool) or int(value) <= 0 for value in values):
            raise ValueError("Geometry dimensions must be positive integers")
        if self.crop_left < 0 or self.crop_top < 0:
            raise ValueError("Crop offsets cannot be negative")
        if self.crop_left + self.crop_width > self.resized_width:
            raise ValueError("Crop extends beyond the resized image width")
        if self.crop_top + self.crop_height > self.resized_height:
            raise ValueError("Crop extends beyond the resized image height")


@dataclass(frozen=True)
class PreparedEvalImage:
    """EXIF-oriented image, model crop, normalized tensor, and geometry."""

    original_rgb: Image.Image
    model_crop_rgb: Image.Image
    tensor: Tensor
    geometry: EvalGeometry


@dataclass(frozen=True)
class MappedHeatmap:
    """A crop CAM mapped to source coordinates with an explicit visibility mask."""

    heatmap: np.ndarray
    crop_mask: np.ndarray


@dataclass(frozen=True)
class GradCAMResult:
    """Output of a single Grad-CAM forward/backward computation."""

    heatmaps: Tensor
    logits: Tensor
    probabilities: Tensor
    predicted_labels: Tensor
    target_labels: Tensor
    target_mode: str


def _module_by_dotted_path(model: nn.Module, path: str) -> nn.Module:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Target-layer path must be a non-empty dotted string")
    if path != path.strip() or path.startswith(".") or path.endswith("."):
        raise ValueError(f"Unsafe target-layer path: {path!r}")
    current: nn.Module = model
    for token in path.split("."):
        if not _SAFE_LAYER_TOKEN.fullmatch(token) or token.startswith("__"):
            raise ValueError(f"Unsafe target-layer token {token!r} in {path!r}")
        # Restrict traversal to registered child modules.  This prevents an
        # explicit path from reaching arbitrary object attributes or methods.
        child = current._modules.get(token)
        if not isinstance(child, nn.Module):
            raise ValueError(
                f"Target-layer path {path!r} does not identify a registered "
                f"module at token {token!r}"
            )
        current = child
    return current


def resolve_target_layer(
    model: nn.Module,
    target_layer: str | None = "auto",
) -> tuple[nn.Module, str]:
    """Resolve the canonical final convolutional block or a safe dotted path.

    Automatic resolution supports the torchvision architectures frozen in
    Stage 3: ``EfficientNet.features[-1]`` and ``ResNet.layer4[-1]``.  Explicit
    paths use registered-module dotted notation, for example ``features.8`` or
    ``layer4.1``; magic attributes and negative indexing are intentionally not
    accepted.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if target_layer not in (None, "auto"):
        module = _module_by_dotted_path(model, target_layer)
        return module, target_layer

    features = getattr(model, "features", None)
    if isinstance(features, nn.Module) and features._modules:
        key = next(reversed(features._modules))
        module = features._modules[key]
        if isinstance(module, nn.Module):
            return module, f"features.{key}"

    layer4 = getattr(model, "layer4", None)
    if isinstance(layer4, nn.Module) and layer4._modules:
        key = next(reversed(layer4._modules))
        module = layer4._modules[key]
        if isinstance(module, nn.Module):
            return module, f"layer4.{key}"

    raise ValueError(
        "Could not identify a supported final convolutional block; supply a "
        "safe registered-module dotted path"
    )


def _normalise_targets(
    logits: Tensor,
    *,
    target_mode: str,
    true_labels: Tensor | Sequence[int] | int | None,
) -> tuple[Tensor, Tensor]:
    if logits.ndim != 2 or logits.shape[0] <= 0 or logits.shape[1] <= 1:
        raise GradCAMError(
            f"Classifier logits must have shape batch x classes, got {tuple(logits.shape)}"
        )
    if not bool(torch.isfinite(logits).all()):
        raise GradCAMError("Classifier produced NaN or infinite logits")
    predicted = torch.argmax(logits, dim=1)
    if target_mode == "predicted":
        return predicted, predicted
    if target_mode != "true":
        raise ValueError("target_mode must be 'predicted' or 'true'")
    if true_labels is None:
        raise ValueError("true_labels are required when target_mode='true'")
    targets = torch.as_tensor(true_labels, dtype=torch.long, device=logits.device)
    if targets.ndim == 0:
        targets = targets.repeat(logits.shape[0])
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("true_labels must contain exactly one label per input")
    if bool(((targets < 0) | (targets >= logits.shape[1])).any()):
        raise ValueError("true_labels contain an out-of-range class index")
    return targets, predicted


class GradCAM:
    """Compute Grad-CAM using captured activations and their score gradients.

    The implementation uses :func:`torch.autograd.grad`, so model parameter
    ``.grad`` buffers are not created, cleared, or changed.  No optimizer is
    accepted and no parameter update is possible.  The model is put in
    evaluation mode and remains there for the lifetime of this object.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str | None = "auto",
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.model.eval()
        self.target_layer, self.target_layer_name = resolve_target_layer(
            model, target_layer
        )
        self._activation: Tensor | None = None
        self._gradient: Tensor | None = None
        self._hook = self.target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        del module, inputs
        if not isinstance(output, Tensor):
            raise GradCAMError("Target layer must return one torch.Tensor")
        self._activation = output
        self._gradient = None

        def capture_gradient(gradient: Tensor) -> Tensor:
            self._gradient = gradient
            return gradient

        if not output.requires_grad:
            raise GradCAMError(
                "Target-layer activations do not require gradients; Grad-CAM "
                "cannot run under inference_mode/no_grad or a detached model"
            )
        output.register_hook(capture_gradient)

    def close(self) -> None:
        """Remove the forward hook; safe to call more than once."""

        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def generate(
        self,
        input_tensor: Tensor,
        *,
        target_mode: str = "predicted",
        true_labels: Tensor | Sequence[int] | int | None = None,
    ) -> GradCAMResult:
        """Return a finite, normalized input-size CAM for each input image."""

        if not isinstance(input_tensor, Tensor):
            raise TypeError("input_tensor must be a torch.Tensor")
        if input_tensor.ndim != 4 or input_tensor.shape[0] <= 0:
            raise ValueError("input_tensor must have shape batch x channels x height x width")
        if not input_tensor.is_floating_point():
            raise TypeError("input_tensor must use a floating-point dtype")
        if not bool(torch.isfinite(input_tensor).all()):
            raise ValueError("input_tensor contains NaN or infinite values")

        self._activation = None
        self._gradient = None
        # Gradients must be enabled even if a caller has an outer no_grad
        # context.  Only the chosen class score is differentiated.
        with torch.enable_grad():
            logits = self.model(input_tensor)
            targets, predicted = _normalise_targets(
                logits, target_mode=target_mode, true_labels=true_labels
            )
            activation = self._activation
            if activation is None:
                raise GradCAMError("Target-layer hook did not capture activations")
            if activation.ndim != 4:
                raise GradCAMError(
                    "Target-layer activation must be BxCxHxW, got "
                    f"{tuple(activation.shape)}"
                )
            if activation.shape[0] != input_tensor.shape[0]:
                raise GradCAMError("Target-layer activation batch size changed")
            scores = logits.gather(1, targets[:, None]).sum()
            gradient = torch.autograd.grad(
                scores,
                activation,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
            # The tensor hook must observe exactly the gradient returned by
            # autograd.  Keeping this invariant catches unsupported hook paths.
            captured = self._gradient
            if captured is None or captured.shape != gradient.shape:
                raise GradCAMError("Target-layer gradient hook did not fire correctly")
            weights = captured.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            cam = functional.interpolate(
                cam,
                size=input_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            flat = cam.flatten(start_dim=1)
            minimum = flat.min(dim=1).values[:, None, None]
            maximum = flat.max(dim=1).values[:, None, None]
            span = maximum - minimum
            cam = torch.where(span > 0, (cam - minimum) / span.clamp_min(1e-12), torch.zeros_like(cam))
            if not bool(torch.isfinite(cam).all()):
                raise GradCAMError("Grad-CAM heatmap contains NaN or infinite values")
            cam = cam.clamp_(0.0, 1.0)
            probabilities = torch.softmax(logits.detach(), dim=1)

        return GradCAMResult(
            heatmaps=cam.detach(),
            logits=logits.detach(),
            probabilities=probabilities,
            predicted_labels=predicted.detach(),
            target_labels=targets.detach(),
            target_mode=target_mode,
        )


def prepare_stage3_eval_image(
    image: Image.Image,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> PreparedEvalImage:
    """Apply the exact Stage 3 eval transform while recording its geometry."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    original = ImageOps.exif_transpose(image).convert("RGB")
    original.load()
    # Match src.deep_learning.transforms._eval_resize_size without importing a
    # private helper: canonical 224 -> 256, miniature tests preserve the ratio.
    resize_size = max(
        image_size,
        round(image_size * DEFAULT_EVAL_RESIZE_SIZE / DEFAULT_IMAGE_SIZE),
    )
    resized = transform_functional.resize(
        original,
        resize_size,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    resized_width, resized_height = resized.size
    crop_left = int(round((resized_width - image_size) / 2.0))
    crop_top = int(round((resized_height - image_size) / 2.0))
    crop = transform_functional.crop(
        resized, crop_top, crop_left, image_size, image_size
    )
    tensor = transform_functional.to_tensor(crop)
    tensor = transform_functional.normalize(
        tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD
    )
    geometry = EvalGeometry(
        original_width=original.width,
        original_height=original.height,
        resized_width=resized_width,
        resized_height=resized_height,
        crop_left=crop_left,
        crop_top=crop_top,
        crop_width=image_size,
        crop_height=image_size,
    )
    return PreparedEvalImage(original, crop, tensor, geometry)


def map_crop_heatmap_to_original(
    crop_heatmap: np.ndarray | Tensor,
    geometry: EvalGeometry,
) -> MappedHeatmap:
    """Map a crop-space CAM to the full source image.

    A nearest-neighbour visibility mask keeps every pixel outside the model's
    centre crop exactly transparent.  Heatmap interpolation is bilinear.
    """

    if isinstance(crop_heatmap, Tensor):
        values = crop_heatmap.detach().cpu().numpy()
    else:
        values = np.asarray(crop_heatmap)
    if values.shape != (geometry.crop_height, geometry.crop_width):
        raise ValueError(
            "crop_heatmap shape must match the recorded model crop: expected "
            f"{(geometry.crop_height, geometry.crop_width)}, got {values.shape}"
        )
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("crop_heatmap contains NaN or infinite values")
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError("crop_heatmap values must lie in [0, 1]")

    full = np.zeros((geometry.resized_height, geometry.resized_width), dtype=np.float32)
    mask = np.zeros_like(full, dtype=np.uint8)
    top, left = geometry.crop_top, geometry.crop_left
    bottom, right = top + geometry.crop_height, left + geometry.crop_width
    full[top:bottom, left:right] = values
    mask[top:bottom, left:right] = 255

    heatmap_image = Image.fromarray(full, mode="F").resize(
        (geometry.original_width, geometry.original_height),
        resample=Image.Resampling.BILINEAR,
    )
    mask_image = Image.fromarray(mask, mode="L").resize(
        (geometry.original_width, geometry.original_height),
        resample=Image.Resampling.NEAREST,
    )
    mapped_mask = np.asarray(mask_image, dtype=np.float32) / 255.0
    mapped = np.asarray(heatmap_image, dtype=np.float32)
    mapped = np.clip(mapped, 0.0, 1.0) * mapped_mask
    if not np.isfinite(mapped).all():
        raise GradCAMError("Mapped Grad-CAM heatmap is not finite")
    return MappedHeatmap(mapped, mapped_mask)


__all__ = [
    "EvalGeometry",
    "GradCAM",
    "GradCAMError",
    "GradCAMResult",
    "MappedHeatmap",
    "PreparedEvalImage",
    "map_crop_heatmap_to_original",
    "prepare_stage3_eval_image",
    "resolve_target_layer",
]
