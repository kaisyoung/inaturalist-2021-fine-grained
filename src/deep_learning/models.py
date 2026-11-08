"""Torchvision model factories for the Stage 3 deep-learning baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

from src.data.common import NUM_CLASSES, sha256_file


_MODEL_ALIASES: Final[dict[str, str]] = {
    "resnet18": "resnet18",
    "resnet_18": "resnet18",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-b0": "efficientnet_b0",
    "efficientnetb0": "efficientnet_b0",
}


class ModelCreationError(RuntimeError):
    """Raised when a requested model or required pretrained weights cannot load."""


@dataclass(frozen=True)
class ModelMetadata:
    """Serializable model construction provenance returned by ``create_model``."""

    architecture: str
    num_classes: int
    pretrained: bool
    freeze_backbone: bool
    weights_enum: str | None
    weights_url: str | None
    weight_file_name: str | None
    weight_file_sha256: str | None
    classifier_path: str
    classifier_in_features: int
    total_parameters: int
    trainable_parameters: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_num_classes(num_classes: int) -> int:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return num_classes


def _normalise_model_name(model_name: str) -> str:
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    key = model_name.strip().lower()
    try:
        return _MODEL_ALIASES[key]
    except KeyError as error:
        raise ValueError(
            f"Unsupported model {model_name!r}; expected 'resnet18' or "
            "'efficientnet_b0'"
        ) from error


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return exact total and trainable parameter counts."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def _freeze_except_classifier(model: nn.Module, classifier: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classifier.parameters():
        parameter.requires_grad = True
    # Private markers are deliberately simple values so checkpoints do not
    # depend on pickling custom wrapper classes.
    model._stage3_backbone_frozen = True  # type: ignore[attr-defined]
    model._stage3_classifier = classifier  # type: ignore[attr-defined]
    set_frozen_backbone_eval(model)


def set_frozen_backbone_eval(model: nn.Module) -> nn.Module:
    """Keep a frozen backbone, including every BatchNorm layer, in eval mode.

    Training loops should call this immediately after ``model.train()``.  The
    classifier remains in training mode, while all other direct model branches
    are switched to evaluation mode.  For an unfrozen model this is a no-op.
    """

    if not getattr(model, "_stage3_backbone_frozen", False):
        return model
    classifier = getattr(model, "_stage3_classifier", None)
    if not isinstance(classifier, nn.Module):
        raise ModelCreationError(
            "Frozen model is missing its Stage 3 classifier marker"
        )
    classifier.train(model.training)
    for child in model.children():
        if child is classifier:
            continue
        child.eval()
    return model


def create_model(
    model_name: str,
    *,
    pretrained: bool,
    freeze_backbone: bool = False,
    num_classes: int = NUM_CLASSES,
) -> tuple[nn.Module, dict[str, object]]:
    """Create a ResNet18 or EfficientNet-B0 with a new classification head.

    Pretrained construction always passes an explicit torchvision weight enum.
    Any cache/download failure is raised as :class:`ModelCreationError`; random
    weights are never substituted.  Unit tests should pass ``pretrained=False``
    and therefore never access the network.
    """

    architecture = _normalise_model_name(model_name)
    classes = _validate_num_classes(num_classes)
    if not isinstance(pretrained, bool):
        raise TypeError("pretrained must be a bool")
    if not isinstance(freeze_backbone, bool):
        raise TypeError("freeze_backbone must be a bool")
    if freeze_backbone and not pretrained:
        # This is almost certainly a configuration error and is not one of the
        # approved experiments.
        raise ValueError("freeze_backbone=True requires pretrained=True")

    weights_enum: ResNet18_Weights | EfficientNet_B0_Weights | None
    weights_name: str | None
    weights_url: str | None
    try:
        if architecture == "resnet18":
            weights_enum = (
                ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            model = resnet18(weights=weights_enum)
            in_features = int(model.fc.in_features)
            model.fc = nn.Linear(in_features, classes)
            classifier = model.fc
            classifier_path = "fc"
        else:
            weights_enum = (
                EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            )
            model = efficientnet_b0(weights=weights_enum)
            final_layer = model.classifier[-1]
            if not isinstance(final_layer, nn.Linear):
                raise ModelCreationError(
                    "torchvision EfficientNet-B0 classifier has an unexpected shape"
                )
            in_features = int(final_layer.in_features)
            model.classifier[-1] = nn.Linear(in_features, classes)
            classifier = model.classifier
            classifier_path = "classifier"
    except ModelCreationError:
        raise
    except Exception as error:
        if pretrained:
            raise ModelCreationError(
                f"Could not load required ImageNet weights for {architecture}; "
                "pretrained runs may not fall back to random weights: "
                f"{error}"
            ) from error
        raise ModelCreationError(f"Could not construct {architecture}: {error}") from error

    if weights_enum is None:
        weights_name = None
        weights_url = None
        weight_file_name = None
        weight_file_sha256 = None
    else:
        weights_name = f"{type(weights_enum).__name__}.{weights_enum.name}"
        weights_url = str(weights_enum.url)
        weight_file_name = Path(urlparse(weights_url).path).name
        cached_weight_path = (
            Path(torch.hub.get_dir()) / "checkpoints" / weight_file_name
        )
        if not cached_weight_path.is_file():
            raise ModelCreationError(
                f"Required pretrained weights loaded for {architecture}, but the "
                f"expected cache artifact is absent: {cached_weight_path}"
            )
        try:
            weight_file_sha256 = sha256_file(cached_weight_path)
        except OSError as error:
            raise ModelCreationError(
                f"Could not hash required pretrained weights "
                f"{cached_weight_path}: {error}"
            ) from error

    model._stage3_architecture = architecture  # type: ignore[attr-defined]
    model._stage3_classifier = classifier  # type: ignore[attr-defined]
    model._stage3_backbone_frozen = False  # type: ignore[attr-defined]
    if freeze_backbone:
        _freeze_except_classifier(model, classifier)

    counts = count_parameters(model)
    metadata = ModelMetadata(
        architecture=architecture,
        num_classes=classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        weights_enum=weights_name,
        weights_url=weights_url,
        weight_file_name=weight_file_name,
        weight_file_sha256=weight_file_sha256,
        classifier_path=classifier_path,
        classifier_in_features=in_features,
        total_parameters=counts["total"],
        trainable_parameters=counts["trainable"],
    )
    return model, metadata.to_dict()


__all__ = [
    "ModelCreationError",
    "ModelMetadata",
    "count_parameters",
    "create_model",
    "set_frozen_backbone_eval",
]
