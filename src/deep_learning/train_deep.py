"""Train reproducible Stage 3 deep-learning experiments.

Only ``train.csv`` and ``val.csv`` are opened by this module. Held-out test
access is exclusively implemented by :mod:`src.deep_learning.evaluate_deep`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from tqdm import tqdm

from src.analysis.metrics import evaluate_aligned_scores
from src.data.common import NUM_CLASSES, sha256_file
from src.deep_learning.dataset import ManifestImageDataset

# Keep downloaded ImageNet weights inside the ignored project artifact tree.
os.environ.setdefault("TORCH_HOME", "results/models/torch_cache")

from src.deep_learning.models import (
    count_parameters,
    create_model,
    set_frozen_backbone_eval,
)
from src.deep_learning.train_utils import (
    amp_enabled_for_device,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_jsonl,
    autocast_context,
    capture_rng_state,
    checkpoint_selection_key,
    clear_device_cache,
    is_out_of_memory_error,
    load_checkpoint,
    make_grad_scaler,
    package_versions,
    restore_rng_state,
    seed_everything,
    seed_worker,
    select_device,
    synchronize_device,
    verify_stage1_artifacts,
)
from src.deep_learning.transforms import get_transform


CONFIG_KEYS = {
    "schema_version",
    "experiment_name",
    "model_name",
    "pretrained",
    "freeze_backbone",
    "num_classes",
    "image_size",
    "batch_size",
    "epochs",
    "min_epochs",
    "patience",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "scheduler",
    "augmentation",
    "gradient_clip_norm",
    "seed",
    "num_workers",
    "device",
    "amp",
    "preflight_batches",
    "split_dir",
    "image_root",
    "model_dir",
    "log_dir",
    "metrics_dir",
    "figure_dir",
}

REQUIRED_CONFIG_KEYS = CONFIG_KEYS - {"schema_version"}

CANONICAL_EXPERIMENT_ORDER = (
    "resnet18_scratch_basic",
    "resnet18_pretrained_basic",
    "resnet18_pretrained_strong_aug",
    "efficientnet_b0_pretrained_strong_aug",
    "resnet18_pretrained_frozen_head",
)
REQUIRED_EXPERIMENTS = frozenset(CANONICAL_EXPERIMENT_ORDER[:3])
OPTIONAL_EXPERIMENTS = frozenset(CANONICAL_EXPERIMENT_ORDER[3:])
TOTAL_TRAINING_GATE_SECONDS = 12 * 60 * 60
MIN_FREE_DISK_BYTES = 20 * 1024**3
SAFE_EXPERIMENT_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*(?:__debug)?$")

_COMMON_CANONICAL_SPEC: dict[str, Any] = {
    "num_classes": 500,
    "image_size": 224,
    "batch_size": 32,
    "optimizer": "adamw",
    "weight_decay": 1e-4,
    "scheduler": "cosine",
    "gradient_clip_norm": 1.0,
    "seed": 42,
    "num_workers": 4,
    "amp": "auto",
    "preflight_batches": 10,
    "split_dir": "data/splits",
    "image_root": "data/processed/images",
    "model_dir": "results/models/deep_learning",
    "log_dir": "results/logs/deep_learning",
    "metrics_dir": "results/metrics/deep_learning",
    "figure_dir": "results/figures/deep_learning",
}
CANONICAL_EXPERIMENT_SPECS: dict[str, dict[str, Any]] = {
    "resnet18_scratch_basic": {
        **_COMMON_CANONICAL_SPEC,
        "model_name": "resnet18",
        "pretrained": False,
        "freeze_backbone": False,
        "epochs": 25,
        "min_epochs": 10,
        "patience": 6,
        "learning_rate": 1e-3,
        "augmentation": "basic_train",
    },
    "resnet18_pretrained_basic": {
        **_COMMON_CANONICAL_SPEC,
        "model_name": "resnet18",
        "pretrained": True,
        "freeze_backbone": False,
        "epochs": 15,
        "min_epochs": 5,
        "patience": 4,
        "learning_rate": 1e-4,
        "augmentation": "basic_train",
    },
    "resnet18_pretrained_strong_aug": {
        **_COMMON_CANONICAL_SPEC,
        "model_name": "resnet18",
        "pretrained": True,
        "freeze_backbone": False,
        "epochs": 15,
        "min_epochs": 5,
        "patience": 4,
        "learning_rate": 1e-4,
        "augmentation": "strong_train",
    },
    "efficientnet_b0_pretrained_strong_aug": {
        **_COMMON_CANONICAL_SPEC,
        "model_name": "efficientnet_b0",
        "pretrained": True,
        "freeze_backbone": False,
        "epochs": 15,
        "min_epochs": 5,
        "patience": 4,
        "learning_rate": 1e-4,
        "augmentation": "strong_train",
    },
    "resnet18_pretrained_frozen_head": {
        **_COMMON_CANONICAL_SPEC,
        "model_name": "resnet18",
        "pretrained": True,
        "freeze_backbone": True,
        "epochs": 10,
        "min_epochs": 3,
        "patience": 3,
        "learning_rate": 1e-3,
        "augmentation": "basic_train",
    },
}


class EpochSeededDataset(Dataset):
    """Make stochastic transforms a pure function of seed, epoch, and row index.

    The sampler sends ``(epoch, index)`` tokens to persistent workers, so no
    mutable worker-local epoch state is involved. RNG states are restored after
    each item so augmentation cannot perturb sampler or model RNG streams.
    """

    def __init__(self, dataset: Dataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int | tuple[int, int]):
        if isinstance(index, tuple):
            epoch, row_index = (int(index[0]), int(index[1]))
        else:
            epoch, row_index = self.epoch, int(index)
        digest = hashlib.sha256(
            f"stage3-augmentation:{self.seed}:{epoch}:{row_index}".encode(
                "ascii"
            )
        ).digest()
        seed64 = int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)
        seed32 = seed64 % (2**32)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(seed64)
            np.random.seed(seed32)
            cpu_generator = torch.Generator(device="cpu").manual_seed(seed64)
            torch.set_rng_state(cpu_generator.get_state())
            return self.dataset[row_index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


class EpochRandomSampler(RandomSampler):
    """Attach the main-process epoch to indices sent to persistent workers."""

    def __iter__(self):
        epoch = int(self.data_source.epoch)
        for index in super().__iter__():
            yield (epoch, int(index))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name", choices=("resnet18", "efficientnet_b0"))
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--augmentation", choices=("basic_train", "strong_train")
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"))
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def load_config(path: Path, overrides: argparse.Namespace | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a YAML mapping")
    unknown = sorted(set(value) - CONFIG_KEYS)
    missing = sorted(REQUIRED_CONFIG_KEYS - set(value))
    if unknown:
        raise ValueError(f"unknown configuration keys: {unknown}")
    if missing:
        raise ValueError(f"missing configuration keys: {missing}")
    config = copy.deepcopy(value)
    if int(config.get("schema_version", 1)) != 1:
        raise ValueError("only Stage 3 configuration schema version 1 is supported")
    if overrides is not None:
        mapping = {
            "model_name": "model_name",
            "pretrained": "pretrained",
            "freeze_backbone": "freeze_backbone",
            "augmentation": "augmentation",
            "batch_size": "batch_size",
            "epochs": "epochs",
            "learning_rate": "learning_rate",
            "optimizer": "optimizer",
            "weight_decay": "weight_decay",
            "num_workers": "num_workers",
            "image_size": "image_size",
            "seed": "seed",
            "device": "device",
        }
        for argument, key in mapping.items():
            candidate = getattr(overrides, argument, None)
            if candidate is not None:
                config[key] = candidate
        if overrides.debug:
            config["experiment_name"] = f"{config['experiment_name']}__debug"
            config["batch_size"] = min(2, int(config["batch_size"]))
            config["epochs"] = 1
            config["min_epochs"] = 1
            config["patience"] = 1
            config["num_workers"] = 0
            config["preflight_batches"] = 1
    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    experiment = config.get("experiment_name")
    if not isinstance(experiment, str) or not SAFE_EXPERIMENT_NAME.fullmatch(experiment):
        raise ValueError("experiment_name must be a safe lowercase identifier")
    if config["num_classes"] != NUM_CLASSES:
        raise ValueError(f"num_classes must remain {NUM_CLASSES}")
    if config["model_name"] not in {"resnet18", "efficientnet_b0"}:
        raise ValueError("unsupported model_name")
    if config["augmentation"] not in {"basic_train", "strong_train"}:
        raise ValueError("training augmentation must be basic_train or strong_train")
    if config["optimizer"] not in {"adamw", "sgd"}:
        raise ValueError("optimizer must be adamw or sgd")
    if config["scheduler"] != "cosine":
        raise ValueError("Stage 3 canonical runs require the cosine scheduler")
    integer_positive = (
        "image_size",
        "batch_size",
        "epochs",
        "min_epochs",
        "patience",
        "preflight_batches",
    )
    for key in integer_positive:
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(config["min_epochs"]) > int(config["epochs"]):
        raise ValueError("min_epochs cannot exceed epochs")
    if int(config["num_workers"]) < 0:
        raise ValueError("num_workers cannot be negative")
    for key in ("learning_rate", "weight_decay", "gradient_clip_norm"):
        if float(config[key]) < 0:
            raise ValueError(f"{key} cannot be negative")
    for key in ("split_dir", "image_root", "model_dir", "log_dir", "metrics_dir", "figure_dir"):
        candidate = Path(str(config[key]))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"{key} must be a safe project-relative path")

    debug = experiment.endswith("__debug")
    canonical_name = experiment.removesuffix("__debug") if debug else experiment
    if canonical_name not in CANONICAL_EXPERIMENT_SPECS:
        raise ValueError(f"unknown canonical Stage 3 experiment_name: {experiment}")
    expected = CANONICAL_EXPERIMENT_SPECS[canonical_name]
    debug_overrides = {
        "batch_size",
        "epochs",
        "min_epochs",
        "patience",
        "num_workers",
        "preflight_batches",
    }
    mismatches = {
        key: {"expected": value, "observed": config.get(key)}
        for key, value in expected.items()
        if (not debug or key not in debug_overrides) and config.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{experiment} violates its fixed canonical experiment spec: {mismatches}"
        )
    if not debug and str(config["device"]) not in {"auto", "mps"}:
        raise ValueError("canonical local Stage 3 runs must resolve through auto or MPS")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_json_document(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _training_implementation_hashes() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    paths = {
        "train_deep.py": Path(__file__).resolve(),
        "train_utils.py": directory / "train_utils.py",
        "dataset.py": directory / "dataset.py",
        "transforms.py": directory / "transforms.py",
        "models.py": directory / "models.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def model_state_sha256(model: nn.Module) -> str:
    """Hash names, dtypes, shapes, and exact initial tensor bytes."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def sampler_order_sha256(dataset_size: int, seed: int) -> str:
    """Hash the epoch-1 RandomSampler order without mutating its live generator."""

    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(int(dataset_size), generator=generator, dtype=torch.int64)
    return hashlib.sha256(order.numpy().tobytes(order="C")).hexdigest()


def _ablation_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"experiment_name", "augmentation"}
    return {
        key: _json_safe(config[key])
        for key in sorted(CONFIG_KEYS - excluded)
        if key in config
    }


def _ablation_provenance(
    config: Mapping[str, Any],
    *,
    initial_model_hash: str,
    dataset_size: int,
) -> dict[str, Any]:
    return {
        "policy": "D2 variants differ only in training transform",
        "comparable_pair": list(CANONICAL_EXPERIMENT_ORDER[1:3]),
        "contract_excluding_experiment_and_augmentation": _ablation_contract(config),
        "contract_sha256": _stable_json_hash(_ablation_contract(config)),
        "initial_model_state_sha256": initial_model_hash,
        "sampler_algorithm": "torch.randperm via RandomSampler with isolated generator",
        "sampler_seed": int(config["seed"]),
        "epoch_1_sampler_order_sha256": sampler_order_sha256(
            dataset_size, int(config["seed"])
        ),
        "dataset_size": int(dataset_size),
    }


def _verify_existing_d2_peer(
    experiment: str,
    provenance: Mapping[str, Any],
    metrics_dir: Path,
) -> dict[str, Any]:
    if experiment not in CANONICAL_EXPERIMENT_ORDER[1:3]:
        return {"status": "not_applicable"}
    peer = (
        CANONICAL_EXPERIMENT_ORDER[2]
        if experiment == CANONICAL_EXPERIMENT_ORDER[1]
        else CANONICAL_EXPERIMENT_ORDER[1]
    )
    peer_path = metrics_dir / f"{peer}_resolved_config.json"
    if not peer_path.is_file():
        return {"status": "peer_not_yet_available", "peer": peer}
    peer_document = _read_json_document(peer_path)
    peer_provenance = peer_document.get("d2_ablation_provenance")
    if not isinstance(peer_provenance, Mapping):
        raise RuntimeError(f"D2 peer lacks ablation provenance: {peer_path}")
    keys = (
        "contract_sha256",
        "initial_model_state_sha256",
        "sampler_seed",
        "epoch_1_sampler_order_sha256",
        "dataset_size",
    )
    mismatches = {
        key: {"current": provenance.get(key), "peer": peer_provenance.get(key)}
        for key in keys
        if provenance.get(key) != peer_provenance.get(key)
    }
    if mismatches:
        raise RuntimeError(f"D2 strict-ablation provenance mismatch: {mismatches}")
    if peer_document.get("augmentation") == str(
        CANONICAL_EXPERIMENT_SPECS[experiment]["augmentation"]
    ):
        raise RuntimeError("D2 strict ablation must use different training transforms")
    return {
        "status": "verified_equal_except_augmentation",
        "peer": peer,
        "peer_resolved_config_sha256": sha256_file(peer_path),
    }


def _status_document(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _read_json_document(path)
    except ValueError:
        return {}


def _compute_suite_gate(
    config: Mapping[str, Any],
    *,
    current_projection_seconds: float,
    phase: str,
) -> tuple[dict[str, Any], Path]:
    """Persist and return the priority, 12-hour, and free-disk decision."""

    experiment = str(config["experiment_name"])
    metrics_dir = Path(str(config["metrics_dir"]))
    gate_path = metrics_dir / "stage3_compute_gate.json"
    gate = _status_document(gate_path)
    entries = gate.get("experiments")
    if not isinstance(entries, dict):
        entries = {}

    index = CANONICAL_EXPERIMENT_ORDER.index(experiment)
    prior = list(CANONICAL_EXPERIMENT_ORDER[:index])
    statuses: dict[str, dict[str, Any]] = {}
    for name in CANONICAL_EXPERIMENT_ORDER:
        statuses[name] = _status_document(metrics_dir / f"{name}_status.json")

    incomplete_prior = [
        name
        for name in prior
        if statuses[name].get("status") not in {"COMPLETE", "NOT RUN", "FAILED"}
    ]
    required_not_complete = [
        name
        for name in prior
        if name in REQUIRED_EXPERIMENTS and statuses[name].get("status") != "COMPLETE"
    ]
    # D1/D2 are mandatory and ordered. Optional experiments may follow a terminal
    # optional predecessor, but never an incomplete/absent one.
    order_ok = not incomplete_prior and not required_not_complete

    committed_seconds = 0.0
    committed: dict[str, float] = {}
    for name in prior:
        status = statuses[name]
        if status.get("status") == "COMPLETE":
            seconds = float(status.get("training_time_seconds", 0.0))
            committed[name] = seconds
            committed_seconds += seconds
    projected_total = committed_seconds + float(current_projection_seconds)
    free_bytes = int(shutil.disk_usage(Path.cwd()).free)
    time_ok = projected_total <= TOTAL_TRAINING_GATE_SECONDS
    disk_ok = free_bytes >= MIN_FREE_DISK_BYTES
    passed = order_ok and time_ok and disk_ok
    reasons: list[str] = []
    if incomplete_prior:
        reasons.append(f"prior experiments have no terminal decision: {incomplete_prior}")
    if required_not_complete:
        reasons.append(f"required prior experiments are incomplete: {required_not_complete}")
    if not time_ok:
        reasons.append(
            f"projected cumulative training {projected_total:.3f}s exceeds "
            f"{TOTAL_TRAINING_GATE_SECONDS}s"
        )
    if not disk_ok:
        reasons.append(
            f"free disk {free_bytes} bytes is below {MIN_FREE_DISK_BYTES} bytes"
        )
    decision = {
        "status": "PASS" if passed else "NOT RUN",
        "phase": phase,
        "experiment_name": experiment,
        "priority_index": index,
        "prior_experiments": prior,
        "committed_training_seconds": committed,
        "committed_total_seconds": committed_seconds,
        "current_projected_training_seconds": float(current_projection_seconds),
        "projected_cumulative_training_seconds": projected_total,
        "training_limit_seconds": TOTAL_TRAINING_GATE_SECONDS,
        "free_disk_bytes": free_bytes,
        "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
        "checks": {"priority": order_ok, "time": time_ok, "disk": disk_ok},
        "reasons": reasons,
    }
    entries[experiment] = decision
    document = {
        "schema_version": 1,
        "policy": "D1 -> D2 basic -> D2 strong -> D3 -> frozen head; 12h training and 20 GiB free",
        "training_limit_seconds": TOTAL_TRAINING_GATE_SECONDS,
        "minimum_free_disk_bytes": MIN_FREE_DISK_BYTES,
        "experiments": entries,
    }
    atomic_write_json(gate_path, document)
    return decision, gate_path


def _validate_resume_before_write(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    stage1: Mapping[str, Any],
    source_config_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    checkpoint = load_checkpoint(Path(args.resume), map_location="cpu")
    required = {
        "checkpoint_kind",
        "epoch",
        "experiment_name",
        "model_name",
        "pretrained",
        "freeze_backbone",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "rng_state",
        "sampler_generator_state",
        "history",
        "resolved_config",
        "resolved_config_sha256",
        "source_config_sha256",
        "stage1_artifacts",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"resume checkpoint is incomplete: {missing}")
    if checkpoint.get("checkpoint_kind") != "last":
        raise RuntimeError("resume accepts only an epoch-boundary last checkpoint")
    if checkpoint.get("experiment_name") != config["experiment_name"]:
        raise RuntimeError("resume checkpoint experiment mismatch")
    if checkpoint.get("source_config_sha256") != source_config_hash:
        raise RuntimeError("resume checkpoint source-config hash mismatch")
    if checkpoint.get("stage1_artifacts") != stage1:
        raise RuntimeError("resume checkpoint Stage 1 provenance mismatch")
    identity = {
        "model_name": config["model_name"],
        "pretrained": bool(config["pretrained"]),
        "freeze_backbone": bool(config["freeze_backbone"]),
    }
    for key, expected in identity.items():
        if checkpoint.get(key) != expected:
            raise RuntimeError(f"resume checkpoint model mismatch: {key}")

    resolved_path = Path(paths["resolved_config"])
    if not resolved_path.is_file():
        raise RuntimeError(f"resume resolved config is missing: {resolved_path}")
    resolved_hash = sha256_file(resolved_path)
    resolved = _read_json_document(resolved_path)
    if (
        checkpoint.get("resolved_config_sha256") != resolved_hash
        or checkpoint.get("resolved_config") != resolved
    ):
        raise RuntimeError("resume checkpoint resolved-config hash mismatch")
    for key in CONFIG_KEYS:
        if key in config and resolved.get(key) != _json_safe(config[key]):
            raise RuntimeError(f"resume current config mismatch: {key}")
    history = checkpoint.get("history")
    epoch = int(checkpoint.get("epoch", -1))
    if (
        epoch < 1
        or not isinstance(history, list)
        or not history
        or int(history[-1].get("epoch", -1)) != epoch
        or len({int(record.get("epoch", -1)) for record in history}) != len(history)
    ):
        raise RuntimeError("resume checkpoint is not a consistent epoch boundary")
    rng_state = checkpoint.get("rng_state")
    if (
        not isinstance(rng_state, Mapping)
        or "dataloader_generator" not in rng_state
        or not torch.equal(
            rng_state["dataloader_generator"], checkpoint["sampler_generator_state"]
        )
    ):
        raise RuntimeError("resume checkpoint sampler state is inconsistent")
    if epoch >= int(config["epochs"]):
        raise RuntimeError("resume checkpoint has already reached the epoch cap")
    return checkpoint, resolved, resolved_hash


def _make_dataset(
    config: dict[str, Any], split: str, transform: Any
) -> ManifestImageDataset:
    return ManifestImageDataset(
        csv_path=Path(config["split_dir"]) / f"{split}.csv",
        image_root=Path(config["image_root"]),
        transform=transform,
        expected_split=split,
        strict=True,
    )


def _make_loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator,
) -> DataLoader:
    # DataLoader otherwise consumes the same generator for worker base seeds and
    # RandomSampler permutations.  A newly created iterator on resume would then
    # shift the permutation relative to uninterrupted persistent workers.
    if shuffle:
        sampler_type = (
            EpochRandomSampler
            if isinstance(dataset, EpochSeededDataset)
            else RandomSampler
        )
        sampler = sampler_type(dataset, generator=generator)
    else:
        sampler = None
    worker_generator = torch.Generator().manual_seed(
        (int(generator.initial_seed()) + 1_000_003) % (2**63 - 1)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=worker_generator,
        drop_last=False,
    )


def _optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("model contains no trainable parameters")
    if config["optimizer"] == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
    return torch.optim.SGD(
        parameters,
        lr=float(config["learning_rate"]),
        momentum=0.9,
        weight_decay=float(config["weight_decay"]),
    )


def _model_and_metadata(config: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    result = create_model(
        str(config["model_name"]),
        pretrained=bool(config["pretrained"]),
        freeze_backbone=bool(config["freeze_backbone"]),
        num_classes=int(config["num_classes"]),
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("create_model must return (model, metadata)")
    model, metadata = result
    return model, _json_safe(metadata)


def run_preflight(
    config: dict[str, Any], device: torch.device, *, debug: bool = False
) -> dict[str, Any]:
    """Resolve a micro-batch with a disposable train-only model."""

    train_transform = get_transform(str(config["augmentation"]), int(config["image_size"]))
    dataset: Dataset = _make_dataset(config, "train", train_transform)
    if debug:
        dataset = Subset(dataset, list(range(min(8, len(dataset)))))
    dataset = EpochSeededDataset(dataset, int(config["seed"]))
    dataset.set_epoch(0)
    attempts: list[dict[str, Any]] = []
    target = int(config["batch_size"])
    candidates = [candidate for candidate in (target, 16, 8) if candidate <= target]
    candidates = list(dict.fromkeys(candidates))
    for micro_batch in candidates:
        seed_everything(int(config["seed"]))
        generator = torch.Generator().manual_seed(int(config["seed"]))
        model: nn.Module | None = None
        try:
            model, model_metadata = _model_and_metadata(config)
            model.to(device)
            optimizer = _optimizer(config, model)
            criterion = nn.CrossEntropyLoss()
            loader = _make_loader(
                dataset,
                batch_size=micro_batch,
                shuffle=True,
                num_workers=int(config["num_workers"]),
                generator=generator,
            )
            accumulation_steps = math.ceil(target / micro_batch)
            probe_batches = max(
                accumulation_steps,
                math.ceil(int(config["preflight_batches"]) / accumulation_steps)
                * accumulation_steps,
            )
            optimizer.zero_grad(set_to_none=True)
            batch_times: list[float] = []
            seen = 0
            iterator = None
            for batch_index in range(probe_batches):
                started = time.perf_counter()
                try:
                    if iterator is None:
                        iterator = iter(loader)
                    images, targets, _ = next(iterator)
                except StopIteration:
                    break
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                loss = criterion(logits, targets) / accumulation_steps
                loss.backward()
                if (batch_index + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                synchronize_device(device)
                batch_times.append(time.perf_counter() - started)
                seen += len(targets)
            if not batch_times:
                raise RuntimeError("preflight produced no batches")
            seconds_per_batch = float(np.median(batch_times))
            steady_times = batch_times[1:] if len(batch_times) > 1 else batch_times
            steady_mean_seconds = float(np.mean(steady_times))
            total_batches = (
                math.ceil(len(dataset) / micro_batch) * int(config["epochs"])
            )
            # Preserve the observed one-time worker/model warm-up, then project
            # the steady mix of backward-only and optimizer-step batches.
            projected = (
                batch_times[0]
                + steady_mean_seconds * max(0, total_batches - 1)
            ) * 1.20
            attempts.append(
                {
                    "micro_batch_size": micro_batch,
                    "status": "PASS",
                    "num_timed_batches": len(batch_times),
                    "num_images": seen,
                    "median_batch_seconds": seconds_per_batch,
                    "steady_mean_batch_seconds": steady_mean_seconds,
                    "first_batch_seconds": batch_times[0],
                }
            )
            return {
                "schema_version": 1,
                "status": "PASS",
                "device": str(device),
                "target_effective_batch_size": target,
                "micro_batch_size": micro_batch,
                "gradient_accumulation_steps": accumulation_steps,
                "projected_training_seconds": projected,
                "projected_training_hours": projected / 3600.0,
                "model_metadata": model_metadata,
                "attempts": attempts,
            }
        except Exception as error:
            oom = is_out_of_memory_error(error)
            attempts.append(
                {
                    "micro_batch_size": micro_batch,
                    "status": "OOM" if oom else "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if not oom:
                raise
        finally:
            del model
            clear_device_cache(device)
    raise RuntimeError(f"preflight exhausted batch sizes: {attempts}")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    scaler: Any,
    amp_enabled: bool,
    accumulation_steps: int,
    gradient_clip_norm: float,
    freeze_backbone: bool,
    description: str,
) -> dict[str, float]:
    model.train()
    if freeze_backbone:
        set_frozen_backbone_eval(model)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    synchronize_device(device)
    started = time.perf_counter()
    progress = tqdm(loader, desc=description, unit="batch")
    for batch_index, (images, targets, _) in enumerate(progress):
        images = images.to(device, non_blocking=False)
        targets = targets.to(device, non_blocking=False)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            raw_loss = criterion(logits, targets)
            scaled_loss = raw_loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        should_step = (batch_index + 1) % accumulation_steps == 0 or (
            batch_index + 1 == len(loader)
        )
        if should_step:
            scaler.unscale_(optimizer)
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    gradient_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(raw_loss.detach().item()) * len(targets)
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += int(len(targets))
        progress.set_postfix(loss=f"{total_loss / total_samples:.4f}")
    synchronize_device(device)
    return {
        "loss": total_loss / total_samples,
        "top1_accuracy": total_correct / total_samples,
        "num_samples": total_samples,
        "time_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def _validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp_enabled: bool,
    description: str,
) -> dict[str, Any]:
    model.eval()
    logits_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0
    forward_seconds = 0.0
    wall_started = time.perf_counter()
    for images, targets, _ in tqdm(loader, desc=description, unit="batch"):
        images = images.to(device, non_blocking=False)
        targets = targets.to(device, non_blocking=False)
        synchronize_device(device)
        forward_started = time.perf_counter()
        with autocast_context(device, amp_enabled):
            logits = model(images)
        synchronize_device(device)
        forward_seconds += time.perf_counter() - forward_started
        loss = criterion(logits, targets)
        total_loss += float(loss.item()) * len(targets)
        total_samples += int(len(targets))
        logits_parts.append(logits.float().cpu().numpy())
        target_parts.append(targets.cpu().numpy())
    logits_array = np.concatenate(logits_parts, axis=0)
    targets_array = np.concatenate(target_parts, axis=0).astype(np.int64, copy=False)
    metrics, _, _, _ = evaluate_aligned_scores(
        targets_array, logits_array, range(NUM_CLASSES)
    )
    metrics["loss"] = total_loss / total_samples
    metrics["forward_time_seconds"] = forward_seconds
    metrics["evaluation_wall_time_seconds"] = time.perf_counter() - wall_started
    return metrics


def _checkpoint_bundle(
    *,
    kind: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: dict[str, Any],
    resolved_config: dict[str, Any],
    resolved_hash: str,
    source_config_path: Path,
    source_config_hash: str,
    stage1: dict[str, Any],
    versions: dict[str, Any],
    device: torch.device,
    parameter_counts: dict[str, int],
    best_epoch: int,
    best_metrics: dict[str, Any],
    history: list[dict[str, Any]],
    early_stopping_counter: int,
    training_time_seconds: float,
    generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "checkpoint_kind": kind,
        "epoch": int(epoch),
        "model_name": config["model_name"],
        "pretrained": bool(config["pretrained"]),
        "freeze_backbone": bool(config["freeze_backbone"]),
        "num_classes": int(config["num_classes"]),
        "image_size": int(config["image_size"]),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_epoch": int(best_epoch),
        "best_metrics": _json_safe(best_metrics),
        "history": _json_safe(history),
        "early_stopping_counter": int(early_stopping_counter),
        "resolved_config": _json_safe(resolved_config),
        "resolved_config_sha256": resolved_hash,
        "source_config_path": source_config_path.as_posix(),
        "source_config_sha256": source_config_hash,
        "stage1_artifacts": stage1,
        "versions": versions,
        "device": str(device),
        "parameter_counts": parameter_counts,
        "training_time_seconds": float(training_time_seconds),
        "rng_state": capture_rng_state(generator),
        "sampler_generator_state": generator.get_state(),
        "dataloader_generator_state": generator.get_state(),
        "augmentation_seed_contract": resolved_config.get(
            "augmentation_seed_contract"
        ),
    }


def _run_training_impl(args: argparse.Namespace) -> int:
    config = load_config(args.config, args)
    experiment = str(config["experiment_name"])
    model_dir = Path(config["model_dir"])
    log_dir = Path(config["log_dir"])
    metrics_dir = Path(config["metrics_dir"])
    figure_dir = Path(config["figure_dir"])
    paths = {
        "best_checkpoint": model_dir / f"{experiment}_best.pt",
        "last_checkpoint": model_dir / f"{experiment}_last.pt",
        "log": log_dir / f"{experiment}_train_log.jsonl",
        "resolved_config": metrics_dir / f"{experiment}_resolved_config.json",
        "best_metrics": metrics_dir / f"{experiment}_best_val_metrics.json",
        "preflight": metrics_dir / f"{experiment}_preflight.json",
        "status": metrics_dir / f"{experiment}_status.json",
        "curves": figure_dir / f"{experiment}_training_curves.png",
    }
    if Path("results/metrics/deep_learning/stage3_freeze.json").exists():
        raise RuntimeError("Stage 3 is already frozen; canonical training is immutable")
    existing = [path for path in paths.values() if path.exists()]
    if existing and args.resume is None and not args.force and not args.preflight_only:
        prior_status = _status_document(paths["status"])
        reusable_preflight = (
            set(existing) <= {paths["preflight"], paths["status"]}
            and prior_status.get("status") in {None, "PREFLIGHT_COMPLETE"}
        )
        if not reusable_preflight:
            raise FileExistsError(
                "experiment artifacts already exist; use --resume or deliberate --force: "
                + ", ".join(str(path) for path in existing)
            )

    stage1 = verify_stage1_artifacts(Path(config["split_dir"]), verify_test=False)
    source_config_hash = sha256_file(args.config)
    resume_checkpoint: dict[str, Any] | None = None
    resolved_config: dict[str, Any] | None = None
    resolved_hash: str | None = None
    if args.resume is not None:
        # This is intentionally the first resume action that could fail. No
        # preflight, status, resolved config, or log is overwritten beforehand.
        resume_checkpoint, resolved_config, resolved_hash = _validate_resume_before_write(
            args,
            config,
            paths,
            stage1=stage1,
            source_config_hash=source_config_hash,
        )

    device = select_device(str(config["device"]))
    print(f"Device: {device}")
    if not args.debug and device.type != "mps":
        raise RuntimeError(
            f"canonical local Stage 3 runs require MPS FP32; resolved device was {device}"
        )
    versions = package_versions()
    implementation_hashes = _training_implementation_hashes()
    if resume_checkpoint is not None:
        assert resolved_config is not None
        if resolved_config.get("versions") != versions:
            raise RuntimeError("resume dependency versions differ from the locked run")
        if resolved_config.get("resolved_device") != str(device):
            raise RuntimeError("resume device differs from the locked run")
        if resolved_config.get("implementation_hashes") != implementation_hashes:
            raise RuntimeError("resume training implementation hashes changed")
    seed_metadata = seed_everything(int(config["seed"]))
    status = {
        "schema_version": 1,
        "experiment_name": experiment,
        "status": "RESUME_VALIDATED" if args.resume is not None else "PREFLIGHT",
        "device": str(device),
    }
    if resume_checkpoint is None:
        atomic_write_json(paths["status"], status)

    if resume_checkpoint is None:
        preflight = run_preflight(config, device, debug=args.debug)
        preflight.update(
            {
                "experiment_name": experiment,
                "timing_scope": (
                    "end-to-end iterator next/load/augmentation, device transfer, "
                    "forward/backward, and optimizer work"
                ),
                "source_config_path": args.config.as_posix(),
                "source_config_sha256": source_config_hash,
                "stage1_artifacts": stage1,
                "versions": versions,
            }
        )
        atomic_write_json(paths["preflight"], preflight)
        print(
            f"Preflight PASS: micro batch {preflight['micro_batch_size']}, "
            f"accumulation {preflight['gradient_accumulation_steps']}, "
            f"projection {preflight['projected_training_hours']:.2f} h"
        )
        if args.preflight_only:
            status.update(
                {
                    "status": "PREFLIGHT_COMPLETE",
                    "projected_training_seconds": preflight[
                        "projected_training_seconds"
                    ],
                }
            )
            atomic_write_json(paths["status"], status)
            return 0
        if args.debug:
            gate_decision = {"status": "BYPASSED_DEBUG", "reasons": []}
            gate_path = metrics_dir / "stage3_compute_gate.json"
        else:
            gate_decision, gate_path = _compute_suite_gate(
                config,
                current_projection_seconds=float(
                    preflight["projected_training_seconds"]
                ),
                phase="pre_training",
            )
        if gate_decision["status"] == "NOT RUN":
            status.update(
                {
                    "status": "NOT RUN",
                    "gate_path": gate_path.as_posix(),
                    "gate_decision": gate_decision,
                }
            )
            atomic_write_json(paths["status"], status)
            print(f"Training NOT RUN: {experiment}; {gate_decision['reasons']}")
            return 0
    else:
        assert resolved_config is not None and resolved_hash is not None
        preflight = {
            "micro_batch_size": int(resolved_config["micro_batch_size"]),
            "target_effective_batch_size": int(
                resolved_config["target_effective_batch_size"]
            ),
            "gradient_accumulation_steps": int(
                resolved_config["gradient_accumulation_steps"]
            ),
            "model_metadata": resolved_config["model_metadata"],
            "projected_training_seconds": float(
                resolved_config["preflight_projected_training_seconds"]
            ),
        }

    seed_everything(int(config["seed"]))
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_dataset: Any = _make_dataset(
        config,
        "train",
        get_transform(str(config["augmentation"]), int(config["image_size"])),
    )
    val_dataset: Any = _make_dataset(
        config, "val", get_transform("eval", int(config["image_size"]))
    )
    if args.debug:
        train_dataset = Subset(train_dataset, list(range(min(8, len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(8, len(val_dataset)))))
    train_dataset = EpochSeededDataset(train_dataset, int(config["seed"]))
    model, model_metadata = _model_and_metadata(config)
    initial_model_hash = model_state_sha256(model)
    if _json_safe(model_metadata) != _json_safe(preflight["model_metadata"]):
        raise RuntimeError("model metadata changed between preflight and canonical run")
    d2_provenance = _ablation_provenance(
        config,
        initial_model_hash=initial_model_hash,
        dataset_size=len(train_dataset),
    )
    d2_peer_check = (
        _verify_existing_d2_peer(experiment, d2_provenance, metrics_dir)
        if not args.debug
        else {"status": "bypassed_debug"}
    )

    if resume_checkpoint is None:
        resolved_config = _json_safe(copy.deepcopy(config))
        resolved_config.update(
            {
                "micro_batch_size": int(preflight["micro_batch_size"]),
                "target_effective_batch_size": int(
                    preflight["target_effective_batch_size"]
                ),
                "gradient_accumulation_steps": int(
                    preflight["gradient_accumulation_steps"]
                ),
                "preflight_projected_training_seconds": float(
                    preflight["projected_training_seconds"]
                ),
                "resolved_device": str(device),
                "model_metadata": preflight["model_metadata"],
                "initial_model_state_sha256": initial_model_hash,
                "augmentation_seed_contract": "sha256(base_seed, epoch, manifest_index)",
                "d2_ablation_provenance": d2_provenance,
                "d2_peer_check": d2_peer_check,
                "implementation_hashes": implementation_hashes,
                "versions": versions,
                "seed_metadata": seed_metadata,
                "source_config_path": args.config.as_posix(),
                "source_config_sha256": source_config_hash,
                "stage1_artifacts": stage1,
            }
        )
        atomic_write_json(paths["resolved_config"], resolved_config)
        resolved_hash = sha256_file(paths["resolved_config"])
    else:
        assert resolved_config is not None and resolved_hash is not None
        if resolved_config.get("initial_model_state_sha256") != initial_model_hash:
            raise RuntimeError("resume initial model state provenance mismatch")
        if resolved_config.get("d2_ablation_provenance") != d2_provenance:
            raise RuntimeError("resume D2/sampler provenance mismatch")

    train_loader = _make_loader(
        train_dataset,
        batch_size=int(preflight["micro_batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        generator=generator,
    )
    val_generator = torch.Generator().manual_seed(int(config["seed"]) + 1)
    val_loader = _make_loader(
        val_dataset,
        batch_size=int(preflight["micro_batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        generator=val_generator,
    )
    model.to(device)
    parameter_counts = count_parameters(model)
    optimizer = _optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(config["epochs"]),
        eta_min=float(config["learning_rate"]) * 0.01,
    )
    amp_enabled = amp_enabled_for_device(device, config["amp"])
    scaler = make_grad_scaler(device, amp_enabled)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_metrics: dict[str, Any] = {}
    best_key: tuple[float, ...] | None = None
    early_counter = 0
    training_time = 0.0
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(resume_checkpoint.get("scaler_state_dict", {}))
        history = list(resume_checkpoint.get("history", []))
        best_epoch = int(resume_checkpoint.get("best_epoch", 0))
        best_metrics = dict(resume_checkpoint.get("best_metrics", {}))
        if best_metrics:
            best_key = checkpoint_selection_key(best_metrics, best_epoch)
        early_counter = int(resume_checkpoint.get("early_stopping_counter", 0))
        training_time = float(resume_checkpoint.get("training_time_seconds", 0.0))
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        restore_rng_state(resume_checkpoint["rng_state"], generator)
        print(f"Resuming {experiment} at epoch {start_epoch}")

    assert resolved_config is not None and resolved_hash is not None
    status.update(
        {
            "status": "RUNNING",
            "resolved_config_sha256": resolved_hash,
            "resume_checkpoint": (
                Path(args.resume).as_posix() if args.resume is not None else None
            ),
        }
    )
    atomic_write_json(paths["status"], status)
    try:
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            train_dataset.set_epoch(epoch)
            train_metrics = _train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler=scaler,
                amp_enabled=amp_enabled,
                accumulation_steps=int(preflight["gradient_accumulation_steps"]),
                gradient_clip_norm=float(config["gradient_clip_norm"]),
                freeze_backbone=bool(config["freeze_backbone"]),
                description=f"{experiment} train {epoch}/{config['epochs']}",
            )
            training_time += float(train_metrics["time_seconds"])
            val_metrics = _validate_epoch(
                model,
                val_loader,
                criterion,
                device,
                amp_enabled=amp_enabled,
                description=f"{experiment} val {epoch}/{config['epochs']}",
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            record = {
                "event": "epoch",
                "experiment_name": experiment,
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_top1": float(train_metrics["top1_accuracy"]),
                "train_num_samples": int(train_metrics["num_samples"]),
                "train_time_seconds": float(train_metrics["time_seconds"]),
                "val_loss": float(val_metrics["loss"]),
                "val_top1": float(val_metrics["top1_accuracy"]),
                "val_top5": float(val_metrics["top5_accuracy"]),
                "val_macro_precision": float(val_metrics["macro_precision"]),
                "val_macro_recall": float(val_metrics["macro_recall"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_forward_time_seconds": float(val_metrics["forward_time_seconds"]),
                "val_wall_time_seconds": float(val_metrics["evaluation_wall_time_seconds"]),
                "learning_rate": learning_rate,
            }
            history.append(record)
            current_metrics = {
                "loss": record["val_loss"],
                "top1_accuracy": record["val_top1"],
                "top5_accuracy": record["val_top5"],
                "overall_accuracy": record["val_top1"],
                "macro_precision": record["val_macro_precision"],
                "macro_recall": record["val_macro_recall"],
                "macro_f1": record["val_macro_f1"],
                "num_samples": len(val_dataset),
                "num_classes": NUM_CLASSES,
                "forward_time_seconds": record["val_forward_time_seconds"],
                "evaluation_wall_time_seconds": record["val_wall_time_seconds"],
            }
            current_key = checkpoint_selection_key(current_metrics, epoch)
            improved = best_key is None or current_key > best_key
            if improved:
                best_key = current_key
                best_epoch = epoch
                best_metrics = current_metrics
                early_counter = 0
                best_bundle = _checkpoint_bundle(
                    kind="best",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    resolved_config=resolved_config,
                    resolved_hash=resolved_hash,
                    source_config_path=args.config,
                    source_config_hash=source_config_hash,
                    stage1=stage1,
                    versions=versions,
                    device=device,
                    parameter_counts=parameter_counts,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                    history=history,
                    early_stopping_counter=early_counter,
                    training_time_seconds=training_time,
                    generator=generator,
                )
                atomic_torch_save(paths["best_checkpoint"], best_bundle)
            else:
                early_counter += 1
            scheduler.step()
            last_bundle = _checkpoint_bundle(
                kind="last",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                resolved_config=resolved_config,
                resolved_hash=resolved_hash,
                source_config_path=args.config,
                source_config_hash=source_config_hash,
                stage1=stage1,
                versions=versions,
                device=device,
                parameter_counts=parameter_counts,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                history=history,
                early_stopping_counter=early_counter,
                training_time_seconds=training_time,
                generator=generator,
            )
            atomic_torch_save(paths["last_checkpoint"], last_bundle)
            atomic_write_jsonl(paths["log"], history)
            print(
                f"Epoch {epoch}: train top1={record['train_top1']:.4f}; "
                f"val top1={record['val_top1']:.4f}; "
                f"val macro-F1={record['val_macro_f1']:.4f}; "
                f"best epoch={best_epoch}"
            )
            if epoch >= int(config["min_epochs"]) and early_counter >= int(config["patience"]):
                print(f"Early stopping after epoch {epoch}")
                break

        if not paths["best_checkpoint"].is_file():
            raise RuntimeError("training completed without a best checkpoint")
        # Preserve the best-epoch model/optimizer tensors while completing its
        # run-level timing and history provenance after early stopping/epoch cap.
        completed_best_bundle = load_checkpoint(paths["best_checkpoint"], map_location="cpu")
        completed_best_bundle["training_time_seconds"] = float(training_time)
        completed_best_bundle["history"] = _json_safe(history)
        completed_best_bundle["completed_epochs"] = len(history)
        completed_best_bundle["stopped_early"] = len(history) < int(config["epochs"])
        atomic_torch_save(paths["best_checkpoint"], completed_best_bundle)
        best_checkpoint_hash = sha256_file(paths["best_checkpoint"])
        best_document = {
            "schema_version": 1,
            "experiment_name": experiment,
            "best_epoch": best_epoch,
            "selection_metric": "macro_f1",
            "selection_value": float(best_metrics["macro_f1"]),
            "metrics": best_metrics,
            "training_time_seconds": training_time,
            "validation_time_seconds": float(best_metrics["forward_time_seconds"]),
            "checkpoint_path": paths["best_checkpoint"].as_posix(),
            "checkpoint_sha256": best_checkpoint_hash,
            "resolved_config_path": paths["resolved_config"].as_posix(),
            "resolved_config_sha256": resolved_hash,
            "stage1_artifacts": stage1,
            "parameter_counts": parameter_counts,
            "completed_epochs": len(history),
            "stopped_early": len(history) < int(config["epochs"]),
        }
        atomic_write_json(paths["best_metrics"], best_document)
        try:
            from src.analysis.plotting import plot_deep_training_curves

            plot_deep_training_curves(
                history,
                paths["curves"],
                title=f"{experiment} training and validation",
            )
        except ImportError:
            print("WARNING: training-curve helper is not yet available", file=sys.stderr)
        status.update(
            {
                "status": "COMPLETE",
                "completed_epochs": len(history),
                "best_epoch": best_epoch,
                "best_checkpoint_sha256": best_checkpoint_hash,
                "training_time_seconds": training_time,
            }
        )
        atomic_write_json(paths["status"], status)
        print(f"Training COMPLETE: {experiment}; best epoch {best_epoch}")
        return 0
    except KeyboardInterrupt:
        status.update(
            {
                "status": "INCOMPLETE",
                "error_type": "KeyboardInterrupt",
                "error": "training interrupted by user",
                "completed_epochs": len(history),
            }
        )
        atomic_write_json(paths["status"], status)
        raise
    except Exception as error:
        status.update(
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_epochs": len(history),
            }
        )
        atomic_write_json(paths["status"], status)
        raise


def _record_terminal_setup_failure(
    args: argparse.Namespace, error: BaseException, *, interrupted: bool
) -> None:
    """Best-effort status recording for failures before the epoch-loop try block."""

    if isinstance(error, FileExistsError):
        return
    try:
        config = load_config(args.config, args)
        experiment = str(config["experiment_name"])
        path = Path(config["metrics_dir"]) / f"{experiment}_status.json"
        existing = _status_document(path)
        # A rejected resume must not mutate the previous run's status artifact.
        if args.resume is not None and path.exists():
            recorded_resume = existing.get("resume_checkpoint")
            if recorded_resume != Path(args.resume).as_posix():
                return
        document = dict(existing)
        document.update(
            {
                "schema_version": 1,
                "experiment_name": experiment,
                "status": "INCOMPLETE" if interrupted else "FAILED",
                "error_type": type(error).__name__,
                "error": str(error) or (
                    "training interrupted by user" if interrupted else "setup failed"
                ),
            }
        )
        atomic_write_json(path, document)
    except Exception:
        # Invalid YAML may not identify a safe experiment/status path. The CLI
        # still reports the original error without masking it.
        return


def run_training(args: argparse.Namespace) -> int:
    try:
        return _run_training_impl(args)
    except KeyboardInterrupt as error:
        _record_terminal_setup_failure(args, error, interrupted=True)
        raise
    except Exception as error:
        _record_terminal_setup_failure(args, error, interrupted=False)
        raise


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return run_training(args)
    except KeyboardInterrupt:
        print("ERROR: training interrupted; status recorded as INCOMPLETE", file=sys.stderr)
        return 130
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
