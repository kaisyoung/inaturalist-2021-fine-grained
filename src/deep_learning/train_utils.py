"""Reproducibility, device, checkpoint, and training support for Stage 3."""

from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from src.data.common import deterministic_json_bytes, sha256_bytes, sha256_file


STAGE1_EXPECTED_HASHES = {
    "selected_classes_seed42.txt": "3c2af417a5c174ef14f2be117a00880b0a08f8dce0c24fad6aaa8ea5dcde8141",
    "class_mapping.json": "2d6e60104b133d2758dffd585d373b085ddb42845d0b4b2c93bf5277ef9a0a1b",
    "train.csv": "d8f4abccc568484f6efaff243c409b8ddcbc8fe46d2146a6922230b4428706c7",
    "val.csv": "4adf287996d35d9e57f829426ff7480bbaf0b74e1c319a29d57fe74b3f0366bb",
    "test.csv": "45ba7673b6b0e27cab8ef5a482ea8142de54e35be05a5b1ba9bb3fe20d5b8f91",
    "split_summary.json": "324413fb3d8d3affe611f8fd1454358cf54f2e6fcd81ef961994be0cc19f0b76",
}

STAGE2_EVIDENCE_PATHS = (
    Path("results/metrics/stage2_freeze.json"),
    Path("results/tables/stage2_traditional_results.csv"),
    Path("results/tables/stage2_feature_ablation_results.csv"),
    Path("STAGE2_TRADITIONAL_REPORT.md"),
    Path("STAGE2_FINAL_ACCEPTANCE_REPORT.md"),
)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write *content* with fsync and atomic replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(Path(path), deterministic_json_bytes(value))


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    payload = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    atomic_write_bytes(Path(path), payload)


def atomic_torch_save(path: Path, value: Any) -> None:
    """Save a PyTorch object atomically without leaving partial checkpoints."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(value, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(deterministic_json_bytes(value))


def verify_stage1_artifacts(
    split_dir: Path = Path("data/splits"), *, verify_test: bool = True
) -> dict[str, dict[str, Any]]:
    """Verify accepted Stage 1 artifacts.

    Training and validation callers pass ``verify_test=False`` so they bind the
    already accepted test-manifest digest without opening ``test.csv``. The
    precheck and post-training freeze perform the full byte verification.
    """

    split_dir = Path(split_dir)
    records: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for name, expected in STAGE1_EXPECTED_HASHES.items():
        path = split_dir / name
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        if name == "test.csv" and not verify_test:
            records[name] = {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": expected,
                "verification": "accepted_precheck_digest; file not opened",
            }
            continue
        observed = sha256_file(path)
        if observed != expected:
            failures.append(f"{path}: expected {expected}, observed {observed}")
        records[name] = {
            "path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": observed,
            "verification": "verified_current_bytes",
        }
    if failures:
        raise RuntimeError("Stage 1 integrity verification failed: " + "; ".join(failures))
    return records


def describe_stage2_evidence(project_root: Path = Path(".")) -> dict[str, dict[str, Any]]:
    project_root = Path(project_root)
    records: dict[str, dict[str, Any]] = {}
    for relative in STAGE2_EVIDENCE_PATHS:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required frozen Stage 2 evidence is missing: {path}")
        records[relative.as_posix()] = {
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed project RNGs and enable best-effort deterministic execution."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    warnings: list[str] = []
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as error:  # pragma: no cover - old/unsupported builds
            warnings.append(f"torch deterministic algorithms unavailable: {error}")
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    return {
        "seed": int(seed),
        "deterministic_algorithms_requested": bool(deterministic),
        "warnings": warnings,
    }


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def select_device(requested: str = "auto") -> torch.device:
    """Resolve CUDA, then MPS, then CPU, or validate an explicit request."""

    normalized = requested.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if normalized not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be one of auto, cuda, mps, or cpu")
    return torch.device(normalized)


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def amp_enabled_for_device(device: torch.device, requested: str | bool = "auto") -> bool:
    if isinstance(requested, bool):
        enabled = requested
    else:
        value = str(requested).lower()
        if value not in {"auto", "true", "false"}:
            raise ValueError("amp must be auto, true, or false")
        enabled = device.type == "cuda" if value == "auto" else value == "true"
    if enabled and device.type != "cuda":
        raise RuntimeError("mixed precision is enabled by default only for CUDA")
    return enabled


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=torch.float16)


def make_grad_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler(device.type, enabled=enabled)


def capture_rng_state(generator: torch.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    mps = getattr(torch, "mps", None)
    if (
        mps is not None
        and torch.backends.mps.is_available()
        and hasattr(mps, "get_rng_state")
    ):
        state["torch_mps"] = mps.get_rng_state()
    if generator is not None:
        state["dataloader_generator"] = generator.get_state()
    return state


def restore_rng_state(state: dict[str, Any], generator: torch.Generator | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    mps = getattr(torch, "mps", None)
    if (
        "torch_mps" in state
        and mps is not None
        and torch.backends.mps.is_available()
        and hasattr(mps, "set_rng_state")
    ):
        mps.set_rng_state(state["torch_mps"])
    if generator is not None and "dataloader_generator" in state:
        generator.set_state(state["dataloader_generator"])


def package_versions() -> dict[str, str]:
    import PIL
    import pandas
    import sklearn
    import torchvision
    import yaml

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "Pillow": PIL.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "PyYAML": yaml.__version__,
    }


def is_out_of_memory_error(error: BaseException) -> bool:
    text = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or any(
        token in text
        for token in (
            "out of memory",
            "mps backend out of memory",
            "cuda error: out of memory",
        )
    )


def checkpoint_selection_key(metrics: dict[str, Any], epoch: int) -> tuple[float, ...]:
    """Higher tuple is better: macro F1, top-1, top-5, -loss, -epoch."""

    return (
        float(metrics["macro_f1"]),
        float(metrics["top1_accuracy"]),
        float(metrics["top5_accuracy"]),
        -float(metrics["loss"]),
        -float(epoch),
    )


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"invalid Stage 3 checkpoint: {path}")
    return checkpoint
