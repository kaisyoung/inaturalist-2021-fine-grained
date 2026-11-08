#!/usr/bin/env python3
"""Run a real-image, no-test-access smoke test of the Stage 3 stack."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.common import sha256_file
from src.deep_learning.dataset import ManifestImageDataset

os.environ.setdefault("TORCH_HOME", "results/models/torch_cache")

from src.deep_learning.models import count_parameters, create_model  # noqa: E402
from src.deep_learning.train_utils import (  # noqa: E402
    atomic_torch_save,
    atomic_write_bytes,
    load_checkpoint,
    package_versions,
    seed_everything,
    select_device,
    synchronize_device,
    verify_stage1_artifacts,
)
from src.deep_learning.transforms import get_transform  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-pretrained", action="store_true")
    parser.add_argument(
        "--report", type=Path, default=Path("DEEP_LEARNING_SMOKE_TEST_REPORT.md")
    )
    return parser.parse_args(argv)


def _batch(dataset: ManifestImageDataset, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    examples = [dataset[index] for index in indices]
    return (
        torch.stack([example[0] for example in examples]),
        torch.tensor([example[1] for example in examples], dtype=torch.long),
        [dict(example[2]) for example in examples],
    )


def _render_report(result: dict[str, Any]) -> bytes:
    lines = [
        "# Deep Learning Smoke Test Report",
        "",
        f"**Status: {result['status']}**",
        "",
        "This smoke test used real materialized training and validation images. It did",
        "not construct a test dataset or open `data/splits/test.csv`.",
        "",
        "## Command",
        "",
        "```text",
        result["command"],
        "```",
        "",
        "## Environment and dataset",
        "",
        "```json",
        json.dumps(result.get("environment", {}), indent=2, sort_keys=True),
        "```",
        "",
        "```json",
        json.dumps(result.get("dataset", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Observed checks",
        "",
        "```json",
        json.dumps(result.get("checks", {}), indent=2, sort_keys=True),
        "```",
    ]
    if result.get("error"):
        lines.extend(
            [
                "",
                "## Failure",
                "",
                "```text",
                f"{result.get('error_type')}: {result['error']}",
                "```",
            ]
        )
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    command = (
        ".venv/bin/python scripts/smoke_test_deep_learning.py "
        f"--device {args.device} --image-size {args.image_size} --seed {args.seed}"
    )
    if args.skip_pretrained:
        command += " --skip-pretrained"
    result: dict[str, Any] = {"status": "RUNNING", "command": command}
    try:
        stage1 = verify_stage1_artifacts(verify_test=False)
        seed_everything(args.seed)
        device = select_device(args.device)
        result["environment"] = {
            "device": str(device),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "versions": package_versions(),
            "stage1_hashes": {name: value["sha256"] for name, value in stage1.items()},
        }
        train_eval = ManifestImageDataset(
            "data/splits/train.csv",
            "data/processed/images",
            get_transform("eval", args.image_size),
            "train",
            strict=True,
        )
        val_eval = ManifestImageDataset(
            "data/splits/val.csv",
            "data/processed/images",
            get_transform("eval", args.image_size),
            "val",
            strict=True,
        )
        train_augmented = ManifestImageDataset(
            "data/splits/train.csv",
            "data/processed/images",
            get_transform("basic_train", args.image_size),
            "train",
            strict=True,
        )
        train_images, train_targets, train_metadata = _batch(train_eval, [0, 40])
        val_images, val_targets, val_metadata = _batch(val_eval, [0, 10])
        augmented_images, augmented_targets, _ = _batch(train_augmented, [0, 40])
        result["dataset"] = {
            "train_rows": len(train_eval),
            "validation_rows": len(val_eval),
            "sample_tensor_shape": list(train_images.shape),
            "sample_train_labels": train_targets.tolist(),
            "sample_validation_labels": val_targets.tolist(),
            "sample_train_image_ids": [item["image_id"] for item in train_metadata],
            "sample_validation_image_ids": [item["image_id"] for item in val_metadata],
        }

        checks: dict[str, Any] = {}
        model_specs = [("resnet18_scratch", "resnet18", False, False)]
        if not args.skip_pretrained:
            model_specs.extend(
                [
                    ("resnet18_pretrained", "resnet18", True, False),
                    ("efficientnet_b0_pretrained", "efficientnet_b0", True, False),
                    ("resnet18_pretrained_frozen_head", "resnet18", True, True),
                ]
            )
        scratch_model: nn.Module | None = None
        for label, architecture, pretrained, frozen in model_specs:
            seed_everything(args.seed)
            model, metadata = create_model(
                architecture,
                pretrained=pretrained,
                freeze_backbone=frozen,
                num_classes=500,
            )
            model.to(device).eval()
            synchronize_device(device)
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(train_images.to(device))
            synchronize_device(device)
            elapsed = time.perf_counter() - started
            if tuple(logits.shape) != (2, 500) or not torch.isfinite(logits).all():
                raise RuntimeError(f"{label} produced invalid logits {tuple(logits.shape)}")
            checks[label] = {
                "forward_shape": list(logits.shape),
                "finite_logits": True,
                "forward_seconds": elapsed,
                "parameter_counts": count_parameters(model),
                "metadata": metadata,
            }
            if label == "resnet18_scratch":
                scratch_model = model
            else:
                del model
        if scratch_model is None:
            raise RuntimeError("scratch model was not constructed")

        seed_everything(args.seed)
        scratch_model.train()
        optimizer = torch.optim.AdamW(scratch_model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad(set_to_none=True)
        logits = scratch_model(augmented_images.to(device))
        loss_before = criterion(logits, augmented_targets.to(device))
        loss_before.backward()
        optimizer.step()
        synchronize_device(device)
        if not torch.isfinite(loss_before):
            raise RuntimeError("one-step training loss is not finite")
        with tempfile.TemporaryDirectory() as temporary_name:
            checkpoint_path = Path(temporary_name) / "smoke_checkpoint.pt"
            atomic_torch_save(
                checkpoint_path,
                {
                    "schema_version": 1,
                    "experiment_name": "deep_learning_smoke",
                    "model_state_dict": scratch_model.state_dict(),
                },
            )
            checkpoint_hash = sha256_file(checkpoint_path)
            loaded = load_checkpoint(checkpoint_path)
            clone, _ = create_model(
                "resnet18", pretrained=False, freeze_backbone=False, num_classes=500
            )
            clone.load_state_dict(loaded["model_state_dict"])
            state_equal = all(
                torch.equal(value.detach().cpu(), clone.state_dict()[key].detach().cpu())
                for key, value in scratch_model.state_dict().items()
            )
        if not state_equal:
            raise RuntimeError("checkpoint round-trip changed model tensors")
        checks["optimizer_step"] = {
            "batch_size": 2,
            "loss": float(loss_before.detach().cpu().item()),
            "finite": True,
        }
        checks["checkpoint_round_trip"] = {
            "sha256": checkpoint_hash,
            "state_dict_equal": True,
        }
        result["checks"] = checks
        result["status"] = "PASS"
        return result
    except Exception as error:
        result["status"] = "FAIL"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        raise
    finally:
        atomic_write_bytes(args.report, _render_report(result))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = run(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Deep-learning smoke test {result['status']}: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
