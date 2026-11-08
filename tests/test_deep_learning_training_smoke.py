"""Focused CPU tests for Stage 3 training and checkpoint utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.analysis.plotting import plot_deep_training_curves
from src.data.common import sha256_file
from src.deep_learning.train_deep import (
    CANONICAL_EXPERIMENT_ORDER,
    TOTAL_TRAINING_GATE_SECONDS,
    EpochSeededDataset,
    _ablation_contract,
    _compute_suite_gate,
    _make_loader,
    _record_terminal_setup_failure,
    _train_epoch,
    _validate_epoch,
    _validate_resume_before_write,
    load_config,
    run_preflight,
)
from src.deep_learning.train_utils import (
    atomic_torch_save,
    capture_rng_state,
    checkpoint_selection_key,
    is_out_of_memory_error,
    load_checkpoint,
    make_grad_scaler,
    restore_rng_state,
    seed_everything,
)


class TinyImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 500))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(images)


class RandomExampleDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        return (
            index,
            float(torch.rand(())),
            float(np.random.random()),
        )


class DeepTrainingSmokeTests(unittest.TestCase):
    def test_one_cpu_minibatch_and_validation_metrics(self) -> None:
        seed_everything(42)
        images = torch.randn(4, 3, 8, 8)
        targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(images, targets, torch.arange(4)), batch_size=2)
        model = TinyImageClassifier()
        before = model.classifier[1].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scaler = make_grad_scaler(torch.device("cpu"), False)
        train = _train_epoch(
            model,
            loader,
            optimizer,
            criterion,
            torch.device("cpu"),
            scaler=scaler,
            amp_enabled=False,
            accumulation_steps=1,
            gradient_clip_norm=1.0,
            freeze_backbone=False,
            description="unit train",
        )
        self.assertTrue(torch.isfinite(torch.tensor(train["loss"])))
        self.assertFalse(torch.equal(before, model.classifier[1].weight))
        validation = _validate_epoch(
            model,
            loader,
            criterion,
            torch.device("cpu"),
            amp_enabled=False,
            description="unit val",
        )
        self.assertEqual(validation["num_samples"], 4)
        self.assertEqual(validation["num_classes"], 500)
        self.assertGreaterEqual(validation["top5_accuracy"], validation["top1_accuracy"])

    def test_atomic_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "checkpoint.pt"
            tensor = torch.arange(8)
            atomic_torch_save(path, {"model_state_dict": {"weight": tensor}})
            first_hash = sha256_file(path)
            loaded = load_checkpoint(path)
            torch.testing.assert_close(loaded["model_state_dict"]["weight"], tensor)
            self.assertEqual(first_hash, sha256_file(path))

    def test_selection_ties_use_accuracy_loss_then_earlier_epoch(self) -> None:
        base = {
            "macro_f1": 0.2,
            "top1_accuracy": 0.3,
            "top5_accuracy": 0.5,
            "loss": 1.0,
        }
        self.assertGreater(checkpoint_selection_key(base, 2), checkpoint_selection_key(base, 3))
        better_top1 = {**base, "top1_accuracy": 0.31}
        self.assertGreater(checkpoint_selection_key(better_top1, 4), checkpoint_selection_key(base, 1))

    def test_oom_detection_is_explicit(self) -> None:
        self.assertTrue(is_out_of_memory_error(RuntimeError("MPS backend out of memory")))
        self.assertFalse(is_out_of_memory_error(RuntimeError("unreadable JPEG")))

    def test_preflight_falls_back_only_on_oom_and_preserves_effective_batch(self) -> None:
        images = torch.randn(16, 3, 8, 8)
        targets = torch.arange(16, dtype=torch.long) % 2
        dataset = TensorDataset(images, targets, torch.arange(16))
        config = {
            "augmentation": "basic_train",
            "image_size": 8,
            "seed": 42,
            "batch_size": 32,
            "num_workers": 0,
            "preflight_batches": 1,
            "epochs": 1,
            "optimizer": "adamw",
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
        }
        original_make_loader = _make_loader

        def loader_with_one_oom(*args, **kwargs):
            if kwargs["batch_size"] == 32:
                raise RuntimeError("MPS backend out of memory")
            return original_make_loader(*args, **kwargs)

        with (
            mock.patch("src.deep_learning.train_deep._make_dataset", return_value=dataset),
            mock.patch(
                "src.deep_learning.train_deep._model_and_metadata",
                side_effect=lambda unused: (TinyImageClassifier(), {"fixture": True}),
            ),
            mock.patch(
                "src.deep_learning.train_deep._make_loader",
                side_effect=loader_with_one_oom,
            ),
        ):
            result = run_preflight(config, torch.device("cpu"))
        self.assertEqual(result["micro_batch_size"], 16)
        self.assertEqual(result["gradient_accumulation_steps"], 2)
        self.assertEqual([item["status"] for item in result["attempts"]], ["OOM", "PASS"])

    def test_epoch_seeded_augmentation_is_resume_independent(self) -> None:
        wrapped = EpochSeededDataset(RandomExampleDataset(), seed=42)
        wrapped.set_epoch(3)
        first = wrapped[5]
        torch.rand(20)
        np.random.random(20)
        second = wrapped[5]
        self.assertEqual(first, second)
        wrapped.set_epoch(4)
        self.assertNotEqual(first[1:], wrapped[5][1:])

    def test_sampler_order_round_trip_is_independent_of_worker_generator(self) -> None:
        dataset = TensorDataset(
            torch.arange(20), torch.arange(20), torch.arange(20)
        )
        generator = torch.Generator().manual_seed(42)
        loader = _make_loader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        list(loader)
        boundary_state = generator.get_state()
        uninterrupted = torch.cat([batch[0] for batch in loader]).tolist()

        resumed_generator = torch.Generator()
        resumed_generator.set_state(boundary_state)
        resumed_loader = _make_loader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            generator=resumed_generator,
        )
        resumed = torch.cat([batch[0] for batch in resumed_loader]).tolist()
        self.assertEqual(uninterrupted, resumed)

    def test_mps_rng_state_is_captured_and_restored_when_supported(self) -> None:
        fixture_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
        with (
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            mock.patch.object(torch.mps, "get_rng_state", return_value=fixture_state),
        ):
            state = capture_rng_state()
        torch.testing.assert_close(state["torch_mps"], fixture_state)
        with (
            mock.patch.object(torch.backends.mps, "is_available", return_value=True),
            mock.patch.object(torch.mps, "set_rng_state") as setter,
        ):
            restore_rng_state(state)
        setter.assert_called_once()
        torch.testing.assert_close(setter.call_args.args[0], fixture_state)

    def test_compute_gate_records_time_disk_and_priority_failures(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = {
                "experiment_name": CANONICAL_EXPERIMENT_ORDER[0],
                "metrics_dir": str(root),
            }
            disk = SimpleNamespace(free=25 * 1024**3)
            with mock.patch("src.deep_learning.train_deep.shutil.disk_usage", return_value=disk):
                decision, path = _compute_suite_gate(
                    config,
                    current_projection_seconds=TOTAL_TRAINING_GATE_SECONDS + 1,
                    phase="unit",
                )
            self.assertEqual(decision["status"], "NOT RUN")
            self.assertFalse(decision["checks"]["time"])
            self.assertTrue(decision["checks"]["disk"])
            self.assertTrue(path.is_file())
            low_disk = SimpleNamespace(free=1)
            with mock.patch(
                "src.deep_learning.train_deep.shutil.disk_usage", return_value=low_disk
            ):
                disk_decision, _ = _compute_suite_gate(
                    config, current_projection_seconds=1.0, phase="unit-disk"
                )
            self.assertFalse(disk_decision["checks"]["disk"])
            d3_config = {
                "experiment_name": CANONICAL_EXPERIMENT_ORDER[3],
                "metrics_dir": str(root),
            }
            with mock.patch(
                "src.deep_learning.train_deep.shutil.disk_usage", return_value=disk
            ):
                priority_decision, _ = _compute_suite_gate(
                    d3_config, current_projection_seconds=1.0, phase="unit-priority"
                )
            self.assertFalse(priority_decision["checks"]["priority"])

    def test_d2_fixed_configs_have_equal_non_augmentation_contract(self) -> None:
        basic = load_config(Path("configs/deep_resnet18_pretrained_basic.yaml"))
        strong = load_config(Path("configs/deep_resnet18_pretrained_strong_aug.yaml"))
        self.assertEqual(_ablation_contract(basic), _ablation_contract(strong))
        self.assertNotEqual(basic["augmentation"], strong["augmentation"])

    def test_canonical_config_rejects_hyperparameter_drift(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            source = Path("configs/deep_resnet18_scratch.yaml").read_text(
                encoding="utf-8"
            )
            path = Path(name) / "changed.yaml"
            path.write_text(source.replace("epochs: 25", "epochs: 24"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed canonical"):
                load_config(path)

    def test_resume_rejects_best_checkpoint_before_resolved_artifact_write(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            checkpoint_path = root / "best.pt"
            resolved_path = root / "resolved.json"
            resolved_path.write_text("{}\n", encoding="utf-8")
            checkpoint = {
                "model_state_dict": {},
                "checkpoint_kind": "best",
                "epoch": 1,
                "experiment_name": "resnet18_scratch_basic",
                "model_name": "resnet18",
                "pretrained": False,
                "freeze_backbone": False,
                "optimizer_state_dict": {},
                "scheduler_state_dict": {},
                "scaler_state_dict": {},
                "rng_state": {},
                "sampler_generator_state": torch.Generator().get_state(),
                "history": [{"epoch": 1}],
                "resolved_config": {},
                "resolved_config_sha256": sha256_file(resolved_path),
                "source_config_sha256": "source",
                "stage1_artifacts": {},
            }
            atomic_torch_save(checkpoint_path, checkpoint)
            before = resolved_path.read_bytes()
            args = SimpleNamespace(resume=checkpoint_path)
            config = {
                "experiment_name": "resnet18_scratch_basic",
                "model_name": "resnet18",
                "pretrained": False,
                "freeze_backbone": False,
                "epochs": 25,
            }
            with self.assertRaisesRegex(RuntimeError, "only an epoch-boundary last"):
                _validate_resume_before_write(
                    args,
                    config,
                    {"resolved_config": resolved_path},
                    stage1={},
                    source_config_hash="source",
                )
            self.assertEqual(before, resolved_path.read_bytes())

    def test_setup_failure_and_interrupt_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = {
                "experiment_name": "resnet18_scratch_basic",
                "metrics_dir": str(root),
            }
            args = SimpleNamespace(
                config=Path("unused.yaml"),
                resume=None,
            )
            with mock.patch("src.deep_learning.train_deep.load_config", return_value=config):
                _record_terminal_setup_failure(args, RuntimeError("download failed"), interrupted=False)
            status_path = root / "resnet18_scratch_basic_status.json"
            self.assertEqual(
                __import__("json").loads(status_path.read_text())["status"], "FAILED"
            )
            with mock.patch("src.deep_learning.train_deep.load_config", return_value=config):
                _record_terminal_setup_failure(args, KeyboardInterrupt(), interrupted=True)
            self.assertEqual(
                __import__("json").loads(status_path.read_text())["status"], "INCOMPLETE"
            )

    def test_canonical_config_and_curve_contract(self) -> None:
        config = load_config(Path("configs/deep_resnet18_scratch.yaml"))
        self.assertEqual(config["num_classes"], 500)
        self.assertEqual(config["augmentation"], "basic_train")
        history = [
            {
                "epoch": 1,
                "train_loss": 2.0,
                "val_loss": 2.1,
                "train_top1": 0.1,
                "val_top1": 0.08,
                "val_macro_f1": 0.06,
                "learning_rate": 1e-3,
            }
        ]
        with tempfile.TemporaryDirectory() as name:
            output = Path(name) / "curves.png"
            plot_deep_training_curves(history, output)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
