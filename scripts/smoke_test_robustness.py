#!/usr/bin/env python3
"""Real-image smoke test for Stage 4 corruptions and frozen-model inference.

The command requires explicit held-out-test acknowledgement.  Model outputs
are always written below ``results/smoke/robustness`` and therefore cannot be
mistaken for canonical full robustness results.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.robustness.corruptions import CORRUPTION_SEVERITIES, apply_corruption, prepare_rgb
from src.robustness.evaluate_robustness import evaluate_one
from src.robustness.freeze_protocol import (
    CANONICAL_PROTOCOL,
    EXPECTED_MODEL_NAMES,
    load_and_verify_protocol,
)


DEFAULT_OUTPUT = Path("results/figures/robustness/corruption_examples.png")
DEFAULT_SMOKE_DIR = Path("results/smoke/robustness")


def _save_grid(
    image: Image.Image,
    *,
    image_id: int,
    output: Path,
    seed: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(4, 5, figsize=(15, 12), constrained_layout=True)
    for row, (corruption_type, values) in enumerate(CORRUPTION_SEVERITIES.items()):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"Original\n{corruption_type}")
        axes[row, 0].axis("off")
        for severity, value in enumerate(values, start=1):
            corrupted = apply_corruption(
                image,
                corruption_type,
                severity,
                seed=seed,
                image_key=image_id,
            )
            axes[row, severity].imshow(corrupted)
            axes[row, severity].set_title(f"s{severity}: {value}")
            axes[row, severity].axis("off")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=output.suffix, dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        figure.savefig(temporary, dpi=160, bbox_inches="tight")
        os.replace(temporary, output)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test all Stage 4 corruptions and selected frozen models."
    )
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--protocol", type=Path, default=CANONICAL_PROTOCOL)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke-output-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    # The traditional feature extractor uses a process pool and therefore
    # requires at least one worker.  A default of one keeps the smoke test
    # portable while remaining valid for every frozen model family.
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-model-evaluation",
        action="store_true",
        help="Generate/validate the corruption grid without running frozen models",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.include_test:
        parser.error("the real-image smoke test requires --include-test")
    if args.seed != 42:
        parser.error("the frozen Stage 4 smoke seed is 42")
    if not 1 <= args.max_samples < 5000:
        parser.error("--max-samples must be between 1 and 4999")
    if args.batch_size < 1 or args.num_workers < 1:
        parser.error("batch-size and num-workers must be positive")

    root = args.project_root.resolve()
    protocol_path = args.protocol if args.protocol.is_absolute() else root / args.protocol
    try:
        protocol, digest = load_and_verify_protocol(protocol_path, project_root=root)
        test_frame = pd.read_csv(root / "data/splits/test.csv", dtype=str, keep_default_na=False)
        first = test_frame.iloc[0]
        relative = PurePosixPath(str(first["local_image_path"]))
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("test",):
            raise ValueError(f"unsafe smoke image path: {relative}")
        image_path = root / "data/processed/images" / Path(*relative.parts)
        with Image.open(image_path) as opened:
            image = prepare_rgb(opened)
        output = args.output if args.output.is_absolute() else root / args.output
        _save_grid(
            image,
            image_id=int(first["image_id"]),
            output=output,
            seed=args.seed,
        )
        print(f"[PASS] All 16 deterministic corruptions produced valid RGB examples: {output}")

        if not args.skip_model_evaluation:
            smoke_output = (
                args.smoke_output_dir
                if args.smoke_output_dir.is_absolute()
                else root / args.smoke_output_dir
            )
            for model_name in EXPECTED_MODEL_NAMES:
                evaluate_one(
                    protocol=protocol,
                    protocol_path=protocol_path.resolve(),
                    protocol_sha256=digest,
                    model_name=model_name,
                    corruption_type="gaussian_noise",
                    severity=1,
                    project_root=root,
                    batch_size=args.batch_size,
                    workers=args.num_workers,
                    device=args.device,
                    max_samples=args.max_samples,
                    output_dir=smoke_output,
                )
                print(f"[PASS] Frozen-model smoke evaluation: {model_name}")
        else:
            print("[SKIP] Frozen-model smoke evaluation disabled by flag")
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print(f"[FAIL] {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
