#!/usr/bin/env python3
"""Run the required non-training Stage 5 verification and capture exact output."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "FINAL_REPRODUCIBILITY_AUDIT.md"
PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "tqdm",
    "scikit-learn",
    "scikit-image",
    "Pillow",
    "matplotlib",
    "joblib",
    "opencv-python-headless",
    "torch",
    "torchvision",
    "PyYAML",
)


@dataclass(frozen=True)
class AuditRun:
    label: str
    command: tuple[str, ...]
    returncode: int
    output: str
    duration_seconds: float


def _atomic_text(path: Path, content: str, *, force: bool = False) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise FileExistsError(f"Refusing unsafe audit report path: {path}")
        if path.read_text(encoding="utf-8") == content:
            return
        if not force:
            raise FileExistsError(
                f"A different reproducibility audit already exists: {path}; "
                "use --force deliberately"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run(label: str, command: list[str]) -> AuditRun:
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "PYTHONHASHSEED": "42"},
    )
    return AuditRun(
        label,
        tuple(command),
        process.returncode,
        process.stdout,
        time.perf_counter() - started,
    )


def _package_versions() -> list[str]:
    versions: list[str] = []
    for name in PACKAGE_NAMES:
        try:
            versions.append(f"- `{name}=={importlib.metadata.version(name)}`")
        except importlib.metadata.PackageNotFoundError:
            versions.append(f"- `{name}: MISSING`")
    return versions


def _render(runs: list[AuditRun], python: Path) -> str:
    passed = all(run.returncode == 0 for run in runs)
    lines = [
        "# Final Reproducibility Audit",
        "",
        f"**Status: {'PASS' if passed else 'FAIL'}**",
        "",
        "These commands perform dependency, compilation, unit-test, report-source,",
        "package-allowlist, and dataset validation checks only. They do not invoke",
        "canonical training, inference, report generation, or ZIP creation.",
        "Synthetic optimizer steps inside unit tests do not modify saved experiments.",
        "",
        "## Environment",
        "",
        f"- Python executable: `{python}`",
        f"- Python version: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        "",
        "### Direct package versions",
        "",
        *_package_versions(),
        "",
        "## Command summary",
        "",
        "| Check | Exit code | Duration (s) | Result |",
        "|---|---:|---:|---|",
    ]
    for run in runs:
        lines.append(
            f"| {run.label} | {run.returncode} | {run.duration_seconds:.3f} | "
            f"{'PASS' if run.returncode == 0 else 'FAIL'} |"
        )
    lines.extend(["", "## Exact command output", ""])
    for run in runs:
        lines.extend(
            [
                f"### {run.label}",
                "",
                "```text",
                "$ " + " ".join(run.command),
                run.output.rstrip(),
                f"[exit code: {run.returncode}]",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Known warnings",
            "",
            "Any warning emitted by the commands is retained verbatim above. A pip cache",
            "permission warning is environmental and is non-fatal when `pip check` reports",
            "no broken requirements.",
            "",
            "## Decision",
            "",
            "All configured final reproducibility checks pass."
            if passed
            else "One or more critical reproducibility checks failed; see exact output above.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(args: argparse.Namespace) -> int:
    # Preserve a virtual-environment launcher instead of resolving its symlink
    # to the base runtime. Python uses the launcher path to select the venv's
    # prefix and site-packages for every child audit command.
    python = args.python.absolute()
    if not python.is_file():
        raise FileNotFoundError(f"Python interpreter not found: {python}")
    commands = [
        ("Dependency consistency", [str(python), "-m", "pip", "check"]),
        (
            "Python compilation",
            [str(python), "-m", "compileall", "-q", "src", "scripts", "tests"],
        ),
        (
            "Full unit-test suite",
            [str(python), "-m", "unittest", "discover", "-s", "tests", "-v"],
        ),
        (
            "Final report source audit",
            [
                str(python),
                "scripts/audit_final_report.py",
                "--no-write-reports",
            ],
        ),
        (
            "Code ZIP allowlist preflight",
            [str(python), "scripts/make_code_zip.py", "--list-only"],
        ),
        (
            "Frozen split and image validation",
            [
                str(python),
                "-m",
                "src.data.validate_splits",
                "--split-dir",
                "data/splits",
                "--train-json",
                "data/raw/train_mini.json",
                "--val-json",
                "data/raw/val.json",
                "--image-root",
                "data/processed/images",
                "--hash-files",
            ],
        ),
    ]
    runs = [_run(label, command) for label, command in commands]
    _atomic_text(
        args.report.resolve(),
        _render(runs, python),
        force=getattr(args, "force", False),
    )
    for run in runs:
        print(f"[{'PASS' if run.returncode == 0 else 'FAIL'}] {run.label}: exit {run.returncode}")
    print(f"Audit report: {args.report.resolve()}")
    return 0 if all(run.returncode == 0 for run in runs) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv/bin/python")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing different audit report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return run_audit(build_parser().parse_args(argv))
    except (OSError, UnicodeError, ValueError) as error:
        print(f"[FAIL] {error}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
