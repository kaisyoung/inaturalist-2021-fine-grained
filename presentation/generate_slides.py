#!/usr/bin/env python3
"""Deterministically rebuild the COMP9517 presentation using artifact-tool.

This wrapper uses Node.js from ``PATH`` (or the ``CODEX_NODE`` compatibility
override), optionally prepares an explicitly configured artifact-tool workspace,
and then runs ``generate_slides.mjs``. It never trains a model, runs inference,
or writes outside ``presentation/``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "COMP9517_Final_Presentation_10min.pptx"
DEFAULT_WORKSPACE = HERE / ".tmp" / "build"
DEFAULT_RENDER_DIR = HERE / ".tmp" / "artifact-render"


def _node_executable() -> Path:
    explicit = os.environ.get("CODEX_NODE")
    if explicit:
        candidate = shutil.which(str(Path(explicit).expanduser()))
        if candidate:
            return Path(candidate)
        raise FileNotFoundError(
            "CODEX_NODE is set but does not name an executable Node.js binary: "
            f"{explicit!r}. Correct the override or unset it to use `node` from PATH."
        )

    candidate = shutil.which("node")
    if candidate:
        return Path(candidate)
    raise FileNotFoundError(
        "Node.js was not found on PATH. Install Node.js and ensure `node` is on "
        "PATH, or set CODEX_NODE to an executable Node.js binary."
    )


def _setup_script() -> Path | None:
    explicit = os.environ.get("PRESENTATIONS_SKILL_DIR")
    if not explicit:
        return None

    candidate = (
        Path(explicit).expanduser()
        / "container_tools"
        / "setup_artifact_tool_workspace.mjs"
    )
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "PRESENTATIONS_SKILL_DIR is set but its artifact-tool workspace setup "
        f"script was not found at {candidate}. Correct the directory, or unset "
        "PRESENTATIONS_SKILL_DIR to use dependencies already installed in the "
        "selected --workspace."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_RENDER_DIR)
    args = parser.parse_args(argv)

    output = args.output.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()
    render_dir = args.render_dir.expanduser().resolve()
    for destination in (output, workspace, render_dir):
        try:
            destination.relative_to(HERE)
        except ValueError as error:
            raise ValueError(
                f"All generated files must remain under {HERE}: {destination}"
            ) from error

    node = _node_executable()
    setup = _setup_script()
    workspace.mkdir(parents=True, exist_ok=True)
    if setup is not None:
        subprocess.run(
            [str(node), str(setup), "--workspace", str(workspace)],
            check=True,
            cwd=HERE,
        )

    command = [
        str(node),
        str(HERE / "generate_slides.mjs"),
        "--output",
        str(output),
        "--workspace",
        str(workspace),
        "--render-dir",
        str(render_dir),
    ]
    subprocess.run(command, check=True, cwd=HERE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
