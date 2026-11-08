#!/usr/bin/env python3
"""Compile the CVPR report and write a complete, auditable build report.

This command never modifies experiment evidence.  It invokes an existing
TeX Live installation, validates the resulting PDF, and atomically installs
the final deliverable under ``report/``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX_DIR = PROJECT_ROOT / "report" / "cvpr_latex"
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "COMP9517_Group_Project_Report.pdf"
DEFAULT_REPORT = PROJECT_ROOT / "LATEX_BUILD_REPORT.md"
PDF_LIMIT_BYTES = 10_000_000
MAIN_TEXT_PAGE_LIMIT = 10


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    cwd: Path
    returncode: int
    output: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
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


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _latex_bin_dir() -> Path | None:
    """Return the optional, validated directory containing TeX executables."""

    configured = os.environ.get("LATEX_BIN_DIR", "")
    if not configured:
        return None
    directory = Path(configured).expanduser().absolute()
    if not directory.is_dir():
        raise NotADirectoryError(
            f"LATEX_BIN_DIR is not a directory: {directory}. "
            "Set LATEX_BIN_DIR to the directory containing the TeX executables, "
            "or unset it to use PATH."
        )
    return directory


def _latex_search_path() -> str | None:
    """Return PATH with LATEX_BIN_DIR first, or None for normal PATH lookup."""

    directory = _latex_bin_dir()
    if directory is None:
        return None
    existing_path = os.environ.get("PATH", "")
    return os.pathsep.join(part for part in (str(directory), existing_path) if part)


def _find_command(command: str, *, search_path: str | None = None) -> Path | None:
    located = (
        shutil.which(command)
        if search_path is None
        else shutil.which(command, path=search_path)
    )
    if located:
        # Preserve the discovered launcher path instead of resolving symlinks.
        # TeX Live's ``bin/<platform>/latexmk`` is a symlink into
        # ``texmf-dist/scripts``; resolving it would make _run prepend the
        # script directory rather than the sibling-binary directory containing
        # pdflatex and BibTeX.
        return Path(located).absolute()
    return None


def _run(command: list[str], cwd: Path) -> CommandResult:
    environment = {**os.environ, "LC_ALL": "C"}
    executable_directory = str(Path(command[0]).parent)
    latex_directory = _latex_bin_dir()
    existing_path = environment.get("PATH", os.defpath)
    prefixes = (
        (str(latex_directory), executable_directory)
        if latex_directory is not None
        else (executable_directory,)
    )
    path_parts: list[str] = []
    for part in (*prefixes, *existing_path.split(os.pathsep)):
        if part and part not in path_parts:
            path_parts.append(part)
    environment["PATH"] = os.pathsep.join(path_parts)
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=environment,
    )
    return CommandResult(tuple(command), cwd, process.returncode, process.stdout)


def _compile(tex_dir: Path, build_dir: Path) -> list[CommandResult]:
    search_path = _latex_search_path()
    pdflatex = _find_command("pdflatex", search_path=search_path)
    bibtex = _find_command("bibtex", search_path=search_path)
    latexmk = _find_command("latexmk", search_path=search_path)
    missing = [
        command
        for command, located in (("pdflatex", pdflatex), ("bibtex", bibtex))
        if located is None
    ]
    if missing:
        raise FileNotFoundError(
            "Required TeX command(s) not found: "
            + ", ".join(missing)
            + ". Install MacTeX on macOS or TeX Live on Linux, then ensure "
            "pdflatex and bibtex are on PATH; alternatively, set LATEX_BIN_DIR "
            "to the directory containing those executables."
        )

    if latexmk is not None:
        command = [
            str(latexmk),
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-outdir={build_dir}",
            "main.tex",
        ]
        return [_run(command, tex_dir)]

    # These are guaranteed by the required-tool check above.
    assert pdflatex is not None
    assert bibtex is not None
    build_dir.mkdir(parents=True, exist_ok=True)
    pdf_command = [
        str(pdflatex),
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={build_dir}",
        "main.tex",
    ]
    results = [_run(pdf_command, tex_dir)]
    if results[-1].returncode != 0:
        return results
    bibtex_target = (build_dir.relative_to(tex_dir) / "main").as_posix()
    results.append(_run([str(bibtex), bibtex_target], tex_dir))
    if results[-1].returncode != 0:
        return results
    results.extend((_run(pdf_command, tex_dir), _run(pdf_command, tex_dir)))
    return results


def _main_text_page(aux_path: Path) -> int | None:
    if not aux_path.is_file():
        return None
    text = aux_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"\\newlabel\{LastPageMainText\}\{\{[^}]*\}\{([0-9]+)\}", text
    )
    return int(match.group(1)) if match else None


def _pdf_pages(pdf_path: Path) -> tuple[int | None, str]:
    pdfinfo = _find_command("pdfinfo")
    if pdfinfo is None:
        return (
            None,
            "pdfinfo is unavailable. Install Poppler (for example, "
            "`brew install poppler` on macOS or the `poppler-utils` package "
            "on Debian/Ubuntu Linux) and ensure pdfinfo is on PATH.",
        )
    result = _run([str(pdfinfo), str(pdf_path)], pdf_path.parent)
    match = re.search(r"^Pages:\s+([0-9]+)\s*$", result.output, re.MULTILINE)
    return (int(match.group(1)) if match else None), result.output


def _render_build_report(
    *,
    status: str,
    commands: list[CommandResult],
    checks: list[tuple[str, str]],
    pdf_path: Path,
    error: str | None,
) -> str:
    lines = [
        "# LaTeX Build Report",
        "",
        f"**Status: {status}**",
        "",
        "The report build performs no model training or inference and does not modify",
        "frozen Stage 1-4 evidence.",
        "",
        "## Commands and exact output",
        "",
    ]
    if commands:
        for index, result in enumerate(commands, start=1):
            transcript = "\n".join(
                line.rstrip() for line in result.output.rstrip().splitlines()
            )
            lines.extend(
                [
                    f"### Command {index}",
                    "",
                    f"Working directory: `{result.cwd}`",
                    "",
                    "```text",
                    "$ " + " ".join(result.command),
                    transcript,
                    f"[exit code: {result.returncode}]",
                    "```",
                    "",
                ]
            )
    else:
        lines.extend(["No compiler command could be started.", ""])
    lines.extend(["## Acceptance checks", "", "| Check | Result |", "|---|---|"])
    lines.extend(f"| {label} | {value} |" for label, value in checks)
    lines.extend(["", "## Deliverable", "", f"`{pdf_path}`", ""])
    if error:
        lines.extend(["## Error", "", "```text", error, "```", ""])
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> int:
    tex_dir = args.tex_dir.resolve()
    main = tex_dir / "main.tex"
    build_dir = tex_dir / "build"
    output = args.output.resolve()
    report_path = args.report.resolve()
    commands: list[CommandResult] = []
    checks: list[tuple[str, str]] = []
    error: str | None = None
    status = "FAIL"
    try:
        if not main.is_file():
            raise FileNotFoundError(f"Missing report entry point: {main}")
        if output.exists() and not args.force:
            raise FileExistsError(f"Final PDF exists; pass --force to replace it: {output}")
        build_dir.mkdir(parents=True, exist_ok=True)
        commands = _compile(tex_dir, build_dir)
        failed = next((result for result in commands if result.returncode != 0), None)
        if failed is not None:
            raise RuntimeError(
                f"LaTeX command failed with exit code {failed.returncode}: "
                + " ".join(failed.command)
            )
        built_pdf = build_dir / "main.pdf"
        if not built_pdf.is_file() or built_pdf.stat().st_size == 0:
            raise FileNotFoundError(f"Compiler did not create {built_pdf}")
        log_path = build_dir / "main.log"
        file_log = (
            log_path.read_text(encoding="utf-8", errors="replace")
            if log_path.is_file()
            else ""
        )
        # latexmk's transcript contains every intermediate pass, where
        # undefined citations/references are expected before BibTeX and the
        # final reruns. Acceptance must be based on the final TeX log; fall
        # back to command output only if no log was produced.
        acceptance_log = file_log or "\n".join(result.output for result in commands)
        unresolved = bool(
            re.search(
                r"undefined (?:citations|references)|Citation .* undefined|Reference .* undefined|There were undefined references",
                acceptance_log,
                flags=re.IGNORECASE,
            )
        )
        checks.append(("Undefined citations/references", "FAIL" if unresolved else "PASS"))
        if unresolved:
            raise RuntimeError("LaTeX log contains unresolved citations or references")

        overfull_hbox = len(re.findall(r"Overfull \\hbox", acceptance_log))
        overfull_vbox = len(re.findall(r"Overfull \\vbox", acceptance_log))
        underfull_hbox = len(re.findall(r"Underfull \\hbox", acceptance_log))
        underfull_vbox = len(re.findall(r"Underfull \\vbox", acceptance_log))
        float_warnings = len(
            re.findall(
                r"Float too large|Too many unprocessed floats|float specifier changed",
                acceptance_log,
                flags=re.IGNORECASE,
            )
        )
        duplicate_warnings = len(
            re.findall(
                r"multiply defined|destination with the same identifier|duplicate destination",
                acceptance_log,
                flags=re.IGNORECASE,
            )
        )
        checks.extend(
            [
                ("Overfull hbox warnings", str(overfull_hbox)),
                ("Overfull vbox warnings", str(overfull_vbox)),
                ("Underfull hbox warnings (recorded)", str(underfull_hbox)),
                ("Underfull vbox warnings (recorded)", str(underfull_vbox)),
                ("Float layout warnings", str(float_warnings)),
                ("Duplicate destination/label warnings", str(duplicate_warnings)),
            ]
        )
        if overfull_hbox or overfull_vbox or float_warnings or duplicate_warnings:
            raise RuntimeError(
                "LaTeX layout acceptance failed: "
                f"overfull_hbox={overfull_hbox}, overfull_vbox={overfull_vbox}, "
                f"float_warnings={float_warnings}, duplicate_warnings={duplicate_warnings}"
            )

        page_count, pdfinfo_output = _pdf_pages(built_pdf)
        main_page = _main_text_page(build_dir / "main.aux")
        size = built_pdf.stat().st_size
        digest = _sha256(built_pdf)
        checks.extend(
            [
                ("PDF exists", "PASS"),
                ("PDF page count", str(page_count) if page_count is not None else "FAIL"),
                (
                    "Main-text last page",
                    str(main_page) if main_page is not None else "NOT CHECKED",
                ),
                (
                    f"Main text <= {MAIN_TEXT_PAGE_LIMIT} pages",
                    "PASS" if main_page is not None and main_page <= MAIN_TEXT_PAGE_LIMIT else "FAIL",
                ),
                ("PDF size bytes", str(size)),
                (
                    f"PDF size < {PDF_LIMIT_BYTES} bytes",
                    "PASS" if size < PDF_LIMIT_BYTES else "FAIL",
                ),
                ("PDF SHA-256", digest),
                (
                    "pdfinfo availability",
                    "PASS" if pdfinfo_output and "Pages:" in pdfinfo_output else "FAIL",
                ),
            ]
        )
        if page_count is None:
            raise RuntimeError(
                "Unable to verify PDF page count with pdfinfo. " + pdfinfo_output
            )
        if main_page is None or main_page > MAIN_TEXT_PAGE_LIMIT:
            raise RuntimeError("Unable to prove the main-text page limit")
        if page_count < main_page:
            raise RuntimeError(
                f"PDF page count {page_count} is smaller than main-text label {main_page}"
            )
        if size >= PDF_LIMIT_BYTES:
            raise RuntimeError("Final report PDF exceeds the 10 MB upload limit")
        # Install the canonical deliverable only after every source-PDF check
        # has passed. A failed rebuild therefore cannot replace a known-good PDF.
        _atomic_copy(built_pdf, output)
        if _sha256(output) != digest or output.stat().st_size != size:
            raise RuntimeError("Atomic PDF installation failed integrity verification")
        status = "PASS"
    except Exception as exception:  # preserve exact failure in the report
        error = f"{type(exception).__name__}: {exception}"
        if not checks:
            checks.append(("Build", "FAIL"))
    finally:
        _atomic_text(
            report_path,
            _render_build_report(
                status=status,
                commands=commands,
                checks=checks,
                pdf_path=output,
                error=error,
            ),
        )
    if status == "PASS":
        print(f"[PASS] Compiled report: {output}")
        print(f"[PASS] Build audit: {report_path}")
        return 0
    print(f"[FAIL] Report build failed: {error}", file=sys.stderr)
    print(f"[INFO] Exact build evidence: {report_path}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex-dir", type=Path, default=DEFAULT_TEX_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return build_report(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
