#!/usr/bin/env python3
"""Validate final COMP9517 code-ZIP and report-PDF upload limits."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_code_zip import (  # noqa: E402
    ARCHIVE_ROOT,
    CODE_ZIP_LIMIT_BYTES,
    DEFAULT_OUTPUT,
    PackagingError,
    collect_allowed_files,
    validate_code_zip,
)


REPORT_PDF_LIMIT_BYTES = 10_000_000
DEFAULT_REPORT_PDF = Path("report/COMP9517_Group_Project_Report.pdf")
DEFAULT_AUDIT_REPORT = Path("SUBMISSION_PACKAGE_AUDIT.md")


def validate_clean_extraction(
    path: str | Path,
) -> dict[str, Any]:
    """Extract into a fresh directory and compile every submitted Python tree.

    ``validate_code_zip`` must be called first. This second, deliberately
    independent acceptance step proves that the archive expands beneath exactly
    one project directory and that its submitted Python files compile in a clean
    source-only checkout using the current interpreter.
    """

    candidate = Path(path).expanduser().resolve()
    # Keep this helper safe when called directly as well as through
    # ``check_submission``; the duplicated validation is cheap for this small,
    # source-only archive and prevents unsafe extraction of an unchecked ZIP.
    validate_code_zip(candidate)
    with tempfile.TemporaryDirectory(prefix="comp9517-code-zip-check-") as directory:
        extraction_root = Path(directory)
        try:
            with zipfile.ZipFile(candidate, "r") as archive:
                archive.extractall(extraction_root)
                expected_files = {
                    Path(*Path(info.filename).parts)
                    for info in archive.infolist()
                    if not info.is_dir()
                }
        except (OSError, zipfile.BadZipFile) as error:
            raise PackagingError(f"Clean ZIP extraction failed: {error}") from error

        top_level = list(extraction_root.iterdir())
        if len(top_level) != 1 or top_level[0].name != ARCHIVE_ROOT:
            observed = sorted(path.name for path in top_level)
            raise PackagingError(
                f"Clean extraction requires exactly one {ARCHIVE_ROOT}/ root; "
                f"observed={observed}"
            )
        project = top_level[0]
        if not project.is_dir() or project.is_symlink():
            raise PackagingError("Extracted project root is not a safe directory")

        observed_files: set[Path] = set()
        for extracted in extraction_root.rglob("*"):
            if extracted.is_symlink():
                raise PackagingError(f"Extracted archive contains a symlink: {extracted}")
            if extracted.is_file():
                observed_files.add(extracted.relative_to(extraction_root))
        if observed_files != expected_files:
            missing = sorted(value.as_posix() for value in expected_files - observed_files)
            extra = sorted(value.as_posix() for value in observed_files - expected_files)
            raise PackagingError(
                f"Clean extraction file mismatch; missing={missing[:5]}, extra={extra[:5]}"
            )

        compile_targets = [
            value
            for value in ("src", "scripts", "tests")
            if (project / value).is_dir()
        ]
        if compile_targets != ["src", "scripts", "tests"]:
            raise PackagingError(
                "Clean extraction is missing one or more Python trees: src, scripts, tests"
            )
        completed = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", *compile_targets],
            cwd=project,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            details = (completed.stdout + completed.stderr).strip()
            raise PackagingError(
                "Extracted source compilation failed"
                + (f": {details}" if details else "")
            )
        return {
            "status": "PASS",
            "top_level_root": ARCHIVE_ROOT,
            "extracted_file_count": len(observed_files),
            "compileall": {
                "status": "PASS",
                "returncode": completed.returncode,
                "targets": compile_targets,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "interpreter": sys.executable,
            },
        }


def validate_report_pdf(
    path: str | Path,
    *,
    project_root: str | Path = PROJECT_ROOT,
    max_bytes: int = REPORT_PDF_LIMIT_BYTES,
) -> dict[str, int | str]:
    """Check that the report is a nonempty PDF within the decimal upload limit."""

    root = Path(project_root).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise PackagingError(f"Report PDF does not exist: {candidate}")
    size = candidate.stat().st_size
    if size <= 8:
        raise PackagingError(f"Report PDF is empty or truncated: {candidate}")
    if size > max_bytes:
        raise PackagingError(
            f"Report PDF is {size:,} bytes; limit is {max_bytes:,} bytes"
        )
    with candidate.open("rb") as stream:
        header = stream.read(8)
        stream.seek(max(0, size - 1024))
        trailer = stream.read()
    if not header.startswith(b"%PDF-"):
        raise PackagingError(f"Report does not have a PDF header: {candidate}")
    if b"%%EOF" not in trailer:
        raise PackagingError(f"Report PDF has no terminal EOF marker: {candidate}")
    return {"report_path": candidate.as_posix(), "report_bytes": size}


def check_submission(
    *,
    project_root: str | Path = PROJECT_ROOT,
    code_zip: str | Path = DEFAULT_OUTPUT,
    report_pdf: str | Path = DEFAULT_REPORT_PDF,
    skip_report: bool = False,
    compare_sources: bool = True,
) -> dict[str, Any]:
    """Return a machine-readable validation result or raise on any failure."""

    root = Path(project_root).expanduser().resolve()
    zip_path = Path(code_zip).expanduser()
    if not zip_path.is_absolute():
        zip_path = root / zip_path
    expected = collect_allowed_files(root) if compare_sources else None
    code = validate_code_zip(
        zip_path,
        expected_files=expected,
        project_root=root if expected is not None else None,
    )
    clean_extraction = validate_clean_extraction(zip_path)
    result: dict[str, Any] = {
        "status": "PASS",
        "limits_are_decimal_bytes": True,
        "code_zip_limit_bytes": CODE_ZIP_LIMIT_BYTES,
        "code_zip": code,
        "clean_extraction": clean_extraction,
    }
    if skip_report:
        result["report_pdf"] = {"status": "SKIP", "reason": "--skip-report"}
    else:
        report = validate_report_pdf(report_pdf, project_root=root)
        result["report_pdf_limit_bytes"] = REPORT_PDF_LIMIT_BYTES
        result["report_pdf"] = report
    if expected is not None:
        result["allowed_files"] = [path.as_posix() for path in expected]
    return result


def render_audit_report(result: dict[str, Any], *, command: Sequence[str]) -> str:
    """Render a factual package audit from one completed checker invocation."""

    code = result["code_zip"]
    report = result["report_pdf"]
    lines = [
        "# Submission Package Audit",
        "",
        "**Status: PASS**",
        "",
        "This report records a completed deterministic package validation. The code",
        "archive and report PDF are separate submission deliverables.",
        "",
        "## Command",
        "",
        "```text",
        shlex.join([str(value) for value in command]),
        "```",
        "",
        "## Size and integrity checks",
        "",
        "| Deliverable | Observed bytes | Limit bytes | Result |",
        "|---|---:|---:|---|",
        (
            f"| Code ZIP `{code['zip_path']}` | {int(code['zip_bytes'])} | "
            f"{CODE_ZIP_LIMIT_BYTES} | PASS |"
        ),
    ]
    if report.get("status") == "SKIP":
        lines.append("| Report PDF | NOT CHECKED | 10000000 | SKIP |")
    else:
        lines.append(
            f"| Report PDF `{report['report_path']}` | {int(report['report_bytes'])} | "
            f"{REPORT_PDF_LIMIT_BYTES} | PASS |"
        )
    lines.extend(
        [
            "",
            f"- Code ZIP files (excluding checksum manifest): **{code['file_count']}**",
            f"- Code ZIP checksum-manifest SHA-256: `{code['manifest_sha256']}`",
            "- ZIP paths, CRCs, member checksums, symlink/traversal safety, and the",
            "  explicit source allowlist were checked.",
            (
                f"- A fresh extraction contained one `{result['clean_extraction']['top_level_root']}/` "
                "root and exactly the archived files."
            ),
            "- `compileall -q src scripts tests` passed inside that clean extraction.",
            "- Images, PDFs, model/checkpoint files, NumPy arrays, raw/processed data,",
            "  predictions, logs, features, caches, and unrelated assets are forbidden",
            "  from the code ZIP.",
            "",
            "## ZIP entries",
            "",
        ]
    )
    entries = code.get("entry_names")
    if isinstance(entries, list):
        lines.extend(f"- `{value}`" for value in entries)
    else:
        lines.append("- NOT LISTED (validator did not return entry names).")
    lines.append("")
    return "\n".join(lines)


def atomic_write_audit(
    path: str | Path,
    content: str,
    *,
    project_root: str | Path = PROJECT_ROOT,
    force: bool = False,
) -> Path:
    """Atomically install a UTF-8 audit report, refusing silent replacement."""

    root = Path(project_root).expanduser().resolve()
    destination = Path(path).expanduser()
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise PackagingError("Audit report must remain inside the project root") from error
    if destination.suffix.lower() != ".md":
        raise PackagingError("Audit report must have a .md extension")
    payload = content.encode("utf-8")
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise PackagingError(f"Unsafe audit-report destination: {destination}")
        if destination.read_bytes() == payload:
            return destination
        if not force:
            raise FileExistsError(
                f"A different audit report already exists: {destination}; use --force deliberately"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate code ZIP (40 MB) and report PDF (10 MB) submissions."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--code-zip", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-pdf", type=Path, default=DEFAULT_REPORT_PDF)
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_AUDIT_REPORT,
        help="Atomic Markdown audit output",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Validate only the code ZIP while the report is still being prepared",
    )
    parser.add_argument(
        "--no-source-compare",
        action="store_true",
        help="Validate a standalone ZIP without comparing it to this workspace",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--force-report",
        action="store_true",
        help="Replace an existing different audit report atomically",
    )
    parser.add_argument(
        "--no-report-file",
        action="store_true",
        help="Print checks without writing the Markdown audit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)
    try:
        result = check_submission(
            project_root=args.project_root,
            code_zip=args.code_zip,
            report_pdf=args.report_pdf,
            skip_report=args.skip_report,
            compare_sources=not args.no_source_compare,
        )
    except (OSError, ValueError, TypeError, PackagingError) as error:
        if args.json:
            print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
        else:
            print(f"[FAIL] {error}", file=os.sys.stderr)
        return 1
    if not args.no_report_file:
        try:
            audit_path = atomic_write_audit(
                args.report,
                render_audit_report(
                    result,
                    command=[sys.executable, str(Path(__file__).resolve()), *raw_argv],
                ),
                project_root=args.project_root,
                force=args.force_report,
            )
            result["audit_report"] = audit_path.as_posix()
        except (OSError, ValueError, TypeError, PackagingError) as error:
            if args.json:
                print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True))
            else:
                print(f"[FAIL] {error}", file=os.sys.stderr)
            return 1
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        code = result["code_zip"]
        print(
            f"[PASS] Code ZIP: {int(code['zip_bytes']):,} / "
            f"{CODE_ZIP_LIMIT_BYTES:,} bytes; {code['file_count']} files"
        )
        clean = result["clean_extraction"]
        print(
            f"[PASS] Clean extraction: {clean['top_level_root']}/; "
            f"{clean['extracted_file_count']} files"
        )
        print("[PASS] Extracted compileall: src scripts tests")
        report = result["report_pdf"]
        if report.get("status") == "SKIP":
            print("[SKIP] Report PDF size check (--skip-report)")
        else:
            print(
                f"[PASS] Report PDF: {int(report['report_bytes']):,} / "
                f"{REPORT_PDF_LIMIT_BYTES:,} bytes"
            )
        if "audit_report" in result:
            print(f"[PASS] Audit report: {result['audit_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REPORT_PDF",
    "DEFAULT_AUDIT_REPORT",
    "REPORT_PDF_LIMIT_BYTES",
    "check_submission",
    "atomic_write_audit",
    "render_audit_report",
    "validate_clean_extraction",
    "validate_report_pdf",
]
