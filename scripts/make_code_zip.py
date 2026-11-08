#!/usr/bin/env python3
"""Build the deterministic, source-only COMP9517 code submission ZIP.

The archive is assembled from an explicit allowlist.  It does not consult Git
or ``.gitignore`` and therefore cannot accidentally include large local
artifacts merely because this workspace is not a Git worktree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("submission/COMP9517_Group_Project_Code.zip")
ARCHIVE_ROOT = "COMP9517_Group_Project_Code"
MANIFEST_NAME = "CODE_ZIP_MANIFEST.sha256"
CODE_ZIP_LIMIT_BYTES = 40_000_000
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

EXACT_ROOT_FILES = (
    Path(".gitignore"),
    Path("README.md"),
    Path("requirements.txt"),
    Path("presentation/generate_slides.py"),
    Path("presentation/generate_slides.mjs"),
    Path("results/tables/final_report_tables_manifest.json"),
)
ALLOWLIST_RULES: tuple[tuple[Path, frozenset[str]], ...] = (
    (Path("src"), frozenset({".py"})),
    (Path("scripts"), frozenset({".py"})),
    (Path("configs"), frozenset({".yaml", ".yml"})),
    (Path("tests"), frozenset({".py"})),
    (Path("data/splits"), frozenset({".csv", ".json", ".txt"})),
    (Path("results/metrics"), frozenset({".json"})),
    (Path("results/tables"), frozenset({".csv", ".md"})),
    (
        Path("report/cvpr_latex"),
        frozenset({".tex", ".bib", ".sty", ".bst", ".md"}),
    ),
)
REQUIRED_SPLIT_FILES = (
    Path("data/splits/selected_classes_seed42.txt"),
    Path("data/splits/class_mapping.json"),
    Path("data/splits/train.csv"),
    Path("data/splits/val.csv"),
    Path("data/splits/test.csv"),
    Path("data/splits/split_summary.json"),
    Path("data/splits/selected_classes_readable.csv"),
)
REQUIRED_REWRITE_EVIDENCE_FILES = (
    Path("results/metrics/stage2_selection_manifest.json"),
    Path("results/metrics/stage2_freeze.json"),
    Path(
        "results/metrics/deep_learning/"
        "stage3_traditional_rewrite_continuity.json"
    ),
    Path("results/tables/traditional_validation_grid.csv"),
    Path("results/tables/stage2_traditional_results.csv"),
    Path("results/tables/stage2_feature_ablation_results.csv"),
    Path("results/tables/final_model_comparison.csv"),
    Path("results/metrics/stage4_robustness_protocol.json"),
    Path("results/tables/robustness_results.csv"),
    Path("results/tables/robustness_drop_summary.csv"),
    Path("results/tables/robustness_model_ranking.csv"),
    Path("results/tables/stage4_summary.csv"),
    Path("results/tables/final_model_comparison_with_robustness.csv"),
    Path("results/tables/final_report_main_results.csv"),
    Path("results/tables/final_report_ablation_results.csv"),
    Path("results/tables/final_report_robustness_summary.csv"),
    Path("results/tables/final_report_tables_manifest.json"),
)
REQUIRED_REPORT_TABLE_OUTPUTS = frozenset(
    {
        "results/tables/final_report_main_results.csv",
        "results/tables/final_report_ablation_results.csv",
        "results/tables/final_report_robustness_summary.csv",
    }
)
FORBIDDEN_LEGACY_EVIDENCE_FILES = frozenset(
    {Path("results/metrics/linear_svm_linearsvc_attempt.json")}
)

MAX_ARCHIVE_MEMBERS = 10_000
MAX_MEMBER_UNCOMPRESSED_BYTES = 20_000_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 40_000_000
MAX_COMPRESSION_RATIO = 200.0
FORBIDDEN_PREFIXES = (
    ".venv/",
    "venv/",
    "env/",
    "data/raw/",
    "data/processed/",
    "docs/",
    "pet-runs/",
    "notebooks/",
    "results/models/",
    "results/features/",
    "results/predictions/",
    "results/logs/",
    "results/figures/",
    "submission/",
)
FORBIDDEN_SUFFIXES = frozenset(
    {
        ".aux",
        ".ckpt",
        ".DS_Store",
        ".jpeg",
        ".jpg",
        ".log",
        ".npy",
        ".npz",
        ".onnx",
        ".pdf",
        ".png",
        ".pt",
        ".pth",
        ".pyc",
        ".tar",
        ".tgz",
        ".zip",
    }
)


class PackagingError(RuntimeError):
    """Raised when a candidate submission violates the package contract."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_file(root: Path, path: Path) -> Path:
    if path.is_symlink():
        raise PackagingError(f"Symlinks are forbidden in the code ZIP: {path}")
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as error:
        raise PackagingError(f"Allowlisted file escapes project root: {path}") from error
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise PackagingError(f"Unsafe allowlisted path: {relative}")
    for ancestor in (root / relative).parents:
        if ancestor == root.parent:
            break
        if ancestor.is_symlink():
            raise PackagingError(f"Symlinked directory is forbidden: {ancestor}")
        if ancestor == root:
            break
    return relative


def _is_hidden_or_cache(relative: Path) -> bool:
    return any(
        part.startswith(".") or part == "__pycache__" for part in relative.parts
    )


def _rule_accepts(relative: Path) -> bool:
    if relative in EXACT_ROOT_FILES:
        return True
    for prefix, suffixes in ALLOWLIST_RULES:
        try:
            inside = relative.relative_to(prefix)
        except ValueError:
            continue
        if not inside.parts or _is_hidden_or_cache(inside):
            return False
        if relative.suffix.lower() not in suffixes:
            return False
        if prefix == Path("tests") and not relative.name.startswith("test_"):
            return False
        return True
    return False


def _assert_not_forbidden(relative: Path) -> None:
    value = relative.as_posix()
    lowered = value.lower()
    if relative in FORBIDDEN_LEGACY_EVIDENCE_FILES:
        raise PackagingError(
            "Retired pre-rewrite evidence is forbidden in the code ZIP: "
            f"{value}; remove it after preserving any required historical audit record"
        )
    if any(lowered.startswith(prefix.lower()) for prefix in FORBIDDEN_PREFIXES):
        raise PackagingError(f"Forbidden path selected for submission: {value}")
    if relative.name == ".DS_Store" or relative.suffix.lower() in {
        suffix.lower() for suffix in FORBIDDEN_SUFFIXES
    }:
        raise PackagingError(f"Forbidden file type selected for submission: {value}")
    if _is_hidden_or_cache(relative) and relative not in EXACT_ROOT_FILES:
        raise PackagingError(f"Hidden/cache path selected for submission: {value}")


def _validated_manifest_path(value: object, *, section: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise PackagingError(
            f"Report-table manifest {section} contains an invalid path: {value!r}"
        )
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise PackagingError(
            f"Report-table manifest {section} contains an unsafe path: {value}"
        )
    return relative


def _validate_report_table_manifest(root: Path, selected: set[Path]) -> None:
    relative_manifest = Path("results/tables/final_report_tables_manifest.json")
    manifest_path = root / relative_manifest
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackagingError(
            f"Report-table manifest is unreadable or invalid JSON: {manifest_path}"
        ) from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise PackagingError("Report-table manifest must use schema_version=1")

    sources = payload.get("sources")
    outputs = payload.get("outputs")
    if not isinstance(sources, dict) or not sources:
        raise PackagingError("Report-table manifest must declare at least one source hash")
    if not isinstance(outputs, dict) or not outputs:
        raise PackagingError("Report-table manifest must declare output hashes")
    missing_outputs = sorted(REQUIRED_REPORT_TABLE_OUTPUTS - set(outputs))
    if missing_outputs:
        raise PackagingError(
            "Report-table manifest omits required canonical CSV output(s): "
            + ", ".join(missing_outputs)
        )

    for section_name, records in (("sources", sources), ("outputs", outputs)):
        for raw_relative, expected_sha in records.items():
            relative = _validated_manifest_path(raw_relative, section=section_name)
            if (
                not isinstance(expected_sha, str)
                or len(expected_sha) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha)
            ):
                raise PackagingError(
                    f"Report-table manifest has invalid SHA-256 for {relative.as_posix()}"
                )
            path = root / relative
            if not path.is_file() or path.is_symlink():
                raise PackagingError(
                    f"Report-table manifest {section_name} file is missing or unsafe: {relative}"
                )
            canonical_relative = _safe_relative_file(root, path)
            if canonical_relative not in selected:
                raise PackagingError(
                    "Report-table manifest references a file outside the code-ZIP allowlist: "
                    f"{relative}"
                )
            actual_sha = _sha256_file(path)
            if actual_sha != expected_sha:
                raise PackagingError(
                    "Report-table manifest hash mismatch for "
                    f"{relative}: expected {expected_sha}, got {actual_sha}"
                )


def collect_allowed_files(project_root: str | Path = PROJECT_ROOT) -> list[Path]:
    """Return the sorted allowed relative file paths for the code ZIP.

    This function is the single source of truth shared by the builder, checker,
    tests, and final evidence inventory.  Returned paths are relative to
    ``project_root`` and use the host :class:`Path` type; archive names are
    converted to portable POSIX form only at the ZIP boundary.
    """

    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise PackagingError(f"Project root is not a directory: {root}")
    selected: set[Path] = set()

    for relative in EXACT_ROOT_FILES:
        path = root / relative
        if not path.is_file():
            raise PackagingError(f"Required submission file is missing: {relative}")
        selected.add(_safe_relative_file(root, path))

    required_files = (*REQUIRED_SPLIT_FILES, *REQUIRED_REWRITE_EVIDENCE_FILES)
    for required in required_files:
        if not (root / required).is_file():
            category = (
                "frozen split"
                if required in REQUIRED_SPLIT_FILES
                else "traditional-rewrite evidence"
            )
            raise PackagingError(f"Required {category} file is missing: {required}")

    for prefix, _ in ALLOWLIST_RULES:
        directory = root / prefix
        if not directory.exists():
            raise PackagingError(f"Required submission directory is missing: {prefix}")
        if not directory.is_dir() or directory.is_symlink():
            raise PackagingError(f"Submission directory is unsafe: {directory}")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise PackagingError(f"Symlinks are forbidden in allowlisted trees: {path}")
            if not path.is_file():
                continue
            relative = _safe_relative_file(root, path)
            if _rule_accepts(relative):
                selected.add(relative)

    for relative in selected:
        if not _rule_accepts(relative):
            raise PackagingError(f"Internal allowlist mismatch for {relative}")
        _assert_not_forbidden(relative)
    missing_required = sorted(set(required_files) - selected)
    if missing_required:
        raise PackagingError(
            "Required submission evidence was not selected: "
            + ", ".join(path.as_posix() for path in missing_required)
        )
    _validate_report_table_manifest(root, selected)
    return sorted(selected, key=lambda path: path.as_posix())


def _manifest_bytes(project_root: Path, relative_files: Iterable[Path]) -> bytes:
    lines = [
        f"{_sha256_file(project_root / relative)}  {relative.as_posix()}"
        for relative in relative_files
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.flag_bits |= 0x800  # UTF-8 names.
    return info


def _archive_name(relative: Path) -> str:
    return f"{ARCHIVE_ROOT}/{relative.as_posix()}"


def _parse_manifest(content: bytes) -> dict[str, str]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PackagingError("ZIP checksum manifest is not valid UTF-8") from error
    records: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        if "  " not in line:
            raise PackagingError(f"Malformed checksum manifest line {line_number}")
        digest, relative = line.split("  ", 1)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise PackagingError(f"Invalid SHA-256 on manifest line {line_number}")
        if relative in records:
            raise PackagingError(f"Duplicate manifest path: {relative}")
        records[relative] = digest
    return records


def _validate_relative_archive_path(value: str) -> str:
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise PackagingError(f"ZIP member uses a non-portable path: {value}")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} or ":" in part for part in raw_parts):
        raise PackagingError(f"ZIP member uses a non-canonical path: {value}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise PackagingError(f"Unsafe ZIP member path: {value}")
    if path.parts[0] != ARCHIVE_ROOT or len(path.parts) < 2:
        raise PackagingError(
            f"Every ZIP member must be inside {ARCHIVE_ROOT}/: {value}"
        )
    return PurePosixPath(*path.parts[1:]).as_posix()


def validate_code_zip(
    zip_path: str | Path,
    *,
    expected_files: Sequence[Path] | None = None,
    project_root: str | Path | None = None,
    max_bytes: int = CODE_ZIP_LIMIT_BYTES,
) -> dict[str, int | str | list[str]]:
    """Strictly validate ZIP safety, allowlist, checksums, and byte limit."""

    path = Path(zip_path).expanduser().resolve()
    if not path.is_file():
        raise PackagingError(f"Code ZIP does not exist: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise PackagingError(
            f"Code ZIP is {size:,} bytes; limit is {max_bytes:,} bytes"
        )
    try:
        archive = zipfile.ZipFile(path, "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise PackagingError(f"Invalid code ZIP {path}: {error}") from error
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise PackagingError(
                f"Code ZIP has {len(infos):,} members; safety cap is "
                f"{MAX_ARCHIVE_MEMBERS:,}"
            )
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise PackagingError("Code ZIP contains duplicate member names")
        expected_manifest_name = f"{ARCHIVE_ROOT}/{MANIFEST_NAME}"
        if names.count(expected_manifest_name) != 1:
            raise PackagingError("Code ZIP must contain exactly one checksum manifest")

        normalized_names: set[str] = set()
        total_uncompressed = 0
        for info in infos:
            relative = _validate_relative_archive_path(info.filename)
            if relative in normalized_names:
                raise PackagingError(
                    f"Code ZIP contains colliding normalized paths: {info.filename}"
                )
            normalized_names.add(relative)
            unix_mode = (info.external_attr >> 16) & 0o170000
            if info.is_dir():
                raise PackagingError(f"Directory entries are not allowed: {info.filename}")
            if unix_mode not in (0, stat.S_IFREG):
                raise PackagingError(
                    f"ZIP member is not a regular file: {info.filename} "
                    f"(mode={oct(unix_mode)})"
                )
            if info.flag_bits & 0x1:
                raise PackagingError(f"Encrypted ZIP members are not allowed: {info.filename}")
            if info.file_size < 0 or info.compress_size < 0:
                raise PackagingError(f"Invalid ZIP member size: {info.filename}")
            if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
                raise PackagingError(
                    f"ZIP member exceeds uncompressed safety cap: {info.filename} "
                    f"({info.file_size:,} > {MAX_MEMBER_UNCOMPRESSED_BYTES:,})"
                )
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise PackagingError(
                    f"Code ZIP expands to more than {MAX_TOTAL_UNCOMPRESSED_BYTES:,} bytes"
                )
            if info.file_size:
                ratio = info.file_size / max(1, info.compress_size)
                if ratio > MAX_COMPRESSION_RATIO:
                    raise PackagingError(
                        f"ZIP member compression ratio is unsafe: {info.filename} "
                        f"({ratio:.1f} > {MAX_COMPRESSION_RATIO:.1f})"
                    )

        bad_member = archive.testzip()
        if bad_member is not None:
            raise PackagingError(f"CRC failure in ZIP member: {bad_member}")

        payloads: dict[str, bytes] = {}
        for info in infos:
            relative = _validate_relative_archive_path(info.filename)
            content = archive.read(info)
            if len(content) != info.file_size:
                raise PackagingError(f"Truncated ZIP member: {info.filename}")
            payloads[relative] = content

        manifest_content = payloads.pop(MANIFEST_NAME, None)
        if manifest_content is None:
            raise PackagingError("Checksum manifest could not be read")
        manifest = _parse_manifest(manifest_content)
        if set(manifest) != set(payloads):
            missing = sorted(set(payloads) - set(manifest))
            extra = sorted(set(manifest) - set(payloads))
            raise PackagingError(
                f"Manifest/member mismatch; missing={missing[:5]}, extra={extra[:5]}"
            )
        for relative_value, content in payloads.items():
            relative = Path(PurePosixPath(relative_value))
            if not _rule_accepts(relative):
                raise PackagingError(f"ZIP member is outside the allowlist: {relative_value}")
            _assert_not_forbidden(relative)
            observed = _sha256_bytes(content)
            if observed != manifest[relative_value]:
                raise PackagingError(f"Checksum mismatch for ZIP member: {relative_value}")

        if expected_files is not None:
            expected_values = {Path(value).as_posix() for value in expected_files}
            if set(payloads) != expected_values:
                raise PackagingError("ZIP contents do not exactly match collect_allowed_files()")
            if project_root is None:
                raise PackagingError("project_root is required with expected_files")
            root = Path(project_root).expanduser().resolve()
            for relative_value, content in payloads.items():
                source = root / Path(PurePosixPath(relative_value))
                if not source.is_file() or source.is_symlink():
                    raise PackagingError(f"ZIP source is missing or unsafe: {relative_value}")
                if _sha256_file(source) != _sha256_bytes(content):
                    raise PackagingError(f"ZIP differs from current source: {relative_value}")

        return {
            "zip_path": path.as_posix(),
            "zip_bytes": size,
            "file_count": len(payloads),
            "uncompressed_payload_bytes": sum(len(value) for value in payloads.values()),
            "manifest_sha256": _sha256_bytes(manifest_content),
            "entry_names": sorted(names),
        }


def _write_zip(
    destination: Path,
    *,
    project_root: Path,
    relative_files: Sequence[Path],
) -> None:
    manifest = _manifest_bytes(project_root, relative_files)
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for relative in relative_files:
            archive.writestr(
                _zip_info(_archive_name(relative)),
                (project_root / relative).read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
        archive.writestr(
            _zip_info(f"{ARCHIVE_ROOT}/{MANIFEST_NAME}"),
            manifest,
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )


def build_code_zip(
    output: str | Path = DEFAULT_OUTPUT,
    *,
    project_root: str | Path = PROJECT_ROOT,
    force: bool = False,
) -> dict[str, int | str | list[str]]:
    """Build and atomically install a validated deterministic code ZIP."""

    root = Path(project_root).expanduser().resolve()
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path = output_path.resolve()
    try:
        output_path.relative_to(root)
    except ValueError as error:
        raise PackagingError("Code ZIP output must remain inside the project root") from error
    if output_path.suffix.lower() != ".zip":
        raise PackagingError("Code ZIP output must have a .zip extension")
    if output_path.is_symlink():
        raise PackagingError(f"Refusing symlink output path: {output_path}")

    relative_files = collect_allowed_files(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        _write_zip(temporary, project_root=root, relative_files=relative_files)
        result = validate_code_zip(
            temporary,
            expected_files=relative_files,
            project_root=root,
        )
        if output_path.exists():
            if not output_path.is_file():
                raise PackagingError(f"Output exists and is not a file: {output_path}")
            if _sha256_file(output_path) == _sha256_file(temporary):
                temporary.unlink()
                existing_result = validate_code_zip(
                    output_path,
                    expected_files=relative_files,
                    project_root=root,
                )
                existing_result["zip_sha256"] = _sha256_file(output_path)
                return existing_result
            if not force:
                raise FileExistsError(
                    f"A different code ZIP already exists: {output_path}; use --force deliberately"
                )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
        result = validate_code_zip(
            output_path,
            expected_files=relative_files,
            project_root=root,
        )
        result["zip_sha256"] = _sha256_file(output_path)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic source-only COMP9517 code ZIP."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing different ZIP after full validation",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the sorted allowlist without creating a ZIP",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.list_only:
            files = collect_allowed_files(args.project_root)
            for path in files:
                print(path.as_posix())
            print(f"ALLOWLIST: {len(files)} files")
            return 0
        result = build_code_zip(
            args.output,
            project_root=args.project_root,
            force=args.force,
        )
    except (OSError, ValueError, TypeError, PackagingError) as error:
        print(f"[FAIL] {error}", file=os.sys.stderr)
        return 1
    print(f"[PASS] Code ZIP: {result['zip_path']}")
    print(f"[PASS] Files: {result['file_count']}")
    print(
        f"[PASS] Size: {int(result['zip_bytes']):,} / "
        f"{CODE_ZIP_LIMIT_BYTES:,} bytes"
    )
    if "zip_sha256" in result:
        print(f"[PASS] SHA-256: {result['zip_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_ROOT",
    "CODE_ZIP_LIMIT_BYTES",
    "DEFAULT_OUTPUT",
    "MANIFEST_NAME",
    "FORBIDDEN_LEGACY_EVIDENCE_FILES",
    "PackagingError",
    "REQUIRED_REWRITE_EVIDENCE_FILES",
    "build_code_zip",
    "collect_allowed_files",
    "validate_code_zip",
]
