"""Safely extract the iNaturalist annotation archives and, optionally, images.

Annotation archives are extracted by default. Image extraction is deliberately opt-in
because the two compressed image archives occupy more than 50 GB.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from tqdm import tqdm


ANNOTATION_ARCHIVES = {
    "train_mini.json.tar.gz": "train_mini.json",
    "val.json.tar.gz": "val.json",
}
IMAGE_ARCHIVES = ("train_mini.tar.gz", "val.tar.gz")


@dataclass
class ExtractionStats:
    """Counts reported after extracting one archive."""

    extracted: int = 0
    skipped: int = 0
    directories: int = 0


@dataclass(frozen=True)
class ArchivePreflight:
    """Validated archive inventory used before any member is written."""

    regular_files: int
    directories: int
    uncompressed_bytes: int
    bytes_to_write: int


def format_bytes(size: int) -> str:
    """Format a byte count using binary units."""

    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _safe_target(output_dir: Path, member: tarfile.TarInfo) -> Path:
    """Validate a tar member and return its destination path."""

    if member.issym() or member.islnk():
        raise ValueError(f"Archive links are not allowed: {member.name!r}")
    if not (member.isdir() or member.isreg()):
        raise ValueError(f"Unsupported archive member type: {member.name!r}")
    if not member.name or "\\" in member.name:
        raise ValueError(f"Unsafe archive member path: {member.name!r}")

    pure_path = PurePosixPath(member.name)
    if (
        pure_path.is_absolute()
        or not pure_path.parts
        or str(pure_path) in {"", "."}
        or ".." in pure_path.parts
    ):
        raise ValueError(f"Unsafe archive member path: {member.name!r}")

    base = output_dir.resolve()
    target = output_dir.joinpath(*pure_path.parts)
    resolved_target = target.resolve(strict=False)
    if resolved_target != base and base not in resolved_target.parents:
        raise ValueError(f"Archive member escapes output directory: {member.name!r}")
    return target


def _write_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    target: Path,
    overwrite: bool,
) -> str:
    """Atomically write one regular tar member, returning extracted or skipped."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise FileExistsError(f"Destination is not a regular file: {target}")
        if not overwrite:
            if target.stat().st_size == member.size:
                return "skipped"
            raise FileExistsError(
                f"Existing file has the wrong size: {target} "
                f"({target.stat().st_size} bytes, expected {member.size}). "
                "Use --overwrite to replace it."
            )

    source = archive.extractfile(member)
    if source is None:
        raise tarfile.ExtractError(f"Could not read archive member: {member.name}")

    temporary_path: Path | None = None
    try:
        with source, tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            shutil.copyfileobj(source, temporary, length=1024 * 1024)
            temporary.flush()
            os.fsync(temporary.fileno())

        if temporary_path.stat().st_size != member.size:
            raise tarfile.ExtractError(
                f"Incomplete extraction for {member.name}: wrote "
                f"{temporary_path.stat().st_size} of {member.size} bytes"
            )
        os.replace(temporary_path, target)
        temporary_path = None
        return "extracted"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _archive_identity(path: Path) -> tuple[int, int, int]:
    status = path.stat()
    return status.st_size, status.st_mtime_ns, status.st_ino


def _preflight_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    expected_member: str | None,
) -> ArchivePreflight:
    """Validate the complete member inventory before creating or replacing files."""

    regular_files = 0
    directories = 0
    uncompressed_bytes = 0
    bytes_to_write = 0
    expected_found = False
    seen_file_targets: set[Path] = set()
    seen_directory_targets: set[Path] = set()
    required_directory_targets: set[Path] = set()
    output_base = output_dir.resolve(strict=False)

    def target_ancestors(target: Path) -> list[Path]:
        ancestors: list[Path] = []
        for ancestor in target.parents:
            ancestors.append(ancestor)
            if ancestor == output_base:
                break
        return ancestors

    def reject_non_directory_ancestors(target: Path) -> None:
        for ancestor in target_ancestors(target):
            if ancestor.exists() and not ancestor.is_dir():
                raise FileExistsError(
                    f"Archive destination has a non-directory ancestor: {ancestor}"
                )
            if ancestor.is_symlink() and not ancestor.is_dir():
                raise FileExistsError(
                    f"Archive destination has an unusable symbolic-link ancestor: "
                    f"{ancestor}"
                )

    with tarfile.open(archive_path, mode="r:gz") as archive:
        progress = tqdm(
            desc=f"Preflight {archive_path.name}", unit="member", leave=False
        )
        try:
            for member in archive:
                target = _safe_target(output_dir, member)
                target_key = target.resolve(strict=False)
                ancestors = target_ancestors(target_key)
                reject_non_directory_ancestors(target_key)

                if member.isdir():
                    if target_key in seen_file_targets or any(
                        ancestor in seen_file_targets for ancestor in ancestors
                    ):
                        raise ValueError(
                            f"File/directory destination collision in "
                            f"{archive_path.name}: {member.name!r}"
                        )
                    if target.exists() and not target.is_dir():
                        raise FileExistsError(
                            f"Archive directory conflicts with existing path: {target}"
                        )
                    seen_directory_targets.add(target_key)
                    required_directory_targets.add(target_key)
                    required_directory_targets.update(ancestors)
                    directories += 1
                    progress.update(1)
                    continue

                if (
                    target_key in seen_file_targets
                    or target_key in seen_directory_targets
                    or target_key in required_directory_targets
                    or any(ancestor in seen_file_targets for ancestor in ancestors)
                ):
                    raise ValueError(
                        f"Duplicate or hierarchical file collision in "
                        f"{archive_path.name}: {member.name!r}"
                    )
                seen_file_targets.add(target_key)
                required_directory_targets.update(ancestors)

                if expected_member is not None:
                    if member.name != expected_member:
                        raise ValueError(
                            f"Unexpected member in {archive_path.name}: {member.name!r}; "
                            f"expected only {expected_member!r}"
                        )
                    expected_found = True

                if target.is_symlink():
                    raise FileExistsError(
                        f"Destination is a symbolic link and will not be replaced: {target}"
                    )
                if target.exists():
                    if not target.is_file():
                        raise FileExistsError(
                            f"Destination is not a regular file: {target}"
                        )
                    if not overwrite and target.stat().st_size != member.size:
                        raise FileExistsError(
                            f"Existing file has the wrong size: {target} "
                            f"({target.stat().st_size} bytes, expected {member.size}). "
                            "Use --overwrite to replace it."
                        )
                    if overwrite:
                        bytes_to_write += member.size
                else:
                    bytes_to_write += member.size

                regular_files += 1
                uncompressed_bytes += member.size
                progress.update(1)
        finally:
            progress.close()

    if expected_member is not None and not expected_found:
        raise ValueError(
            f"Expected member {expected_member!r} was not found in {archive_path}"
        )
    return ArchivePreflight(
        regular_files=regular_files,
        directories=directories,
        uncompressed_bytes=uncompressed_bytes,
        bytes_to_write=bytes_to_write,
    )


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    expected_member: str | None = None,
) -> ExtractionStats:
    """Safely extract one archive into *output_dir*."""

    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    identity = _archive_identity(archive_path)
    preflight = _preflight_archive(
        archive_path,
        output_dir,
        overwrite=overwrite,
        expected_member=expected_member,
    )
    if _archive_identity(archive_path) != identity:
        raise OSError(f"Archive changed during preflight: {archive_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    free_space = shutil.disk_usage(output_dir).free
    if preflight.bytes_to_write > free_space:
        raise OSError(
            f"Insufficient free space for {archive_path.name}: need "
            f"{format_bytes(preflight.bytes_to_write)}, have {format_bytes(free_space)}"
        )
    print(
        f"[PREFLIGHT] {archive_path.name}: {preflight.regular_files} files, "
        f"{format_bytes(preflight.uncompressed_bytes)} uncompressed, "
        f"{format_bytes(preflight.bytes_to_write)} to write"
    )
    if _archive_identity(archive_path) != identity:
        raise OSError(f"Archive changed before extraction: {archive_path}")

    stats = ExtractionStats()
    expected_found = False
    seen_file_targets: set[Path] = set()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        progress = tqdm(desc=archive_path.name, unit="file", leave=False)
        try:
            for member in archive:
                target = _safe_target(output_dir, member)
                if expected_member is not None:
                    if member.isdir():
                        continue
                    if member.name != expected_member:
                        raise ValueError(
                            f"Unexpected member in {archive_path.name}: {member.name!r}; "
                            f"expected only {expected_member!r}"
                        )
                    expected_found = True

                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    stats.directories += 1
                    continue

                target_key = target.resolve(strict=False)
                if target_key in seen_file_targets:
                    raise ValueError(
                        f"Duplicate file destination in {archive_path.name}: {member.name!r}"
                    )
                seen_file_targets.add(target_key)

                result = _write_member(archive, member, target, overwrite)
                if result == "extracted":
                    stats.extracted += 1
                else:
                    stats.skipped += 1
                progress.update(1)
        finally:
            progress.close()

    if expected_member is not None and not expected_found:
        raise ValueError(
            f"Expected member {expected_member!r} was not found in {archive_path}"
        )
    return stats


def _show_image_warning(archives: Iterable[Path], output_dir: Path) -> None:
    """Display archive sizes, destination space, and overwrite behaviour."""

    archive_list = list(archives)
    total_size = sum(path.stat().st_size for path in archive_list)
    free_space = shutil.disk_usage(output_dir).free
    print("\nWARNING: image extraction is a large operation.")
    for path in archive_list:
        print(f"  - {path.name}: {format_bytes(path.stat().st_size)} compressed")
    print(f"  Total compressed size: {format_bytes(total_size)}")
    print(f"  Available space at destination: {format_bytes(free_space)}")
    print("  Existing files are skipped only when their size matches the archive.")
    if free_space < total_size:
        print("  WARNING: available space is smaller than the compressed archive total.")


def _confirm_image_extraction() -> bool:
    """Obtain an explicit interactive confirmation."""

    if not sys.stdin.isatty():
        print(
            "Image extraction requires an interactive terminal or the explicit --yes flag.",
            file=sys.stderr,
        )
        return False
    try:
        answer = input("Type EXTRACT to continue, or press Enter to cancel: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nImage extraction cancelled.")
        return False
    return answer == "EXTRACT"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Safely extract iNaturalist-2021 Mini annotations. Image archives are "
            "extracted only when --extract-images is supplied."
        )
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing the .tar.gz archives (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Extraction destination (default: data/raw).",
    )
    parser.add_argument(
        "--extract-images",
        action="store_true",
        help="Also extract the large train_mini and val image archives.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Acknowledge the image-size warning without an interactive prompt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace conflicting existing files instead of failing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run archive extraction."""

    args = build_parser().parse_args(argv)
    archive_dir = args.archive_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    try:
        for archive_name, expected_member in ANNOTATION_ARCHIVES.items():
            path = archive_dir / archive_name
            stats = extract_archive(
                path,
                output_dir,
                overwrite=args.overwrite,
                expected_member=expected_member,
            )
            print(
                f"[OK] {archive_name}: {stats.extracted} extracted, "
                f"{stats.skipped} skipped"
            )

        if not args.extract_images:
            print("[SKIP] Image archives were not extracted (use --extract-images to opt in).")
            return 0

        image_paths = [archive_dir / name for name in IMAGE_ARCHIVES]
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing image archive(s): {', '.join(missing)}")
        _show_image_warning(image_paths, output_dir)
        if not args.yes and not _confirm_image_extraction():
            print("[CANCELLED] Image archives were not extracted.")
            return 0
        if args.yes:
            print("[WARNING] --yes supplied; interactive confirmation bypassed.")

        for path in image_paths:
            stats = extract_archive(path, output_dir, overwrite=args.overwrite)
            print(
                f"[OK] {path.name}: {stats.extracted} extracted, "
                f"{stats.skipped} skipped, {stats.directories} directories"
            )
        return 0
    except (FileNotFoundError, FileExistsError, OSError, tarfile.TarError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
