"""Materialize only the images referenced by the split manifests.

The iNaturalist image archives are large gzip-compressed tar files. This module
streams each required archive once and writes only selected regular-file members;
it never performs a full archive extraction. Because gzip-compressed tar archives
do not provide random access, the complete compressed streams still have to be
read to locate all requested images.

The split manifests are treated as immutable inputs. ``image_path`` identifies the
original archive member, while ``local_image_path`` identifies the destination
relative to ``--output-dir`` (for example, ``train/<uuid>.jpg``).
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

import pandas as pd
from tqdm import tqdm

from .common import (
    CSV_COLUMNS,
    LOCAL_IMAGE_PATH_COLUMN,
    NEW_LABEL_COLUMN,
    is_safe_relative_posix_path,
)


SPLIT_NAMES = ("train", "val", "test")
SOURCE_ARCHIVES = {
    "train_mini": "train_mini.tar.gz",
    "val": "val.tar.gz",
}
EXPECTED_SOURCE_BY_SPLIT = {
    "train": "train_mini",
    "val": "train_mini",
    "test": "val",
}


@dataclass(frozen=True)
class ImageRequest:
    """One archive member and its validated local destination."""

    split: str
    source_split: str
    archive_member: str
    local_path: PurePosixPath
    target: Path


@dataclass
class MaterializationStats:
    """Extraction counts and byte totals for one source archive."""

    requested: int = 0
    found: int = 0
    extracted: int = 0
    skipped: int = 0
    selected_bytes: int = 0
    written_bytes: int = 0


def format_bytes(size: int) -> str:
    """Format a non-negative byte count using binary units."""

    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _archive_identity(path: Path) -> tuple[int, int, int]:
    """Return fields used to detect replacement or mutation during a scan."""

    status = path.stat()
    return status.st_size, status.st_mtime_ns, status.st_ino


def _validate_archive_member(member: tarfile.TarInfo) -> PurePosixPath:
    """Reject links, special files, and unsafe member names."""

    if member.issym() or member.islnk():
        raise ValueError(f"Archive links are not allowed: {member.name!r}")
    if not (member.isdir() or member.isreg()):
        raise ValueError(f"Unsupported archive member type: {member.name!r}")
    if not is_safe_relative_posix_path(member.name):
        raise ValueError(f"Unsafe archive member path: {member.name!r}")
    return PurePosixPath(member.name)


def _prepare_output_root(output_dir: Path) -> Path:
    """Create a real output directory and return its resolved path."""

    if output_dir.is_symlink():
        raise ValueError(f"Output directory must not be a symbolic link: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir.resolve()


def _destination_for(
    output_dir: Path,
    output_root: Path,
    local_path: PurePosixPath,
) -> Path:
    """Resolve a validated manifest path without following output symlinks."""

    target = output_dir.joinpath(*local_path.parts)
    resolved_target = target.resolve(strict=False)
    if output_root not in resolved_target.parents:
        raise ValueError(f"Local image path escapes output directory: {local_path}")

    current = target.parent
    while True:
        if current.is_symlink():
            raise ValueError(f"Destination ancestor is a symbolic link: {current}")
        if current.exists() and not current.is_dir():
            raise NotADirectoryError(
                f"Destination ancestor is not a directory: {current}"
            )
        if current == output_dir:
            break
        if output_dir not in current.parents:
            raise ValueError(f"Destination is outside output directory: {target}")
        current = current.parent

    if target.is_symlink():
        raise ValueError(f"Destination must not be a symbolic link: {target}")
    if target.exists() and not target.is_file():
        raise FileExistsError(f"Destination is not a regular file: {target}")
    return target


def load_requests(
    split_dir: Path,
    output_dir: Path,
) -> dict[str, dict[str, ImageRequest]]:
    """Load and validate all split manifests before opening either archive."""

    output_root = _prepare_output_root(output_dir)
    requests_by_source: dict[str, dict[str, ImageRequest]] = {
        source: {} for source in SOURCE_ARCHIVES
    }
    seen_local_paths: set[str] = set()
    total_rows = 0

    for expected_split in SPLIT_NAMES:
        manifest_path = split_dir / f"{expected_split}.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
        try:
            frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        except (ValueError, UnicodeError, pd.errors.ParserError) as error:
            raise ValueError(f"Could not read {manifest_path}: {error}") from error

        if list(frame.columns) != CSV_COLUMNS:
            raise ValueError(
                f"{manifest_path} has an unexpected schema: expected "
                f"{CSV_COLUMNS}, got {list(frame.columns)}"
            )
        if frame.empty:
            raise ValueError(f"Split manifest is empty: {manifest_path}")

        for row_number, row in enumerate(frame.to_dict(orient="records"), start=2):
            split = str(row["split"])
            source_split = str(row["source_split"])
            archive_member = str(row["image_path"])
            local_value = str(row[LOCAL_IMAGE_PATH_COLUMN])
            location = f"{manifest_path}:{row_number}"

            if split != expected_split:
                raise ValueError(
                    f"{location}: split is {split!r}, expected {expected_split!r}"
                )
            expected_source = EXPECTED_SOURCE_BY_SPLIT[split]
            if source_split != expected_source:
                raise ValueError(
                    f"{location}: source_split is {source_split!r}, "
                    f"expected {expected_source!r}"
                )
            if not is_safe_relative_posix_path(archive_member):
                raise ValueError(
                    f"{location}: unsafe archive image_path {archive_member!r}"
                )
            archive_path = PurePosixPath(archive_member)
            if archive_path.parts[:1] != (source_split,):
                raise ValueError(
                    f"{location}: image_path does not match source_split: "
                    f"{archive_member!r}"
                )
            if not str(row[NEW_LABEL_COLUMN]).strip():
                raise ValueError(f"{location}: {NEW_LABEL_COLUMN} is empty")
            if not is_safe_relative_posix_path(local_value):
                raise ValueError(
                    f"{location}: unsafe local_image_path {local_value!r}"
                )
            local_path = PurePosixPath(local_value)
            expected_local_path = PurePosixPath(split, archive_path.name)
            if local_path != expected_local_path:
                raise ValueError(
                    f"{location}: local_image_path must be "
                    f"{str(expected_local_path)!r}, got {local_value!r}"
                )
            if local_value in seen_local_paths:
                raise ValueError(
                    f"{location}: duplicate local_image_path {local_value!r}"
                )

            source_requests = requests_by_source[source_split]
            if archive_member in source_requests:
                raise ValueError(
                    f"{location}: duplicate archive image_path {archive_member!r}"
                )
            target = _destination_for(
                output_dir, output_root, local_path
            )
            source_requests[archive_member] = ImageRequest(
                split=split,
                source_split=source_split,
                archive_member=archive_member,
                local_path=local_path,
                target=target,
            )
            seen_local_paths.add(local_value)
            total_rows += 1

    if total_rows != len(seen_local_paths):
        raise AssertionError("Manifest destination uniqueness check is inconsistent")
    return requests_by_source


def _check_free_space(target_parent: Path, required_bytes: int) -> None:
    """Fail before writing a member if there is insufficient current free space."""

    free_bytes = shutil.disk_usage(target_parent).free
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient disk space for atomic extraction: "
            f"{format_bytes(free_bytes)} free, at least "
            f"{format_bytes(required_bytes)} required"
        )


def _write_selected_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    target: Path,
    *,
    overwrite: bool,
) -> str:
    """Atomically write one selected member and return extracted or skipped."""

    if target.is_symlink():
        raise ValueError(f"Destination must not be a symbolic link: {target}")
    if target.exists():
        if not target.is_file():
            raise FileExistsError(f"Destination is not a regular file: {target}")
        if not overwrite:
            existing_size = target.stat().st_size
            if existing_size == member.size:
                return "skipped"
            raise FileExistsError(
                f"Existing file has the wrong size: {target} "
                f"({existing_size} bytes, expected {member.size}). "
                "Use --overwrite to replace it."
            )

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.is_symlink():
        raise ValueError(f"Destination directory is a symbolic link: {target.parent}")

    # Atomic replacement temporarily needs enough space for the complete new file.
    _check_free_space(target.parent, member.size)
    source = archive.extractfile(member)
    if source is None:
        raise tarfile.ExtractError(
            f"Could not read selected archive member: {member.name!r}"
        )

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

        actual_size = temporary_path.stat().st_size
        if actual_size != member.size:
            raise tarfile.ExtractError(
                f"Incomplete extraction for {member.name!r}: wrote "
                f"{actual_size} of {member.size} bytes"
            )
        os.chmod(temporary_path, member.mode & 0o666 or 0o600)
        os.replace(temporary_path, target)
        temporary_path = None
        return "extracted"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def materialize_archive(
    archive_path: Path,
    requests: dict[str, ImageRequest],
    *,
    overwrite: bool,
) -> MaterializationStats:
    """Stream one archive and materialize exactly its requested members."""

    if not archive_path.is_file():
        raise FileNotFoundError(f"Image archive not found: {archive_path}")
    if archive_path.is_symlink():
        raise ValueError(f"Image archive must not be a symbolic link: {archive_path}")

    identity_before = _archive_identity(archive_path)
    stats = MaterializationStats(requested=len(requests))
    found_members: set[str] = set()
    archive_size = archive_path.stat().st_size

    with archive_path.open("rb") as raw_stream:
        with tqdm(
            total=archive_size,
            desc=f"Scanning {archive_path.name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            previous_position = 0
            with tarfile.open(fileobj=raw_stream, mode="r|gz") as archive:
                for member in archive:
                    _validate_archive_member(member)
                    request = requests.get(member.name)
                    if request is not None:
                        if member.name in found_members:
                            raise ValueError(
                                f"Archive contains duplicate selected member: "
                                f"{member.name!r}"
                            )
                        if not member.isreg():
                            raise ValueError(
                                f"Selected archive member is not a regular file: "
                                f"{member.name!r}"
                            )

                        outcome = _write_selected_member(
                            archive,
                            member,
                            request.target,
                            overwrite=overwrite,
                        )
                        found_members.add(member.name)
                        stats.found += 1
                        stats.selected_bytes += member.size
                        if outcome == "extracted":
                            stats.extracted += 1
                            stats.written_bytes += member.size
                        else:
                            stats.skipped += 1

                    current_position = raw_stream.tell()
                    progress.update(max(0, current_position - previous_position))
                    previous_position = current_position
                    progress.set_postfix(
                        selected=f"{stats.found}/{stats.requested}",
                        extracted=stats.extracted,
                        skipped=stats.skipped,
                        refresh=False,
                    )

            current_position = raw_stream.tell()
            progress.update(max(0, current_position - previous_position))

    if _archive_identity(archive_path) != identity_before:
        raise OSError(f"Archive changed while it was being read: {archive_path}")

    missing = sorted(set(requests).difference(found_members))
    if missing:
        examples = ", ".join(repr(value) for value in missing[:3])
        raise FileNotFoundError(
            f"{archive_path.name} is missing {len(missing)} requested members; "
            f"examples: {examples}"
        )
    return stats


def materialize_subset(
    split_dir: Path,
    archive_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, MaterializationStats]:
    """Validate manifests, selectively materialize images, and return statistics."""

    requests_by_source = load_requests(split_dir, output_dir)
    total_requests = sum(len(requests) for requests in requests_by_source.values())
    free_bytes = shutil.disk_usage(output_dir).free

    print("Materialization preflight:")
    print(f"  Split directory: {split_dir}")
    print(f"  Archive directory: {archive_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Selected images: {total_requests:,}")
    print(f"  Available disk space: {format_bytes(free_bytes)}")
    for source_split, archive_name in SOURCE_ARCHIVES.items():
        archive_path = archive_dir / archive_name
        if not archive_path.is_file():
            raise FileNotFoundError(f"Image archive not found: {archive_path}")
        print(
            f"  {archive_name}: {format_bytes(archive_path.stat().st_size)} compressed; "
            f"{len(requests_by_source[source_split]):,} selected images"
        )

    results: dict[str, MaterializationStats] = {}
    for source_split, archive_name in SOURCE_ARCHIVES.items():
        requests = requests_by_source[source_split]
        results[source_split] = materialize_archive(
            archive_dir / archive_name,
            requests,
            overwrite=overwrite,
        )

    print("\nMaterialization complete:")
    for source_split in SOURCE_ARCHIVES:
        stats = results[source_split]
        print(
            f"  {source_split}: {stats.found:,}/{stats.requested:,} found, "
            f"{stats.extracted:,} extracted, {stats.skipped:,} skipped, "
            f"{format_bytes(stats.selected_bytes)} selected, "
            f"{format_bytes(stats.written_bytes)} written"
        )
    print(f"  Output directory: {output_dir}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only images referenced by train.csv, val.csv, and test.csv. "
            "The complete compressed tar streams are scanned, but unselected "
            "images are never written."
        )
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing train.csv, val.csv, and test.csv.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing train_mini.tar.gz and val.tar.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/images"),
        help=(
            "Root for local_image_path destinations "
            "(default: data/processed/images)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing selected images instead of size-based skips.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        materialize_subset(
            args.split_dir.expanduser(),
            args.archive_dir.expanduser(),
            args.output_dir.expanduser(),
            overwrite=args.overwrite,
        )
    except (
        EOFError,
        FileNotFoundError,
        FileExistsError,
        NotADirectoryError,
        OSError,
        ValueError,
        tarfile.TarError,
    ) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
