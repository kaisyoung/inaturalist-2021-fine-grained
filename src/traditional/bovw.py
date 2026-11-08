"""SIFT Bag-of-Visual-Words primitives for the optional Stage 2 baseline.

The functions in this module are intentionally independent of the CLI so they
can be unit-tested on tiny images without touching the 30,000-image dataset.
"""

from __future__ import annotations

import hashlib
import platform
import resource
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import MiniBatchKMeans


EPS = 1e-12
SIFT_DIM = 128


@dataclass(frozen=True)
class BoVWConfig:
    """Frozen SIFT and visual-vocabulary parameters."""

    max_side: int = 256
    sift_nfeatures: int = 300
    vocabulary_size: int = 256
    vocabulary_images_per_class: int = 4
    pilot_images_per_class: int = 2
    vocabulary_descriptors_per_image: int = 100
    batch_size: int = 4096
    n_init: int = 3
    soft_assign_k: int = 3
    pyramid_grid: int = 2
    use_spatial_pyramid: bool = True
    seed: int = 42

    @property
    def n_regions(self) -> int:
        return 1 + self.pyramid_grid**2 if self.use_spatial_pyramid else 1

    @property
    def feature_dim(self) -> int:
        return self.vocabulary_size * self.n_regions

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if name in {"seed", "use_spatial_pyramid"}:
                continue
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.soft_assign_k > self.vocabulary_size:
            raise ValueError("soft_assign_k cannot exceed vocabulary_size")


@dataclass(frozen=True)
class SiftResult:
    """SIFT descriptors and keypoint geometry for spatial encoding."""

    descriptors: np.ndarray
    points: np.ndarray
    height: int
    width: int

    @property
    def is_empty(self) -> bool:
        return len(self.descriptors) == 0


@dataclass
class FittedVocabulary:
    """Fitted MiniBatchKMeans plus train-fitted soft-assignment bandwidth."""

    kmeans: MiniBatchKMeans
    sigma: float
    config: BoVWConfig

    @property
    def n_clusters(self) -> int:
        return int(self.kmeans.n_clusters)

    @property
    def cluster_centers_(self) -> np.ndarray:
        return self.kmeans.cluster_centers_

    def predict(self, descriptors: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(descriptors)

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        return self.kmeans.transform(descriptors)


def load_grayscale(path: Path, max_side: int = 256) -> np.ndarray:
    """Decode, orient, grayscale, and aspect-preservingly resize one image."""

    if max_side <= 0:
        raise ValueError("max_side must be positive")
    try:
        with Image.open(path) as source:
            oriented = ImageOps.exif_transpose(source)
            grayscale = oriented.convert("L")
            width, height = grayscale.size
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid image dimensions {grayscale.size}")
            scale = max_side / max(width, height)
            target = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            resized = grayscale.resize(target, resample=Image.Resampling.LANCZOS)
            array = np.asarray(resized, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Could not decode image {path}: {exc}") from exc
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"Decoded image is not a non-empty grayscale array: {path}")
    return np.ascontiguousarray(array)


def create_sift(nfeatures: int = 300) -> Any:
    if nfeatures <= 0:
        raise ValueError("nfeatures must be positive")
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "This OpenCV build does not expose SIFT_create; install "
            "opencv-python-headless from requirements.txt"
        )
    return cv2.SIFT_create(nfeatures=int(nfeatures))


def extract_sift_descriptors(
    image: np.ndarray, *, nfeatures: int = 300, sift: Any | None = None
) -> np.ndarray | None:
    """Return float32 SIFT descriptors or ``None`` for a featureless image."""

    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("SIFT input must be a two-dimensional uint8 array")
    result = extract_sift_result(image, nfeatures=nfeatures, sift=sift)
    if result.is_empty:
        return None
    return result.descriptors


def extract_sift_result(
    image: np.ndarray, *, nfeatures: int = 300, sift: Any | None = None
) -> SiftResult:
    """Return descriptors together with keypoint coordinates."""

    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("SIFT input must be a two-dimensional uint8 array")
    detector = sift if sift is not None else create_sift(nfeatures)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    if descriptors is None or len(descriptors) == 0:
        return SiftResult(
            np.empty((0, SIFT_DIM), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            int(image.shape[0]),
            int(image.shape[1]),
        )
    matrix = np.asarray(descriptors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != SIFT_DIM:
        raise ValueError(f"Unexpected SIFT descriptor shape: {matrix.shape}")
    points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    if points.shape != (len(matrix), 2):
        raise ValueError("SIFT keypoint coordinates do not match descriptors")
    return SiftResult(matrix, points, int(image.shape[0]), int(image.shape[1]))


def deterministic_balanced_indices(
    labels: np.ndarray,
    images_per_class: int,
    *,
    seed: int,
    source_tag: int,
) -> np.ndarray:
    """Choose a deterministic, balanced subset without changing manifest order."""

    if images_per_class <= 0:
        raise ValueError("images_per_class must be positive")
    selected: list[int] = []
    for raw_label in np.unique(labels):
        label = int(raw_label)
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < images_per_class:
            raise ValueError(
                f"Class {label} has {len(candidates)} images; "
                f"{images_per_class} required"
            )
        generator = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([seed, label, source_tag]))
        )
        positions = generator.choice(
            len(candidates), size=images_per_class, replace=False
        )
        selected.extend(int(candidates[int(position)]) for position in positions)
    return np.asarray(sorted(selected), dtype=np.int64)


def fit_vocabulary(
    descriptors: np.ndarray,
    *,
    vocabulary_size: int,
    batch_size: int,
    n_init: int,
    seed: int,
    config: BoVWConfig | None = None,
) -> FittedVocabulary:
    """Fit the train-only visual vocabulary."""

    descriptors = np.asarray(descriptors, dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[1] != 128:
        raise ValueError(f"Expected an Nx128 descriptor matrix, got {descriptors.shape}")
    if len(descriptors) < vocabulary_size:
        raise ValueError(
            f"Need at least {vocabulary_size} descriptors, got {len(descriptors)}"
        )
    kmeans = MiniBatchKMeans(
        n_clusters=vocabulary_size,
        batch_size=batch_size,
        n_init=n_init,
        random_state=seed,
    )
    kmeans.fit(descriptors)
    sample_step = max(1, len(descriptors) // 20_000)
    nearest = kmeans.transform(descriptors[::sample_step]).min(axis=1)
    positive = nearest[nearest > EPS]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    sigma = max(sigma, EPS)
    resolved = config or BoVWConfig(
        vocabulary_size=vocabulary_size,
        batch_size=batch_size,
        n_init=n_init,
        soft_assign_k=min(3, vocabulary_size),
        seed=seed,
    )
    resolved.validate()
    if resolved.vocabulary_size != vocabulary_size:
        raise ValueError("config vocabulary_size does not match fitted vocabulary")
    return FittedVocabulary(kmeans=kmeans, sigma=sigma, config=resolved)


def encode_descriptors(
    descriptors: np.ndarray | SiftResult | None,
    vocabulary: MiniBatchKMeans | FittedVocabulary,
) -> np.ndarray:
    """Encode descriptors with train-fitted soft assignment and a 1+2x2 pyramid.

    Raw arrays and simple vocabulary fixtures retain the legacy one-region
    behavior for small callers; production extraction passes :class:`SiftResult`
    and a :class:`FittedVocabulary`, yielding exactly 1,280 raw features.
    """

    vocabulary_size = int(vocabulary.n_clusters)
    config = vocabulary.config if isinstance(vocabulary, FittedVocabulary) else None
    result = descriptors if isinstance(descriptors, SiftResult) else None
    matrix = result.descriptors if result is not None else descriptors
    spatial = result is not None and config is not None and config.use_spatial_pyramid
    regions = config.n_regions if spatial else 1
    output = np.zeros((regions, vocabulary_size), dtype=np.float64)
    if matrix is None or len(matrix) == 0:
        return output.ravel().astype(np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)

    soft_k = int(config.soft_assign_k) if config is not None else 1
    if soft_k > 1 and hasattr(vocabulary, "cluster_centers_"):
        centers = np.ascontiguousarray(
            np.asarray(vocabulary.cluster_centers_, dtype=np.float32)
        )
        if (
            centers.ndim != 2
            or centers.shape != (vocabulary_size, matrix.shape[1])
            or not np.isfinite(centers).all()
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("BoVW descriptors or vocabulary centers are invalid")
        # OpenCV returns the exact top-k Euclidean distances and indices without
        # materialising the full descriptor-by-vocabulary distance matrix.  This
        # also avoids spurious Accelerate/BLAS floating-point status warnings
        # observed from sklearn's otherwise equivalent matrix multiplication on
        # Apple Silicon.
        selected32, nearest32 = cv2.batchDistance(
            np.ascontiguousarray(matrix),
            centers,
            cv2.CV_32F,
            normType=cv2.NORM_L2,
            K=soft_k,
        )
        selected = np.asarray(selected32, dtype=np.float64)
        nearest = np.asarray(nearest32, dtype=np.int64)
        if (
            selected.shape != (len(matrix), soft_k)
            or nearest.shape != selected.shape
            or not np.isfinite(selected).all()
            or np.any(nearest < 0)
            or np.any(nearest >= vocabulary_size)
        ):
            raise ValueError("Top-k BoVW assignment produced invalid distances")
        sigma = float(getattr(vocabulary, "sigma", 1.0))
        logits = -(selected**2) / (2.0 * sigma**2 + EPS)
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), EPS)
    else:
        nearest = np.asarray(vocabulary.predict(matrix), dtype=np.int64)[:, None]
        weights = np.ones(nearest.shape, dtype=np.float64)

    if not spatial:
        for column in range(nearest.shape[1]):
            np.add.at(output[0], nearest[:, column], weights[:, column])
        return output.ravel().astype(np.float32)

    assert result is not None and config is not None
    grid = config.pyramid_grid
    rows = np.clip(
        (result.points[:, 1] / max(result.height, 1) * grid).astype(int),
        0,
        grid - 1,
    )
    columns = np.clip(
        (result.points[:, 0] / max(result.width, 1) * grid).astype(int),
        0,
        grid - 1,
    )
    cells = 1 + rows * grid + columns
    for column in range(nearest.shape[1]):
        np.add.at(output, (cells, nearest[:, column]), weights[:, column])
    output[0] = output[1:].sum(axis=0)
    return output.ravel().astype(np.float32)


def bovw_configuration(config: BoVWConfig) -> dict[str, Any]:
    """Return the serialisable 1,280-dimensional BoVW feature contract."""

    config.validate()
    return {
        "feature_type": "sift_bovw",
        "feature_dim": config.feature_dim,
        "image_size": f"max side {config.max_side}",
        "resize_policy": "preserve aspect ratio; longest side equals max_side",
        "concatenation_order": ["bovw"],
        "components": {
            "bovw": {
                "start": 0,
                "stop": config.feature_dim,
                "dimension": config.feature_dim,
                "parameters": asdict(config),
            }
        },
        "component_offsets": {"bovw": [0, config.feature_dim]},
        "configuration": asdict(config),
        "normalisation": "applied by the model pipeline, not at extraction time",
    }


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on macOS and Linux."""

    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.  platform.system is preferable to a
    # numeric heuristic because small Linux processes can be below one MiB.
    return raw if platform.system() == "Darwin" else raw * 1024


def evaluate_pilot_gate(
    *,
    decode_failures: int,
    classes_with_descriptors: int,
    expected_classes: int,
    descriptor_count: int,
    vocabulary_size: int,
    available_disk_bytes: int,
    peak_memory_bytes: int,
    projected_seconds: float,
    benchmark_succeeded: bool = True,
    minimum_disk_bytes: int = 5 * 1024**3,
    maximum_peak_bytes: int = 12 * 1024**3,
    maximum_projected_seconds: float = 3 * 60 * 60,
) -> dict[str, Any]:
    """Apply all mandatory BoVW feasibility checks and explain every result."""

    checks = {
        "zero_decode_failures": decode_failures == 0,
        "descriptors_from_every_class": classes_with_descriptors == expected_classes,
        "enough_descriptors": descriptor_count >= vocabulary_size,
        "minimum_free_disk": available_disk_bytes >= minimum_disk_bytes,
        "peak_memory_limit": peak_memory_bytes <= maximum_peak_bytes,
        "runtime_projection_limit": projected_seconds <= maximum_projected_seconds,
        "pilot_benchmark_succeeded": benchmark_succeeded,
    }
    return {
        "status": "PASS" if all(checks.values()) else "NOT RUN",
        "passed": bool(all(checks.values())),
        "checks": checks,
        "measurements": {
            "decode_failures": int(decode_failures),
            "classes_with_descriptors": int(classes_with_descriptors),
            "expected_classes": int(expected_classes),
            "descriptor_count": int(descriptor_count),
            "vocabulary_size": int(vocabulary_size),
            "available_disk_bytes": int(available_disk_bytes),
            "peak_memory_bytes": int(peak_memory_bytes),
            "projected_seconds": float(projected_seconds),
            "benchmark_succeeded": bool(benchmark_succeeded),
        },
        "thresholds": {
            "minimum_disk_bytes": int(minimum_disk_bytes),
            "maximum_peak_bytes": int(maximum_peak_bytes),
            "maximum_projected_seconds": float(maximum_projected_seconds),
        },
    }


def project_bovw_runtime(
    *,
    pilot_extraction_seconds: float,
    pilot_encoding_seconds: float,
    pilot_clustering_seconds: float,
    pilot_images: int,
    pilot_vocabulary_descriptors: int,
    total_encoding_images: int,
    vocabulary_training_images: int,
    vocabulary_descriptors_per_image: int,
    safety_factor: float = 1.25,
) -> dict[str, float | int | str]:
    """Project the complete BoVW feature run from auditable pilot timings.

    SIFT is run once for every encoded image and once more for each image used
    to train the visual vocabulary.  Prediction/histogram encoding is needed
    only for the encoded splits.  MiniBatchKMeans time is scaled by the
    conservative maximum number of vocabulary descriptors in the full run.
    """

    numeric_values = (
        pilot_extraction_seconds,
        pilot_encoding_seconds,
        pilot_clustering_seconds,
        safety_factor,
    )
    if any(not np.isfinite(value) or value < 0 for value in numeric_values):
        raise ValueError("BoVW projection timings must be finite and non-negative")
    if pilot_images <= 0 or pilot_vocabulary_descriptors <= 0:
        raise ValueError("BoVW projection pilot counts must be positive")
    if total_encoding_images <= 0 or vocabulary_training_images <= 0:
        raise ValueError("BoVW projection full-run image counts must be positive")
    if vocabulary_descriptors_per_image <= 0 or safety_factor < 1.0:
        raise ValueError("Invalid BoVW projection descriptor cap or safety factor")

    full_sift_image_passes = total_encoding_images + vocabulary_training_images
    full_vocabulary_descriptor_upper_bound = (
        vocabulary_training_images * vocabulary_descriptors_per_image
    )
    projected_extraction = (
        pilot_extraction_seconds * full_sift_image_passes / pilot_images
    )
    projected_encoding = (
        pilot_encoding_seconds * total_encoding_images / pilot_images
    )
    projected_clustering = (
        pilot_clustering_seconds
        * full_vocabulary_descriptor_upper_bound
        / pilot_vocabulary_descriptors
    )
    unadjusted = projected_extraction + projected_encoding + projected_clustering
    projected = safety_factor * unadjusted
    return {
        "pilot_images": int(pilot_images),
        "total_encoding_images": int(total_encoding_images),
        "vocabulary_training_images": int(vocabulary_training_images),
        "full_sift_image_passes": int(full_sift_image_passes),
        "pilot_vocabulary_descriptors": int(pilot_vocabulary_descriptors),
        "full_vocabulary_descriptor_upper_bound": int(
            full_vocabulary_descriptor_upper_bound
        ),
        "safety_factor": float(safety_factor),
        "projected_extraction_seconds": float(projected_extraction),
        "projected_encoding_seconds": float(projected_encoding),
        "projected_clustering_seconds": float(projected_clustering),
        "unadjusted_projected_seconds": float(unadjusted),
        "projected_seconds": float(projected),
        "formula": (
            "safety_factor * (pilot_extraction_seconds * "
            "(total_encoding_images + vocabulary_training_images) / pilot_images "
            "+ pilot_encoding_seconds * total_encoding_images / pilot_images "
            "+ pilot_clustering_seconds * "
            "(vocabulary_training_images * descriptors_per_image) / "
            "pilot_vocabulary_descriptors)"
        ),
    }


def sha256_array(array: np.ndarray) -> str:
    """Hash a descriptor sample including its dtype and shape."""

    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def run_descriptor_pass(
    paths: Sequence[Path],
    labels: np.ndarray,
    indices: Iterable[int],
    *,
    config: BoVWConfig,
    descriptors_per_image: int | None,
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract a bounded train-only descriptor matrix and pass statistics."""

    config.validate()
    detector = create_sift(config.sift_nfeatures)
    chunks: list[np.ndarray] = []
    classes: set[int] = set()
    failures: list[dict[str, Any]] = []
    zero_descriptor_images = 0
    descriptor_counts: list[int] = []
    iterable = list(int(index) for index in indices)
    iterator = progress(iterable) if progress is not None else iterable
    start = time.perf_counter()
    for index in iterator:
        try:
            image = load_grayscale(paths[index], config.max_side)
            descriptors = extract_sift_descriptors(
                image, nfeatures=config.sift_nfeatures, sift=detector
            )
        except Exception as exc:
            failures.append(
                {"index": index, "path": paths[index].as_posix(), "error": str(exc)}
            )
            descriptor_counts.append(-1)
            continue
        if descriptors is None:
            zero_descriptor_images += 1
            descriptor_counts.append(0)
            continue
        classes.add(int(labels[index]))
        if descriptors_per_image is not None:
            descriptors = descriptors[:descriptors_per_image]
        descriptor_counts.append(int(len(descriptors)))
        chunks.append(descriptors)
    elapsed = float(time.perf_counter() - start)
    matrix = (
        np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        if chunks
        else np.empty((0, 128), dtype=np.float32)
    )
    stats = {
        "images_requested": len(iterable),
        "images_decoded": len(iterable) - len(failures),
        "decode_failures": failures,
        "zero_descriptor_images": zero_descriptor_images,
        "classes_with_descriptors": len(classes),
        "descriptor_count": int(len(matrix)),
        "descriptor_counts_per_requested_image": descriptor_counts,
        "extraction_seconds": elapsed,
        "descriptor_sha256": sha256_array(matrix),
    }
    return matrix, stats


def make_pilot_gate(
    paths: Sequence[Path],
    labels: np.ndarray,
    *,
    config: BoVWConfig,
    output_dir: Path,
    total_encoding_images: int,
    progress: Callable[[Iterable[int]], Iterable[int]] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Run the balanced train-only pilot and produce the auditable gate record."""

    indices = deterministic_balanced_indices(
        labels,
        config.pilot_images_per_class,
        seed=config.seed,
        source_tag=2,
    )
    descriptors, stats = run_descriptor_pass(
        paths,
        labels,
        indices,
        config=config,
        descriptors_per_image=None,
        progress=progress,
    )

    descriptor_chunks: list[np.ndarray | None] = []
    offset = 0
    for count in stats["descriptor_counts_per_requested_image"]:
        if count < 0 or count == 0:
            descriptor_chunks.append(None)
            continue
        descriptor_chunks.append(descriptors[offset : offset + count])
        offset += count
    if offset != len(descriptors):
        raise RuntimeError("Pilot descriptor accounting is inconsistent")

    capped_chunks = [
        chunk[: config.vocabulary_descriptors_per_image]
        for chunk in descriptor_chunks
        if chunk is not None
    ]
    vocabulary_descriptors = (
        np.concatenate(capped_chunks, axis=0).astype(np.float32, copy=False)
        if capped_chunks
        else np.empty((0, 128), dtype=np.float32)
    )
    benchmark_succeeded = len(vocabulary_descriptors) >= config.vocabulary_size
    benchmark_error: str | None = None
    clustering_seconds = 0.0
    encoding_seconds = 0.0
    projection: dict[str, Any] | None = None
    if benchmark_succeeded:
        try:
            started = time.perf_counter()
            pilot_vocabulary = fit_vocabulary(
                vocabulary_descriptors,
                vocabulary_size=config.vocabulary_size,
                batch_size=config.batch_size,
                n_init=config.n_init,
                seed=config.seed,
                config=config,
            )
            clustering_seconds = float(time.perf_counter() - started)
            started = time.perf_counter()
            for chunk in descriptor_chunks:
                encode_descriptors(chunk, pilot_vocabulary)
            encoding_seconds = float(time.perf_counter() - started)
            vocabulary_training_images = (
                config.vocabulary_images_per_class * len(np.unique(labels))
            )
            projection = project_bovw_runtime(
                pilot_extraction_seconds=stats["extraction_seconds"],
                pilot_encoding_seconds=encoding_seconds,
                pilot_clustering_seconds=clustering_seconds,
                pilot_images=len(indices),
                pilot_vocabulary_descriptors=len(vocabulary_descriptors),
                total_encoding_images=total_encoding_images,
                vocabulary_training_images=vocabulary_training_images,
                vocabulary_descriptors_per_image=(
                    config.vocabulary_descriptors_per_image
                ),
            )
        except Exception as exc:
            benchmark_succeeded = False
            benchmark_error = f"{type(exc).__name__}: {exc}"
    else:
        benchmark_error = (
            f"Only {len(vocabulary_descriptors)} capped descriptors were available; "
            f"{config.vocabulary_size} are required"
        )

    projected_seconds = float(projection["projected_seconds"]) if projection else 0.0
    gate = evaluate_pilot_gate(
        decode_failures=len(stats["decode_failures"]),
        classes_with_descriptors=stats["classes_with_descriptors"],
        expected_classes=len(np.unique(labels)),
        descriptor_count=len(vocabulary_descriptors),
        vocabulary_size=config.vocabulary_size,
        available_disk_bytes=shutil.disk_usage(output_dir.parent).free,
        peak_memory_bytes=peak_rss_bytes(),
        projected_seconds=projected_seconds,
        benchmark_succeeded=benchmark_succeeded,
    )
    gate.update(
        {
            "schema_version": 1,
            "configuration": asdict(config),
            "pilot_indices_sha256": sha256_array(indices),
            "pilot_statistics": stats,
            "pilot_benchmark": {
                "status": "PASS" if benchmark_succeeded else "FAILED",
                "error": benchmark_error,
                "vocabulary_descriptor_count": int(len(vocabulary_descriptors)),
                "clustering_seconds": clustering_seconds,
                "prediction_and_histogram_encoding_seconds": encoding_seconds,
            },
            "runtime_projection": projection,
        }
    )
    return gate, vocabulary_descriptors
