"""
Phase 1c — signed weighted graph construction.

This is the SINGLE place the graph is built (docs/phase1c_design.md
section 18). Both the correlation-clustering method and the baseline
consume the exact same Node/graph output produced here — neither
recomputes similarity, threshold, signs, or weights.

Implements docs/phase1c_design.md sections 4 (corrected pixel distance),
5 (normalization), 6 (Otsu threshold + flat-curve fallback), 7 (signed
edge rule), 8 (edge weights). Does not import or modify anything in
phase0/ — phase0/'s pixel_delta_size has a documented int8-wraparound
defect (design doc section 4a) that is corrected here, but that
correction is not applied retroactively to phase0/.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PHASE0_ORIGINALS = Path(__file__).resolve().parent.parent / "phase0" / "data" / "originals"
PHASE0_TRANSFORMED = Path(__file__).resolve().parent.parent / "phase0" / "data" / "transformed"

OTSU_BINS = 256
FLAT_OTSU_TOLERANCE = 1e-9


@dataclass
class Node:
    index: int
    path: Path
    stem: str
    source_id: str      # e.g. "001" — the original an image derives from
    transform: str       # "original" or one of resize/jpegq50/crop/bright/rotate/png


@dataclass
class Graph:
    nodes: list[Node]
    r_matrix: np.ndarray        # (n, n) float64, symmetric, diagonal unused
    sign_matrix: np.ndarray     # (n, n) int8, +1/-1, diagonal unused
    weight_matrix: np.ndarray   # (n, n) float64, symmetric, diagonal unused
    tau: float
    threshold_method: str        # "otsu" or "range_midpoint_fallback"
    n_positive_edges: int
    n_negative_edges: int
    build_seconds: float


TRANSFORM_SUFFIXES = ["resize", "jpegq50", "crop", "bright", "rotate", "png"]


def _stem_to_source_and_transform(stem: str) -> tuple[str, str]:
    for suffix in TRANSFORM_SUFFIXES:
        marker = f"_{suffix}"
        if stem.endswith(marker):
            return stem[: -len(marker)], suffix
    return stem, "original"


def discover_images(source_ids: set[str] | None = None) -> list[Path]:
    """List Phase 0 image paths (unmodified, read-only), deterministically
    sorted. If source_ids is given, restrict to images whose source_id is
    in that set (used by the pilot, design doc section 16).
    """
    paths = sorted(p for p in PHASE0_ORIGINALS.glob("*") if p.is_file())
    paths += sorted(p for p in PHASE0_TRANSFORMED.glob("*") if p.is_file())

    if source_ids is None:
        return paths

    kept = []
    for p in paths:
        sid, _ = _stem_to_source_and_transform(p.stem)
        if sid in source_ids:
            kept.append(p)
    return kept


def _load_nodes(paths: list[Path]) -> tuple[list[Node], list[np.ndarray]]:
    nodes = []
    arrays = []
    for i, p in enumerate(paths):
        sid, transform = _stem_to_source_and_transform(p.stem)
        nodes.append(Node(index=i, path=p, stem=p.stem, source_id=sid, transform=transform))
        with Image.open(p) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        arrays.append(arr)
    return nodes, arrays


def pairwise_distance(arr_a: np.ndarray, arr_b: np.ndarray) -> float:
    """Corrected, symmetric pixel distance (design doc sections 4-5).

    Subtraction happens in int16 (wide enough to hold the full [-255,255]
    range), THEN abs() is taken, THEN the result is narrowed to uint8 —
    in that order, so nothing wraps. Dimension mismatches are handled by
    comparing only the overlapping top-left region (same non-goal Phase 0
    already committed to: no silent resizing).
    """
    ha, wa = arr_a.shape[0], arr_a.shape[1]
    hb, wb = arr_b.shape[0], arr_b.shape[1]

    if (ha, wa) != (hb, wb):
        h = min(ha, hb)
        w = min(wa, wb)
        a = arr_a[:h, :w, :]
        b = arr_b[:h, :w, :]
    else:
        a = arr_a
        b = arr_b

    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))  # int16, range [0, 255] after abs
    diff = diff.astype(np.uint8)                              # safe to narrow: values already in [0, 255]

    compressed_len = len(zlib.compress(diff.tobytes(), level=9))
    denom = max(wa * ha * 3, wb * hb * 3)
    return compressed_len / denom


def compute_otsu_threshold(r_values: np.ndarray) -> tuple[float, str]:
    """Design doc section 6: Otsu's method over a fixed 256-bin histogram
    of r_values, with a deterministic range-midpoint fallback if the
    between-class-variance curve is flat (range < 1e-9).

    Uses ONLY r_values — no label of any kind is accessible here.
    Returns (tau, threshold_method) where threshold_method is exactly
    "otsu" or "range_midpoint_fallback".
    """
    hist, edges = np.histogram(r_values, bins=OTSU_BINS)
    hist = hist.astype(np.float64)
    total = hist.sum()
    bin_centers = (edges[:-1] + edges[1:]) / 2.0

    # Candidate splits: t in 1..OTSU_BINS-1, i.e. the internal bin
    # boundaries of a 256-bin histogram (t=0 or t=OTSU_BINS would leave
    # one side empty and contribute no separating power).
    between_class_variance = np.zeros(OTSU_BINS - 1, dtype=np.float64)
    for t in range(1, OTSU_BINS):
        w0 = hist[:t].sum() / total
        w1 = hist[t:].sum() / total
        if w0 == 0 or w1 == 0:
            between_class_variance[t - 1] = 0.0
            continue
        mu0 = (hist[:t] * bin_centers[:t]).sum() / (w0 * total)
        mu1 = (hist[t:] * bin_centers[t:]).sum() / (w1 * total)
        between_class_variance[t - 1] = w0 * w1 * (mu0 - mu1) ** 2

    variance_range = between_class_variance.max() - between_class_variance.min()

    if variance_range < FLAT_OTSU_TOLERANCE:
        tau = (float(r_values.min()) + float(r_values.max())) / 2.0
        return tau, "range_midpoint_fallback"

    best_t = int(np.argmax(between_class_variance)) + 1  # +1 to undo the t-1 offset above
    tau = float(edges[best_t])
    return tau, "otsu"


def build_graph(paths: list[Path]) -> Graph:
    """Design doc section 15, steps 2-4: build the complete graph
    (pairwise distances), compute the threshold (Otsu or fallback), and
    assign signs/weights. This is the only function that constructs the
    graph both downstream methods consume.
    """
    start = time.perf_counter()

    nodes, arrays = _load_nodes(paths)
    n = len(nodes)

    r_matrix = np.zeros((n, n), dtype=np.float64)
    pair_values = np.empty(n * (n - 1) // 2, dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            r = pairwise_distance(arrays[i], arrays[j])
            r_matrix[i, j] = r
            r_matrix[j, i] = r
            pair_values[k] = r
            k += 1

    tau, threshold_method = compute_otsu_threshold(pair_values)

    sign_matrix = np.zeros((n, n), dtype=np.int8)
    weight_matrix = np.zeros((n, n), dtype=np.float64)
    n_pos = 0
    n_neg = 0
    for i in range(n):
        for j in range(i + 1, n):
            r = r_matrix[i, j]
            sign = 1 if r < tau else -1
            weight = abs(tau - r)
            sign_matrix[i, j] = sign
            sign_matrix[j, i] = sign
            weight_matrix[i, j] = weight
            weight_matrix[j, i] = weight
            if sign == 1:
                n_pos += 1
            else:
                n_neg += 1

    build_seconds = time.perf_counter() - start

    return Graph(
        nodes=nodes,
        r_matrix=r_matrix,
        sign_matrix=sign_matrix,
        weight_matrix=weight_matrix,
        tau=tau,
        threshold_method=threshold_method,
        n_positive_edges=n_pos,
        n_negative_edges=n_neg,
        build_seconds=build_seconds,
    )
