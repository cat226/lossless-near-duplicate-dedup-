"""
Phase 2 — candidate-pair generation (docs/phase2_design.md sections 6, 17.5-17.6).

Candidate sets are boolean strictly-upper-triangular (n, n) matrices. This module
takes only signature distances; it has no access to family or transformation
labels (ground truth is used only to score the sets, in phase2.metrics).
"""

from __future__ import annotations

import hashlib

import numpy as np

from cc_dedup.graph import compute_otsu_threshold  # Phase 1c Otsu, unchanged

SEED = 12345
M1_PRIMARY_R = 10
M1_SECONDARY_R = (4, 6, 8, 10, 12, 14, 16)
M2_SECONDARY_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)


def n_pairs(n: int) -> int:
    return n * (n - 1) // 2


def upper_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu(np.ones(matrix.shape, dtype=bool), 1)]


def m1_candidates(hamming: np.ndarray, radius: int) -> np.ndarray:
    return np.triu(hamming <= radius, 1)


def m2_threshold(distance: np.ndarray) -> tuple[float, str]:
    """Primary M2 operating point: Otsu over all-pairs signature distances."""
    return compute_otsu_threshold(upper_values(distance))


def m2_candidates(distance: np.ndarray, tau_sig: float) -> np.ndarray:
    return np.triu(distance < tau_sig, 1)


def m3_candidates(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return m1 | m2


def _row_starts(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.int64)
    return i * (2 * n - i - 1) // 2


def flat_to_pair(flat: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-major upper-triangle flat index -> (i, j), exact integer arithmetic."""
    starts = _row_starts(n)
    i = np.searchsorted(starts, flat, side="right") - 1
    j = flat - starts[i] + i + 1
    return i, j


def pair_flat_indices(candidates: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(candidates)
    return _row_starts(candidates.shape[0])[rows] + (cols - rows - 1)


def candidate_hash(candidates: np.ndarray) -> str:
    """SHA-256 of the sorted flat-index array (int64, little endian)."""
    flat = np.sort(pair_flat_indices(candidates)).astype("<i8")
    return hashlib.sha256(flat.tobytes()).hexdigest()


def random_control(n: int, k: int, seed: int = SEED) -> np.ndarray:
    """B1: k pairs drawn uniformly without replacement from the n(n-1)/2 pairs."""
    rng = np.random.default_rng(seed)
    flat = np.sort(rng.choice(n_pairs(n), size=k, replace=False))
    rows, cols = flat_to_pair(flat, n)
    out = np.zeros((n, n), dtype=bool)
    out[rows, cols] = True
    return out
