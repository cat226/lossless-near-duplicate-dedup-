"""
Phase 2 — expensive verification stage (docs/phase2_design.md sections 8, 17.9).

Thin wrapper over the UNCHANGED Phase 1c cc_dedup.graph.pairwise_distance.
Verified pairs are judged against the fixed Phase 1c tau at every n.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cc_dedup.graph import pairwise_distance  # Phase 1c, unchanged
from phase2.parallel_matrix import load_rgb

SEED = 12345
SAMPLE_SIZE = 200


def verify_pair(path_i: Path, path_j: Path) -> float:
    return pairwise_distance(load_rgb(path_i), load_rgb(path_j))


def sample_pair_indices(n_candidates: int, seed: int = SEED, size: int = SAMPLE_SIZE) -> np.ndarray:
    """Fixed sample of candidate-pair positions (sorted), seed 12345."""
    k = min(size, n_candidates)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_candidates, size=k, replace=False))


def time_verification(paths: list[Path], pairs: list[tuple[int, int]]) -> list[float]:
    """Single-process wall time of each pairwise_distance call on already-decoded
    arrays (image decode excluded). Caller must ensure no parallel job is running."""
    times = []
    for i, j in pairs:
        a, b = load_rgb(paths[i]), load_rgb(paths[j])  # decoded outside the timed region
        t0 = time.perf_counter()
        pairwise_distance(a, b)
        times.append(time.perf_counter() - t0)
    return times
