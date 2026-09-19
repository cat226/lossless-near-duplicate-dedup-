"""
Phase 1c — weighted correlation-clustering disagreement objective.

Design doc docs/phase1c_design.md section 11:

D(C) = sum over positive edges cut by C of w_ij
     + sum over negative edges kept inside C of w_ij

Used identically to score BOTH methods' output clusterings — this is
the only place the objective is computed, so it can't drift between
the two evaluations.
"""

from __future__ import annotations

import numpy as np


def compute_disagreement(
    clusters: list[int],
    sign_matrix: np.ndarray,
    weight_matrix: np.ndarray,
) -> float:
    n = len(clusters)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            same_cluster = clusters[i] == clusters[j]
            sign = sign_matrix[i, j]
            if sign == 1 and not same_cluster:
                total += weight_matrix[i, j]
            elif sign == -1 and same_cluster:
                total += weight_matrix[i, j]
    return float(total)  # native Python float, not numpy.float64 (JSON-serializable)


def total_edge_weight(weight_matrix: np.ndarray) -> float:
    """Sum of w_ij over all unique pairs (i<j) — used to normalize D(C)."""
    n = weight_matrix.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += weight_matrix[i, j]
    return float(total)  # native Python float
