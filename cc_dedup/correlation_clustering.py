"""
Phase 1c — KwikCluster / Pivot algorithm (Ailon, Charikar, Newman, 2005).

Design doc docs/phase1c_design.md section 9. A known, practical
correlation-clustering heuristic — not invented for this project, and
not a claim of exact NP-hard optimality.

Procedure: while unclustered vertices remain, pick a uniformly random
unclustered pivot; cluster it with every remaining unclustered vertex
having a positive edge to it; remove that cluster; repeat.

Reads only the sign_matrix from an already-built cc_dedup.graph.Graph —
never modifies the graph.
"""

from __future__ import annotations

import random

import numpy as np


def kwik_cluster(sign_matrix: np.ndarray, seed: int) -> list[int]:
    """Returns a list of cluster ids, one per node index (0..n-1).

    Deterministic given a fixed seed (design doc section 9: randomized
    pivot choice, seed 12345, reused from Phase 0's dataset seed).
    """
    n = sign_matrix.shape[0]
    rng = random.Random(seed)

    unclustered = list(range(n))  # kept sorted for reproducible rng.choice population order
    clusters = [-1] * n
    next_cluster_id = 0

    while unclustered:
        pivot = rng.choice(unclustered)
        cluster_members = [pivot]
        for u in unclustered:
            if u != pivot and sign_matrix[pivot, u] == 1:
                cluster_members.append(u)

        for m in cluster_members:
            clusters[m] = next_cluster_id
        next_cluster_id += 1

        member_set = set(cluster_members)
        unclustered = [u for u in unclustered if u not in member_set]

    assert all(c != -1 for c in clusters), "KwikCluster left a node unclustered"
    return clusters
