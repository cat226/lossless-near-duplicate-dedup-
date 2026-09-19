"""
Phase 1c — naive threshold-based chaining baseline.

Design doc docs/phase1c_design.md section 10: connected components of
the positive-edge subgraph G+ = (V, E+), via union-find (path
compression + union by rank). Reads only the sign_matrix from an
already-built cc_dedup.graph.Graph — same graph, same signs, same
weights as the correlation-clustering method; never modifies the graph
and uses no separate threshold or similarity computation.
"""

from __future__ import annotations

import numpy as np


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def threshold_chaining(sign_matrix: np.ndarray) -> list[int]:
    """Returns a list of cluster ids, one per node index (0..n-1)."""
    n = sign_matrix.shape[0]
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if sign_matrix[i, j] == 1:
                uf.union(i, j)

    # Normalize root ids to a compact 0..k-1 cluster-id range.
    root_to_cluster_id: dict[int, int] = {}
    clusters = [0] * n
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_cluster_id:
            root_to_cluster_id[root] = len(root_to_cluster_id)
        clusters[i] = root_to_cluster_id[root]

    return clusters
