"""
Phase 3 end-to-end clustering analysis (docs/phase3_design.md section 8, Addendum A.4).

Full graph from the frozen verifier with the fixed Phase 1c tau (sign +1 iff r < tau, weight
|tau - r|); candidate graph = +1 only for pairs in the primary M2 candidate set with r < tau.
KwikCluster (unchanged) with 20 frozen seeds on both graphs, scored by D(C) on the FULL graph;
union-find baseline on the same graph. Phase 1c and Phase 2 code is imported, never edited.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baselines.threshold_chaining import threshold_chaining
from cc_dedup.correlation_clustering import kwik_cluster
from cc_dedup.disagreement import compute_disagreement
from cc_dedup.graph import pairwise_distance
from phase2 import candidates as cand
from phase2.consistency import full_graph_from_matrix
from phase2.parallel_matrix import build_distance_matrix, load_rgb

REPO = Path(__file__).resolve().parent.parent
TAU_1C = json.loads((REPO / "results" / "phase1c_results.json").read_text())["threshold"]
SEEDS = tuple(range(12345, 12365))  # 20 frozen KwikCluster seeds
ASSERT_SEED = 12345
PRIMARY_SAMPLE, ALIGNED_SAMPLE = 500, 200


def all_pairs_sample(n: int, k: int, seed: int = ASSERT_SEED) -> list[tuple[int, int]]:
    flat = np.sort(np.random.default_rng(seed).choice(cand.n_pairs(n), size=k, replace=False))
    rows, cols = cand.flat_to_pair(flat, n)
    return list(zip(rows.tolist(), cols.tolist()))


def full_graph(paths: list[Path], workers: int, log=lambda m: None) -> np.ndarray:
    return build_distance_matrix(paths, workers=workers, log=log)


def primary_exactness(paths: list[Path], r: np.ndarray, k: int = PRIMARY_SAMPLE) -> dict:
    """Assert: the parallel graph equals single-process pairwise_distance, bitwise, on a seeded sample."""
    pairs = all_pairs_sample(len(paths), min(k, cand.n_pairs(len(paths))))
    single = np.array([pairwise_distance(load_rgb(paths[i]), load_rgb(paths[j])) for i, j in pairs])
    par = np.array([r[i, j] for i, j in pairs])
    return {"pairs_checked": len(pairs), "pairs_equal": int((single == par).sum()),
            "max_abs_difference": float(np.abs(single - par).max()), "bitwise_equal": bool(np.array_equal(single, par))}


def cluster_comparison(r: np.ndarray, candidate_flats: np.ndarray) -> dict:
    n = r.shape[0]
    sign_full, weight = full_graph_from_matrix(r, TAU_1C)
    P = np.zeros((n, n), dtype=bool)
    rows, cols = cand.flat_to_pair(candidate_flats, n)
    P[rows, cols] = True
    sign_cand = np.where((P | P.T) & (sign_full == 1), 1, -1).astype(np.int8)
    np.fill_diagonal(sign_cand, 0)

    def stats(sign):
        d, k = [], []
        for s in SEEDS:
            clusters = kwik_cluster(sign, seed=s)
            d.append(compute_disagreement(clusters, sign_full, weight))
            k.append(len(set(clusters)))
        return {"D_mean": float(np.mean(d)), "D_sd": float(np.std(d, ddof=1)), "clusters_mean": float(np.mean(k)), "seeds": len(SEEDS)}

    uf_full = threshold_chaining(sign_full)
    uf_cand = threshold_chaining(sign_cand)
    return {
        "positive_edges_full": int((sign_full[np.triu_indices(n, 1)] == 1).sum()),
        "positive_edges_candidate_graph": int((sign_cand[np.triu_indices(n, 1)] == 1).sum()),
        "kwik_full_graph": stats(sign_full),
        "kwik_candidate_graph": stats(sign_cand),
        "union_find_full_graph": {"D": compute_disagreement(uf_full, sign_full, weight), "clusters": len(set(uf_full))},
        "union_find_candidate_graph": {"D": compute_disagreement(uf_cand, sign_full, weight), "clusters": len(set(uf_cand))},
    }
