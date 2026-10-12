"""
Phase 2 — consistency check against Phase 1c (docs/phase2_design.md 4 and 17.8).

Hash matching is impossible (Phase 1c recorded no image hashes), so the check is
on the rebuilt graph: tau, edge counts, and the two clusterings' disagreement.
Reads results/phase1c_results.json read-only. Uses the unmodified Phase 1c
threshold, clustering, baseline and objective code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from baselines.threshold_chaining import threshold_chaining
from cc_dedup.correlation_clustering import kwik_cluster
from cc_dedup.disagreement import compute_disagreement
from cc_dedup.graph import compute_otsu_threshold

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE1C_RESULTS = REPO_ROOT / "results" / "phase1c_results.json"
SEED = 12345


def full_graph_from_matrix(r: np.ndarray, tau: float):
    """Phase 1c sign/weight rule: sign +1 iff r < tau, weight |tau - r|."""
    sign = np.where(r < tau, 1, -1).astype(np.int8)
    np.fill_diagonal(sign, 0)
    weight = np.abs(tau - r)
    np.fill_diagonal(weight, 0.0)
    return sign, weight


def check_against_phase1c(r: np.ndarray) -> dict:
    ref = json.loads(PHASE1C_RESULTS.read_text())
    n = r.shape[0]
    upper = r[np.triu_indices(n, k=1)]
    tau, method = compute_otsu_threshold(upper)
    sign, weight = full_graph_from_matrix(r, tau)
    n_pos = int((sign[np.triu_indices(n, k=1)] == 1).sum())

    kwik = kwik_cluster(sign, seed=SEED)
    base = threshold_chaining(sign)
    d_kwik = compute_disagreement(kwik, sign, weight)
    d_base = compute_disagreement(base, sign, weight)

    result = {
        "tau": tau,
        "threshold_method": method,
        "positive_edge_count": n_pos,
        "kwik_clusters": len(set(kwik)),
        "baseline_clusters": len(set(base)),
        "kwik_disagreement": d_kwik,
        "baseline_disagreement": d_base,
        "phase1c_reference": {
            "tau": ref["threshold"],
            "positive_edge_count": ref["positive_edge_count"],
            "kwik_clusters": ref["correlation_clustering"]["cluster_count"],
            "baseline_clusters": ref["baseline"]["cluster_count"],
            "kwik_disagreement": ref["correlation_clustering"]["disagreement"],
            "baseline_disagreement": ref["baseline"]["disagreement"],
        },
    }
    r_ref = result["phase1c_reference"]
    result["matches"] = {
        "tau": tau == r_ref["tau"],
        "positive_edge_count": n_pos == r_ref["positive_edge_count"],
        "kwik_clusters": result["kwik_clusters"] == r_ref["kwik_clusters"],
        "baseline_clusters": result["baseline_clusters"] == r_ref["baseline_clusters"],
        "kwik_disagreement": d_kwik == r_ref["kwik_disagreement"],
        "baseline_disagreement": d_base == r_ref["baseline_disagreement"],
    }
    result["all_match"] = all(result["matches"].values())
    return result
