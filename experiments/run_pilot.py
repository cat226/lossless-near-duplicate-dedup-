"""
Phase 1c — pilot validation run.

Design doc docs/phase1c_design.md section 16. Purpose: implementation
correctness only, NEVER parameter tuning. Runs on the first 3 originals
(001, 002, 003) by filename order plus their 18 transformed variants =
21 nodes, C(21,2) = 210 pairs.

This script's numeric results (threshold, D(C), clusters) are NOT
interpretable findings and are written only to
results/pilot_validation.json — never merged into
results/phase1c_results.json. If any checklist item fails, the fix is a
code-correctness fix applied uniformly, not a parameter change.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from cc_dedup.graph import build_graph, discover_images
from cc_dedup.correlation_clustering import kwik_cluster
from cc_dedup.disagreement import compute_disagreement
from baselines.threshold_chaining import threshold_chaining

PILOT_SOURCE_IDS = {"001", "002", "003"}
SEED = 12345


def main():
    checklist = {}

    paths = discover_images(source_ids=PILOT_SOURCE_IDS)
    checklist["node_count_is_21"] = (len(paths) == 21)

    expected_pairs = 21 * 20 // 2
    graph = build_graph(paths)
    n = len(graph.nodes)
    actual_pairs = n * (n - 1) // 2
    checklist["pair_count_is_210"] = (actual_pairs == expected_pairs == 210)

    upper = graph.r_matrix[np.triu_indices(n, k=1)]
    checklist["distances_finite"] = bool(np.all(np.isfinite(upper)))

    upper_w = graph.weight_matrix[np.triu_indices(n, k=1)]
    checklist["weights_finite"] = bool(np.all(np.isfinite(upper_w)))

    checklist["threshold_method_valid"] = graph.threshold_method in ("otsu", "range_midpoint_fallback")
    checklist["threshold_is_finite"] = bool(np.isfinite(graph.tau))

    checklist["positive_and_negative_edges_present"] = (graph.n_positive_edges > 0 and graph.n_negative_edges > 0)

    t0 = time.perf_counter()
    baseline_clusters = threshold_chaining(graph.sign_matrix)
    baseline_seconds = time.perf_counter() - t0
    checklist["union_find_terminates_valid_partition"] = (
        len(baseline_clusters) == n and all(c is not None for c in baseline_clusters)
    )

    t0 = time.perf_counter()
    kwik_clusters = kwik_cluster(graph.sign_matrix, seed=SEED)
    kwik_seconds = time.perf_counter() - t0
    checklist["kwikcluster_terminates_valid_partition"] = (
        len(kwik_clusters) == n and all(c != -1 for c in kwik_clusters)
    )

    baseline_disagreement = compute_disagreement(baseline_clusters, graph.sign_matrix, graph.weight_matrix)
    kwik_disagreement = compute_disagreement(kwik_clusters, graph.sign_matrix, graph.weight_matrix)
    checklist["disagreement_finite_nonnegative"] = bool(
        np.isfinite(baseline_disagreement) and baseline_disagreement >= 0
        and np.isfinite(kwik_disagreement) and kwik_disagreement >= 0
    )

    all_passed = all(checklist.values())

    result = {
        "status": "DISCARDED — validation only, not experimental evidence",
        "checklist": checklist,
        "all_checks_passed": all_passed,
        "diagnostic_only_values": {
            "node_count": n,
            "pair_count": actual_pairs,
            "tau": graph.tau,
            "threshold_method": graph.threshold_method,
            "n_positive_edges": graph.n_positive_edges,
            "n_negative_edges": graph.n_negative_edges,
            "graph_build_seconds": graph.build_seconds,
            "baseline_seconds": baseline_seconds,
            "kwikcluster_seconds": kwik_seconds,
            "baseline_disagreement": baseline_disagreement,
            "kwikcluster_disagreement": kwik_disagreement,
        },
    }

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "pilot_validation.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"Pilot: {n} nodes, {actual_pairs} pairs")
    print(f"Checklist: {checklist}")
    print(f"ALL CHECKS PASSED: {all_passed}")
    print(f"Written to {out_path} (discarded, not used as Phase 1c evidence)")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
