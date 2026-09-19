"""
Phase 1c — full 280-node experiment.

Design doc docs/phase1c_design.md section 15 defines the frozen
execution order this script follows exactly:

  1. Load dataset (Phase 0 machinery, unchanged)
  2. Build complete graph: compute r_ij for all 39,060 pairs
  3. Compute threshold tau (Otsu or fallback) — input is ONLY r_ij values
  4. Assign signs and weights
  5. Run baseline (union-find) on the graph from steps 2-4
  6. Run KwikCluster (seed 12345) on the same graph
  7. Compute D(C) for both -> PRIMARY GATE DECISION MADE HERE
  8. Only after step 7: compute secondary metrics (pairwise P/R/F1,
     diagnostic label midpoint)

No step is re-run with different parameters after its output has been
used by a later step. phase0/ is not read except as read-only input
image files, and is not modified.
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
from cc_dedup.disagreement import compute_disagreement, total_edge_weight
from cc_dedup.metrics import pairwise_precision_recall_f1, label_midpoint_threshold, cluster_count
from baselines.threshold_chaining import threshold_chaining

SEED = 12345


def main():
    # --- Step 1: load dataset (Phase 0 machinery, unchanged) ---
    paths = discover_images(source_ids=None)
    assert len(paths) == 280, f"expected 280 images, found {len(paths)}"

    # --- Steps 2-4: build complete graph, threshold, signs/weights ---
    graph = build_graph(paths)
    n = len(graph.nodes)
    n_pairs = n * (n - 1) // 2
    assert n_pairs == 39_060, f"expected 39,060 pairs, computed {n_pairs}"

    upper = graph.r_matrix[np.triu_indices(n, k=1)]
    upper_w = graph.weight_matrix[np.triu_indices(n, k=1)]
    assert np.all(np.isfinite(upper)), "non-finite distance found"
    assert np.all(np.isfinite(upper_w)), "non-finite weight found"

    # --- Step 5: baseline (union-find on E+) ---
    t0 = time.perf_counter()
    baseline_clusters = threshold_chaining(graph.sign_matrix)
    baseline_seconds = time.perf_counter() - t0

    # --- Step 6: KwikCluster, seed 12345, same graph ---
    t0 = time.perf_counter()
    kwik_clusters = kwik_cluster(graph.sign_matrix, seed=SEED)
    kwik_seconds = time.perf_counter() - t0

    # --- Step 7: D(C) for both -> PRIMARY GATE DECISION ---
    baseline_disagreement = compute_disagreement(baseline_clusters, graph.sign_matrix, graph.weight_matrix)
    kwik_disagreement = compute_disagreement(kwik_clusters, graph.sign_matrix, graph.weight_matrix)
    total_weight = total_edge_weight(graph.weight_matrix)

    gate_passed = kwik_disagreement < baseline_disagreement

    # --- Step 8: secondary metrics, ONLY after step 7 ---
    source_ids = [node.source_id for node in graph.nodes]
    baseline_prf1 = pairwise_precision_recall_f1(baseline_clusters, source_ids)
    kwik_prf1 = pairwise_precision_recall_f1(kwik_clusters, source_ids)
    diagnostic_midpoint = label_midpoint_threshold(graph.r_matrix, source_ids)

    results = {
        "node_count": n,
        "edge_count": n_pairs,
        "positive_edge_count": graph.n_positive_edges,
        "negative_edge_count": graph.n_negative_edges,
        "threshold": graph.tau,
        "threshold_method": graph.threshold_method,
        "random_seed": SEED,
        "graph_build_seconds": graph.build_seconds,
        "total_edge_weight": total_weight,
        "baseline": {
            "algorithm": "threshold_chaining (union-find on E+)",
            "cluster_count": cluster_count(baseline_clusters),
            "runtime_seconds": baseline_seconds,
            "disagreement": baseline_disagreement,
            "normalized_disagreement": baseline_disagreement / total_weight if total_weight > 0 else float("nan"),
            "secondary_pairwise_metrics": baseline_prf1,
        },
        "correlation_clustering": {
            "algorithm": "KwikCluster / Pivot (Ailon, Charikar, Newman 2005)",
            "cluster_count": cluster_count(kwik_clusters),
            "runtime_seconds": kwik_seconds,
            "disagreement": kwik_disagreement,
            "normalized_disagreement": kwik_disagreement / total_weight if total_weight > 0 else float("nan"),
            "secondary_pairwise_metrics": kwik_prf1,
        },
        "diagnostic_label_midpoint_threshold": diagnostic_midpoint,
        "primary_gate": {
            "criterion": "D(correlation_clustering) < D(baseline)",
            "correlation_clustering_disagreement": kwik_disagreement,
            "baseline_disagreement": baseline_disagreement,
            "passed": gate_passed,
        },
    }

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "phase1c_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"\nWritten to {out_path}")
    print(f"\nPRIMARY GATE: {'PASSED' if gate_passed else 'FAILED'}")


if __name__ == "__main__":
    main()
