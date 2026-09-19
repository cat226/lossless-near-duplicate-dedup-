"""
Phase 1c — secondary, ground-truth-based evaluation.

Design doc docs/phase1c_design.md section 14. MUST only be called after
both clustering methods are frozen and have produced final clusterings
(section 15, step 8) — nothing here may feed back into the graph,
threshold, or either clustering method. This module does not import
cc_dedup.graph or cc_dedup.correlation_clustering; it only consumes
already-produced cluster assignments and source_id labels.
"""

from __future__ import annotations

import math


def pairwise_precision_recall_f1(clusters: list[int], source_ids: list[str]) -> dict:
    """Standard pairwise clustering evaluation against a known partition
    (source_id = ground-truth cluster membership by dataset construction).
    """
    n = len(clusters)
    tp = fp = fn = tn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_gt = source_ids[i] == source_ids[j]
            same_pred = clusters[i] == clusters[j]
            if same_gt and same_pred:
                tp += 1
            elif (not same_gt) and same_pred:
                fp += 1
            elif same_gt and (not same_pred):
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = math.nan
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_pairs": tp,
        "false_positive_pairs": fp,
        "false_negative_pairs": fn,
        "true_negative_pairs": tn,
    }


def label_midpoint_threshold(r_matrix, source_ids: list[str]) -> float:
    """Design doc section 14: diagnostic-only same/different distance
    midpoint, computed AFTER the primary graph/threshold/clustering are
    already frozen. Never fed back into thresholding or clustering.
    """
    n = len(source_ids)
    same_vals = []
    diff_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            r = r_matrix[i, j]
            if source_ids[i] == source_ids[j]:
                same_vals.append(r)
            else:
                diff_vals.append(r)

    mean_same = sum(same_vals) / len(same_vals) if same_vals else math.nan
    mean_diff = sum(diff_vals) / len(diff_vals) if diff_vals else math.nan
    return (mean_same + mean_diff) / 2.0


def cluster_count(clusters: list[int]) -> int:
    return len(set(clusters))
