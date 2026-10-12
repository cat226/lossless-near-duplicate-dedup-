"""
Phase 2 — scoring (docs/phase2_design.md sections 7, 12, 17.7-17.8).

The ONLY module that sees ground truth (family and transformation labels), and
only to score candidate sets that have already been fixed.
"""

from __future__ import annotations

import numpy as np

TRANSFORMS = ("resize", "jpegq50", "crop", "bright", "rotate", "png")
GATE_RECALL = 0.95
GATE_REDUCTION = 0.90
GATE_PER_TYPE_RECALL = 0.85


def n_pairs(n: int) -> int:
    return n * (n - 1) // 2


def truth_matrix(family: np.ndarray) -> np.ndarray:
    """Strictly-upper boolean matrix: pair is a true near-duplicate iff same family."""
    return np.triu(family[:, None] == family[None, :], 1)


def _ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else float("nan")


def score(candidates: np.ndarray, truth: np.ndarray, transform: np.ndarray) -> dict:
    """transform: array of str labels per node ('original' or one of TRANSFORMS)."""
    n = candidates.shape[0]
    total = n_pairs(n)
    n_true = int(truth.sum())
    n_cand = int(candidates.sum())
    hit = candidates & truth

    is_orig = transform == "original"
    ot = truth & (is_orig[:, None] ^ is_orig[None, :])
    tt = truth & ~is_orig[:, None] & ~is_orig[None, :]

    per_type = {}
    for t in TRANSFORMS:
        has = transform == t
        mask = truth & (has[:, None] | has[None, :])
        per_type[t] = _ratio(int((candidates & mask).sum()), int(mask.sum()))

    return {
        "n": n,
        "all_pairs": total,
        "true_pairs": n_true,
        "candidate_pairs": n_cand,
        "true_pairs_found": int(hit.sum()),
        "recall": _ratio(int(hit.sum()), n_true),
        "reduction": 1.0 - _ratio(n_cand, total),
        "candidates_per_true_pair": _ratio(n_cand, n_true),
        "recall_original_transform": _ratio(int((candidates & ot).sum()), int(ot.sum())),
        "recall_transform_transform": _ratio(int((candidates & tt).sum()), int(tt.sum())),
        "recall_per_transform": per_type,
        "min_recall_per_transform": min(per_type.values()),
    }


def passes_gate(s: dict) -> dict:
    """Per-size gate components (section 12). A method passes only if every
    size passes all three."""
    recall_ok = s["recall"] >= GATE_RECALL
    reduction_ok = s["reduction"] >= GATE_REDUCTION
    per_type_ok = s["min_recall_per_transform"] >= GATE_PER_TYPE_RECALL
    return {
        "recall_ge_0.95": bool(recall_ok),
        "reduction_ge_0.90": bool(reduction_ok),
        "per_transform_recall_ge_0.85": bool(per_type_ok),
        "pass": bool(recall_ok and reduction_ok and per_type_ok),
    }
