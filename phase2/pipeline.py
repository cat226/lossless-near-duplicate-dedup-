"""
Phase 2 — shared candidate-generation pipeline.

Signatures and candidate sets are produced here from images alone; labels are
only extracted separately (node_labels) for scoring in phase2.metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phase2 import candidates as cand
from phase2 import signatures as sig
from phase2.generate import parse_stem


def node_labels(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth for SCORING ONLY: (family id per node, transformation per node)."""
    parsed = [parse_stem(p.stem) for p in paths]
    family = np.array([int(f) for f, _ in parsed])
    transform = np.array([t for _, t in parsed])
    return family, transform


def primary_state(hashes: np.ndarray, coarse: np.ndarray) -> dict:
    """Distances, per-n Otsu tau_sig, and the three primary candidate sets."""
    ham = sig.hamming_matrix(hashes)
    dist = sig.l2_matrix(coarse)
    tau_sig, method = cand.m2_threshold(dist)
    m1 = cand.m1_candidates(ham, cand.M1_PRIMARY_R)
    m2 = cand.m2_candidates(dist, tau_sig)
    return {
        "ham": ham,
        "dist": dist,
        "tau_sig": tau_sig,
        "threshold_method": method,
        "M1": m1,
        "M2": m2,
        "M3": cand.m3_candidates(m1, m2),
    }
