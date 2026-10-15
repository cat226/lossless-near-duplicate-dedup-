"""
The audit's independent primitives must agree with the run's code on the same inputs (otherwise an
audit mismatch would not mean anything).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

from phase2 import candidates as cand
from phase2 import metrics as met
from phase2 import signatures as sig
from phase3 import blockwise as bw
from phase3 import hashes as hs
from phase3.splits import REPO

spec = importlib.util.spec_from_file_location("audit_phase3", REPO / "experiments" / "audit_phase3.py")
audit = importlib.util.module_from_spec(spec)
sys.modules["audit_phase3"] = audit
spec.loader.exec_module(audit)


def _sigs(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 1056)).astype(np.float32)
    for lo, hi in ((0, 1024), (1024, 1048), (1048, 1056)):
        x[:, lo:hi] /= np.linalg.norm(x[:, lo:hi], axis=1, keepdims=True)
    return x


def test_independent_hamming_equals_phase2_hamming_path():
    rng = np.random.default_rng(1)
    h = rng.integers(0, 2**63, size=300, dtype=np.int64).astype(np.uint64)
    h[10] = h[3] ^ np.uint64(0b1011)  # a near pair
    mine = audit.hamming_flats(h, (4, 10), block=64)
    ham = sig.hamming_matrix(h)
    for R in (4, 10):
        assert np.array_equal(mine[R], np.sort(cand.pair_flat_indices(np.triu(ham <= R, 1))))


def test_independent_m2_and_flat_index_equal_blockwise():
    x = _sigs(250, seed=4)
    a = x.astype(np.float64)
    tau = float(np.quantile(audit.distances_upper(a, block=64), 0.05))
    assert np.array_equal(audit.m2_flats(a, [tau], block=64)[tau], bw.m2_candidate_flats(x, tau))
    n = 250
    rows, cols = np.array([0, 3, 100, 248]), np.array([1, 7, 200, 249])
    assert np.array_equal(audit.flat_idx(rows, cols, n), cand._row_starts(n)[rows] + (cols - rows - 1))


def test_independent_exact_and_sha_equal_phase3():
    sha = np.array(["a", "b", "a", "c", "a"])
    starts = cand._row_starts(5)
    assert np.array_equal(audit.exact_flats(sha), hs.exact_candidate_flats(sha.tolist(), starts))
    flat = np.array([5, 1, 3], dtype=np.int64)
    from phase3.evaluate import flat_hash
    assert audit.sha_of(flat) == flat_hash(flat)


def test_independent_scoring_equals_phase2_metrics():
    names = [f"i{k:05d}" for k in range(14)]
    fam = [("A", "original"), ("A", "resize"), ("A", "crop"), ("A", "png"), ("B", "original"), ("B", "bright"), ("B", "rotate"),
           (None, "singleton"), (None, "singleton"), (None, "singleton"), ("C", "original"), ("C", "jpegq50"), ("C", "png"), (None, "singleton")]
    labels = {nm: {"source": nm, "family": f, "transform": t} for nm, (f, t) in zip(names, fam)}
    paths = [Path(f"{nm}.png") for nm in names]
    T = audit.true_info(labels, paths)
    family = np.array([ord(f) - 65 if f else -1 - k for k, (f, _) in enumerate(fam)])
    truth = met.truth_matrix(family)
    transform = np.array([t for _, t in fam])
    n = len(names)
    rng = np.random.default_rng(0)
    for trial in range(5):
        P = np.triu(rng.random((n, n)) < 0.25, 1)
        flat = np.sort(cand.pair_flat_indices(P))
        theirs = met.score(P, truth, transform)
        mine = audit.score_independent(flat, T)
        for k in ("candidate_pairs", "recall", "reduction", "recall_original_transform", "recall_transform_transform",
                  "recall_per_transform", "min_recall_per_transform"):
            assert audit.same(mine[k], theirs[k]), (trial, k)
        assert mine["gate_pass"] == met.passes_gate(theirs)["pass"]
