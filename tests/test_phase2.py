"""
Phase 2 unit tests: hand-computed metric cases, candidate-set plumbing, and
equality of the parallel wrapper with the unchanged Phase 1c distance.

Run from the repo root:  python -m pytest tests -q
"""

import itertools
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from cc_dedup.graph import pairwise_distance
from phase2 import candidates as cand
from phase2 import metrics as met
from phase2 import signatures as sig
from phase2.parallel_matrix import build_distance_matrix, load_rgb


def tiny_world():
    """2 families x (original + resize + crop) = 6 nodes.
    Nodes: 0=A orig, 1=A resize, 2=A crop, 3=B orig, 4=B resize, 5=B crop."""
    family = np.array([0, 0, 0, 1, 1, 1])
    transform = np.array(["original", "resize", "crop", "original", "resize", "crop"])
    return family, transform


def mask(n, pairs):
    m = np.zeros((n, n), dtype=bool)
    for i, j in pairs:
        m[min(i, j), max(i, j)] = True
    return m


def test_hand_computed_counts():
    family, transform = tiny_world()
    truth = met.truth_matrix(family)
    # 6 nodes -> 15 pairs; each family has 3 true pairs -> 6 true pairs.
    assert met.n_pairs(6) == 15
    assert int(truth.sum()) == 6

    # P = {(0,1) A-orig/resize [true OT], (1,2) A-resize/crop [true TT],
    #      (0,3) A-orig/B-orig [false], (4,5) B-resize/crop [true TT]}
    P = mask(6, [(0, 1), (1, 2), (0, 3), (4, 5)])
    s = met.score(P, truth, transform)
    assert s["candidate_pairs"] == 4
    assert s["true_pairs_found"] == 3
    assert s["recall"] == 3 / 6
    assert s["reduction"] == 1 - 4 / 15
    assert s["candidates_per_true_pair"] == 4 / 6
    # OT true pairs: (0,1),(0,2),(3,4),(3,5) -> found (0,1) -> 1/4
    assert s["recall_original_transform"] == 1 / 4
    # TT true pairs: (1,2),(4,5) -> both found -> 2/2
    assert s["recall_transform_transform"] == 1.0
    # resize true pairs: (0,1),(1,2),(3,4),(4,5) -> found (0,1),(1,2),(4,5) = 3/4
    assert s["recall_per_transform"]["resize"] == 3 / 4
    # crop true pairs: (0,2),(1,2),(3,5),(4,5) -> found (1,2),(4,5) = 2/4
    assert s["recall_per_transform"]["crop"] == 2 / 4
    # transforms absent from this world have no true pairs -> NaN, not a pass
    assert np.isnan(s["recall_per_transform"]["bright"])


def test_extreme_candidate_sets():
    family, transform = tiny_world()
    truth = met.truth_matrix(family)
    everything = np.triu(np.ones((6, 6), dtype=bool), 1)
    s = met.score(everything, truth, transform)
    assert s["recall"] == 1.0 and s["reduction"] == 0.0
    s = met.score(np.zeros((6, 6), dtype=bool), truth, transform)
    assert s["recall"] == 0.0 and s["reduction"] == 1.0


def test_flat_index_round_trip_and_hash_is_order_free():
    n = 37
    rows, cols = np.triu_indices(n, 1)
    flat = cand._row_starts(n)[rows] + (cols - rows - 1)
    assert np.array_equal(flat, np.arange(n * (n - 1) // 2))
    i, j = cand.flat_to_pair(flat, n)
    assert np.array_equal(i, rows) and np.array_equal(j, cols)
    m = mask(n, [(3, 9), (0, 36), (10, 11)])
    assert cand.candidate_hash(m) == cand.candidate_hash(m.copy())


def test_random_control_size_and_determinism():
    a = cand.random_control(50, 100)
    b = cand.random_control(50, 100)
    assert int(a.sum()) == 100 and np.array_equal(a, b)
    assert not np.tril(a).any()  # strictly upper triangular


def test_hamming_and_phash_shapes():
    h = np.array([0, 1, 3, 2**63], dtype=np.uint64)
    ham = sig.hamming_matrix(h)
    assert ham[0, 1] == 1 and ham[0, 2] == 2 and ham[0, 3] == 1 and ham[1, 3] == 2
    assert ham.dtype == np.uint8 and np.array_equal(ham, ham.T)


def test_signature_blocks_are_unit_norm_and_brightness_scaling_is_invisible_to_gray_block():
    rng = np.random.default_rng(0)
    gray = rng.uniform(10, 200, size=(32, 32))
    assert abs(np.linalg.norm(sig.block_gray(gray)) - 1) < 1e-12
    assert np.allclose(sig.block_gray(gray), sig.block_gray(gray * 1.15))


def test_parallel_wrapper_equals_phase1c_distance():
    rng = np.random.default_rng(1)
    shapes = [(40, 50), (40, 50), (30, 44), (40, 50)]  # includes a dimension mismatch
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for k, (h, w) in enumerate(shapes):
            p = Path(tmp) / f"{k}.png"
            Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)).save(p)
            paths.append(p)
        matrix = build_distance_matrix(paths, workers=2, chunk=2, log=lambda m: None)
        arrays = [load_rgb(p) for p in paths]
        for i, j in itertools.combinations(range(len(paths)), 2):
            expected = pairwise_distance(arrays[i], arrays[j])
            assert matrix[i, j] == expected and matrix[j, i] == expected
        assert np.all(np.diag(matrix) == 0)
