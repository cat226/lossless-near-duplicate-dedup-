"""
Phase 3 unit tests: block-wise equivalence with Phase 2, hash definitions, split verification and its
tamper behaviour, the test-split lock, the label guard and scoring arithmetic.

Run from the repo root:  python -m pytest tests -q
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cc_dedup.graph import compute_otsu_threshold
from phase2 import candidates as cand
from phase2 import signatures as sig
from phase3 import blockwise as bw
from phase3 import evaluate as ev
from phase3 import hashes as hs
from phase3 import labels as lb
from phase3 import splits as sp


def random_sigs(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 1056)).astype(np.float32)
    for lo, hi in ((0, 1024), (1024, 1048), (1048, 1056)):
        x[:, lo:hi] /= np.linalg.norm(x[:, lo:hi], axis=1, keepdims=True)
    return x


def test_otsu_from_histogram_equals_phase1c_otsu():
    rng = np.random.default_rng(1)
    v = np.concatenate([rng.normal(0.4, 0.05, 500), rng.normal(1.2, 0.2, 5000)])
    hist, edges = np.histogram(v, bins=256)
    assert bw.otsu_from_histogram(hist, edges, v.min(), v.max()) == compute_otsu_threshold(v)
    flat = np.full(100, 0.7)
    h2, e2 = np.histogram(flat, bins=256)
    assert bw.otsu_from_histogram(h2, e2, 0.7, 0.7) == compute_otsu_threshold(flat) == (0.7, "range_midpoint_fallback")


def test_blockwise_candidates_equal_phase2_l2_matrix_path():
    x = random_sigs(300)
    dist = sig.l2_matrix(x)
    tau = float(np.quantile(cand.upper_values(dist), 0.05))
    expected = np.sort(cand.pair_flat_indices(cand.m2_candidates(dist, tau)))
    assert np.array_equal(bw.m2_candidate_flats(x, tau, block=64), expected)
    assert np.array_equal(bw.m2_candidate_flats(x, tau, block=300), expected)


def test_blockwise_otsu_matches_full_array_otsu():
    x = random_sigs(400, seed=3)
    full = compute_otsu_threshold(cand.upper_values(sig.l2_matrix(x)))
    blk = bw.otsu_blockwise(x, block=96)
    assert blk[1] == full[1] and abs(blk[0] - full[0]) < 1e-9


def test_pair_distances_match_row_blocks():
    x = random_sigs(120, seed=5)
    a, sq = bw.prepare(x)
    full = bw.l2_rows(a, sq, 0, 120)
    rows, cols = np.array([0, 3, 50, 7]), np.array([5, 99, 51, 8])
    assert np.array_equal(bw.pair_distances(x, rows, cols, block=32), full[rows, cols])


def test_dhash_definition_hand_checked():
    dec = Image.fromarray(np.tile(np.linspace(255, 0, 90).astype(np.uint8), (80, 1)))     # decreasing left to right
    inc = Image.fromarray(np.tile(np.linspace(0, 255, 90).astype(np.uint8), (80, 1)))     # increasing
    assert hs.dhash64(dec) == np.uint64(2**64 - 1)   # left > right everywhere: all 64 bits set
    assert hs.dhash64(inc) == np.uint64(0)


def test_exact_candidate_flats_hand_checked():
    n = 5
    starts = cand._row_starts(n)
    flats = hs.exact_candidate_flats(["a", "b", "a", "c", "a"], starts)
    rows, cols = cand.flat_to_pair(flats, n)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 2), (0, 4), (2, 4)]


def test_flat_hash_equals_phase2_candidate_hash():
    n = 40
    P = np.triu(np.random.default_rng(2).random((n, n)) < 0.1, 1)
    assert ev.flat_hash(ev.flats_of(P)) == cand.candidate_hash(P)
    assert np.array_equal(ev.matrix_of(ev.flats_of(P), n), P)


def test_startup_verification_passes_and_checks_every_hash():
    s = sp.verify_startup()
    assert all(d["hashes_checked"] == 13 and d["disjoint"] for d in s.report["draws"].values())
    assert all(s.report["manifest_hashes_in_design"].values())
    assert s.dev(1) != s.dev(2) and not set(s.dev(1)) & set(s.dev_singletons(2)[:0])


def test_startup_verification_stops_on_a_tampered_hash(tmp_path, monkeypatch):
    doc = sp.DESIGN.read_text(encoding="utf-8")
    i = doc.index("draw 2 (seed 12346)")
    j = doc.index("  dev             perm[0:1500]       ", i) + len("  dev             perm[0:1500]       ")
    bad = doc[:j] + ("0" if doc[j] != "0" else "1") + doc[j + 1:]
    fake = tmp_path / "design.md"
    fake.write_text(bad, encoding="utf-8")
    monkeypatch.setattr(sp, "DESIGN", fake)
    with pytest.raises(sp.SplitMismatch):
        sp.verify_startup()


def test_test_split_is_locked_without_approval():
    assert not sp.TEST_APPROVAL_FILE.exists()
    s = sp.verify_startup()
    for fn in (s.test_sources, s.test_singletons, s.test_photos):
        with pytest.raises(PermissionError):
            fn(1)


def test_labels_cannot_be_opened_before_candidate_hashes(tmp_path):
    f = tmp_path / "labels.json"
    f.write_text(json.dumps({"images": {}}))
    with pytest.raises(lb.LabelAccessBeforeCandidates):
        lb.open_labels(f, {})
    assert lb.open_labels(f, {"M1_R10": "x"}) == {}


def test_tau_b_calibration_refuses_a_non_dev_directory(tmp_path):
    d = tmp_path / "draw1_test"
    d.mkdir()
    with pytest.raises(PermissionError):
        ev.calibrate_tau_b([], d / "labels.json")


def test_score_sets_hand_computed_arithmetic(tmp_path):
    # 2 families of (original, resize, crop) + 2 singletons = 8 nodes; ids i00000..i00007
    names = [f"i{k:05d}.png" for k in range(8)]
    paths = [Path(n) for n in names]
    lab = {}
    for k, (fam, tr) in enumerate([("A", "original"), ("A", "resize"), ("A", "crop"), ("B", "original"),
                                   ("B", "resize"), ("B", "crop"), (None, "singleton"), (None, "singleton")]):
        lab[Path(names[k]).stem] = {"source": f"s{k}", "family": fam, "transform": tr}
    lf = tmp_path / "labels.json"
    lf.write_text(json.dumps({"images": lab}))
    n = 8
    starts = cand._row_starts(n)
    flat = lambda i, j: int(starts[i] + (j - i - 1))
    # P = {(0,1) true OT, (1,2) true TT, (0,3) false, (4,5) true OT-ish (B resize/crop = TT)}
    sets = {"X": np.sort(np.array([flat(0, 1), flat(1, 2), flat(0, 3), flat(4, 5)], dtype=np.int64))}
    stage = {"n": n, "sets": sets, "hashes": {"X": ev.flat_hash(sets["X"])}, "tau": {}}
    out = ev.score_sets(stage, lf, paths, draw_seed=12345)
    s = out["sets"]["X"]
    assert out["true_pairs"] == 6                      # 3 per family
    assert s["candidate_pairs"] == 4 and s["recall"] == 3 / 6
    assert s["reduction"] == 1 - 4 / 28
    assert s["recall_original_transform"] == 1 / 4 and s["recall_transform_transform"] == 1.0
    assert s["recall_per_transform"]["resize"] == 3 / 4 and s["recall_per_transform"]["crop"] == 2 / 4
    assert s["membership_recall"] == s["recall"]
