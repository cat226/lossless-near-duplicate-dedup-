"""
Phase 3 run-machinery tests: per-image features, the ladder builder, verifier metrics, the parallel
primary verifier, aggregation and the gates that keep the test split locked.

Run from the repo root:  python -m pytest tests -q
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from phase2 import candidates as cand
from phase2 import signatures as sig
from phase3 import dataset as ds
from phase3 import evaluate as ev
from phase3 import features as ft
from phase3 import hashes as hs
from phase3 import parallel as par
from phase3 import runner as rn
from phase3 import splits as sp
from phase3 import verifier_metrics as vm


def _make_images(folder, n=10):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    paths = []
    for k in range(n):
        base = (rng.random((40 + 3 * k, 56, 3)) * 255).astype(np.uint8)
        p = folder / f"i{k:05d}.png"
        Image.fromarray(base).save(p)
        paths.append(p)
    dup = folder / f"i{n:05d}.png"
    shutil.copyfile(paths[0], dup)  # a byte-identical duplicate of image 0
    paths.append(dup)
    return paths


def test_features_parallel_equals_single_equals_candidate_stage(tmp_path):
    paths = _make_images(tmp_path / "imgs")
    one = ft.extract(paths, workers=1)
    two = ft.extract(paths, workers=2, chunk=3)
    for k in one:
        assert np.array_equal(one[k], two[k])
    p_h, coarse = sig.compute_signatures(paths)  # the unmodified Phase 2 path
    assert np.array_equal(one["phash"], p_h) and np.array_equal(one["coarse"], coarse)
    assert np.array_equal(one["dhash"], hs.compute_dhashes(paths))
    a = ev.candidate_stage(paths, 0.5)
    b = ev.candidate_stage(paths, 0.5, features=one)
    assert a["hashes"] == b["hashes"]
    rows, cols = cand.flat_to_pair(b["sets"]["EXACT"], len(paths))
    assert list(zip(rows.tolist(), cols.tolist())) == [(0, len(paths) - 1)]


def test_ladder_membership_nests_and_singletons_keep_pool_bytes(tmp_path):
    split = sp.verify_startup()
    src, single = split.dev_sources(1)[:3], split.dev_singletons(1)[:10]
    mem = {7: (1, 3), 21: (3, 10)}
    out = ds.build_d3a_set(tmp_path / "ladder", src, single, 12345, membership=mem)
    m = json.loads((tmp_path / "ladder" / "membership.json").read_text())
    assert len(m["7"]) == 7 + 3 and len(m["21"]) == 21 + 10 and set(m["7"]) < set(m["21"])
    assert out["n"] == 31
    labs = json.loads(out["labels_path"].read_text())["images"]
    stems7 = {Path(n).stem for n in m["7"]}
    assert {labs[s]["source"] for s in stems7 if labs[s]["family"]} == {src[0]}
    assert sum(1 for s in stems7 if labs[s]["family"] is None) == 3
    one = next(p for p in out["paths"] if labs[p.stem]["family"] is None)
    assert hs.file_sha256(one) == hs.file_sha256(ds.POOL_DIR / labs[one.stem]["source"])
    again = ds.build_d3a_set(tmp_path / "ladder", src, single, 12345, membership=mem)  # reuse path
    assert again["paths"] == out["paths"]


def test_auc_hand_checked_and_equals_brute_force():
    assert vm.auc_smaller_is_positive(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0
    assert vm.auc_smaller_is_positive(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 0.0
    assert vm.auc_smaller_is_positive(np.array([1.0, 1.0]), np.array([1.0])) == 0.5
    rng = np.random.default_rng(3)
    pos, neg = rng.integers(0, 6, 40).astype(float), rng.integers(0, 6, 55).astype(float)
    brute = np.mean([(p < q) + 0.5 * (p == q) for p in pos for q in neg])
    assert abs(vm.auc_smaller_is_positive(pos, neg) - brute) < 1e-12
    assert vm.auc_smaller_is_positive(np.array([np.inf]), np.array([np.inf, 2.0])) == 0.25  # a failure ties with a failure


def test_true_flats_in_size_and_false_accept_sample():
    L = 12
    nodes = np.array([0, 1, 2, 3, 4, 6, 7, 9])  # a nested subset keeps node order
    tp = np.array([[0, 1], [1, 2], [5, 8], [6, 9], [3, 7]])
    flat, pos = vm.true_flats_in_size(tp, nodes, L, len(nodes))
    assert pos.tolist() == [0, 1, 3, 4]  # (5, 8) falls outside the subset
    rows, cols = cand.flat_to_pair(flat, len(nodes))
    assert [(int(nodes[r]), int(nodes[c])) for r, c in zip(rows, cols)] == [(0, 1), (1, 2), (6, 9), (3, 7)]
    n = len(nodes)
    allc = np.arange(cand.n_pairs(n), dtype=np.int64)
    s1 = vm.fa_sample_pairs(allc, flat, n, k=10)
    assert s1 == vm.fa_sample_pairs(allc, flat, n, k=10) and len(s1) == 10
    starts = cand._row_starts(n)
    assert not {int(starts[i] + (j - i - 1)) for i, j in s1} & set(flat.tolist())


def test_set_metrics_hand_computed():
    true_flat = np.array([0, 1, 2, 3])
    r = np.array([0.1, 0.9, 0.2, 0.3])  # tau 0.5: verifier passes 3 of 4
    m = vm.primary_set_metrics(true_flat, r, np.array([0, 1, 5]), 0.5)
    assert m["verifier_cap"] == 0.75 and m["confirmed_recall"] == 0.25  # only pair 0 is a candidate and passes
    ok = np.array([True, True, False, True])
    dist = np.array([0.1, 0.9, np.nan, 0.3])
    a = vm.aligned_set_metrics(true_flat, ok, dist, np.array([0, 2, 3]), 0.5)
    assert a["alignment_failure_rate"] == 0.25 and a["aligned_cap"] == 0.5 and a["confirmed_recall"] == 0.5
    fa = vm.false_accept(np.array([0.1, 0.7, 0.2, 0.9]), 0.5)
    assert fa["false_accept_rate"] == 0.5


def test_primary_pairs_parallel_equals_single_and_batches(tmp_path):
    paths = _make_images(tmp_path / "imgs", n=7)
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)][:20]
    single = par.primary_single(paths, pairs)
    for max_images in (800, 3):  # one batch, then many
        assert np.array_equal(single, par.primary_pairs_parallel(paths, pairs, 2, chunk=4, max_images=max_images))
    assert par.assert_primary_exact(paths, pairs, single, k=10)["bitwise_equal"]
    with pytest.raises(AssertionError):
        par.assert_primary_exact(paths, pairs, single + 1e-12, k=10)


def test_true_pairs_from_groups_by_family():
    fam = np.array([0, -2, 1, 0, 1, -6, 0])
    assert rn.true_pairs_from(fam).tolist() == [[0, 3], [0, 6], [3, 6], [2, 4]]


def _cell(passes, above=True, recall=0.99):
    return {"gate": {"pass": passes}, "above_b1": above, "recall": recall, "reduction": 0.99,
            "min_recall_per_transform": 0.9, "candidate_pairs": 10}


def test_aggregate_strict_gate_is_every_cell():
    parts = {}
    for d in (1, 2, 3):
        parts[f"ladder_d{d}"] = {"sizes": {str(n): {"score": {"sets": {name: _cell(not (name == "M1_R10" and d == 2 and n == 20))
                                                                         for name in ev.PRIMARY.values()}}} for n in (10, 20)}}
        parts[f"variant_d{d}"] = {"score": {"sets": {name: _cell(name != "dHash_R10") for name in ev.PRIMARY.values()}}}
        parts[f"d3b_d{d}"] = {"score": {"sets": {name: _cell(True, recall=0.94 if name == "EXACT" else 0.96) for name in ev.PRIMARY.values()}}}
    out = rn.aggregate(parts, [10, 20], 99)
    m = out["methods"]
    assert m["M1"]["strict_pass_ladder"] is False and m["M2_B"]["strict_pass_ladder"] is True
    assert m["dHash"]["strict_pass_ladder"] is True and m["dHash"]["strict_pass_ladder_and_variant"] is False
    assert m["EXACT"]["d3b"]["strict_pass"] is False and m["M3"]["d3b"]["strict_pass"] is True
    assert out["any_gated_method_passes_strict_ladder"] is True
    rec = m["M2_B"]["by_size"]["10"]["recall"]
    assert abs(rec["mean"] - 0.99) < 1e-12 and rec["min"] == rec["max"] == 0.99


def test_plan_uses_dev_in_rehearsal_and_test_is_locked_in_full(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "TEST_APPROVAL_FILE", tmp_path / "no_approval_here")  # independent of the local approval file
    s = sp.verify_startup()
    reh = rn.Plan("rehearsal", s)
    assert reh.sources(1) == s.dev_sources(1) and reh.singletons(1) == s.dev_singletons(1)
    assert not set(reh.d3b_singletons(1)) & set(reh.sources(1))
    full = rn.Plan("full", s)
    for fn in (full.sources, full.singletons, full.variant, full.d3b_singletons, full.ext):
        with pytest.raises(PermissionError):
            fn(1)
    assert max(F for F, _ in rn.REHEARSAL["ladder"].values()) <= 150
    assert max(S for _, S in rn.REHEARSAL["ladder"].values()) <= 1350
    assert rn.FULL["ladder"] == {1000: (50, 650), 2500: (125, 1625), 5000: (250, 3250), 10000: (500, 6500)}
    assert rn.FULL["variant"] == (8000, 114, 7202) and rn.FULL["ext"] == (5000, 250, 3250)


def test_disk_guard_stops_the_run(monkeypatch):
    ctx = rn.Ctx("rehearsal", sp.verify_startup(), tag="_unit")
    try:
        class U:
            free = 1_000_000_000

        monkeypatch.setattr(rn.shutil, "disk_usage", lambda p: U)
        with pytest.raises(rn.DiskGuard):
            ctx.guard()
    finally:
        monkeypatch.undo()
        shutil.rmtree(ctx.art, ignore_errors=True)


def test_full_mode_refuses_without_go_and_leaves_the_approval_file_alone():
    before = (sp.TEST_APPROVAL_FILE.exists(), sp.TEST_APPROVAL_FILE.stat().st_mtime_ns if sp.TEST_APPROVAL_FILE.exists() else None)
    r = subprocess.run([sys.executable, "experiments/run_phase3.py", "--mode", "full"], cwd=sp.REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "--go is required" in (r.stderr + r.stdout)
    after = (sp.TEST_APPROVAL_FILE.exists(), sp.TEST_APPROVAL_FILE.stat().st_mtime_ns if sp.TEST_APPROVAL_FILE.exists() else None)
    assert before == after  # neither created nor touched
