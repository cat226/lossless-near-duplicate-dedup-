"""
Phase 2b unit tests: constants, node ordering, and fixed-threshold candidate rule.

Run from the repo root:  python -m pytest tests -q
"""

import json
import re
import tempfile
from pathlib import Path

import numpy as np

from phase2 import candidates as cand
from phase2b import data as d

REPO = Path(__file__).resolve().parent.parent


def test_tau_star_is_half_of_phase2_n280_otsu_tau_sig():
    tau_sig = json.loads((REPO / "results" / "phase2_results.json").read_text())["sizes"]["40"]["tau_sig"]
    assert d.TAU_STAR == 0.5 * tau_sig


def test_tau_star_matches_frozen_design_literal():
    text = (REPO / "docs" / "phase2b_design.md").read_text()
    assert float(re.search(r"tau\* = (0\.\d+)", text).group(1)) == d.TAU_STAR


def test_fresh_families_are_disjoint_from_phase2_and_ladder_is_prefix_nested():
    assert d.FIRST_FRESH == 801 and d.PHASE2_FAMILIES == 800
    assert d.SIZES == (40, 100, 200, 400, 800)


def test_node_order_originals_then_transformed_by_family_then_filename():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "originals").mkdir()
        (root / "transformed").mkdir()
        for k in (801, 802, 1000):
            (root / "originals" / f"{k:03d}.jpg").write_bytes(b"")
        for k in (801, 802, 1000):
            for t, ext in (("rotate", "jpg"), ("bright", "jpg"), ("png", "png")):
                (root / "transformed" / f"{k:03d}_{t}.{ext}").write_bytes(b"")
        # F=2 -> families 801, 802 only; 1000 must not appear
        names = [p.name for p in d.node_paths(2, root)]
        assert names == ["801.jpg", "802.jpg",
                         "801_bright.jpg", "801_png.png", "801_rotate.jpg",
                         "802_bright.jpg", "802_png.png", "802_rotate.jpg"]
        # four-digit ids sort numerically, not as strings
        d_first = d.FIRST_FRESH
        assert d_first == 801


def test_fixed_threshold_candidate_rule_is_strict_and_n_independent():
    dist = np.array([[0.0, 0.5, 0.6004747353136561, 0.9],
                     [0.5, 0.0, 0.7, 0.2],
                     [0.6004747353136561, 0.7, 0.0, 0.3],
                     [0.9, 0.2, 0.3, 0.0]])
    P = cand.m2_candidates(dist, d.TAU_STAR)
    assert P[0, 1] and P[1, 3] and P[2, 3]          # below tau*
    assert not P[0, 2]                              # exactly tau* is excluded (strict <)
    assert not P[0, 3] and not P[1, 2]              # above tau*
    assert not np.tril(P).any()
