"""
Phase 2b — F=3 pilot (docs/phase2b_design.md section 9).

Validation only: fresh families 801-803 (21 images, 210 pairs, 63 true pairs). Needs
the two small pilot datasets produced by `python phase2b/data.py 3|5 <dir>`. Its numbers
are not evidence and nothing here is committed. Exits non-zero if any item fails.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from phase2 import candidates as cand
from phase2 import metrics as met
from phase2 import signatures as sig
from phase2.generate import sha256_file
from phase2.pipeline import node_labels
from phase2b.data import DEFAULT_DATASET_DIR, FIRST_FRESH, PHASE2_FAMILIES, TAU_STAR, node_paths

PILOT = REPO_ROOT / "phase2b" / "artifacts" / "pilot"
N3, N5 = PILOT / "N3", PILOT / "N5"


def run_once(paths):
    hashes, coarse = sig.compute_signatures(paths)
    dist = sig.l2_matrix(coarse)
    return hashes, coarse, dist, cand.m2_candidates(dist, TAU_STAR)


def main():
    checks: dict[str, bool] = {}
    notes: dict[str, object] = {}

    m3 = json.loads((N3 / "manifest.json").read_text())
    m5 = json.loads((N5 / "manifest.json").read_text())
    checks["fresh families start at 801 (disjoint from Phase 2's 1-800)"] = (
        m3["first_family"] == FIRST_FRESH == PHASE2_FAMILIES + 1 and m5["first_family"] == FIRST_FRESH
    )
    checks["regenerated Phase 2 families 1-800 matched the Phase 2 manifest (N3 and N5 runs)"] = (
        m3["phase2_families_verified_against_manifest"] == PHASE2_FAMILIES
        and m5["phase2_families_verified_against_manifest"] == PHASE2_FAMILIES
    )

    paths = node_paths(3, N3)
    n = len(paths)
    checks["node_count_is_21"] = n == 21
    checks["pair_count_is_210"] = cand.n_pairs(n) == 210
    checks["node order: 3 originals first, then transformed by (family, filename)"] = (
        [p.name for p in paths[:3]] == ["801.jpg", "802.jpg", "803.jpg"]
        and [p.name for p in paths[3:9]] == [f"801_{t}" + (".png" if t == "png" else ".jpg")
                                              for t in ("bright", "crop", "jpegq50", "png", "resize", "rotate")]
    )

    # nestedness: the 3-family run is the first 3 families of the 5-family run (byte for byte)
    h3 = {p.name: sha256_file(p) for p in paths}
    h5 = {p.name: sha256_file(p) for p in node_paths(3, N5)}
    checks["nested: N3 equals the first 3 families of N5"] = h3 == h5
    if (DEFAULT_DATASET_DIR / "manifest.json").exists():
        full = json.loads((DEFAULT_DATASET_DIR / "manifest.json").read_text())["sha256"]
        checks["nested: N3 equals the first 3 families of the full fresh800 dataset"] = all(
            full[f"{p.parent.name}/{p.name}"] == h3[p.name] for p in paths
        )

    hashes, coarse, dist, P = run_once(paths)
    checks["hash dtype/shape"] = hashes.dtype == np.uint64 and hashes.shape == (21,)
    checks["coarse dtype/shape"] = coarse.dtype == np.float32 and coarse.shape == (21, sig.COARSE_DIM)
    checks["coarse total norm^2 is 3"] = bool(np.allclose((coarse.astype(np.float64) ** 2).sum(1), 3.0, atol=1e-4))
    checks["distance matrix symmetric, zero diagonal"] = bool(
        np.array_equal(dist, dist.T) and np.all(np.diag(dist) == 0)
    )

    hashes2, coarse2, dist2, P2 = run_once(paths)
    checks["determinism: signatures identical across two runs"] = bool(
        np.array_equal(hashes, hashes2) and np.array_equal(coarse, coarse2)
    )
    checks["determinism: candidate sets identical across two runs"] = cand.candidate_hash(P) == cand.candidate_hash(P2)

    # the signature code path reproduces Phase 2's cached signatures exactly
    z = np.load(REPO_ROOT / "phase2" / "artifacts" / "signatures_F800.npz", allow_pickle=False)
    names2 = list(z["names"])
    ok = True
    for name in ("001.jpg", "100_rotate.jpg", "800_png.png", "400_crop.jpg"):
        i = names2.index(name)
        sub = "originals" if "_" not in name else "transformed"
        h, c = sig.signature_of(REPO_ROOT / "phase2" / "data" / "F800" / sub / name)
        ok &= bool(h == z["hashes"][i] and np.array_equal(c, z["coarse"][i]))
    checks["signature code reproduces Phase 2's cached signatures (4 images)"] = ok

    # tau* provenance
    tau_sig_280 = json.loads((REPO_ROOT / "results" / "phase2_results.json").read_text())["sizes"]["40"]["tau_sig"]
    checks["TAU_STAR == 0.5 x Phase 2 n=280 tau_sig (exact)"] = TAU_STAR == 0.5 * tau_sig_280
    design = (REPO_ROOT / "docs" / "phase2b_design.md").read_text()
    checks["TAU_STAR equals the literal in docs/phase2b_design.md"] = (
        re.search(r"tau\* = (0\.\d+)", design) is not None
        and float(re.search(r"tau\* = (0\.\d+)", design).group(1)) == TAU_STAR
    )

    # hand-checked arithmetic on the pilot
    family, transform = node_labels(paths)
    truth = met.truth_matrix(family)
    checks["true_pairs_is_63"] = int(truth.sum()) == 63
    s_true = met.score(truth.copy(), truth, transform)
    checks["P=T: recall 1, reduction 1-63/210"] = s_true["recall"] == 1.0 and s_true["reduction"] == 1 - 63 / 210
    fam1 = truth & (family[:, None] == FIRST_FRESH) & (family[None, :] == FIRST_FRESH)
    s_fam1 = met.score(fam1, truth, transform)
    checks["P=family 801 (21 pairs): recall 1/3, reduction 0.9"] = (
        int(fam1.sum()) == 21 and s_fam1["recall"] == 1 / 3 and s_fam1["reduction"] == 1 - 21 / 210
    )
    s_none = met.score(np.zeros((n, n), dtype=bool), truth, transform)
    checks["P=empty: recall 0, reduction 1"] = s_none["recall"] == 0.0 and s_none["reduction"] == 1.0
    s_all = met.score(np.triu(np.ones((n, n), dtype=bool), 1), truth, transform)
    checks["P=all: recall 1, reduction 0"] = s_all["recall"] == 1.0 and s_all["reduction"] == 0.0
    checks["B1 has the same size as the method set"] = int(cand.random_control(n, int(P.sum())).sum()) == int(P.sum())

    notes.update(candidate_pairs=int(P.sum()), tau_star=TAU_STAR)
    PILOT.mkdir(parents=True, exist_ok=True)
    (PILOT / "pilot_report.json").write_text(json.dumps({"checks": checks, "notes": notes}, indent=1))
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    print("notes (not evidence):", notes)
    failed = [k for k, v in checks.items() if not v]
    print("PILOT:", "ALL CHECKS PASSED" if not failed else f"FAILED {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
