"""
Phase 2 — F=3 pilot (docs/phase2_design.md section 13).

Validation only: 21 images, 210 pairs, 63 true pairs. Its numbers are not
evidence and nothing here is committed (output goes to phase2/artifacts/pilot/,
which is gitignored). Exits non-zero if any item fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from cc_dedup.graph import build_graph  # Phase 1c, unchanged
from phase2 import candidates as cand
from phase2 import metrics as met
from phase2 import signatures as sig
from phase2.generate import DEFAULT_DATASET_DIR, generate_dataset, node_paths, sha256_file
from phase2.parallel_matrix import build_distance_matrix
from phase2.pipeline import node_labels, primary_state

PILOT_DIR = REPO_ROOT / "phase2" / "artifacts" / "pilot"
F = 3


def family_hashes(dataset_dir: Path, n_families: int) -> dict:
    return {p.name: sha256_file(p) for p in node_paths(n_families, dataset_dir)}


def run_once(paths):
    hashes, coarse = sig.compute_signatures(paths)
    state = primary_state(hashes, coarse)
    return hashes, coarse, state


def main():
    checks: dict[str, bool] = {}
    notes: dict[str, object] = {}

    # --- nestedness: the first 3 families of an F=5 run equal an F=3 run ---
    f3_dir, f5_dir = PILOT_DIR / "F3", PILOT_DIR / "F5"
    generate_dataset(3, f3_dir)
    generate_dataset(5, f5_dir)
    h3 = family_hashes(f3_dir, 3)
    h5_prefix = family_hashes(f5_dir, 3)
    checks["nested_F3_equals_prefix_of_F5"] = h3 == h5_prefix
    if (DEFAULT_DATASET_DIR / "manifest.json").exists():
        checks["nested_F3_equals_prefix_of_F800"] = h3 == family_hashes(DEFAULT_DATASET_DIR, 3)

    paths = node_paths(3, f3_dir)
    n = len(paths)
    checks["node_count_is_21"] = n == 21
    checks["pair_count_is_210"] = cand.n_pairs(n) == 210

    # --- signature shapes / dtypes ---
    hashes, coarse, state = run_once(paths)
    checks["hash_dtype_shape"] = hashes.dtype == np.uint64 and hashes.shape == (21,)
    checks["coarse_dtype_shape"] = coarse.dtype == np.float32 and coarse.shape == (21, sig.COARSE_DIM)
    checks["coarse_total_norm_sq_is_3"] = bool(np.allclose((coarse.astype(np.float64) ** 2).sum(1), 3.0, atol=1e-4))
    checks["distance_matrices_symmetric"] = bool(
        np.array_equal(state["dist"], state["dist"].T) and np.array_equal(state["ham"], state["ham"].T)
    )

    # --- determinism across two runs ---
    hashes2, coarse2, state2 = run_once(paths)
    checks["determinism_signatures"] = bool(np.array_equal(hashes, hashes2) and np.array_equal(coarse, coarse2))
    checks["determinism_tau_and_candidates"] = bool(
        state["tau_sig"] == state2["tau_sig"]
        and all(cand.candidate_hash(state[m]) == cand.candidate_hash(state2[m]) for m in ("M1", "M2", "M3"))
    )

    # --- Otsu on signature distances ---
    checks["otsu_tau_finite"] = bool(np.isfinite(state["tau_sig"]))
    checks["threshold_method_recorded"] = state["threshold_method"] in ("otsu", "range_midpoint_fallback")

    # --- hand-checked candidate / true / reduction arithmetic on the pilot ---
    family, transform = node_labels(paths)
    truth = met.truth_matrix(family)
    checks["true_pairs_is_63"] = int(truth.sum()) == 63
    s_true = met.score(truth.copy(), truth, transform)
    checks["P=T: recall 1, reduction 1-63/210"] = s_true["recall"] == 1.0 and s_true["reduction"] == 1 - 63 / 210
    fam1 = truth & (family[:, None] == 1) & (family[None, :] == 1)
    s_fam1 = met.score(fam1, truth, transform)
    checks["P=family1 (21 pairs): recall 1/3, reduction 0.9"] = (
        int(fam1.sum()) == 21 and s_fam1["recall"] == 1 / 3 and s_fam1["reduction"] == 1 - 21 / 210
    )
    s_none = met.score(np.zeros((n, n), dtype=bool), truth, transform)
    checks["P=empty: recall 0, reduction 1"] = s_none["recall"] == 0.0 and s_none["reduction"] == 1.0
    s_all = met.score(np.triu(np.ones((n, n), dtype=bool), 1), truth, transform)
    checks["P=all: recall 1, reduction 0"] = s_all["recall"] == 1.0 and s_all["reduction"] == 0.0
    k = int(state["M3"].sum())
    b1 = cand.random_control(n, k)
    checks["B1 has the same size as the method set"] = int(b1.sum()) == k

    # --- parallel wrapper identical to the unchanged Phase 1c pipeline ---
    r_parallel = build_distance_matrix(paths, workers=4, chunk=20, log=lambda m: None)
    r_phase1c = build_graph(paths).r_matrix
    checks["distances_identical_to_phase1c"] = bool(np.array_equal(r_parallel, r_phase1c))

    # --- pilot images equal the Phase 1c images when those are on disk ---
    phase0 = REPO_ROOT / "phase0" / "data"
    if phase0.exists():
        same = all(
            sha256_file(p) == sha256_file(phase0 / ("originals" if p.parent.name == "originals" else "transformed") / p.name)
            for p in paths
        )
        checks["pilot_images_byte_identical_to_phase0_data"] = same

    notes.update(
        tau_sig=state["tau_sig"],
        threshold_method=state["threshold_method"],
        candidate_pairs={m: int(state[m].sum()) for m in ("M1", "M2", "M3")},
    )

    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    (PILOT_DIR / "pilot_report.json").write_text(json.dumps({"checks": checks, "notes": notes}, indent=1))
    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    print("notes (not evidence):", notes)
    failed = [name for name, ok in checks.items() if not ok]
    print("PILOT:", "ALL CHECKS PASSED" if not failed else f"FAILED {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
