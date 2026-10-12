"""
Phase 2 — full deterministic run (docs/phase2_design.md sections 6-12, 17).

For each size F in {40,100,200,400,800}: signatures -> candidate sets (M1, M2, M3)
-> score against ground truth (only after the sets are fixed) -> gate. At n=280
it also performs the section-11 analysis from the persisted distance matrix.

Records NO timings and nothing machine-specific, so the output is byte-for-byte
reproducible. Timings live in experiments/run_phase2_timing.py.

Usage:
    python experiments/run_phase2.py                      # all sizes
    python experiments/run_phase2.py --only 40 --out X    # one size (audit rerun)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from cc_dedup.correlation_clustering import kwik_cluster  # Phase 1c, unchanged
from cc_dedup.disagreement import compute_disagreement  # Phase 1c, unchanged
from phase2 import candidates as cand
from phase2 import metrics as met
from phase2 import signatures as sig
from phase2.consistency import SEED, check_against_phase1c, full_graph_from_matrix
from phase2.generate import node_paths
from phase2.pipeline import node_labels, primary_state

SIZES = (40, 100, 200, 400, 800)
METHODS = ("M1", "M2", "M3")
ARTIFACTS = REPO_ROOT / "phase2" / "artifacts"
MATRIX = REPO_ROOT / "results" / "phase2_n280_distance_matrix.npy"
MATRIX_META = REPO_ROOT / "results" / "phase2_n280_distance_matrix.meta.json"
PHASE1C = REPO_ROOT / "results" / "phase1c_results.json"
DEFAULT_OUT = REPO_ROOT / "results" / "phase2_results.json"


def py(obj):
    """Convert numpy scalars/containers to plain JSON-serialisable Python."""
    if isinstance(obj, dict):
        return {str(k): py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_signatures():
    cache = ARTIFACTS / "signatures_F800.npz"
    paths = node_paths(800)
    names = [p.name for p in paths]
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        assert list(z["names"]) == names, "signature cache does not match the node order"
        return names, z["hashes"], z["coarse"]
    hashes = np.empty(len(paths), dtype=np.uint64)
    coarse = np.empty((len(paths), sig.COARSE_DIM), dtype=np.float32)
    for s in range(0, len(paths), 400):
        h, c = sig.compute_signatures(paths[s : s + 400])
        hashes[s : s + 400], coarse[s : s + 400] = h, c
        print(f"  signatures {min(s + 400, len(paths))}/{len(paths)}", flush=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.savez(cache, names=np.array(names), hashes=hashes, coarse=coarse)
    return names, hashes, coarse


def brief(s: dict, n_pairs_all: int) -> dict:
    return {
        "candidate_pairs": s["candidate_pairs"],
        "recall": s["recall"],
        "reduction": s["reduction"],
        "min_recall_per_transform": s["min_recall_per_transform"],
        "expected_random_recall": s["candidate_pairs"] / n_pairs_all,
    }


def evaluate_size(F, names, H, C, name_to_idx):
    paths = node_paths(F)
    idx = [name_to_idx[p.name] for p in paths]
    n = len(paths)
    state = primary_state(H[idx], C[idx])

    # Candidate sets are fixed here, before any label is read.
    sets = {m: state[m] for m in METHODS}
    set_hashes = {m: cand.candidate_hash(sets[m]) for m in METHODS}

    family, transform = node_labels(paths)
    truth = met.truth_matrix(family)
    all_pairs = cand.n_pairs(n)

    methods = {}
    for m in METHODS:
        s = met.score(sets[m], truth, transform)
        b1 = met.score(cand.random_control(n, s["candidate_pairs"]), truth, transform)
        methods[m] = {
            "primary": {
                **s,
                "gate": met.passes_gate(s),
                "candidate_set_sha256": set_hashes[m],
                "b1_recall": b1["recall"],
                "above_b1": bool(s["recall"] > b1["recall"]),
            }
        }
    methods["M1"]["curve"] = [
        {"R": r, **brief(met.score(cand.m1_candidates(state["ham"], r), truth, transform), all_pairs)}
        for r in cand.M1_SECONDARY_R
    ]
    methods["M2"]["curve"] = [
        {
            "tau_sig_factor": f,
            "tau_sig": state["tau_sig"] * f,
            **brief(met.score(cand.m2_candidates(state["dist"], state["tau_sig"] * f), truth, transform), all_pairs),
        }
        for f in cand.M2_SECONDARY_FACTORS
    ]
    entry = {
        "n": n,
        "all_pairs": all_pairs,
        "true_pairs": int(truth.sum()),
        "perfect_filter_reduction": 1 - int(truth.sum()) / all_pairs,
        "tau_sig": state["tau_sig"],
        "threshold_method": state["threshold_method"],
        "methods": methods,
    }
    return entry, sets, truth, paths


def section_11(sets, truth, paths):
    """Design doc 17.8. Needs the persisted n=280 distance matrix."""
    meta = json.loads(MATRIX_META.read_text())
    assert meta["nodes"] == [p.name for p in paths], "matrix node order differs from the pipeline's"
    r = np.load(MATRIX)
    tau = json.loads(PHASE1C.read_text())["threshold"]  # fixed Phase 1c tau
    consistency = check_against_phase1c(r)
    sign, weight = full_graph_from_matrix(r, tau)
    V = np.triu(sign == 1, 1)

    out = {
        "fixed_tau": tau,
        "consistency_with_phase1c": consistency,
        "images_byte_identical_to_phase0_data": meta["images_byte_identical_to_phase0_data"],
        "positive_edges_V": int(V.sum()),
        "verifier_cap": int((truth & V).sum()) / int(truth.sum()),
        "methods": {},
    }
    for m, P in sets.items():
        cand_sign = np.where((P | P.T) & (sign == 1), 1, -1).astype(np.int8)
        np.fill_diagonal(cand_sign, 0)
        clusters = kwik_cluster(cand_sign, seed=SEED)
        out["methods"][m] = {
            "e_plus_retention": int((P & V).sum()) / int(V.sum()),
            "verifier_confirmed_recall": int((P & truth & V).sum()) / int(truth.sum()),
            "retention_of_verifiable_true_pairs": int((P & truth & V).sum()) / int((truth & V).sum()),
            "kwikcluster_clusters_on_candidate_graph": len(set(clusters)),
            "kwikcluster_D_on_full_graph": compute_disagreement(clusters, sign, weight),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache-only", action="store_true", help="only compute and cache signatures")
    args = ap.parse_args()
    sizes = (args.only,) if args.only else SIZES

    names, H, C = load_signatures()
    if args.cache_only:
        print(f"cached {len(names)} signatures: hashes {H.dtype}{H.shape}, coarse {C.dtype}{C.shape}")
        return
    name_to_idx = {nm: i for i, nm in enumerate(names)}

    result = {"design": "docs/phase2_design.md", "seed": SEED, "sizes": {}}
    for F in sizes:
        print(f"F={F}", flush=True)
        entry, sets, truth, paths = evaluate_size(F, names, H, C, name_to_idx)
        if F == 40:
            if MATRIX.exists() and MATRIX_META.exists():
                entry["section_11_n280"] = section_11(sets, truth, paths)
            else:
                entry["section_11_n280"] = "MISSING: run experiments/run_phase2_n280_matrix.py first"
        result["sizes"][str(F)] = entry
        for m in METHODS:
            p = entry["methods"][m]["primary"]
            print(f"  {m}: recall={p['recall']:.4f} reduction={p['reduction']:.4f} "
                  f"min_type_recall={p['min_recall_per_transform']:.4f} "
                  f"|P|={p['candidate_pairs']} gate_pass={p['gate']['pass']}", flush=True)

    if not args.only:
        gate = {}
        for m in METHODS:
            per_size = {str(F): result["sizes"][str(F)]["methods"][m]["primary"]["gate"] for F in SIZES}
            gate[m] = {"per_size": per_size, "passes_every_size": all(g["pass"] for g in per_size.values())}
        result["gate"] = gate
        result["phase2_succeeds"] = any(g["passes_every_size"] for g in gate.values())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(py(result), indent=1))
    print(f"written {args.out}")
    if not args.only:
        print("PHASE 2 GATE:", {m: result["gate"][m]["passes_every_size"] for m in METHODS},
              "-> succeeds:", result["phase2_succeeds"])


if __name__ == "__main__":
    main()
