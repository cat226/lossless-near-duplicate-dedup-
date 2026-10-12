"""
Phase 2b — fixed-threshold M2* on held-out families 801-1600 (docs/phase2b_design.md).

For each size F in {40,100,200,400,800}: signatures of the first F fresh families ->
candidate set {L2 distance < TAU_STAR} -> hash it -> only then read labels and score.
Records NO timings and nothing machine-specific, so the output is byte-for-byte
reproducible. Timings live in experiments/run_phase2b_timing.py.

    python experiments/run_phase2b.py                      # all sizes
    python experiments/run_phase2b.py --only 40 --out X    # one size (audit rerun)
    python experiments/run_phase2b.py --cache-only         # compute signatures only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from phase2 import candidates as cand
from phase2 import metrics as met
from phase2 import signatures as sig
from phase2.pipeline import node_labels
from phase2b.data import DEFAULT_DATASET_DIR, FIRST_FRESH, SIZES, TAU_STAR, node_paths

SEED = 12345
ARTIFACTS = REPO_ROOT / "phase2b" / "artifacts"
DEFAULT_OUT = REPO_ROOT / "results" / "phase2b_results.json"
PHASE2_RESULTS = REPO_ROOT / "results" / "phase2_results.json"


def py(obj):
    if isinstance(obj, dict):
        return {str(k): py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [py(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_signatures(dataset_dir: Path = DEFAULT_DATASET_DIR, cache_name: str = "signatures_fresh800.npz"):
    paths = node_paths(max(SIZES), dataset_dir)
    names = [p.name for p in paths]
    cache = ARTIFACTS / cache_name
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


def evaluate_size(F, names, C, name_to_idx, dataset_dir: Path = DEFAULT_DATASET_DIR):
    paths = node_paths(F, dataset_dir)
    idx = [name_to_idx[p.name] for p in paths]
    n = len(paths)
    dist = sig.l2_matrix(C[idx])

    # The candidate set is fixed and hashed here, before any label is read.
    P = cand.m2_candidates(dist, TAU_STAR)
    set_hash = cand.candidate_hash(P)

    family, transform = node_labels(paths)
    truth = met.truth_matrix(family)
    s = met.score(P, truth, transform)
    b1 = met.score(cand.random_control(n, s["candidate_pairs"]), truth, transform)
    all_pairs = cand.n_pairs(n)
    return {
        "n": n,
        "families": [FIRST_FRESH, FIRST_FRESH + F - 1],
        "all_pairs": all_pairs,
        "true_pairs": int(truth.sum()),
        "perfect_filter_reduction": 1 - int(truth.sum()) / all_pairs,
        "tau_star": TAU_STAR,
        "M2_star": {
            **s,
            "gate": met.passes_gate(s),
            "candidate_set_sha256": set_hash,
            "retained_fraction": s["candidate_pairs"] / all_pairs,
            "b1_recall": b1["recall"],
            "above_b1": bool(s["recall"] > b1["recall"]),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache-only", action="store_true")
    args = ap.parse_args()
    sizes = (args.only,) if args.only else SIZES

    names, H, C = load_signatures()
    if args.cache_only:
        print(f"cached {len(names)} signatures: hashes {H.dtype}{H.shape}, coarse {C.dtype}{C.shape}")
        return
    name_to_idx = {nm: i for i, nm in enumerate(names)}

    result = {"design": "docs/phase2b_design.md", "seed": SEED, "tau_star": TAU_STAR,
              "tau_star_source": "0.5 x phase2 sizes['40'].tau_sig", "sizes": {}}
    for F in sizes:
        print(f"F={F}", flush=True)
        entry = evaluate_size(F, names, C, name_to_idx)
        result["sizes"][str(F)] = entry
        p = entry["M2_star"]
        print(f"  M2*: recall={p['recall']:.4f} reduction={p['reduction']:.4f} "
              f"min_type_recall={p['min_recall_per_transform']:.4f} |P|={p['candidate_pairs']} "
              f"gate_pass={p['gate']['pass']}", flush=True)

    if not args.only:
        per_size = {str(F): result["sizes"][str(F)]["M2_star"]["gate"] for F in SIZES}
        result["gate"] = {"per_size": per_size, "passes_every_size": all(g["pass"] for g in per_size.values())}
        result["hypothesis_supported"] = result["gate"]["passes_every_size"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(py(result), indent=1))
    print(f"written {args.out}")
    if not args.only:
        print("PHASE 2b HYPOTHESIS SUPPORTED:", result["hypothesis_supported"])


if __name__ == "__main__":
    main()
