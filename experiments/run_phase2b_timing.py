"""
Phase 2b — single-process timings (docs/phase2b_design.md section 7).

Run only after every heavy job has finished. Single process, BLAS threads pinned to 1.
For each size: the cheap-stage cost (all-pairs signature distances + candidate rule) and a
PROJECTION of verification time (|P| x mean of a fixed 200-pair sample of candidate pairs,
seed 12345, pairwise_distance only). The all-pairs baseline above n=280 is EXTRAPOLATED
at 0.1300 s/pair; n=280 uses the recorded Phase 1c measurement.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"  # set before numpy loads

import json

import numpy as np

from phase2 import candidates as cand
from phase2 import signatures as sig
from phase2.verify import sample_pair_indices, time_verification
from phase2b.data import SIZES, TAU_STAR, node_paths
from run_phase2_timing import PHASE1C_ALL_PAIRS_SECONDS, PHASE1C_SECONDS_PER_PAIR, hardware
from run_phase2b import load_signatures

OUT = REPO_ROOT / "results" / "phase2b_timing.json"


def main():
    names, H, C = load_signatures()
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    out = {"hardware": hardware(),
           "projection_note": "'projected_*' values are projections from a 200-pair sample (seed 12345), "
                              "not measurements of the whole candidate set.", "sizes": {}}

    paths280 = node_paths(40)
    t0 = time.perf_counter()
    sig.compute_signatures(paths280)
    total = time.perf_counter() - t0
    out["signature_extraction"] = {"images": len(paths280), "total_seconds": total,
                                   "seconds_per_image": total / len(paths280)}

    for F in SIZES:
        paths = node_paths(F)
        idx = [name_to_idx[p.name] for p in paths]
        n = len(paths)
        t0 = time.perf_counter()
        dist = sig.l2_matrix(C[idx])
        P = cand.m2_candidates(dist, TAU_STAR)
        cheap = time.perf_counter() - t0
        flat = np.sort(cand.pair_flat_indices(P))
        entry = {"n": n, "all_pairs": cand.n_pairs(n), "candidate_pairs": int(flat.size),
                 "cheap_stage_seconds": cheap,
                 "extrapolated_all_pairs_verification_seconds_at_0.13s_per_pair": cand.n_pairs(n) * PHASE1C_SECONDS_PER_PAIR}
        if n == 280:
            entry["measured_phase1c_all_pairs_seconds"] = PHASE1C_ALL_PAIRS_SECONDS
        if flat.size:
            chosen = flat[sample_pair_indices(flat.size)]
            rows, cols = cand.flat_to_pair(chosen, n)
            times = np.array(time_verification(paths, list(zip(rows.tolist(), cols.tolist()))))
            entry.update(
                sample_size=int(times.size), sample_mean_seconds=float(times.mean()),
                sample_std_seconds=float(times.std()),
                projected_verification_seconds=float(flat.size * times.mean()),
                projected_vs_extrapolated_all_pairs_speedup=float(
                    cand.n_pairs(n) * PHASE1C_SECONDS_PER_PAIR / (flat.size * times.mean())),
            )
        out["sizes"][str(F)] = entry
        print(f"F={F}: |P|={flat.size} cheap={cheap:.2f}s", flush=True)

    OUT.write_text(json.dumps(out, indent=1))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
