"""
Phase 2 — single-process timings (docs/phase2_design.md sections 10, 17.9).

Must be run only AFTER the parallel n=280 matrix job has finished and nothing
else heavy is running. Refuses to start if the matrix result is missing.

Reports: (a) cheap-stage cost (signature extraction, all-pairs signature
distances + Otsu); (b) for each (size, primary method) the mean time of a fixed
sample of 200 candidate-pair verifications and the resulting PROJECTION of the
verification time for the whole candidate set. All projections are labelled as
such; none is a measurement of the full candidate set.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"  # single process AND single BLAS thread, set before numpy loads

import numpy as np
import PIL

from phase2 import candidates as cand
from phase2 import signatures as sig
from phase2.generate import node_paths
from phase2.pipeline import primary_state
from phase2.verify import sample_pair_indices, time_verification

sys.path.insert(0, str(REPO_ROOT / "experiments"))
from run_phase2 import MATRIX, SIZES, load_signatures  # noqa: E402

OUT = REPO_ROOT / "results" / "phase2_timing.json"
PHASE1C_ALL_PAIRS_SECONDS = 5079.1731483  # recorded Phase 1c measurement (parallel-free)
PHASE1C_SECONDS_PER_PAIR = 0.1300


def hardware() -> dict:
    import os

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "parallelism": "none (single process, BLAS threads pinned to 1)",
    }


def main():
    if not MATRIX.exists():
        sys.exit("refusing to time: run the n=280 matrix job to completion first")

    names, H, C = load_signatures()
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    out = {"hardware": hardware(), "projection_note": "All 'projected_*' values are projections "
           "from a 200-pair sample (seed 12345), not measurements of the whole candidate set.", "sizes": {}}

    # (a) cheap-stage cost, signatures re-extracted for the n=280 images
    paths280 = node_paths(40)
    t0 = time.perf_counter()
    sig.compute_signatures(paths280)
    sig_seconds = time.perf_counter() - t0
    out["signature_extraction"] = {
        "images": len(paths280),
        "total_seconds": sig_seconds,
        "seconds_per_image": sig_seconds / len(paths280),
    }

    for F in SIZES:
        paths = node_paths(F)
        idx = [name_to_idx[p.name] for p in paths]
        n = len(paths)
        t0 = time.perf_counter()
        state = primary_state(H[idx], C[idx])
        cheap_seconds = time.perf_counter() - t0
        entry = {
            "n": n,
            "all_pairs": cand.n_pairs(n),
            "all_pairs_signature_distances_plus_otsu_seconds": cheap_seconds,
            "extrapolated_all_pairs_verification_seconds_at_0.13s_per_pair": cand.n_pairs(n) * PHASE1C_SECONDS_PER_PAIR,
            "methods": {},
        }
        if n == 280:
            entry["measured_phase1c_all_pairs_seconds"] = PHASE1C_ALL_PAIRS_SECONDS
        for m in ("M1", "M2", "M3"):
            P = state[m]
            flat = np.sort(cand.pair_flat_indices(P))
            if flat.size == 0:
                entry["methods"][m] = {"candidate_pairs": 0, "note": "empty candidate set"}
                continue
            chosen = flat[sample_pair_indices(flat.size)]
            rows, cols = cand.flat_to_pair(chosen, n)
            times = np.array(time_verification(paths, list(zip(rows.tolist(), cols.tolist()))))
            entry["methods"][m] = {
                "candidate_pairs": int(flat.size),
                "sample_size": int(times.size),
                "sample_mean_seconds": float(times.mean()),
                "sample_std_seconds": float(times.std()),
                "projected_verification_seconds": float(flat.size * times.mean()),
                "projected_vs_extrapolated_all_pairs_speedup": float(
                    cand.n_pairs(n) * PHASE1C_SECONDS_PER_PAIR / (flat.size * times.mean())
                ),
            }
            print(f"F={F} {m}: |P|={flat.size} sample_mean={times.mean():.4f}s", flush=True)
        out["sizes"][str(F)] = entry

    OUT.write_text(json.dumps(out, indent=1))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
