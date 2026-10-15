"""
Phase 3 timing (docs/phase3_design.md section 9). Run ONLY after every parallel job has finished.

Single process, BLAS pinned to one thread (set before numpy is imported). Reports, per draw and
ladder rung: the cheap stage (feature extraction + candidate generation) wall time, the primary
verifier's mean per-pair cost on a fixed 200-pair sample of candidate pairs (seed 12345,
pairwise_distance only, decode excluded), and the projected verification time = |P| x that mean
(a PROJECTION). The all-pairs baseline uses a seeded 200-pair sample of real pairs, extrapolated
(an EXTRAPOLATION). Also the aligned verifier's per-pair cost on a 100-pair candidate sample.
Timings are machine-specific, so they live in results/phase3_timing.json, outside the
byte-comparable results file.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from phase2 import candidates as cand  # noqa: E402
from phase2.parallel_matrix import load_rgb  # noqa: E402
from phase2.verify import sample_pair_indices, time_verification  # noqa: E402
from phase3 import e2e, runner  # noqa: E402
from phase3 import evaluate as ev  # noqa: E402
from phase3 import features as ft  # noqa: E402
from phase3.aligned_verifier import AlignConfig, aligned_distance, prepare  # noqa: E402
from phase3.splits import verify_startup  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rehearsal", "full"], required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--draws", default="1,2,3")
    args = ap.parse_args()
    split = verify_startup()
    ctx = runner.Ctx(args.mode, split, draws=[int(x) for x in args.draws.split(",")])
    out: dict = {"mode": args.mode, "machine": {"platform": platform.platform(), "python": platform.python_version(),
                                                "numpy": np.__version__, "cpus": os.cpu_count()},
                 "blas_threads_pinned_to": 1, "note": "single process, run after all parallel jobs; timings are not reproducible bit for bit",
                 "cells": {}}
    cfg = AlignConfig()
    for d in ctx.draws:
        ladder = ctx.load(f"ladder_d{d}")
        paths_L, nodes, _ = runner.ladder_nodes(ctx, d)
        tau_b = ladder["tau_b"]
        for n in ctx.plan.sizes:
            paths = [paths_L[i] for i in nodes[n]]
            t0 = time.perf_counter()
            feats = ft.extract(paths, workers=1)
            t_feat = time.perf_counter() - t0
            t0 = time.perf_counter()
            ev.candidate_stage(paths, tau_b, features=feats, curves=False)
            t_cand = time.perf_counter() - t0
            sets = ladder["sizes"][str(n)]["score"]["sets"]
            m2 = runner.load_sets(ctx, f"ladder_d{d}", n)["M2_B"]
            pick = sample_pair_indices(m2.size)
            rr, cc = cand.flat_to_pair(m2[pick], n)
            t_cand_pairs = np.array(time_verification(paths, list(zip(rr.tolist(), cc.tolist()))))
            t_rand = np.array(time_verification(paths, e2e.all_pairs_sample(n, 200)))
            arrays = {}
            al_times, prep_times = [], []
            for i, j in list(zip(rr.tolist(), cc.tolist()))[:100]:
                for k in (i, j):
                    if k not in arrays:
                        a = load_rgb(paths[k])
                        t1 = time.perf_counter()
                        arrays[k] = prepare(a, cfg)
                        prep_times.append(time.perf_counter() - t1)
                t1 = time.perf_counter()
                aligned_distance(arrays[i], arrays[j], cfg)
                al_times.append(time.perf_counter() - t1)
            total = cand.n_pairs(n)
            out["cells"][f"d{d}_n{n}"] = {
                "images": n, "all_pairs": total,
                "cheap_stage_seconds": {"feature_extraction_decode_included": t_feat, "candidate_generation_one_operating_point_each": t_cand},
                "primary_verifier_seconds_per_pair": {"candidate_sample_mean": float(t_cand_pairs.mean()), "candidate_sample_pairs": int(t_cand_pairs.size),
                                                      "random_real_pairs_mean": float(t_rand.mean()), "random_pairs": int(t_rand.size)},
                "aligned_verifier_seconds": {"per_pair_mean": float(np.mean(al_times)), "prepare_per_image_mean": float(np.mean(prep_times)),
                                             "pairs": len(al_times)},
                "projection_primary_verification_seconds": {m: sets[name]["candidate_pairs"] * float(t_cand_pairs.mean()) for m, name in ev.PRIMARY.items()},
                "extrapolation_all_pairs_primary_seconds": total * float(t_rand.mean()),
                "labels": {"projection": "|P| x candidate-sample mean", "extrapolation": "all pairs x random-pair mean"},
            }
            print(f"d{d} n{n} done", flush=True)
    dest = Path(args.out) if args.out else (REPO / "results" / "phase3_timing.json" if args.mode == "full" else REPO / "phase3" / "artifacts" / "rehearsal_timing.json")
    dest.write_text(json.dumps(out, indent=1, sort_keys=True))
    print("wrote", dest)


if __name__ == "__main__":
    main()
