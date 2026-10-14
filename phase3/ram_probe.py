"""
Peak-RAM probe for the largest block-wise steps (n = 10,000), on RANDOM signatures: it depends only
on the array shapes, so no photograph (and no test-split image) is touched.

    python -m phase3.ram_probe m2      # block-wise candidate generation
    python -m phase3.ram_probe otsu    # block-wise per-run Otsu (two passes)
    python -m phase3.ram_probe score   # scoring with n x n boolean matrices (phase2.metrics.score)
"""

from __future__ import annotations

import json
import sys

import numpy as np

from phase3 import blockwise as bw

N = 10000


def random_signatures(n: int, seed: int = 12345) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 1056)).astype(np.float32)
    for lo, hi in ((0, 1024), (1024, 1048), (1048, 1056)):  # three unit-norm blocks, like the real signature
        x[:, lo:hi] /= np.linalg.norm(x[:, lo:hi], axis=1, keepdims=True)
    return x


def main(stage: str) -> dict:
    out = {"stage": stage, "n": N, "baseline_peak_mb": bw.peak_working_set_mb()}
    if stage == "m2":
        x = random_signatures(N)
        a, sq = bw.prepare(x[:600])
        d = bw.l2_rows(a, sq, 0, 100)
        tau = float(np.quantile(d[:, 100:], 0.01))
        flats = bw.m2_candidate_flats(x, tau)
        out.update(tau=tau, candidate_pairs=int(flats.size))
    elif stage == "otsu":
        tau, method = bw.otsu_blockwise(random_signatures(N))
        out.update(tau=tau, method=method)
    elif stage == "score":
        from phase2 import candidates as cand
        from phase2 import metrics as met
        family = np.concatenate([np.repeat(np.arange(500), 7), -1 - np.arange(6500)])
        transform = np.array((["original", "resize", "jpegq50", "crop", "bright", "rotate", "png"] * 500) + ["singleton"] * 6500)
        truth = met.truth_matrix(family)
        P = cand.random_control(N, 500000)
        s = met.score(P, truth, transform)
        out.update(recall=s["recall"], candidate_pairs=s["candidate_pairs"])
    else:
        raise SystemExit("stage must be m2, otsu or score")
    out["peak_mb"] = bw.peak_working_set_mb()
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1])))
