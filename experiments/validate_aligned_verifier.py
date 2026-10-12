"""
Validate the draft aligned verifier on SYNTHETIC pairs from Phase 2 families 1-40 ONLY
(user instruction 2026-10-12: not families 801-1600, not any Phase 3 data).

Pairs: all 21 true pairs per family (840) plus 840 different-family pairs (seed 12345).
Reports alignment success by transformation, distance separation, and a comparison with
the frozen Phase 1c distance on the same pairs. Output: phase3/artifacts/ (gitignored).
Descriptive validation only: it informs which parameters get frozen; it is not a result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from phase2.generate import node_paths, sha256_file
from phase2.parallel_matrix import load_rgb
from phase2.pipeline import node_labels
from phase3.aligned_verifier import AlignConfig, aligned_distance, prepare

N_FAMILIES = 40  # families 1..40 only
DATASET_DIR = REPO_ROOT / "phase3" / "artifacts" / "synth40"  # regenerated with the unmodified Phase 0 generator
PHASE2_MANIFEST = REPO_ROOT / "phase2" / "data" / "F800" / "manifest.json"
TAU_PHASE1C = json.loads((REPO_ROOT / "results" / "phase1c_results.json").read_text())["threshold"]
OUT = REPO_ROOT / "phase3" / "artifacts"


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos < score_neg), ties half; lower distance = more similar. inf allowed."""
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort(kind="stable").argsort() + 1.0
    # handle ties by average rank
    order = np.sort(allv)
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    first = np.cumsum(cnt) - cnt + 1
    avg = first + (cnt - 1) / 2.0
    ranks = avg[inv]
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return 1.0 - u / (len(pos) * len(neg))


def main(cfg: AlignConfig = AlignConfig()):
    paths = node_paths(N_FAMILIES, DATASET_DIR)
    assert len(paths) == 280
    p2 = json.loads(PHASE2_MANIFEST.read_text())["sha256"]
    images_match_phase2 = all(sha256_file(p) == p2[f"{p.parent.name}/{p.name}"] for p in paths)
    assert images_match_phase2, "validation images are not Phase 2 families 1-40"
    family, transform = node_labels(paths)
    r1c = np.load(REPO_ROOT / "results" / "phase2_n280_distance_matrix.npy")

    n = len(paths)
    iu, ju = np.triu_indices(n, 1)
    same = family[iu] == family[ju]
    true_idx = np.flatnonzero(same)
    rng = np.random.default_rng(12345)
    false_idx = np.sort(rng.choice(np.flatnonzero(~same), size=len(true_idx), replace=False))

    prepared: dict[int, dict] = {}

    def get(i):
        if i not in prepared:
            prepared[i] = prepare(load_rgb(paths[i]), cfg)
        return prepared[i]

    def run(idx):
        out = []
        for k in idx:
            i, j = int(iu[k]), int(ju[k])
            res = aligned_distance(get(i), get(j), cfg)
            res.update(i=i, j=j, r_phase1c=float(r1c[i, j]))
            out.append(res)
        return out

    true_res, false_res = run(true_idx), run(false_idx)

    def dist_or_inf(rs):
        return np.array([r["distance"] if r["status"] == "ok" else np.inf for r in rs])

    d_true, d_false = dist_or_inf(true_res), dist_or_inf(false_res)
    r_true = np.array([r["r_phase1c"] for r in true_res])
    r_false = np.array([r["r_phase1c"] for r in false_res])

    def by_type(res):
        groups = {}
        for r in res:
            ti, tj = transform[r["i"]], transform[r["j"]]
            if "original" in (ti, tj):
                key = "OT:" + (tj if ti == "original" else ti)
            else:
                key = "TT"
            groups.setdefault(key, []).append(r)
        return groups

    summary = {"config": cfg.__dict__ | {"scale_range": list(cfg.scale_range)},
               "data": "synthetic Phase 2 families 1-40 only (280 images, byte-identical to the Phase 2 manifest)",
               "images_match_phase2_manifest": images_match_phase2,
               "n_true_pairs": len(true_res), "n_false_pairs": len(false_res),
               "true_pairs_alignment_ok": float(np.isfinite(d_true).mean()),
               "false_pairs_alignment_ok": float(np.isfinite(d_false).mean()),
               "auc_aligned_failures_as_worst": auc(d_true, d_false),
               "auc_phase1c_distance": auc(r_true, r_false),
               "true_pairs_passing_phase1c_tau": float((r_true < TAU_PHASE1C).mean()),
               "failure_reasons_true": {}, "by_type": {}}
    for r in true_res:
        if r["status"] != "ok":
            summary["failure_reasons_true"][r["status"]] = summary["failure_reasons_true"].get(r["status"], 0) + 1
    for key, rs in sorted(by_type(true_res).items()):
        ok = [r for r in rs if r["status"] == "ok"]
        summary["by_type"][key] = {
            "pairs": len(rs),
            "alignment_failures": len(rs) - len(ok),
            "alignment_ok": len(ok) / len(rs),
            "median_aligned_distance": float(np.median([r["distance"] for r in ok])) if ok else None,
            "phase1c_pass_rate": float(np.mean([r["r_phase1c"] < TAU_PHASE1C for r in rs])),
        }
    ok_false = d_false[np.isfinite(d_false)]
    ok_true = d_true[np.isfinite(d_true)]
    summary["aligned_distance_quantiles_true_ok"] = np.quantile(ok_true, [0.1, 0.5, 0.9]).tolist() if ok_true.size else None
    summary["aligned_distance_quantiles_false_ok"] = np.quantile(ok_false, [0.1, 0.5, 0.9]).tolist() if ok_false.size else None
    OUT.mkdir(parents=True, exist_ok=True)
    return summary


if __name__ == "__main__":
    s = main()
    manifest = REPO_ROOT / "phase3" / "manifests" / "aligned_verifier_validation.json"
    manifest.write_text(json.dumps(s, indent=1, sort_keys=True))
    print(json.dumps(s, indent=1))
