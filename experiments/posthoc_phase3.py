"""
Phase 3 POST-HOC analyses. NOT PRE-REGISTERED. Nothing here may change a gate verdict.

Run only after the audit has passed. Reads the committed run's persisted candidate sets and the labels
(the test split is already open), recomputes verifier values for the n = 1,000 primary-M2 candidate
pairs (bitwise-asserted against single-process), and writes results/phase3_posthoc.json.

(a) candidate-graph clustering quality: pairwise precision / recall / F1, ARI, cluster count and
    giant-cluster size, KwikCluster (20 frozen seeds, mean and sd) and union-find, graph = +1 iff the
    pair is in the primary M2 candidate set and r < tau_1c (the run's candidate graph).
(b) family-level bootstrap 95% CIs (percentile, 2,000 resamples, seeded) for M2-at-tau_B and M3 recall
    and per-transformation recall at every n, per draw and for the mean over the three draws.
(c) EXPLORATORY: alignment success as an edge rule (+1 iff the aligned verifier reports status ok, for
    pairs in the primary M2 candidate set), same clustering metrics.
The full-graph KwikCluster is NOT rerun (decision of the user).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from sklearn.metrics import adjusted_rand_score

from baselines.threshold_chaining import threshold_chaining
from cc_dedup.correlation_clustering import kwik_cluster
from phase2 import candidates as cand
from phase3 import e2e, runner
from phase3 import parallel as par
from phase3.labels import label_arrays
from phase3.splits import verify_startup

OUT = REPO / "results" / "phase3_posthoc.json"
BOOT_B = 2000
BOOT_SEED = 12345
TRANSFORMS = ("resize", "jpegq50", "crop", "bright", "rotate", "png")
METHODS = {"M2_B": "M2_B_x1.0", "M3": "M3"}


def comb2(x):
    x = np.asarray(x, dtype=np.int64)
    return x * (x - 1) // 2


def cluster_metrics(pred, fam) -> dict:
    pred = np.asarray(pred)
    _, pi = np.unique(pred, return_inverse=True)
    _, fi = np.unique(fam, return_inverse=True)
    sizes_p = np.bincount(pi)
    sizes_f = np.bincount(fi)
    cell = np.bincount(pi * (fi.max() + 1) + fi)
    tp, pp, tt = int(comb2(cell).sum()), int(comb2(sizes_p).sum()), int(comb2(sizes_f).sum())
    prec = tp / pp if pp else float("nan")
    rec = tp / tt if tt else float("nan")
    f1 = 2 * tp / (pp + tt) if (pp + tt) else float("nan")  # = 2PR/(P+R); 0 when no predicted pair is correct
    return {"precision": prec, "recall": rec, "f1": f1, "ari": float(adjusted_rand_score(fi, pi)),
            "clusters": int(sizes_p.size), "giant_cluster": int(sizes_p.max()), "pred_pairs": pp, "true_pairs": tt, "tp_pairs": tp}


def summarise(per_seed: list[dict]) -> dict:
    keys = ("precision", "recall", "f1", "ari", "clusters", "giant_cluster")
    return {k: {"mean": float(np.nanmean([m[k] for m in per_seed])), "sd": float(np.nanstd([m[k] for m in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0}
            for k in keys} | {"seeds": len(per_seed)}


def cluster_graph(sign: np.ndarray, fam: np.ndarray) -> dict:
    kw = summarise([cluster_metrics(kwik_cluster(sign, seed=s), fam) for s in e2e.SEEDS])
    uf = cluster_metrics(threshold_chaining(sign), fam)
    return {"kwik_20_seeds": kw, "union_find": uf, "positive_edges": int((sign[np.triu_indices(sign.shape[0], 1)] == 1).sum())}


def sign_matrix(n: int, pairs: list, positive: np.ndarray) -> np.ndarray:
    s = np.full((n, n), -1, dtype=np.int8)
    for (i, j), p in zip(pairs, positive):
        if p:
            s[i, j] = s[j, i] = 1
    np.fill_diagonal(s, 0)
    return s


def part_a_c(ctx, d: int, stored_e2e: dict) -> dict:
    n = ctx.plan.e2e_n
    paths_L, nodes, labels_path = runner.ladder_nodes(ctx, d)
    labels = json.loads(labels_path.read_text())["images"]
    paths = [paths_L[i] for i in nodes[n]]
    fam, _ = label_arrays(labels, paths)
    flats = runner.load_sets(ctx, f"ladder_d{d}", n)["M2_B"]
    rr, cc = cand.flat_to_pair(flats, n)
    pairs = list(zip(rr.tolist(), cc.tolist()))
    r = par.primary_pairs_parallel(paths, pairs, ctx.workers)
    ex_p = par.assert_primary_exact(paths, pairs, r, k=200)
    al = par.aligned_pairs_parallel(paths, pairs, ctx.workers)
    ex_a = par.assert_aligned_exact(paths, pairs, al, k=100)
    pos_primary = r < e2e.TAU_1C
    s_primary = sign_matrix(n, pairs, pos_primary)
    stored_pos = stored_e2e["clustering"]["positive_edges_candidate_graph"]
    assert int(pos_primary.sum()) == stored_pos, f"draw {d}: positive edges {int(pos_primary.sum())} != stored {stored_pos}"
    ok = np.array([x["status"] == "ok" for x in al])
    s_align = sign_matrix(n, pairs, ok)
    is_true = fam[rr] == fam[cc]
    return {"draw": d, "n": n, "candidate_pairs": len(pairs), "true_pairs_in_candidates": int(is_true.sum()),
            "exactness": {"primary": ex_p, "aligned": ex_a},
            "a_candidate_graph_primary_edges": cluster_graph(s_primary, fam),
            "c_exploratory_alignment_success_edges": {**cluster_graph(s_align, fam),
                                                      "aligned_ok": int(ok.sum()), "aligned_ok_true": int((ok & is_true).sum()),
                                                      "aligned_ok_non_true": int((ok & ~is_true).sum()),
                                                      "true_candidates_not_aligned": int((~ok & is_true).sum())},
            "reference_all_singletons": cluster_metrics(np.arange(n), fam)}


def bootstrap_sources(ctx, d: int, n: int, method_key: str):
    """Per-family hit and total counts: columns 0 = all pairs, 1..6 = per transformation."""
    paths_L, nodes, labels_path = runner.ladder_nodes(ctx, d)
    labels = json.loads(labels_path.read_text())["images"]
    nd = nodes[n]
    paths = [paths_L[i] for i in nd]
    fam, trans = label_arrays(labels, paths)
    tp = runner.true_pairs_from(fam)
    starts = cand._row_starts(n)
    flat = starts[tp[:, 0]] + (tp[:, 1] - tp[:, 0] - 1)
    sets = runner.load_sets(ctx, f"ladder_d{d}", n)
    hit = np.isin(flat, sets[method_key])
    _, fidx = np.unique(fam[tp[:, 0]], return_inverse=True)
    F = int(fidx.max()) + 1
    hits = np.zeros((F, 7))
    tots = np.zeros((F, 7))
    np.add.at(hits[:, 0], fidx, hit)
    np.add.at(tots[:, 0], fidx, 1)
    ta, tb = trans[tp[:, 0]], trans[tp[:, 1]]
    for k, t in enumerate(TRANSFORMS, start=1):
        sel = (ta == t) | (tb == t)
        np.add.at(hits[:, k], fidx[sel], hit[sel])
        np.add.at(tots[:, k], fidx[sel], 1)
    return hits, tots


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def part_b(ctx, parts) -> dict:
    out = {}
    cols = ("all",) + TRANSFORMS
    for mname, mkey in METHODS.items():
        out[mname] = {}
        for n in ctx.plan.sizes:
            per_draw = {}
            for d in ctx.draws:
                hits, tots = bootstrap_sources(ctx, d, n, mname)
                stored = parts[f"ladder_d{d}"]["sizes"][str(n)]["score"]["sets"][mkey]
                point = hits.sum(0) / tots.sum(0)
                assert abs(point[0] - stored["recall"]) < 1e-12, (mname, n, d, point[0], stored["recall"])
                for k, t in enumerate(TRANSFORMS, start=1):
                    assert abs(point[k] - stored["recall_per_transform"][t]) < 1e-12, (mname, n, d, t)
                per_draw[d] = (hits, tots, point)
            rng = np.random.default_rng([BOOT_SEED, n, 0 if mname == "M2_B" else 1])
            boots = {}
            for d, (hits, tots, point) in per_draw.items():
                F = hits.shape[0]
                idx = rng.integers(0, F, size=(BOOT_B, F))
                boots[d] = hits[idx].sum(axis=1) / tots[idx].sum(axis=1)  # (B, 7)
            pooled = np.mean([boots[d] for d in ctx.draws], axis=0)
            point_mean = np.mean([per_draw[d][2] for d in ctx.draws], axis=0)
            entry = {"families_per_draw": int(per_draw[ctx.draws[0]][0].shape[0]),
                     "mean_over_draws": {c: {"point": float(point_mean[k]), "ci95": ci(pooled[:, k])} for k, c in enumerate(cols)},
                     "per_draw": {f"d{d}": {c: {"point": float(per_draw[d][2][k]), "ci95": ci(boots[d][:, k])} for k, c in enumerate(cols)}
                                  for d in ctx.draws}}
            out[mname][str(n)] = entry
            print(f"  bootstrap {mname} n={n} done", flush=True)
    return out


def main() -> None:
    split = verify_startup()
    base = runner.Ctx("full", split, workers=10)
    parts = runner.collect_parts(base)
    out = {"NOT_PRE_REGISTERED": True,
           "notice": "Post-hoc analyses added after the results were reviewed. They cannot change any pre-registered gate verdict.",
           "run_commit": "47f1b879c51808fec6f9b7b40e30c8ee18932c54",
           "bootstrap": {"resamples": BOOT_B, "seed": BOOT_SEED, "unit": "family (source photograph)", "interval": "percentile 95%",
                         "note": "pooled interval = mean over the three draws, each draw resampled independently; draws share the pool so they are not fully independent"},
           "a_and_c": {}, "b": None}
    for d in base.draws:
        out["a_and_c"][f"d{d}"] = part_a_c(base, d, parts[f"e2e_d{d}"])
        print(f"  clustering draw {d} done", flush=True)
    out["b"] = part_b(base, parts)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
