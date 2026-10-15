"""
Phase 3 verifier metrics (docs/phase3_design.md sections 5.1, 6 and Addendum A.3, B).

Pure arithmetic on already-computed verifier outputs; it computes no distance itself and opens no
label file (the caller passes true-pair indices that came from the scoring stage).

Primary verifier = the Phase 1c distance with the fixed tau; aligned verifier = configuration C with
the per-draw Otsu threshold tau_al of A.3. Alignment failures (status != "ok") are counted and are
never given a distance.
"""

from __future__ import annotations

import numpy as np

from phase2 import candidates as cand

FA_SAMPLE = 1000
FA_SEED = 12345


def aligned_arrays(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    ok = np.array([r["status"] == "ok" for r in results], dtype=bool)
    dist = np.array([r["distance"] if r["status"] == "ok" else np.nan for r in results], dtype=np.float64)
    return ok, dist


def status_counts(results: list[dict]) -> dict:
    out: dict = {}
    for r in results:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return dict(sorted(out.items()))


def auc_smaller_is_positive(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(d_pos < d_neg) + 0.5 P(d_pos == d_neg): the AUC of "smaller distance = true pair" (Mann-Whitney)."""
    pos, neg = np.asarray(pos, dtype=np.float64), np.asarray(neg, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size, dtype=np.float64)
    sorted_v = allv[order]
    # average ranks over ties
    _, first, counts = np.unique(sorted_v, return_index=True, return_counts=True)
    avg = first + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    r_pos = ranks[: pos.size].sum()
    u_pos_greater = r_pos - pos.size * (pos.size + 1) / 2.0  # pairs with d_pos > d_neg (ties 0.5)
    return float(1.0 - u_pos_greater / (pos.size * neg.size))


def aligned_auc(true_ok: np.ndarray, true_dist: np.ndarray, neg_ok: np.ndarray, neg_dist: np.ndarray) -> dict:
    """Two declared variants: failures dropped (ok pairs only) and failures counted as +inf (worst)."""
    pos_ok, neg_okd = true_dist[true_ok], neg_dist[neg_ok]
    pos_inf = np.where(true_ok, true_dist, np.inf)
    neg_inf = np.where(neg_ok, neg_dist, np.inf)
    return {"auc_ok_pairs_only": auc_smaller_is_positive(pos_ok, neg_okd),
            "auc_failures_as_inf": auc_smaller_is_positive(pos_inf, neg_inf),
            "true_pairs": int(true_ok.size), "true_ok": int(true_ok.sum()),
            "non_true_pairs": int(neg_ok.size), "non_true_ok": int(neg_ok.sum())}


def fa_sample_pairs(cand_flats: np.ndarray, true_flats: np.ndarray, n: int, k: int = FA_SAMPLE,
                    seed: int = FA_SEED) -> list[tuple[int, int]]:
    """Seeded sample (without replacement, sorted) of NON-true candidate pairs, as (i, j) node pairs."""
    non_true = cand_flats[~np.isin(cand_flats, true_flats)]
    if non_true.size == 0:
        return []
    pick = np.sort(np.random.default_rng(seed).choice(non_true.size, size=min(k, non_true.size), replace=False))
    rows, cols = cand.flat_to_pair(non_true[pick], n)
    return list(zip(rows.tolist(), cols.tolist()))


def true_flats_in_size(true_pairs_L: np.ndarray, nodes_L: np.ndarray, total_L: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Map true pairs given in the largest set's indices to a nested size. Returns (flat index in the
    size's pair order, positions of those pairs in `true_pairs_L`). Node order is preserved by nesting,
    so i < j is kept."""
    pos = np.full(total_L, -1, dtype=np.int64)
    pos[nodes_L] = np.arange(n)
    pa, pb = pos[true_pairs_L[:, 0]], pos[true_pairs_L[:, 1]]
    inside = (pa >= 0) & (pb >= 0)
    assert np.all(pa[inside] < pb[inside])
    starts = cand._row_starts(n)
    flat = starts[pa[inside]] + (pb[inside] - pa[inside] - 1)
    return flat, np.flatnonzero(inside)


def primary_set_metrics(true_flat: np.ndarray, r_true: np.ndarray, cand_flats: np.ndarray, tau: float) -> dict:
    confirmed = np.isin(true_flat, cand_flats) & (r_true < tau)
    return {"true_pairs": int(true_flat.size), "verifier_cap": float((r_true < tau).mean()) if r_true.size else float("nan"),
            "confirmed_recall": float(confirmed.mean()) if r_true.size else float("nan")}


def aligned_set_metrics(true_flat: np.ndarray, ok: np.ndarray, dist: np.ndarray, cand_flats: np.ndarray, tau_al: float) -> dict:
    accepted = ok & (np.where(ok, dist, np.inf) < tau_al)
    in_p = np.isin(true_flat, cand_flats)
    return {"true_pairs": int(true_flat.size), "alignment_failure_rate": float(1.0 - ok.mean()) if ok.size else float("nan"),
            "aligned_cap": float(accepted.mean()) if ok.size else float("nan"),
            "confirmed_recall": float((in_p & accepted).mean()) if ok.size else float("nan")}


def false_accept(values: np.ndarray, tau: float) -> dict:
    return {"sampled_non_true_candidate_pairs": int(values.size),
            "false_accept_rate": float((values < tau).mean()) if values.size else float("nan")}


def false_accept_aligned(results: list[dict], tau_al: float) -> dict:
    ok, dist = aligned_arrays(results)
    accepted = ok & (np.where(ok, dist, np.inf) < tau_al)
    return {"sampled_non_true_candidate_pairs": len(results), "alignment_failure_rate": float(1.0 - ok.mean()) if ok.size else float("nan"),
            "false_accept_rate": float(accepted.mean()) if ok.size else float("nan"), "status_counts": status_counts(results)}
