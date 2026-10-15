"""
Phase 3 run library (docs/phase3_design.md sections 3-8, 11-12 and Addenda A, B).

A run is a sequence of resumable PARTS written under phase3/artifacts/run_<mode>/ (one JSON file per
part, byte-reproducible: no timings, no timestamps). Wall-clock events go to events.jsonl, which is
NOT part of the results. Order: dev tau_B per draw, then D3a ladder (+ verifier metrics) for every
draw, the low-prevalence variant, D3b, the secondary pool run, and the end-to-end run.

mode "rehearsal": exactly the same code on the DEV split only (smaller rungs, dev photographs standing
in for the test lists). It exists to validate the code; its numbers are not evidence.
mode "full": the test split. Reachable only through experiments/run_phase3.py after its gates pass.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

from cc_dedup.graph import compute_otsu_threshold
from phase2 import candidates as cand
from phase3 import e2e
from phase3 import evaluate as ev
from phase3 import features as ft
from phase3 import parallel as par
from phase3 import verifier_metrics as vm
from phase3.dataset import COPYDAYS_DIR, build_d3a_set, build_d3b_set
from phase3.labels import label_arrays, open_labels
from phase3.splits import DRAW_SEEDS, REPO, VerifiedSplit

GATED = ("M1", "dHash", "EXACT", "M2_B", "M3")  # M2_A1 is a non-gated comparison (Addendum A.1)
VERIFIED_METHODS = ("M1", "dHash", "M2_B", "M3")  # false-accept samples (EXACT has no non-true pairs by construction)
MIN_FREE_C_GB, MIN_FREE_D_GB = 1.5, 3.0  # runtime guard, below the start thresholds of 2 and 6


class DiskGuard(RuntimeError):
    pass


# ----------------------------------------------------------------------------- plan
FULL = {
    "ladder": {1000: (50, 650), 2500: (125, 1625), 5000: (250, 3250), 10000: (500, 6500)},
    "variant": (8000, 114, 7202),
    "d3b_singletons": 4614,
    "ext": (5000, 250, 3250),
}
# Rehearsal: the dev split has 150 sources and 1,350 singletons per draw, so the rungs are smaller.
REHEARSAL = {
    "ladder": {400: (20, 260), 1000: (50, 650), 2000: (100, 1300)},
    "variant": (1400, 20, 1260),
    "d3b_singletons": 300,
    "ext": (500, 25, 325),
}


class Plan:
    def __init__(self, mode: str, split: VerifiedSplit):
        assert mode in ("full", "rehearsal")
        self.mode, self.split = mode, split
        self.cfg = FULL if mode == "full" else REHEARSAL
        self.sizes = sorted(self.cfg["ladder"])
        self.e2e_n = self.sizes[0]

    def sources(self, d):
        s = self.split
        return s.test_sources(d) if self.mode == "full" else s.dev_sources(d)

    def singletons(self, d):
        s = self.split
        return s.test_singletons(d) if self.mode == "full" else s.dev_singletons(d)

    def variant(self, d):
        n, f, sg = self.cfg["variant"]
        if self.mode == "full":
            return self.split.variant_sources(d)[:f], self.split.variant_singletons(d)[:sg]
        return self.split.dev_sources(d)[:f], self.split.dev_singletons(d)[:sg]

    def d3b_singletons(self, d):
        k = self.cfg["d3b_singletons"]
        if self.mode == "full":
            return self.split.d3b_singletons(d)[:k]
        return self.split.dev_singletons(d)[1000 : 1000 + k]

    def ext(self, d):
        n, f, sg = self.cfg["ext"]
        if self.mode == "full":
            return self.split.ext_sources250(d)[:f], self.split.ext_singletons3250(d)[:sg]
        return self.split.ext_dev_sources(d)[:f], self.split.ext_dev_singletons(d)[:sg]


# ----------------------------------------------------------------------------- context, io
class Ctx:
    def __init__(self, mode: str, split: VerifiedSplit, workers: int = 10, draws=(1, 2, 3), tag: str = "",
                 only_sizes=None):
        self.plan = Plan(mode, split)
        self.mode, self.workers, self.draws, self.tag = mode, workers, tuple(draws), tag
        self.only_sizes = list(only_sizes) if only_sizes else None  # audit re-runs: a subset of the ladder, no cache
        self.art = REPO / "phase3" / "artifacts" / f"run_{mode}{tag}"
        self.data = REPO / "phase3" / "data" / f"run_{mode}"
        self.art.mkdir(parents=True, exist_ok=True)
        self.data.mkdir(parents=True, exist_ok=True)

    def event(self, name: str, **kw) -> None:
        rec = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": name, **kw}
        with open(self.art / "events.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{rec['t']}] {name} " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)

    def part_path(self, name: str) -> Path:
        return self.art / f"{name}.json"

    def have(self, name: str) -> bool:
        return self.part_path(name).exists()

    def save(self, name: str, obj) -> None:
        tmp = self.art / f"{name}.json.tmp"
        tmp.write_text(json.dumps(obj, sort_keys=True, indent=1))
        os.replace(tmp, self.part_path(name))

    def load(self, name: str):
        return json.loads(self.part_path(name).read_text())

    def guard(self) -> None:
        c, d = shutil.disk_usage("C:\\").free / 1e9, shutil.disk_usage("D:\\").free / 1e9
        if c < MIN_FREE_C_GB or d < MIN_FREE_D_GB:
            raise DiskGuard(f"free disk fell below the run guard: C: {c:.2f} GB (min {MIN_FREE_C_GB}), D: {d:.2f} GB (min {MIN_FREE_D_GB})")


def free_gb() -> dict:
    return {"C:": round(shutil.disk_usage("C:\\").free / 1e9, 2), "D:": round(shutil.disk_usage("D:\\").free / 1e9, 2)}


def true_pairs_from(family: np.ndarray) -> np.ndarray:
    """All within-family node pairs (i < j), grouped by family so a batch shares images; singletons have none."""
    groups: dict[int, list[int]] = {}
    for i, f in enumerate(family.tolist()):
        if f >= 0:
            groups.setdefault(f, []).append(i)
    out = [(a, b) for f in sorted(groups) for a, b in combinations(groups[f], 2)]
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def primary_sets(stage: dict) -> dict:
    return {m: stage["sets"][name] for m, name in ev.PRIMARY.items()}


def save_sets(ctx: Ctx, name: str, per_size: dict) -> None:
    np.savez_compressed(ctx.art / f"sets_{name}.npz", **{f"n{n}__{m}": v for n, sets in per_size.items() for m, v in sets.items()})


def load_sets(ctx: Ctx, name: str, n: int) -> dict:
    z = np.load(ctx.art / f"sets_{name}.npz")
    return {m: z[f"n{n}__{m}"] for m in ev.PRIMARY}


# ----------------------------------------------------------------------------- tau_B (dev only)
def run_tau_b(ctx: Ctx, d: int, ext: bool = False) -> None:
    name = f"tau_b_{'ext_' if ext else ''}d{d}"
    if ctx.have(name):
        return
    sp = ctx.plan.split
    src, single = (sp.ext_dev_sources(d), sp.ext_dev_singletons(d)) if ext else (sp.dev_sources(d), sp.dev_singletons(d))
    root = ctx.data / f"draw{d}_{'ext_' if ext else ''}dev"  # directory name ends in _dev (calibrate_tau_b requires it)
    dev = build_d3a_set(root, src, single, DRAW_SEEDS[d])
    cache = None if ctx.tag else root / "sig.npz"  # tagged (audit) re-runs recompute the signatures
    info = ev.calibrate_tau_b(dev["paths"], dev["labels_path"], sig_cache=cache)
    info.update(draw=d, extended_pool=ext, dev_sources=len(src), dev_singletons=len(single))
    ctx.save(name, info)
    ctx.event("tau_b_written", part=name, tau_b=info["tau_b"], dev_recall_at_tau=info["dev_recall_at_tau"])


# ----------------------------------------------------------------------------- one evaluated set
def evaluate_set(ctx: Ctx, tag: str, paths, feats, tau_b: float, labels_path: Path, seed: int):
    """Stage 1 (label-free) -> hashes are final -> stage 2 opens the labels. Returns (stage, scored)."""
    stage = ev.candidate_stage(paths, tau_b, features=feats, curves=True)
    ctx.event("candidate_hashes_final", part=tag, n=len(paths), sets=len(stage["hashes"]))
    ctx.event("labels_opened", part=tag)
    return stage, ev.score_sets(stage, labels_path, paths, seed)


# ----------------------------------------------------------------------------- verifier metrics
def verify_true_pairs(ctx: Ctx, paths, true_pairs: np.ndarray) -> dict:
    pairs = [tuple(p) for p in true_pairs.tolist()]
    r = par.primary_pairs_parallel(paths, pairs, ctx.workers)
    exact_p = par.assert_primary_exact(paths, pairs, r, k=500)
    al = par.aligned_pairs_parallel(paths, pairs, ctx.workers)
    exact_a = par.assert_aligned_exact(paths, pairs, al, k=200)
    ok, dist = vm.aligned_arrays(al)
    return {"r": r, "ok": ok, "dist": dist, "status_counts": vm.status_counts(al),
            "exactness": {"primary": exact_p, "aligned": exact_a}}


def method_metrics(true_flat, r, ok, dist, sets: dict, tau_al) -> dict:
    out = {}
    for m in ev.PRIMARY:
        flats = sets[m]
        entry = {"primary": vm.primary_set_metrics(true_flat, r, flats, e2e.TAU_1C)}
        if tau_al is not None:
            entry["aligned"] = vm.aligned_set_metrics(true_flat, ok, dist, flats, tau_al)
        out[m] = entry
    return out


def false_accept_all(ctx: Ctx, paths, sets: dict, true_flat_all: np.ndarray, tau_al) -> dict:
    n = len(paths)
    out = {}
    for m in VERIFIED_METHODS:
        pairs = vm.fa_sample_pairs(sets[m], true_flat_all, n)
        if not pairs:
            out[m] = {"sampled_non_true_candidate_pairs": 0}
            continue
        r = par.primary_pairs_parallel(paths, pairs, ctx.workers)
        entry = {"primary": vm.false_accept(r, e2e.TAU_1C),
                 "exactness_primary": par.assert_primary_exact(paths, pairs, r, k=100)}
        if tau_al is not None:
            al = par.aligned_pairs_parallel(paths, pairs, ctx.workers)
            entry["aligned"] = vm.false_accept_aligned(al, tau_al)
            entry["exactness_aligned"] = par.assert_aligned_exact(paths, pairs, al, k=50)
        out[m] = entry
    return out


def tau_al_from(ctx: Ctx, d: int):
    part = ctx.load(f"verify_d{d}")["tau_al"]
    return part["tau_al"]


# ----------------------------------------------------------------------------- D3a ladder + verifier metrics
def ladder_dir(ctx, d):
    return ctx.data / f"draw{d}_ladder"


def ladder_nodes(ctx: Ctx, d: int):
    root = ladder_dir(ctx, d)
    mem = json.loads((root / "membership.json").read_text())
    paths_L = sorted((root / "images").glob("i*"))
    idx = {p.name: k for k, p in enumerate(paths_L)}
    nodes = {int(n): np.array([idx[nm] for nm in names], dtype=np.int64) for n, names in mem.items()}
    return paths_L, nodes, root / "labels.json"


def run_ladder(ctx: Ctx, d: int) -> None:
    name = f"ladder_d{d}"
    if ctx.have(name):
        return
    ctx.guard()
    plan = ctx.plan
    seed = DRAW_SEEDS[d]
    tau_b = ctx.load(f"tau_b_d{d}")["tau_b"]
    Fmax, Smax = plan.cfg["ladder"][plan.sizes[-1]]
    build_d3a_set(ladder_dir(ctx, d), plan.sources(d)[:Fmax], plan.singletons(d)[:Smax], seed, membership=plan.cfg["ladder"])
    paths_L, nodes, labels_path = ladder_nodes(ctx, d)
    ctx.event("ladder_built", draw=d, images=len(paths_L), free_gb=free_gb())
    run_sizes = ctx.only_sizes or plan.sizes
    if ctx.only_sizes:  # audit re-run: features of the chosen rungs only, extracted afresh (no cache)
        feats_L = None
    else:
        feats_L = ft.extract_cached(paths_L, ladder_dir(ctx, d) / "features.npz", ctx.workers)
    out = {"draw": d, "seed": seed, "tau_b": tau_b, "sizes": {}}
    kept = {}
    for n in run_sizes:
        ctx.guard()
        nd = nodes[n]
        sub = {k: v[nd] for k, v in feats_L.items()} if feats_L is not None else ft.extract([paths_L[i] for i in nd], ctx.workers)
        stage, scored = evaluate_set(ctx, f"{name}/n{n}", [paths_L[i] for i in nd], sub, tau_b, labels_path, seed)
        out["sizes"][str(n)] = {"images": int(nd.size), "candidate_hashes": stage["hashes"], "score": scored}
        kept[n] = primary_sets(stage)
        if n == plan.sizes[0]:  # the smallest rung keeps every operating point, so the audit can compare curves set by set
            kept[n].update(stage["sets"])
        del stage
    save_sets(ctx, f"ladder_d{d}", kept)
    ctx.save(name, out)


def run_verify(ctx: Ctx, d: int) -> None:
    """Verifier metrics for the ladder of draw d, and tau_al (A.3) from the smallest rung's candidate pairs."""
    name = f"verify_d{d}"
    if ctx.have(name):
        return
    ctx.guard()
    plan = ctx.plan
    ladder = ctx.load(f"ladder_d{d}")
    paths_L, nodes, labels_path = ladder_nodes(ctx, d)
    labels = open_labels(labels_path, ladder["sizes"][str(plan.sizes[0])]["candidate_hashes"])
    fam_L, _ = label_arrays(labels, paths_L)
    L = len(paths_L)
    out = {"draw": d}

    # --- tau_al (A.3): aligned distances of the primary-M2 candidate pairs at the smallest rung (the end-to-end run)
    n0 = plan.e2e_n
    paths0 = [paths_L[i] for i in nodes[n0]]
    m2 = load_sets(ctx, f"ladder_d{d}", n0)["M2_B"]
    rr, cc = cand.flat_to_pair(m2, n0)
    pairs0 = list(zip(rr.tolist(), cc.tolist()))
    al0 = par.aligned_pairs_parallel(paths0, pairs0, ctx.workers)
    exact_a = par.assert_aligned_exact(paths0, pairs0, al0, k=200)
    ok0, dist0 = vm.aligned_arrays(al0)
    fam0, _ = label_arrays(labels, paths0)
    true0 = true_pairs_from(fam0)
    true0_flat = np.sort(cand._row_starts(n0)[true0[:, 0]] + (true0[:, 1] - true0[:, 0] - 1))
    is_true0 = np.isin(m2, true0_flat)
    if ok0.sum() >= 2:
        thr, method = compute_otsu_threshold(dist0[ok0])
        tau_al = float(thr)
    else:
        tau_al, method = None, "no_alignment_ok_pairs"
    out["tau_al"] = {"tau_al": tau_al, "threshold_method": method, "rung": n0, "candidate_pairs": len(pairs0),
                     "alignment_ok": int(ok0.sum()), "status_counts": vm.status_counts(al0), "exactness": exact_a,
                     "auc": vm.aligned_auc(ok0[is_true0], dist0[is_true0], ok0[~is_true0], dist0[~is_true0])}
    ctx.event("tau_al_written", draw=d, tau_al=tau_al)

    # --- true pairs of the largest rung (every smaller rung is a node subset of it)
    tp = true_pairs_from(fam_L)
    v = verify_true_pairs(ctx, paths_L, tp)
    out["largest_rung_true_pairs"] = {"images": L, "true_pairs": int(tp.shape[0]), "alignment_status_counts": v["status_counts"],
                                      "exactness": v["exactness"]}
    out["sizes"] = {}
    for n in plan.sizes:
        tf, pos = vm.true_flats_in_size(tp, nodes[n], L, n)
        sets = load_sets(ctx, f"ladder_d{d}", n)
        out["sizes"][str(n)] = {"methods": method_metrics(tf, v["r"][pos], v["ok"][pos], v["dist"][pos], sets, out["tau_al"]["tau_al"])}
    # --- false-accept samples at the largest rung
    nL = plan.sizes[-1]
    sets_L = load_sets(ctx, f"ladder_d{d}", nL)
    tfL, _ = vm.true_flats_in_size(tp, nodes[nL], L, nL)
    out["false_accept_largest_rung"] = {"rung": nL, **false_accept_all(ctx, [paths_L[i] for i in nodes[nL]], sets_L, tfL, out["tau_al"]["tau_al"])}
    ctx.save(name, out)


# ----------------------------------------------------------------------------- variant, D3b, secondary pool
def run_single_set(ctx: Ctx, kind: str, d: int) -> None:
    name = f"{kind}_d{d}"
    if ctx.have(name):
        return
    ctx.guard()
    plan, seed = ctx.plan, DRAW_SEEDS[d]
    if kind == "variant":
        src, single = plan.variant(d)
        tau_b = ctx.load(f"tau_b_d{d}")["tau_b"]
        built = build_d3a_set(ctx.data / f"draw{d}_variant", src, single, seed)
    elif kind == "ext":
        src, single = plan.ext(d)
        tau_b = ctx.load(f"tau_b_ext_d{d}")["tau_b"]
        built = build_d3a_set(ctx.data / f"draw{d}_ext", src, single, seed)
    else:  # d3b
        tau_b = ctx.load(f"tau_b_d{d}")["tau_b"]
        rows = [json.loads(l) for l in open(REPO / "phase3" / "manifests" / "copydays.jsonl")]
        blocks = sorted({r["block"] for r in rows if r["accepted"]})
        built = build_d3b_set(ctx.data / f"draw{d}_d3b", blocks, plan.d3b_singletons(d), seed)
    paths, labels_path = built["paths"], built["labels_path"]
    feats = ft.extract_cached(paths, labels_path.parent / "features.npz", ctx.workers)
    stage, scored = evaluate_set(ctx, name, paths, feats, tau_b, labels_path, seed)
    out = {"draw": d, "kind": kind, "seed": seed, "tau_b": tau_b, "images": len(paths),
           "candidate_hashes": stage["hashes"], "score": scored}
    sets = primary_sets(stage)
    save_sets(ctx, name, {len(paths): sets})
    labels = open_labels(labels_path, stage["hashes"])
    fam, _ = label_arrays(labels, paths)
    tp = true_pairs_from(fam)
    tau_al = tau_al_from(ctx, d)
    v = verify_true_pairs(ctx, paths, tp)
    tf = cand._row_starts(len(paths))[tp[:, 0]] + (tp[:, 1] - tp[:, 0] - 1)  # aligned with v['r'] (not sorted)
    out["verifier"] = {"alignment_status_counts": v["status_counts"], "exactness": v["exactness"], "tau_al": tau_al,
                       "methods": method_metrics(tf, v["r"], v["ok"], v["dist"], sets, tau_al),
                       "false_accept": false_accept_all(ctx, paths, sets, tf, tau_al)}
    ctx.save(name, out)


# ----------------------------------------------------------------------------- end-to-end clustering
def run_e2e(ctx: Ctx, d: int) -> None:
    name = f"e2e_d{d}"
    if ctx.have(name):
        return
    ctx.guard()
    n = ctx.plan.e2e_n
    paths_L, nodes, labels_path = ladder_nodes(ctx, d)
    paths = [paths_L[i] for i in nodes[n]]
    flats = load_sets(ctx, f"ladder_d{d}", n)["M2_B"]
    ctx.event("e2e_graph_start", draw=d, n=n, pairs=cand.n_pairs(n))
    r = e2e.full_graph(paths, ctx.workers)
    exact = e2e.primary_exactness(paths, r)
    if not exact["bitwise_equal"]:
        raise AssertionError(f"parallel full graph differs from single-process: {exact}")
    out = {"draw": d, "n": n, "exactness_primary": exact, "tau_1c": e2e.TAU_1C}
    out["clustering"] = e2e.cluster_comparison(r, flats)
    ladder = ctx.load(f"ladder_d{d}")
    labels = open_labels(labels_path, ladder["sizes"][str(n)]["candidate_hashes"])
    fam, _ = label_arrays(labels, paths)
    iu = np.triu_indices(n, 1)
    truth = (fam[:, None] == fam[None, :])[iu]
    cand_mask = np.zeros(cand.n_pairs(n), dtype=bool)
    cand_mask[flats] = True  # flat index == position in the row-major upper triangle
    rv = r[iu]
    out["primary_auc_on_candidate_pairs"] = {
        "auc": vm.auc_smaller_is_positive(rv[cand_mask & truth], rv[cand_mask & ~truth]),
        "true_candidate_pairs": int((cand_mask & truth).sum()), "non_true_candidate_pairs": int((cand_mask & ~truth).sum())}
    out["primary_auc_all_pairs"] = vm.auc_smaller_is_positive(rv[truth], rv[~truth])
    out["true_pairs"] = int(truth.sum())
    out["positive_edges_among_true_pairs"] = int((rv[truth] < e2e.TAU_1C).sum())
    out["positive_edges_among_non_true_pairs"] = int((rv[~truth] < e2e.TAU_1C).sum())
    ctx.save(name, out)


# ----------------------------------------------------------------------------- aggregation
def _stats(vals):
    a = np.array(vals, dtype=np.float64)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def aggregate(parts: dict, sizes: list[int], variant_n: int | None, draws=(1, 2, 3)) -> dict:
    """parts: {'ladder_d1': ..., 'variant_d1': ..., 'd3b_d1': ..., 'ext_d1': ..., 'e2e_d1': ..., 'verify_d1': ...}.
    Gates (Addendum B.4): per cell recall >= 0.95, reduction >= 0.90, per-transformation recall >= 0.85;
    strict = every cell passes. M2_A1 is non-gated."""
    out: dict = {"gated_methods": list(GATED), "non_gated_comparison": ["M2_A1"], "methods": {}}
    for m, name in ev.PRIMARY.items():
        cells = {}
        for d in draws:
            for n in sizes:
                s = parts[f"ladder_d{d}"]["sizes"][str(n)]["score"]["sets"][name]
                cells[f"ladder_n{n}_d{d}"] = s
            if variant_n is not None and f"variant_d{d}" in parts:
                cells[f"variant_n{variant_n}_d{d}"] = parts[f"variant_d{d}"]["score"]["sets"][name]
        ladder_cells = {k: v for k, v in cells.items() if k.startswith("ladder")}
        entry = {
            "ladder_cells_pass": {k: v["gate"]["pass"] for k, v in ladder_cells.items()},
            "strict_pass_ladder": bool(all(v["gate"]["pass"] for v in ladder_cells.values())),
            "strict_pass_ladder_and_variant": bool(all(v["gate"]["pass"] for v in cells.values())),
            "cells_pass_all": {k: v["gate"]["pass"] for k, v in cells.items()},
            "above_random_control_every_cell": bool(all(v["above_b1"] for v in cells.values())),
            "by_size": {},
        }
        for n in sizes:
            vs = [cells[f"ladder_n{n}_d{d}"] for d in draws]
            entry["by_size"][str(n)] = {k: _stats([v[k] for v in vs]) for k in ("recall", "reduction", "min_recall_per_transform", "candidate_pairs")}
        if variant_n is not None and any(k.startswith("variant") for k in cells):
            vs = [v for k, v in cells.items() if k.startswith("variant")]
            entry["variant"] = {k: _stats([v[k] for v in vs]) for k in ("recall", "reduction", "min_recall_per_transform", "candidate_pairs")}
        d3b = []
        for d in draws:
            if f"d3b_d{d}" in parts:
                s = parts[f"d3b_d{d}"]["score"]["sets"][name]
                d3b.append({"draw": d, "recall": s["recall"], "reduction": s["reduction"], "above_b1": s["above_b1"],
                            "pass": bool(s["recall"] >= 0.95 and s["reduction"] >= 0.90)})
        if d3b:
            entry["d3b"] = {"per_draw": d3b, "strict_pass": bool(all(x["pass"] for x in d3b))}
        ext = [parts[f"ext_d{d}"]["score"]["sets"][name] for d in draws if f"ext_d{d}" in parts]
        if ext:
            entry["secondary_pool_not_gated"] = {k: _stats([v[k] for v in ext]) for k in ("recall", "reduction", "min_recall_per_transform", "candidate_pairs")}
        out["methods"][m] = entry
    gated = [out["methods"][m] for m in GATED]
    out["any_gated_method_passes_strict_ladder"] = bool(any(e["strict_pass_ladder"] for e in gated))
    out["all_gated_methods_fail_strict_ladder"] = not out["any_gated_method_passes_strict_ladder"]
    out["tau_b"] = {f"d{d}": parts[f"tau_b_d{d}"]["tau_b"] for d in draws if f"tau_b_d{d}" in parts}
    out["tau_b_extended_pool"] = {f"d{d}": parts[f"tau_b_ext_d{d}"]["tau_b"] for d in draws if f"tau_b_ext_d{d}" in parts}
    out["tau_al"] = {f"d{d}": parts[f"verify_d{d}"]["tau_al"] for d in draws if f"verify_d{d}" in parts}
    if all(f"e2e_d{d}" in parts for d in draws):
        out["end_to_end"] = {f"d{d}": parts[f"e2e_d{d}"] for d in draws}
    return out


PART_NAMES = ("tau_b_d{d}", "tau_b_ext_d{d}", "ladder_d{d}", "verify_d{d}", "variant_d{d}", "d3b_d{d}", "ext_d{d}", "e2e_d{d}")


def collect_parts(ctx: Ctx) -> dict:
    parts = {}
    for d in ctx.draws:
        for tmpl in PART_NAMES:
            nm = tmpl.format(d=d)
            if ctx.have(nm):
                parts[nm] = ctx.load(nm)
    return parts


def run_all(ctx: Ctx) -> dict:
    """The full sequence for the chosen mode; every part is resumable."""
    t0 = time.time()
    ctx.event("run_start", mode=ctx.mode, draws=list(ctx.draws), free_gb=free_gb())
    for d in ctx.draws:
        run_tau_b(ctx, d)
        run_tau_b(ctx, d, ext=True)
    for d in ctx.draws:
        run_ladder(ctx, d)
        run_verify(ctx, d)
    for d in ctx.draws:
        run_single_set(ctx, "variant", d)
    for d in ctx.draws:
        run_single_set(ctx, "d3b", d)
    for d in ctx.draws:
        run_single_set(ctx, "ext", d)
    for d in ctx.draws:
        run_e2e(ctx, d)
    res = aggregate(collect_parts(ctx), ctx.plan.sizes, ctx.plan.cfg["variant"][0], ctx.draws)
    ctx.save("results", res)
    ctx.event("run_done", seconds=round(time.time() - t0), free_gb=free_gb())
    return res
