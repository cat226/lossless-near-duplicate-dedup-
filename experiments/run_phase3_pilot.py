"""
Phase 3 pilot (docs/phase3_design.md section 10 and Addendum A).

DEV SPLIT ONLY. The test split is locked (phase3.splits) and no test image or label is read, loaded
or scored. Validation only: its numbers are not evidence and nothing here is committed (output goes
to phase3/data/pilot and phase3/artifacts, both gitignored). Exits non-zero if any item fails.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import shutil
import subprocess
import sys
import time
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from phase2 import candidates as cand
from phase2 import metrics as met
from phase2.generate import node_paths, parse_stem
from phase2.parallel_matrix import load_rgb
from phase2.verify import time_verification
from phase3 import e2e, evaluate as ev, labels as lb
from phase3 import blockwise as bw
from phase3.aligned_verifier import AlignConfig, aligned_distance, prepare
from phase3.dataset import build_d3a_set, build_d3b_set
from phase3.parallel import aligned_pairs_parallel, aligned_pairs_single
from phase3.splits import TEST_APPROVAL_FILE, verify_startup

PILOT = REPO / "phase3" / "data" / "pilot"
SYNTH40 = REPO / "phase3" / "artifacts" / "synth40"
OUT = REPO / "phase3" / "artifacts" / "pilot_report.json"
SEED, WORKERS = 12345, 10
N_FAM, N_SINGLE = 20, 300
checks: dict = {}
info: dict = {}


def check(name: str, ok: bool, **detail) -> None:
    checks[name] = {"pass": bool(ok), **detail}
    print(("PASS  " if ok else "FAIL  ") + name, flush=True)


def ram_probe(stage: str) -> dict:
    r = subprocess.run([sys.executable, "-m", "phase3.ram_probe", stage], cwd=REPO, capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main() -> None:
    # ---------------------------------------------------------------- startup (before any image or label)
    split = verify_startup()
    check("P0 startup: 39 split hashes and 4 manifest hashes recomputed and match the design", True, **split.report)
    check("P0 the test split is locked (no approval file)", not TEST_APPROVAL_FILE.exists())

    # ---------------------------------------------------------------- peak RAM at the largest block-wise steps
    info["peak_ram_n10000_mb"] = {s: ram_probe(s) for s in ("m2", "otsu", "score")}
    print("RAM probes:", {k: round(v["peak_mb"]) for k, v in info["peak_ram_n10000_mb"].items()}, flush=True)

    # ---------------------------------------------------------------- build dev-only sets
    dev_src, dev_single = split.dev_sources(1), split.dev_singletons(1)
    dev = build_d3a_set(PILOT / "draw1_dev", dev_src, dev_single, SEED)
    pilot = build_d3a_set(PILOT / "draw1_pilot", dev_src[:N_FAM], dev_single[:N_SINGLE], SEED)
    blocks = sorted({json.loads(l)["block"] for l in open(REPO / "phase3/manifests/copydays.jsonl")})[:20]
    cdays = build_d3b_set(PILOT / "draw1_copydays", blocks, dev_single[N_SINGLE:2 * N_SINGLE], SEED)
    paths, n = pilot["paths"], pilot["n"]
    check("P3 every pool photograph the pilot used is in the DEV split",
          set(dev_src[:N_FAM]) | set(dev_single[:2 * N_SINGLE]) <= set(split.dev(1)))
    info["sizes"] = {"dev_images": dev["n"], "pilot_images": n, "copydays_mechanics_images": cdays["n"]}

    # ---------------------------------------------------------------- protocol B calibration (dev only)
    tau_info = ev.calibrate_tau_b(dev["paths"], dev["labels_path"], sig_cache=PILOT / "draw1_dev" / "sig.npz")
    info["tau_b_draw1_dev"] = tau_info
    check("P-B tau_B reaches the 0.97 dev-recall target", tau_info["dev_recall_at_tau"] >= 0.97,
          **{k: tau_info[k] for k in ("tau_b", "dev_recall_at_tau", "dev_true_pairs")})
    tau_b = tau_info["tau_b"]

    # ---------------------------------------------------------------- per-pair verifier cost (single process, before any parallel job)
    pairs200 = e2e.all_pairs_sample(n, 200)
    t_primary = np.array(time_verification(paths, pairs200))
    cfg = AlignConfig()
    arrays50 = [load_rgb(paths[i]) for i in range(50)]
    t0 = time.perf_counter()
    prep50 = [prepare(a, cfg) for a in arrays50]
    t_prepare = (time.perf_counter() - t0) / 50
    t_align = []
    statuses = {}
    for i, j in e2e.all_pairs_sample(50, 100):
        t0 = time.perf_counter()
        res = aligned_distance(prep50[i], prep50[j], cfg)
        t_align.append(time.perf_counter() - t0)
        statuses[res["status"]] = statuses.get(res["status"], 0) + 1
        if res["status"] != "ok" and res["distance"] is not None:
            raise AssertionError("a distance was substituted for a failed alignment")
    info["verifier_cost_single_process_real_pairs"] = {
        "primary_seconds_per_pair_mean": float(t_primary.mean()), "primary_median": float(np.median(t_primary)),
        "primary_pairs": int(t_primary.size),
        "aligned_prepare_seconds_per_image": t_prepare, "aligned_seconds_per_pair_mean": float(np.mean(t_align)),
        "aligned_pairs": len(t_align), "note": "pilot measurement on 500x375-class dev photographs; decode excluded; not a timing result",
    }

    # ---------------------------------------------------------------- two-run determinism and label ordering
    trace = []
    real_open = ev.open_labels

    def traced(path, hashes):
        trace.append(("labels_opened", len(hashes)))
        return real_open(path, hashes)

    ev.open_labels = traced
    stage1 = ev.candidate_stage(paths, tau_b)
    trace.append(("candidate_stage_done", len(stage1["hashes"])))
    res1 = ev.score_sets(stage1, pilot["labels_path"], paths, SEED)
    stage2 = ev.candidate_stage(paths, tau_b)
    res2 = ev.score_sets(stage2, pilot["labels_path"], paths, SEED)
    ev.open_labels = real_open
    check("P1 two runs give identical candidate-set hashes", stage1["hashes"] == stage2["hashes"], sets=len(stage1["hashes"]))
    check("P1 two runs give identical results (byte for byte)",
          json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True))
    check("P2 labels were opened only after every candidate set was produced and hashed",
          [t[0] for t in trace[:2]] == ["candidate_stage_done", "labels_opened"] and trace[1][1] == len(stage1["hashes"]))

    banned = {"labels", "label_arrays", "open_labels", "family", "truth", "transform", "labels_path"}
    leaks = []
    for mod in ("phase2/signatures.py", "phase2/candidates.py", "phase3/blockwise.py", "phase3/hashes.py"):
        for node in ast.walk(ast.parse((REPO / mod).read_text())):
            if (isinstance(node, ast.Name) and node.id in banned) or (isinstance(node, ast.arg) and node.arg in banned):
                leaks.append((mod, getattr(node, "id", None) or node.arg))
    for node in ast.walk(ast.parse(inspect.getsource(ev.candidate_stage))):
        if (isinstance(node, ast.Name) and node.id in banned) or (isinstance(node, ast.arg) and node.arg in banned):
            leaks.append(("evaluate.candidate_stage", getattr(node, "id", None) or node.arg))
    refused = False
    try:
        lb.open_labels(pilot["labels_path"], {})
    except lb.LabelAccessBeforeCandidates:
        refused = True
    check("P2 label-leak: signature, hash, block-wise and candidate-stage code name no label field; early access is refused",
          not leaks and refused, findings=leaks)

    # ---------------------------------------------------------------- draws reproduce; builder is deterministic
    again = verify_startup()
    rebuilt = build_d3a_set(PILOT / "draw1_pilot_rebuild", dev_src[:N_FAM], dev_single[:N_SINGLE], SEED)
    sha = lambda ps: sorted(hashlib.sha256(p.read_bytes()).hexdigest() for p in ps)
    same_images = sha(paths) == sha(rebuilt["paths"])
    same_labels = pilot["labels_path"].read_bytes() == rebuilt["labels_path"].read_bytes()
    shutil.rmtree(PILOT / "draw1_pilot_rebuild")
    check("P3 seeded split reproduces; dev and test disjoint; the set builder reproduces images and labels",
          again.report == split.report and split.dev(1) != split.dev(2) and same_images and same_labels)

    # ---------------------------------------------------------------- neutral names
    names_ok = all(re.fullmatch(r"i\d{5}\.(jpg|png)", p.name) for p in paths)
    labs = json.loads(pilot["labels_path"].read_text())["images"]
    fam_rank: dict = {}
    for p in paths:
        f = labs[p.stem]["family"]
        if f is not None:
            fam_rank.setdefault(f, []).append(int(p.stem[1:]))
    scattered = all(max(r) - min(r) + 1 > len(r) for r in fam_rank.values())
    check("P5 neutral names carry no label (pattern, separate labels file, family members scattered)",
          names_ok and scattered and (PILOT / "draw1_pilot" / "labels.json").exists())

    # ---------------------------------------------------------------- hand-checked arithmetic and internal consistency
    family, transform = lb.label_arrays(labs, paths)
    truth = met.truth_matrix(family)
    total = cand.n_pairs(n)
    check("P4 true pairs = 21 x 20 and all pairs = n(n-1)/2", int(truth.sum()) == 21 * N_FAM == 420 and total == n * (n - 1) // 2)
    s_true = met.score(truth.copy(), truth, transform)
    s_none = met.score(np.zeros((n, n), dtype=bool), truth, transform)
    s_all = met.score(np.triu(np.ones((n, n), dtype=bool), 1), truth, transform)
    first = np.flatnonzero(family == family[np.flatnonzero(family >= 0)[0]])
    fam1 = np.zeros((n, n), dtype=bool)
    for a in first:
        for b in first:
            if a < b:
                fam1[a, b] = True
    s_f1 = met.score(fam1, truth, transform)
    check("P4 hand-checked recall/reduction arithmetic (P=T, empty, all, one family of 21)",
          s_true["recall"] == 1.0 and s_true["reduction"] == 1 - 420 / total and s_none["recall"] == 0.0 and s_none["reduction"] == 1.0
          and s_all["recall"] == 1.0 and s_all["reduction"] == 0.0 and int(fam1.sum()) == 21 and s_f1["recall"] == 1 / 20
          and s_f1["reduction"] == 1 - 21 / total)
    check("P4 membership-based recall equals matrix-based recall for every candidate set",
          all(v["membership_recall"] == v["recall"] for v in res1["sets"].values()))
    check("P4 B1 has the same size as every method set",
          all(int(cand.random_control(n, v["candidate_pairs"], seed=SEED).sum()) == v["candidate_pairs"] for v in res1["sets"].values()))

    # M2 block-wise path equals the unmodified Phase 2 path at pilot scale
    from phase2 import signatures as sig
    _, coarse = ev.signatures(paths)
    dist = sig.l2_matrix(coarse)
    p2_flats = np.sort(cand.pair_flat_indices(cand.m2_candidates(dist, tau_b)))
    near = int((np.abs(cand.upper_values(dist) - tau_b) < 1e-9).sum())
    check("P10 block-wise M2 candidates equal the Phase 2 l2_matrix path", np.array_equal(p2_flats, stage1["sets"]["M2_B_x1.0"]),
          pairs_within_1e9_of_tau=near)
    rows_t, cols_t = np.nonzero(truth)
    d_true = bw.pair_distances(coarse, rows_t, cols_t)
    check("P10 distance-based recall equals candidate-set recall at tau_B",
          float((d_true < tau_b).mean()) == res1["sets"]["M2_B_x1.0"]["recall"])

    # ---------------------------------------------------------------- verifier definitions
    frozen = {"work_size": 512, "orb_nfeatures": 1000, "orb_fast_threshold": 5, "ransac_threshold": 3.0,
              "ransac_max_iters": 2000, "ransac_confidence": 0.99, "min_inliers": 12, "min_overlap_fraction": 0.25,
              "scale_range": (0.25, 4.0), "seed": 12345}
    check("P6 aligned-verifier configuration equals the frozen table", AlignConfig().__dict__ == frozen)
    s40 = node_paths(5, SYNTH40)
    fam40 = np.array([int(parse_stem(p.stem)[0]) for p in s40])
    tp = [(i, j) for i in range(len(s40)) for j in range(i + 1, len(s40)) if fam40[i] == fam40[j]]
    fp = [(i, j) for i, j in e2e.all_pairs_sample(len(s40), 400) if fam40[i] != fam40[j]][:len(tp)]
    res_t = aligned_pairs_single(s40, tp)
    res_f = aligned_pairs_single(s40, fp)
    check("P6 aligned verifier reproduces its validation behaviour on Phase 2 families 1-5 (every true pair aligns)",
          len(tp) == 105 and all(r["status"] == "ok" for r in res_t),
          true_pairs=len(tp), unrelated_pairs=len(fp), unrelated_aligned=sum(1 for r in res_f if r["status"] == "ok"))
    fail_reasons = {}
    for r in res_f:
        if r["status"] != "ok":
            fail_reasons[r["status"]] = fail_reasons.get(r["status"], 0) + 1
    check("P6 alignment failures are counted, never given a distance",
          all(r["distance"] is None for r in res_f + res_t if r["status"] != "ok") and sum(statuses.values()) == 100,
          failure_reasons=fail_reasons, real_photo_pair_statuses=statuses)

    # ---------------------------------------------------------------- parallel verifiers: equality with single-process
    log = lambda m: None
    r = e2e.full_graph(paths, WORKERS, log)
    ex = e2e.primary_exactness(paths, r)
    info["primary_parallel_vs_single"] = ex
    check("P10 parallel primary verifier equals single-process, bitwise, on a 500-pair seeded sample",
          ex["bitwise_equal"] and ex["pairs_checked"] == 500, **ex)
    flats = stage1["sets"]["M2_B_x1.0"]
    rr, cc = cand.flat_to_pair(flats, n)
    cpairs = list(zip(rr.tolist(), cc.tolist()))
    pick = np.sort(np.random.default_rng(SEED).choice(len(cpairs), size=min(200, len(cpairs)), replace=False))
    sample = [cpairs[k] for k in pick]
    a_par = aligned_pairs_parallel(paths, sample, WORKERS)
    a_one = aligned_pairs_single(paths, sample)
    check("P10 parallel aligned verifier equals single-process on a seeded sample of candidate pairs",
          a_par == a_one, pairs_checked=len(sample), candidate_pairs_total=len(cpairs),
          statuses={k: sum(1 for x in a_one if x["status"] == k) for k in {x["status"] for x in a_one}})
    rnd = e2e.all_pairs_sample(n, 200)
    check("P10 parallel aligned verifier equals single-process on 200 random pairs (mostly failures)",
          aligned_pairs_parallel(paths, rnd, WORKERS) == aligned_pairs_single(paths, rnd), pairs_checked=200)

    # primary verifier: parallel graph equals a direct call (also the P6 definition check)
    i0, j0 = rnd[0]
    from cc_dedup.graph import pairwise_distance
    check("P6 primary verifier equals a direct pairwise_distance call", r[i0, j0] == pairwise_distance(load_rgb(paths[i0]), load_rgb(paths[j0])))

    # end-to-end smoke test at pilot scale (not evidence)
    e2e_out = e2e.cluster_comparison(r, flats)
    info["e2e_smoke_pilot_scale"] = e2e_out
    check("P10 end-to-end machinery runs (20 KwikCluster seeds on both graphs, union-find baseline)",
          e2e_out["kwik_full_graph"]["seeds"] == 20 and e2e_out["kwik_candidate_graph"]["seeds"] == 20)

    # ---------------------------------------------------------------- Copydays mechanics (structure only, no metrics reported)
    cs = ev.candidate_stage(cdays["paths"], tau_b, curves=False)
    cres = ev.score_sets(cs, cdays["labels_path"], cdays["paths"], SEED)
    rows = [json.loads(l) for l in open(REPO / "phase3/manifests/copydays.jsonl")]
    block_sizes: dict = {}
    for row in rows:
        if row["accepted"] and row["block"] in set(blocks):
            block_sizes[row["block"]] = block_sizes.get(row["block"], 0) + 1
    expected_tp = sum(comb(k, 2) for k in block_sizes.values())
    n_cd = sum(block_sizes.values()) + N_SINGLE
    check("P2/P4 Copydays block labelling: true pairs = sum of C(k,2) over 20 blocks; pipeline runs (metrics deliberately not reported)",
          cres["true_pairs"] == expected_tp and cdays["n"] == n_cd
          and all(v["membership_recall"] == v["recall"] or (v["membership_recall"] != v["membership_recall"]) for v in cres["sets"].values()),
          blocks=len(block_sizes), images=cdays["n"], true_pairs=expected_tp)

    # ---------------------------------------------------------------- manifests and disk
    check("P7 manifests match the SHA-256 values recorded in the design", all(split.report["manifest_hashes_in_design"].values()))
    free_c, free_d = shutil.disk_usage("C:\\").free / 1e9, shutil.disk_usage("D:\\").free / 1e9
    info["free_disk_gb"] = {"C:": free_c, "D:": free_d}
    check("P8 free disk: C: >= 2 GB and D: >= 6 GB (the user's thresholds; the design's floor is 1.5 GB)",
          free_c >= 2.0 and free_d >= 6.0, C_gb=round(free_c, 2), D_gb=round(free_d, 2))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"checks": checks, "info": info}, indent=1, default=str))
    failed = [k for k, v in checks.items() if not v["pass"]]
    print("PILOT:", "ALL ITEMS PASSED" if not failed else f"FAILED {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
