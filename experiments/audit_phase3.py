"""
Phase 3 audit (docs/phase3_design.md section 11).

    python experiments/audit_phase3.py --mode rehearsal      # validates the audit itself on the dev-only rehearsal
    python experiments/audit_phase3.py --mode full

Independent recomputation: candidate sets (SHA-256), recall, reduction, per-transformation recall and
gate verdicts at every rung and draw are rebuilt from the persisted per-image features and the label
file with DIFFERENT code from the run (lookup-table popcount instead of np.bitwise_count, scipy cdist
instead of the Gram identity, plain index arithmetic instead of phase2.metrics/candidates/blockwise).
Also: tau_B recomputed from the dev files; split hashes recomputed with separate code; a fresh-process
byte-identical re-run of one configuration; git diff of every frozen path; a label-leak scan; label
opening always after candidate hashing; tau_B written before the first test-split image was touched.

Verifier metrics (parallel runs) are covered by the bitwise exactness assertions recorded in the
results, not recomputed here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
from scipy.spatial.distance import cdist

from cc_dedup.graph import compute_otsu_threshold  # Phase 1c Otsu, unchanged (the run used its own block-wise version)

POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
TRANSFORMS = ("resize", "jpegq50", "crop", "bright", "rotate", "png")
checks: dict = {}


def check(name: str, ok: bool, **detail) -> None:
    checks[name] = {"pass": bool(ok), **detail}
    print(("PASS  " if ok else "FAIL  ") + name, flush=True)


# ----------------------------------------------------------------------------- independent primitives
def flat_idx(i, j, n):
    i = np.asarray(i, dtype=np.int64)
    return i * n - i * (i + 1) // 2 + (np.asarray(j, dtype=np.int64) - i - 1)


def sha_of(flat: np.ndarray) -> str:
    return hashlib.sha256(np.sort(flat).astype("<i8").tobytes()).hexdigest()


def hamming_flats(h: np.ndarray, radii, block: int = 256) -> dict:
    n = len(h)
    out = {R: [] for R in radii}
    for s in range(0, n, block):
        e = min(n, s + block)
        x = h[s:e, None] ^ h[None, :]
        d = POP8[x.view(np.uint8).reshape(e - s, n, 8)].sum(axis=2, dtype=np.int32)
        gi = np.arange(s, e)[:, None]
        upper = np.arange(n)[None, :] > gi
        for R in radii:
            ii, jj = np.nonzero((d <= R) & upper)
            out[R].append(flat_idx(ii + s, jj, n))
    return {R: np.sort(np.concatenate(v)) for R, v in out.items()}


def distances_upper(a: np.ndarray, block: int = 256) -> np.ndarray:
    n = len(a)
    parts = []
    for s in range(0, n - 1, block):
        e = min(n, s + block)
        d = cdist(a[s:e], a)
        upper = np.arange(n)[None, :] > np.arange(s, e)[:, None]
        parts.append(d[upper])
    return np.concatenate(parts)


def m2_flats(a: np.ndarray, taus, block: int = 256) -> dict:
    n = len(a)
    out = {t: [] for t in taus}
    for s in range(0, n, block):
        e = min(n, s + block)
        d = cdist(a[s:e], a)
        upper = np.arange(n)[None, :] > np.arange(s, e)[:, None]
        for t in taus:
            ii, jj = np.nonzero((d < t) & upper)
            out[t].append(flat_idx(ii + s, jj, n))
    return {t: np.sort(np.concatenate(v)) for t, v in out.items()}


def near_threshold_counts(a: np.ndarray, taus: dict, block: int = 256, width: float = 1e-9) -> dict:
    """Number of pairs (i < j) whose Euclidean distance lies within `width` of each named threshold."""
    n = len(a)
    out = {k: 0 for k in taus}
    for s in range(0, n, block):
        e = min(n, s + block)
        d = cdist(a[s:e], a)
        upper = np.arange(n)[None, :] > np.arange(s, e)[:, None]
        for k, t in taus.items():
            out[k] += int((np.abs(d - t) < width)[upper].sum())
    return out


def exact_flats(sha: np.ndarray) -> np.ndarray:
    groups: dict = {}
    for i, h in enumerate(sha.tolist()):
        groups.setdefault(h, []).append(i)
    n = len(sha)
    fl = [flat_idx(i, j, n) for g in groups.values() for k, i in enumerate(g) for j in g[k + 1:]]
    return np.sort(np.array(fl, dtype=np.int64))


def true_info(labels: dict, paths: list):
    fam, trans = {}, []
    for k, p in enumerate(paths):
        lab = labels[p.stem]
        trans.append(lab["transform"])
        if lab["family"] is not None:
            fam.setdefault(lab["family"], []).append(k)
    a, b = [], []
    for members in fam.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a.append(members[x]), b.append(members[y])
    a, b = np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)
    n = len(paths)
    return {"a": a, "b": b, "flat": flat_idx(a, b, n), "trans": np.array(trans), "n": n}


def ratio(num, den):
    return float(num) / float(den) if den else float("nan")


def score_independent(flat: np.ndarray, T: dict) -> dict:
    n = T["n"]
    total = n * (n - 1) // 2
    hit = np.isin(T["flat"], flat)
    tr = T["trans"]
    ta, tb = tr[T["a"]], tr[T["b"]]
    orig_a, orig_b = ta == "original", tb == "original"
    ot, tt = orig_a ^ orig_b, ~orig_a & ~orig_b
    per = {t: ratio(hit[(ta == t) | (tb == t)].sum(), ((ta == t) | (tb == t)).sum()) for t in TRANSFORMS}
    s = {"candidate_pairs": int(flat.size), "recall": ratio(hit.sum(), hit.size), "reduction": 1.0 - ratio(flat.size, total),
         "recall_original_transform": ratio(hit[ot].sum(), ot.sum()), "recall_transform_transform": ratio(hit[tt].sum(), tt.sum()),
         "recall_per_transform": per}
    mn = min(per.values())
    s["min_recall_per_transform"] = mn
    s["gate_pass"] = bool(s["recall"] >= 0.95 and s["reduction"] >= 0.90 and mn >= 0.85)
    return s


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


# ----------------------------------------------------------------------------- audit of one evaluated set
def audit_cell(tag, stored_score, stored_hashes, feats, labels, paths, npz, n_key, all_sets) -> dict:
    n = len(paths)
    T = true_info(labels, paths)
    radii = (4, 6, 8, 10, 12, 14, 16) if all_sets else (10,)
    ham_m1, ham_dh = hamming_flats(feats["phash"], radii), hamming_flats(feats["dhash"], radii)
    a = feats["coarse"].astype(np.float64)
    tau_b = stored_score["tau"]["tau_b"]
    factors = (0.5, 0.75, 1.0, 1.25, 1.5) if all_sets else (1.0,)
    m2 = m2_flats(a, [tau_b * f for f in factors])
    tau_a1, method_a1 = compute_otsu_threshold(distances_upper(a))
    a1 = m2_flats(a, [tau_a1])[tau_a1]
    mine = {f"M1_R{R}": v for R, v in ham_m1.items()}
    mine.update({f"dHash_R{R}": v for R, v in ham_dh.items()})
    mine.update({f"M2_B_x{f}": m2[tau_b * f] for f in factors})
    mine["EXACT"] = exact_flats(feats["sha256"])
    mine["M2_A1"] = a1
    mine["M3"] = np.union1d(mine["M1_R10"], mine["M2_B_x1.0"])
    # Persisted sets (the run's own) versus the independent ones. The two distance routines (Gram identity in the run,
    # direct Euclidean norm here) can round differently for a pair whose distance equals a threshold to ~1e-12; such a
    # pair may fall on either side. A difference is accepted ONLY if every differing pair lies within 1e-9 of the
    # threshold of that set; the independent scoring below then runs on the run's persisted set. Hamming-based sets and
    # the exact-hash set must be identical.
    short = {"M1": "M1_R10", "dHash": "dHash_R10", "M2_B": "M2_B_x1.0"}
    stored = {}
    for key in npz.files:
        if key.startswith(f"n{n_key}__"):
            nm = key.split("__", 1)[1]
            stored[short.get(nm, nm)] = npz[key]
    thresholds = {f"M2_B_x{f}": tau_b * f for f in factors}
    thresholds.update({"M2_A1": tau_a1, "M3": tau_b})
    near_counts = near_threshold_counts(a, {k: v for k, v in thresholds.items() if k != "M3"})  # M3 shares the tau_B pairs
    starts = np.arange(n, dtype=np.int64) * n - np.arange(n, dtype=np.int64) * (np.arange(n, dtype=np.int64) + 1) // 2
    mismatch_npz, near_flips = [], {}
    for name in list(mine):
        st = stored.get(name)
        if st is None or np.array_equal(st, mine[name]):
            continue
        thr = thresholds.get(name)
        if thr is None:
            mismatch_npz.append(name)
            continue
        diff = np.setxor1d(st, mine[name])
        i = np.searchsorted(starts, diff, side="right") - 1
        j = diff - starts[i] + i + 1
        dist = np.linalg.norm(a[i] - a[j], axis=1)
        if np.all(np.abs(dist - thr) < 1e-9):
            near_flips[name] = int(diff.size)
            mine[name] = st
        else:
            mismatch_npz.append(name)
    bad_hash, bad_score = [], []
    for name, flat in mine.items():
        if sha_of(flat) != stored_hashes[name]:
            bad_hash.append(name)
        sc = score_independent(flat, T)
        st = stored_score["sets"][name]
        for k in ("candidate_pairs", "recall", "reduction", "recall_original_transform", "recall_transform_transform",
                  "recall_per_transform", "min_recall_per_transform"):
            if not same(sc[k], st[k]):
                bad_score.append((name, k))
        if sc["gate_pass"] != st["gate"]["pass"]:
            bad_score.append((name, "gate"))
    tau_ok = abs(tau_a1 - stored_score["tau"]["a1"]) < 1e-9 and method_a1 == stored_score["tau"]["a1_method"]
    return {"tag": tag, "images": n, "true_pairs": int(T["flat"].size), "sets_recomputed": len(mine), "hash_mismatch": bad_hash,
            "score_mismatch": bad_score, "persisted_set_mismatch": mismatch_npz, "near_threshold_flips": near_flips, "near_threshold_pairs_within_1e-9": near_counts,
            "a1_threshold_equal": bool(tau_ok),
            "true_pairs_equal": T["flat"].size == stored_score["true_pairs"]}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rehearsal", "full"], required=True)
    ap.add_argument("--skip-rerun", action="store_true")
    args = ap.parse_args()
    from phase3 import runner
    from phase3.splits import DESIGN, DRAW_SEEDS, MANIFESTS, REPO as SPLIT_REPO, TEST_APPROVAL_FILE, verify_startup

    split = verify_startup()
    ctx = runner.Ctx(args.mode, split, tag="")
    res_path = ctx.art / "results.json"
    results = json.loads(res_path.read_text())
    parts = runner.collect_parts(ctx)
    check("A0 results file and every part exist", all(f"ladder_d{d}" in parts and f"e2e_d{d}" in parts and f"verify_d{d}" in parts
                                                       and f"variant_d{d}" in parts and f"d3b_d{d}" in parts and f"ext_d{d}" in parts
                                                       for d in ctx.draws))

    # -- 1. splits recomputed with separate code
    doc = DESIGN.read_text(encoding="utf-8")
    pool = [json.loads(l)["name"] for l in open(MANIFESTS / "distractor_pool.jsonl")]
    ok_split, lists = True, {}
    for d, seed in DRAW_SEEDS.items():
        block = doc[doc.index(f"draw {d} (seed {seed})"):]
        block = block[: block.index("\n\n")]
        rec = {m.group(1): m.group(2) for m in re.finditer(r"^\s{2}(\w+)\s+(?:perm\[[^\]]*\]\s+)?([0-9a-f]{64})\s*$", block, re.M)}
        perm = np.random.default_rng(seed).permutation(10000)
        pn = [pool[i] for i in perm]
        h = lambda names: hashlib.sha256("\n".join(names).encode()).hexdigest()
        want = {"perm_sha256": hashlib.sha256(perm.astype("<i8").tobytes()).hexdigest(), "dev": h(pn[:1500]), "dev_sources": h(pn[:150]),
                "dev_singletons": h(pn[150:1500]), "test_sources": h(pn[1500:2000]), "test_singletons": h(pn[2000:8500]),
                "d3b_singletons": h(pn[1500:6114]), "variant_sources": h(pn[1500:1614]), "variant_singletons": h(pn[1614:8816])}
        ok_split &= all(rec.get(k) == v for k, v in want.items())
        lists[d] = {"dev": set(pn[:1500]), "test": set(pn[1500:])}
        ok_split &= not lists[d]["dev"] & lists[d]["test"]
    check("A1 split hashes recomputed with separate code equal the design; dev and test disjoint", ok_split)

    # -- 2. independent recomputation of every cell
    cells = []
    for d in ctx.draws:
        paths_L, nodes, labels_path = runner.ladder_nodes(ctx, d)
        labels = json.loads(labels_path.read_text())["images"]
        feats_L = np.load(runner.ladder_dir(ctx, d) / "features.npz", allow_pickle=False)
        feats_L = {k: feats_L[k] for k in ("phash", "coarse", "dhash", "sha256")}
        npz = np.load(ctx.art / f"sets_ladder_d{d}.npz")
        for n in ctx.plan.sizes:
            nd = nodes[n]
            cells.append(audit_cell(f"ladder_d{d}_n{n}", {**parts[f"ladder_d{d}"]["sizes"][str(n)]["score"]},
                                    parts[f"ladder_d{d}"]["sizes"][str(n)]["candidate_hashes"],
                                    {k: v[nd] for k, v in feats_L.items()}, labels, [paths_L[i] for i in nd], npz, n,
                                    all_sets=(n == ctx.plan.sizes[0])))
            print("  audited", cells[-1]["tag"], "| pairs within 1e-9 of a threshold:", cells[-1]["near_threshold_pairs_within_1e-9"],
                  "| flips vs the run:", cells[-1]["near_threshold_flips"] or 0, flush=True)
        for kind, folder in (("variant", f"draw{d}_variant"), ("d3b", f"draw{d}_d3b"), ("ext", f"draw{d}_ext")):
            root = ctx.data / folder
            paths = sorted((root / "images").glob("i*"))
            lab = json.loads((root / "labels.json").read_text())["images"]
            f = np.load(root / "features.npz", allow_pickle=False)
            part = parts[f"{kind}_d{d}"]
            npz = np.load(ctx.art / f"sets_{kind}_d{d}.npz")
            cells.append(audit_cell(f"{kind}_d{d}", part["score"], part["candidate_hashes"], {k: f[k] for k in ("phash", "coarse", "dhash", "sha256")},
                                    lab, paths, npz, len(paths), all_sets=False))
            print("  audited", cells[-1]["tag"], "| pairs within 1e-9 of a threshold:", cells[-1]["near_threshold_pairs_within_1e-9"],
                  "| flips vs the run:", cells[-1]["near_threshold_flips"] or 0, flush=True)
    check("A2 candidate-set SHA-256 recomputed independently equals the run's for every audited set",
          all(not c["hash_mismatch"] for c in cells), cells_audited=len(cells), mismatches=[(c["tag"], c["hash_mismatch"]) for c in cells if c["hash_mismatch"]])
    check("A2 recall, reduction, original/transform splits, per-transformation recall and gate verdicts equal",
          all(not c["score_mismatch"] and c["true_pairs_equal"] and c["a1_threshold_equal"] for c in cells),
          mismatches=[(c["tag"], c["score_mismatch"]) for c in cells if c["score_mismatch"]])
    check("A2 persisted candidate sets equal the independent ones (differences only for pairs within 1e-9 of the set's threshold)",
          all(not c["persisted_set_mismatch"] for c in cells),
          near_threshold_flips={c["tag"]: c["near_threshold_flips"] for c in cells if c["near_threshold_flips"]})

    # -- 3. strict gate verdicts rebuilt from the recomputed gate flags
    agg_ok = True
    for m, name in __import__("phase3.evaluate", fromlist=["PRIMARY"]).PRIMARY.items():
        ladder_flags = []
        for d in ctx.draws:
            for n in ctx.plan.sizes:
                sc = next(c for c in cells if c["tag"] == f"ladder_d{d}_n{n}")
                ladder_flags.append(parts[f"ladder_d{d}"]["sizes"][str(n)]["score"]["sets"][name]["gate"]["pass"])
        agg_ok &= results["methods"][m]["strict_pass_ladder"] == all(ladder_flags)
    check("A3 aggregated strict ladder verdicts equal the conjunction of the per-cell verdicts", agg_ok)

    # -- 4. tau_B recomputed from the dev files
    tb_ok, tb_detail = True, {}
    for d in ctx.draws:
        for ext in (False, True):
            part = parts[f"tau_b_{'ext_' if ext else ''}d{d}"]
            root = ctx.data / f"draw{d}_{'ext_' if ext else ''}dev"
            z = np.load(root / "sig.npz", allow_pickle=False)
            labs = json.loads((root / "labels.json").read_text())["images"]
            paths = sorted((root / "images").glob("i*"))
            T = true_info(labs, paths)
            dd = np.array([np.linalg.norm(z["coarse"][i].astype(np.float64) - z["coarse"][j].astype(np.float64)) for i, j in zip(T["a"], T["b"])])
            q = float(np.quantile(dd, 0.97, method="higher"))
            tau = float(np.nextafter(q, np.inf))
            ok = abs(tau - part["tau_b"]) < 1e-9 and float((dd < part["tau_b"] + 1e-9).mean()) >= 0.97  # the quantile pair sits at tau_B to rounding
            tb_ok &= ok
            tb_detail[f"d{d}{'_ext' if ext else ''}"] = {"stored": part["tau_b"], "recomputed": tau}
    check("A4 tau_B recomputed from the dev files equals the stored value and reaches dev recall 0.97", tb_ok, **tb_detail)

    # -- 5. event order: hashes before labels; tau_B before the first test-split image
    ev = [json.loads(l) for l in open(ctx.art / "events.jsonl")]
    order_ok, seen = True, set()
    for e in ev:
        if e["event"] == "candidate_hashes_final":
            seen.add(e["part"])
        if e["event"] == "labels_opened" and e["part"] not in seen:
            order_ok = False
    check("A5 every labels_opened event is preceded by its candidate_hashes_final event", order_ok, events=len(ev))
    tau_idx = [k for k, e in enumerate(ev) if e["event"] == "tau_b_written"]
    first_set = next(k for k, e in enumerate(ev) if e["event"] == "ladder_built")
    unlock = [k for k, e in enumerate(ev) if e["event"] == "test_split_unlocked"]
    check("A5 tau_B (all draws, both pools) was written before the first ladder image was built"
          + (" and after the unlock" if args.mode == "full" else " (rehearsal: dev only)"),
          bool(tau_idx) and max(tau_idx) < first_set and (args.mode != "full" or (unlock and unlock[0] < min(tau_idx))))
    if args.mode == "full":
        check("A5 the approval file exists and records the head commit", TEST_APPROVAL_FILE.exists()
              and json.loads(TEST_APPROVAL_FILE.read_text()).get("head"))

    # -- 6. photographs used belong to the right split
    ok_use = True
    for d in ctx.draws:
        for folder in (f"draw{d}_ladder", f"draw{d}_variant", f"draw{d}_d3b"):
            labs = json.loads((ctx.data / folder / "labels.json").read_text())["images"]
            srcs = {v["source"] for v in labs.values() if not v["source"].startswith("copydays_")}
            if args.mode == "full":
                ok_use &= srcs <= lists[d]["test"] and not srcs & lists[d]["dev"]
            else:
                ok_use &= srcs <= lists[d]["dev"]
    check("A6 every photograph in the evaluated sets lies in the intended split (test for full, dev for rehearsal)", ok_use)

    # -- 7. frozen values
    from phase3.aligned_verifier import AlignConfig
    from phase3 import e2e
    phase1c = json.loads((REPO / "results" / "phase1c_results.json").read_text())["threshold"]
    check("A7 seeds, tau, aligned configuration and rung tables equal the frozen values",
          e2e.TAU_1C == phase1c == 0.5805796715948317 and tuple(DRAW_SEEDS.values()) == (12345, 12346, 12347)
          and e2e.SEEDS == tuple(range(12345, 12365)) and AlignConfig().__dict__["work_size"] == 512
          and runner.FULL["ladder"] == {1000: (50, 650), 2500: (125, 1625), 5000: (250, 3250), 10000: (500, 6500)}
          and runner.FULL["variant"] == (8000, 114, 7202) and runner.FULL["ext"] == (5000, 250, 3250))

    # -- 8. git: frozen paths
    frozen = ["phase0", "cc_dedup", "baselines", "phase2", "phase2b", "results/phase1c_results.json", "docs/phase1c_design.md",
              "docs/phase2_design.md", "docs/phase2b_design.md", "experiments/run_phase1c.py", "experiments/run_pilot.py"]
    existing = [p for p in frozen if (REPO / p).exists()]
    g = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    check("A8 git diff against the design commit is empty for every frozen path", not g("diff", "624283f", "--", *existing) and not g("status", "--porcelain", "--", *existing),
          paths=existing)

    # -- 9. label leak scan
    banned = {"labels", "label_arrays", "open_labels", "family", "truth", "transform", "labels_path"}
    leaks = []
    for mod in ("phase2/signatures.py", "phase2/candidates.py", "phase3/blockwise.py", "phase3/hashes.py", "phase3/features.py"):
        for node in ast.walk(ast.parse((REPO / mod).read_text())):
            if (isinstance(node, ast.Name) and node.id in banned) or (isinstance(node, ast.arg) and node.arg in banned):
                leaks.append((mod, getattr(node, "id", None) or node.arg))
    from phase3 import evaluate as ev_mod
    for node in ast.walk(ast.parse(inspect.getsource(ev_mod.candidate_stage))):
        if (isinstance(node, ast.Name) and node.id in banned) or (isinstance(node, ast.arg) and node.arg in banned):
            leaks.append(("evaluate.candidate_stage", getattr(node, "id", None) or node.arg))
    check("A9 no label field is reachable from signature, feature, hash, block-wise or candidate-stage code", not leaks, findings=leaks)

    # -- 10. fresh-process byte-identical re-run of one configuration (draw 1, smallest rung)
    if not args.skip_rerun:
        n0 = ctx.plan.sizes[0]
        cmd = [sys.executable, "experiments/run_phase3.py", "--mode", args.mode, "--tag", "_audit", "--draws", "1", "--only-sizes", str(n0)]
        if args.mode == "full":
            cmd.append("--go")
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        rerun = REPO / "phase3" / "artifacts" / f"run_{args.mode}_audit"
        same_tau = r.returncode == 0 and (rerun / "tau_b_d1.json").read_bytes() == (ctx.art / "tau_b_d1.json").read_bytes()
        new = json.loads((rerun / "ladder_d1.json").read_text())["sizes"][str(n0)] if r.returncode == 0 else None
        old = parts["ladder_d1"]["sizes"][str(n0)]
        same_cell = new is not None and json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)
        check("A10 a fresh process reproduces tau_B and the smallest-rung results of draw 1 byte for byte", same_tau and same_cell,
              returncode=r.returncode, stderr_tail=r.stderr[-300:] if r.returncode else "")
    all_passed = all(v["pass"] for v in checks.values())
    out = {"mode": args.mode, "all_passed": bool(all_passed), "checks": checks, "cells_audited": [c["tag"] for c in cells],
           "near_threshold_pairs_within_1e-9_per_cell": {c["tag"]: c["near_threshold_pairs_within_1e-9"] for c in cells},
           "near_threshold_flips_per_cell": {c["tag"]: c["near_threshold_flips"] for c in cells},
           "statements": [
               "The run's computation (block-wise Gram identity, phase3/blockwise.py, as pre-registered) is the OFFICIAL one. The audit's "
               "scipy cdist route is an independent cross-check and never replaces the run's candidate sets or results.",
               "A pair whose distance is within 1e-9 of a threshold may fall on either side of it between the two routes; the audit accepts a "
               "difference only for such pairs and then scores the run's persisted set. Per-cell counts of such pairs and of actual flips are recorded.",
               "Limitation: the verifier metrics of the parallel runs (true-pair verification, false-accept samples, tau_al, the end-to-end graph) are "
               "NOT recomputed by this audit; they are covered only by the bitwise exactness assertions recorded in the results (parallel vs "
               "single-process on seeded samples)."]}
    dest = REPO / ("phase3/artifacts/rehearsal_audit.json" if args.mode == "rehearsal" else "results/phase3_audit.json")
    dest.write_text(json.dumps(out, indent=1, sort_keys=True))
    for st in out["statements"]:
        print("NOTE:", st)
    print("AUDIT:", "ALL CHECKS PASSED" if all_passed else f"FAILED {[k for k, v in checks.items() if not v['pass']]}", "->", dest)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
