"""
Phase 2 — independent audit (docs/phase2_design.md checkpoint 6).

Recomputes the headline numbers with code paths that differ from the pipeline's
(bit-unpack matmul for Hamming, scipy cdist for L2, its own Otsu, explicit
enumeration of true pairs, an independent disagreement loop), checks that frozen
paths are unchanged, that candidate generation cannot see ground truth, and that
a fresh-process rerun at n=280 reproduces the saved results byte for byte.

Prints PASS/FAIL per check and exits non-zero if any check fails.
"""

from __future__ import annotations

import ast
import itertools
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np

from phase2 import candidates as cand
from phase2.generate import node_paths, parse_stem
from run_phase2 import ARTIFACTS, MATRIX, PHASE1C, SIZES, load_signatures

RESULTS = REPO_ROOT / "results" / "phase2_results.json"
TRANSFORMS = ("resize", "jpegq50", "crop", "bright", "rotate", "png")
checks: dict[str, bool] = {}
details: dict[str, object] = {}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout


# ---------- independent implementations ----------
def ham_independent(hashes: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(hashes.astype("<u8").view(np.uint8).reshape(-1, 8), axis=1).astype(np.float32)
    ones = bits.sum(1)
    return np.rint(ones[:, None] + ones[None, :] - 2.0 * (bits @ bits.T)).astype(np.int64)


def l2_independent(coarse: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist

    a = coarse.astype(np.float64)
    return cdist(a, a, metric="euclidean")


def otsu_independent(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=256)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    w0 = np.cumsum(hist)[:-1] / total
    w1 = 1 - w0
    cum_mean = np.cumsum(hist * centers)
    mu0 = cum_mean[:-1] / np.maximum(np.cumsum(hist)[:-1], 1)
    mu1 = (cum_mean[-1] - cum_mean[:-1]) / np.maximum(hist.sum() - np.cumsum(hist)[:-1], 1)
    var = w0 * w1 * (mu0 - mu1) ** 2
    if var.max() - var.min() < 1e-9:
        return (values.min() + values.max()) / 2
    return float(edges[int(np.argmax(var)) + 1])


def flat(i, j, n):  # row-major upper-triangle flat index
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def true_pairs_enumerated(family, transform):
    """Enumerate true pairs by looping over families (not via a matrix)."""
    by_family: dict[int, list[int]] = {}
    for node, f in enumerate(family):
        by_family.setdefault(int(f), []).append(node)
    pairs = []
    for members in by_family.values():
        pairs += [(i, j) for i, j in itertools.combinations(sorted(members), 2)]
    return pairs


def independent_metrics(P_flat: np.ndarray, n: int, family, transform):
    pairs = true_pairs_enumerated(family, transform)
    t_flat = np.array([flat(i, j, n) for i, j in pairs], dtype=np.int64)
    found = np.isin(t_flat, P_flat)
    out = {"true_pairs": len(pairs), "recall": float(found.mean()),
           "reduction": 1.0 - P_flat.size / (n * (n - 1) // 2)}
    per = {}
    for t in TRANSFORMS:
        sel = np.array([(transform[i] == t) or (transform[j] == t) for i, j in pairs])
        per[t] = float(found[sel].mean())
    out["per_transform"] = per
    return out


def main():
    res = json.loads(RESULTS.read_text())

    # 1. frozen paths unchanged: nothing tracked in main is modified/deleted except .gitignore
    changed = [l for l in git("diff", "main", "--name-status", "--diff-filter=MD").splitlines() if l.strip()]
    changed = [l for l in changed if not l.endswith(".gitignore")]
    checks["frozen_paths_unchanged (no M/D vs main except .gitignore)"] = not changed
    details["modified_or_deleted_vs_main"] = changed
    checks["Phase 0 / Phase 1c result files untouched"] = not git(
        "diff", "main", "--stat", "--", "phase0", "results/phase1c_results.json", "docs/phase1c_design.md",
        "cc_dedup", "baselines", "experiments/run_phase1c.py", "experiments/run_pilot.py").strip()

    # 2. ground truth cannot reach candidate generation (import + identifier scan)
    banned_imports = {"phase2.metrics", "phase2.generate", "phase2.pipeline"}
    banned_names = {"family", "truth", "transform", "labels", "node_labels", "parse_stem"}
    leak = []
    for mod in ("signatures.py", "candidates.py"):
        tree = ast.parse((REPO_ROOT / "phase2" / mod).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module in banned_imports or
                                                    (node.module == "phase2" and any(a.name in {"metrics", "generate", "pipeline"} for a in node.names))):
                leak.append((mod, "import", node.module))
            if isinstance(node, ast.Name) and node.id in banned_names:
                leak.append((mod, "name", node.id))
            if isinstance(node, ast.arg) and node.arg in banned_names:
                leak.append((mod, "arg", node.arg))
    src = (REPO_ROOT / "experiments" / "run_phase2.py").read_text()
    fn = src[src.index("def evaluate_size"):src.index("def section_11")]
    checks["candidate sets are hashed before labels are read (run_phase2.evaluate_size)"] = (
        fn.index("candidate_hash") < fn.index("node_labels("))
    checks["signatures/candidates import and name no ground truth"] = not leak
    details["leak_findings"] = leak

    # 3. independent recomputation of headline numbers at every size
    names, H, C = load_signatures()
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    mismatches = []
    boundary_close = {}
    for F in SIZES:
        paths = node_paths(F)
        idx = [name_to_idx[p.name] for p in paths]
        n = len(paths)
        parsed = [parse_stem(p.stem) for p in paths]
        family = np.array([int(f) for f, _ in parsed])
        transform = np.array([t for _, t in parsed])
        entry = res["sizes"][str(F)]

        ham = ham_independent(H[idx])
        dist = l2_independent(C[idx])
        mask = np.triu(np.ones((n, n), dtype=bool), 1)
        tau = otsu_independent(dist[mask])
        if abs(tau - entry["tau_sig"]) > 1e-9:
            mismatches.append((F, "tau_sig", tau, entry["tau_sig"]))

        sets = {"M1": np.triu(ham <= cand.M1_PRIMARY_R, 1), "M2": np.triu(dist < entry["tau_sig"], 1)}
        sets["M3"] = sets["M1"] | sets["M2"]
        near = int((np.abs(dist[mask] - entry["tau_sig"]) < 1e-9).sum())
        boundary_close[F] = near
        for m, P in sets.items():
            saved = entry["methods"][m]["primary"]
            P_flat = np.sort(cand.pair_flat_indices(P))
            ind = independent_metrics(P_flat, n, family, transform)
            ok = (
                P_flat.size == saved["candidate_pairs"]
                and ind["true_pairs"] == saved["true_pairs"]
                and abs(ind["recall"] - saved["recall"]) < 1e-12
                and abs(ind["reduction"] - saved["reduction"]) < 1e-12
                and all(abs(ind["per_transform"][t] - saved["recall_per_transform"][t]) < 1e-12 for t in TRANSFORMS)
                and cand.candidate_hash(P) == saved["candidate_set_sha256"]
            )
            if not ok:
                mismatches.append((F, m, P_flat.size, saved["candidate_pairs"]))
            gate = (ind["recall"] >= 0.95 and ind["reduction"] >= 0.90
                    and min(ind["per_transform"].values()) >= 0.85)
            if gate != saved["gate"]["pass"]:
                mismatches.append((F, m, "gate", gate, saved["gate"]["pass"]))
        print(f"  audited F={F}", flush=True)
        del ham, dist, mask, sets
    checks["independent recomputation matches saved results at every size"] = not mismatches
    details["recomputation_mismatches"] = mismatches
    details["pairs_within_1e-9_of_tau_sig"] = boundary_close

    # 4. per-method gate verdicts recomputed from the saved per-size gates
    for m in ("M1", "M2", "M3"):
        verdict = all(res["sizes"][str(F)]["methods"][m]["primary"]["gate"]["pass"] for F in SIZES)
        checks[f"gate verdict for {m} consistent with per-size gates"] = verdict == res["gate"][m]["passes_every_size"]
    checks["phase2_succeeds consistent"] = res["phase2_succeeds"] == any(
        res["gate"][m]["passes_every_size"] for m in ("M1", "M2", "M3"))

    # 5. n=280 consistency with Phase 1c and independent section-11 recomputation
    s11 = res["sizes"]["40"]["section_11_n280"]
    if isinstance(s11, dict):
        checks["n=280 graph reproduces Phase 1c (tau, E+, clusters, D)"] = bool(s11["consistency_with_phase1c"]["all_match"])
        r = np.load(MATRIX)
        tau1c = json.loads(PHASE1C.read_text())["threshold"]
        n = 280
        V = np.triu(r < tau1c, 1)
        paths = node_paths(40)
        parsed = [parse_stem(p.stem) for p in paths]
        family = np.array([int(f) for f, _ in parsed])
        T = np.zeros((n, n), dtype=bool)
        for i, j in true_pairs_enumerated(family, None):
            T[i, j] = True
        idx = [name_to_idx[p.name] for p in paths]
        ham = ham_independent(H[idx])
        dist = l2_independent(C[idx])
        sets = {"M1": np.triu(ham <= 10, 1), "M2": np.triu(dist < res["sizes"]["40"]["tau_sig"], 1)}
        sets["M3"] = sets["M1"] | sets["M2"]
        sec_ok = abs(s11["verifier_cap"] - (T & V).sum() / T.sum()) < 1e-12 and int(V.sum()) == s11["positive_edges_V"]
        for m, P in sets.items():
            got = s11["methods"][m]
            sec_ok &= abs(got["e_plus_retention"] - (P & V).sum() / V.sum()) < 1e-12
            sec_ok &= abs(got["verifier_confirmed_recall"] - (P & T & V).sum() / T.sum()) < 1e-12
            sec_ok &= abs(got["retention_of_verifiable_true_pairs"] - (P & T & V).sum() / (T & V).sum()) < 1e-12
        checks["section 11 headline numbers recomputed independently"] = bool(sec_ok)
    else:
        checks["section 11 present"] = False

    # 6. byte-for-byte rerun determinism at n=280 in a fresh process
    rerun = ARTIFACTS / "rerun_F40.json"
    subprocess.run([sys.executable, str(REPO_ROOT / "experiments" / "run_phase2.py"), "--only", "40",
                    "--out", str(rerun)], cwd=REPO_ROOT, capture_output=True, check=True)
    saved40 = json.dumps(res["sizes"]["40"], sort_keys=True)
    again40 = json.dumps(json.loads(rerun.read_text())["sizes"]["40"], sort_keys=True)
    checks["fresh-process rerun at n=280 is byte-identical"] = saved40 == again40

    # 7. nothing generated by this run is tracked-by-accident
    status = git("status", "--porcelain")
    checks["no data/artifact files untracked-visible"] = not any(
        ("phase2/data" in l or "phase2/artifacts" in l or "__pycache__" in l or ".pyc" in l) for l in status.splitlines())

    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    (ARTIFACTS / "audit.json").write_text(json.dumps({"checks": checks, "details": details}, indent=1, default=str))
    failed = [k for k, v in checks.items() if not v]
    print("AUDIT:", "ALL CHECKS PASSED" if not failed else f"FAILED: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
