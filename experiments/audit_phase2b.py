"""
Phase 2b — independent audit (docs/phase2b_design.md section 10).

Recomputes the M2* candidate sets and headline numbers with different code (scipy cdist
for L2, enumerated true pairs), re-derives the gate, checks tau* provenance, Phase 2 /
Phase 2b family disjointness, that no Phase 2 file was edited, that candidate generation
cannot see ground truth, and that a fresh-process rerun at n=280 is byte-identical.

Prints PASS/FAIL per check and exits non-zero if any check fails.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np

from audit_phase2 import TRANSFORMS, git, independent_metrics, l2_independent
from phase2 import candidates as cand
from phase2.generate import parse_stem
from phase2b.data import DEFAULT_DATASET_DIR, FIRST_FRESH, PHASE2_FAMILIES, SIZES, TAU_STAR, node_paths
from run_phase2b import ARTIFACTS, load_signatures

PHASE2_TIP = "2151373"  # the audited Phase 2 branch tip
RESULTS = REPO_ROOT / "results" / "phase2b_results.json"
checks: dict[str, bool] = {}
details: dict[str, object] = {}


def main():
    res = json.loads(RESULTS.read_text())

    # 1. nothing from earlier phases was edited (additions are fine; .gitignore may change)
    changed = [l for l in git("diff", PHASE2_TIP, "--name-status", "--diff-filter=MD").splitlines()
               if l.strip() and not l.endswith(".gitignore")]
    checks["no tracked file modified/deleted since the Phase 2 tip (except .gitignore)"] = not changed
    details["modified_or_deleted_since_phase2_tip"] = changed
    checks["Phase 0/1c/2 code, designs and results untouched"] = not git(
        "diff", PHASE2_TIP, "--stat", "--", "phase0", "cc_dedup", "baselines", "phase2",
        "experiments/run_phase1c.py", "experiments/run_pilot.py", "experiments/run_phase2.py",
        "docs/phase1c_design.md", "docs/phase2_design.md", "docs/phase2_report.md",
        "results/phase1c_results.json", "results/phase2_results.json").strip()

    # 2. tau* provenance and family disjointness
    p2 = json.loads((REPO_ROOT / "results" / "phase2_results.json").read_text())
    checks["TAU_STAR == 0.5 x Phase 2 n=280 tau_sig (exact)"] = TAU_STAR == 0.5 * p2["sizes"]["40"]["tau_sig"]
    design = (REPO_ROOT / "docs" / "phase2b_design.md").read_text()
    checks["TAU_STAR equals the literal in the frozen design"] = (
        float(re.search(r"tau\* = (0\.\d+)", design).group(1)) == TAU_STAR)
    checks["results record tau_star unchanged"] = res["tau_star"] == TAU_STAR and all(
        res["sizes"][str(F)]["tau_star"] == TAU_STAR for F in SIZES)
    fresh_manifest = json.loads((DEFAULT_DATASET_DIR / "manifest.json").read_text())
    checks["fresh dataset verified 800 Phase 2 originals against the Phase 2 manifest"] = (
        fresh_manifest["phase2_families_verified_against_manifest"] == PHASE2_FAMILIES)
    fresh_families = {int(parse_stem(Path(k).stem)[0]) for k in fresh_manifest["sha256"]}
    checks["fresh families are exactly 801..1600 and disjoint from 1..800"] = (
        fresh_families == set(range(FIRST_FRESH, FIRST_FRESH + 800)) and not (fresh_families & set(range(1, 801))))
    p2_manifest = json.loads((REPO_ROOT / "phase2" / "data" / "F800" / "manifest.json").read_text())["sha256"]
    checks["no fresh image is byte-identical to a Phase 2 image"] = not (
        set(fresh_manifest["sha256"].values()) & set(p2_manifest.values()))

    # 3. ground truth cannot reach candidate generation
    leak = []
    for mod in ("signatures.py", "candidates.py"):
        tree = ast.parse((REPO_ROOT / "phase2" / mod).read_text())
        banned_names = {"family", "truth", "transform", "labels", "node_labels", "parse_stem"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name,)) and node.id in banned_names:
                leak.append((mod, node.id))
            if isinstance(node, ast.arg) and node.arg in banned_names:
                leak.append((mod, node.arg))
    src = (REPO_ROOT / "experiments" / "run_phase2b.py").read_text()
    fn = src[src.index("def evaluate_size"):src.index("def main")]
    checks["candidate sets are hashed before labels are read (run_phase2b.evaluate_size)"] = (
        fn.index("candidate_hash") < fn.index("node_labels("))
    checks["signature/candidate code names no ground truth"] = not leak

    # 4. independent recomputation at every size
    names, H, C = load_signatures()
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    mismatches, near = [], {}
    for F in SIZES:
        paths = node_paths(F)
        idx = [name_to_idx[p.name] for p in paths]
        n = len(paths)
        parsed = [parse_stem(p.stem) for p in paths]
        family = np.array([int(f) for f, _ in parsed])
        transform = np.array([t for _, t in parsed])
        dist = l2_independent(C[idx])
        mask = np.triu(np.ones((n, n), dtype=bool), 1)
        P = np.triu(dist < TAU_STAR, 1)
        near[F] = int((np.abs(dist[mask] - TAU_STAR) < 1e-9).sum())
        saved = res["sizes"][str(F)]["M2_star"]
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
        gate = (ind["recall"] >= 0.95 and ind["reduction"] >= 0.90 and min(ind["per_transform"].values()) >= 0.85)
        if not ok or gate != saved["gate"]["pass"]:
            mismatches.append((F, P_flat.size, saved["candidate_pairs"], gate, saved["gate"]["pass"]))
        print(f"  audited F={F}", flush=True)
        del dist, mask, P
    checks["independent recomputation matches saved results at every size"] = not mismatches
    details["recomputation_mismatches"] = mismatches
    details["pairs_within_1e-9_of_tau_star"] = near
    verdict = all(res["sizes"][str(F)]["M2_star"]["gate"]["pass"] for F in SIZES)
    checks["gate verdict consistent with per-size gates"] = (
        verdict == res["gate"]["passes_every_size"] == res["hypothesis_supported"])

    # 5. byte-for-byte rerun in a fresh process at n=280
    rerun = ARTIFACTS / "rerun_F40.json"
    subprocess.run([sys.executable, str(REPO_ROOT / "experiments" / "run_phase2b.py"), "--only", "40",
                    "--out", str(rerun)], cwd=REPO_ROOT, capture_output=True, check=True)
    checks["fresh-process rerun at n=280 is byte-identical"] = (
        json.dumps(res["sizes"]["40"], sort_keys=True)
        == json.dumps(json.loads(rerun.read_text())["sizes"]["40"], sort_keys=True))

    # 6. hygiene
    status = git("status", "--porcelain")
    checks["no data/artifact/cache files visible to git"] = not any(
        k in l for l in status.splitlines() for k in ("phase2b/data", "phase2b/artifacts", "__pycache__", ".pyc"))

    for name, ok in checks.items():
        print(("PASS  " if ok else "FAIL  ") + name)
    (ARTIFACTS / "audit.json").write_text(json.dumps({"checks": checks, "details": details}, indent=1, default=str))
    failed = [k for k, v in checks.items() if not v]
    print("AUDIT:", "ALL CHECKS PASSED" if not failed else f"FAILED: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
