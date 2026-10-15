"""
Phase 3 main run (docs/phase3_design.md, Addenda A and B).

    python experiments/run_phase3.py --mode rehearsal          # DEV split only; validates the code, not evidence
    python experiments/run_phase3.py --mode full --go          # test split; refuses unless every gate below passes

Gates for --mode full (all must hold, otherwise nothing is read and nothing is created):
  * free space C: >= 2 GB and D: >= 6 GB; TEMP and TMP on D:
  * all 39 split hashes and 4 manifest hashes recomputed and equal to the design; per-file data integrity
  * setup_verification.json passed; the dev-only pilot report has no failed item
  * the rehearsal ran and its audit passed (phase3/artifacts/rehearsal_audit.json)
  * the working tree has no modified tracked file, and no frozen path differs from the design commit
  * --go, i.e. the user's explicit go; only then is phase3/.test_split_approved created
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from phase3 import runner  # noqa: E402
from phase3.integrity import verify_data_integrity  # noqa: E402
from phase3.splits import TEST_APPROVAL_FILE, verify_startup  # noqa: E402

FROZEN = ["phase0", "cc_dedup", "baselines", "phase2", "phase2b", "results/phase1c_results.json", "docs/phase1c_design.md",
          "docs/phase2_design.md", "docs/phase2b_design.md", "experiments/run_phase1c.py", "experiments/run_pilot.py"]
DESIGN_COMMIT = "624283f"
ALLOWED_UNTRACKED = {"project_handoff.md", "results/pilot_validation.json", "phase3/manifests/distractors_trial.jsonl",
                     "phase3/manifests/distractors_trial.summary.json"}  # never committed (the user's handoff note, pilot/trial output)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def frozen_paths_clean() -> dict:
    existing = [p for p in FROZEN if (REPO / p).exists()]
    diff = git("diff", DESIGN_COMMIT, "--", *existing).strip()
    status = git("status", "--porcelain", "--", *existing).strip()
    return {"frozen_paths_checked": existing, "differs_from_design_commit": bool(diff), "uncommitted_changes": bool(status)}


def preflight_common(mode: str) -> tuple:
    rep: dict = {"mode": mode, "repo": str(REPO), "free_gb": runner.free_gb()}
    for var in ("TEMP", "TMP"):
        if not os.environ.get(var, "").upper().startswith("D:"):
            raise SystemExit(f"{var} must point to D: for this run (it is {os.environ.get(var)!r})")
    if rep["free_gb"]["C:"] < 2.0 or rep["free_gb"]["D:"] < 6.0:
        raise SystemExit(f"free space below the start thresholds (C: >= 2 GB, D: >= 6 GB): {rep['free_gb']}")
    split = verify_startup()  # before any image or label; raises SplitMismatch
    rep["startup"] = {"draws": {d: v["hashes_checked"] for d, v in split.report["draws"].items()},
                      "manifest_hashes_in_design": split.report["manifest_hashes_in_design"]}
    rep["integrity"] = verify_data_integrity()
    rep["head"] = git("rev-parse", "HEAD").strip()
    rep["frozen"] = frozen_paths_clean()
    if rep["frozen"]["differs_from_design_commit"] or rep["frozen"]["uncommitted_changes"]:
        raise SystemExit(f"a frozen path changed: {rep['frozen']}")
    return rep, split


def full_gates(rep: dict) -> None:
    art = REPO / "phase3" / "artifacts"
    setup = json.loads((art / "setup_verification.json").read_text())
    if not setup.get("all_passed"):
        raise SystemExit("setup verification has not passed")
    pilot = json.loads((art / "pilot_report.json").read_text())["checks"]
    failed = [k for k, v in pilot.items() if not v["pass"]]
    if failed:
        raise SystemExit(f"the dev-only pilot has failed items: {failed}")
    audit = art / "rehearsal_audit.json"
    if not audit.exists() or not json.loads(audit.read_text()).get("all_passed"):
        raise SystemExit("the rehearsal audit (phase3/artifacts/rehearsal_audit.json) is missing or failed")
    lines = git("status", "--porcelain").splitlines()
    dirty = [l for l in lines if not l.startswith("??")]
    stray = [l[3:].strip() for l in lines if l.startswith("??") and l[3:].strip() not in ALLOWED_UNTRACKED]
    if dirty or stray:
        raise SystemExit(f"the working tree must be committed before the run: modified {dirty}, untracked {stray}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rehearsal", "full"], required=True)
    ap.add_argument("--go", action="store_true", help="the user's explicit go for the test split (full mode only)")
    ap.add_argument("--draws", default="1,2,3")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tag", default="", help="separate artifacts directory (audit re-runs)")
    ap.add_argument("--only-sizes", default="", help="comma list of ladder rungs (audit re-runs)")
    args = ap.parse_args()
    if args.mode == "full" and not args.go:
        raise SystemExit("--go is required for the test split; the user has not given it yet")

    rep, split = preflight_common(args.mode)
    if args.mode == "full" and args.tag:  # audit re-run: the unlock already happened in the main run
        if not TEST_APPROVAL_FILE.exists():
            raise SystemExit("a tagged re-run needs the approval file written by the main run")
    elif args.mode == "full":
        full_gates(rep)
        if not TEST_APPROVAL_FILE.exists():
            TEST_APPROVAL_FILE.write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                                      "head": rep["head"], "command": " ".join(sys.argv)}))
    elif TEST_APPROVAL_FILE.exists():
        print("note: the test-approval file exists; rehearsal mode still reads only the dev split", flush=True)

    ctx = runner.Ctx(args.mode, split, workers=args.workers, draws=[int(x) for x in args.draws.split(",")], tag=args.tag,
                     only_sizes=[int(x) for x in args.only_sizes.split(",")] if args.only_sizes else None)
    ctx.event("preflight_passed", **{k: v for k, v in rep.items() if k in ("free_gb", "head", "mode")})
    if args.mode == "full":
        ctx.event("test_split_unlocked", approval_file=str(TEST_APPROVAL_FILE))
    if args.only_sizes:  # audit re-run: tau_B and the chosen ladder rungs only
        for d in ctx.draws:
            runner.run_tau_b(ctx, d)
            runner.run_ladder(ctx, d)
        return
    res = runner.run_all(ctx)
    if args.mode == "full":
        out = {"results": res, "parts": runner.collect_parts(ctx)}
        dest = REPO / "results" / "phase3_results.json"
        dest.write_text(json.dumps(out, sort_keys=True, indent=1))
        print("wrote", dest, flush=True)
    print("DONE", args.mode, flush=True)


if __name__ == "__main__":
    main()
