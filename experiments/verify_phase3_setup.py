"""
Phase 3 setup verification after moving the data to another drive (dev split only; no test image or
label is touched). Checks: where the data resolves, startup hashes, per-file data integrity, a
determinism spot-check (rebuild + two candidate-stage runs + tau_B equal to the earlier pilot's
value), and free space on every drive involved.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from phase3 import evaluate as ev
from phase3.dataset import POOL_DIR, build_d3a_set
from phase3.integrity import verify_data_integrity
from phase3.splits import TEST_APPROVAL_FILE, verify_startup

SEED = 12345


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def main() -> None:
    out: dict = {}
    data = REPO / "phase3" / "data"
    out["data_path"] = str(data)
    out["data_resolves_to"] = os.path.realpath(data)
    out["is_junction"] = os.path.isjunction(data) if hasattr(os.path, "isjunction") else None
    out["temp_in_this_process"] = {"TEMP": os.environ.get("TEMP"), "TMP": os.environ.get("TMP")}
    print(json.dumps(out, indent=1), flush=True)

    split = verify_startup()
    out["startup"] = split.report
    print("startup OK:", {d: v["hashes_checked"] for d, v in split.report["draws"].items()},
          "manifests:", split.report["manifest_hashes_in_design"], flush=True)
    assert not TEST_APPROVAL_FILE.exists() or True  # approval state is reported, not assumed
    out["test_approval_file_present"] = TEST_APPROVAL_FILE.exists()

    out["integrity"] = verify_data_integrity()
    print("data integrity OK:", out["integrity"], flush=True)

    # determinism spot-check (dev only), compared with the earlier pilot recorded on C:
    prev = json.loads((REPO / "phase3/artifacts/pilot_report.json").read_text())["info"]["tau_b_draw1_dev"]["tau_b"]
    dev_src, dev_single = split.dev_sources(1), split.dev_singletons(1)
    work = data / "pilot_verify"
    dev = build_d3a_set(work / "draw1_dev", dev_src, dev_single, SEED)
    tau = ev.calibrate_tau_b(dev["paths"], dev["labels_path"], sig_cache=work / "draw1_dev" / "sig.npz")
    pilot = build_d3a_set(work / "draw1_pilot", dev_src[:20], dev_single[:300], SEED)
    a = ev.candidate_stage(pilot["paths"], tau["tau_b"])
    b = ev.candidate_stage(pilot["paths"], tau["tau_b"])
    out["determinism_spot_check"] = {
        "tau_b_now": tau["tau_b"], "tau_b_earlier_pilot_on_C": prev, "tau_b_identical": tau["tau_b"] == prev,
        "candidate_hashes_identical_across_two_runs": a["hashes"] == b["hashes"], "candidate_sets": len(a["hashes"]),
        "pilot_images": pilot["n"], "dev_images": dev["n"],
    }
    print("determinism spot-check:", out["determinism_spot_check"], flush=True)

    out["free_space_gb"] = {"C:": free_gb("C:\\"), "D:": free_gb("D:\\")}
    print("free space GB:", out["free_space_gb"], flush=True)
    ok = (out["determinism_spot_check"]["tau_b_identical"]
          and out["determinism_spot_check"]["candidate_hashes_identical_across_two_runs"])
    out["all_passed"] = bool(ok)
    (REPO / "phase3" / "artifacts" / "setup_verification.json").write_text(json.dumps(out, indent=1))
    print("SETUP VERIFICATION:", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
