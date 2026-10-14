"""
Phase 3 data integrity (run after any move of phase3/data): every pool photograph and every
Copydays image is re-hashed and compared with the SHA-256 recorded in its manifest.
Reads files only; changes nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from phase3.dataset import COPYDAYS_DIR, POOL_DIR
from phase3.hashes import file_sha256
from phase3.splits import MANIFESTS


def verify_data_integrity() -> dict:
    initial = {Path(r["key"]).name: r["out_sha256"] for r in map(json.loads, open(MANIFESTS / "distractors.jsonl"))
               if r["accepted"]}
    pool = [json.loads(l) for l in open(MANIFESTS / "distractor_pool.jsonl")]
    excluded = [json.loads(l) for l in open(MANIFESTS / "distractor_pool_excluded.jsonl")]
    expected = {r["name"]: r["out_sha256"] for r in pool}
    for r in excluded:
        expected[r["name"]] = initial[r["name"]]
    bad, missing = [], []
    for name, want in expected.items():
        p = POOL_DIR / name
        if not p.exists():
            missing.append(name)
        elif file_sha256(p) != want:
            bad.append(name)
    cd_bad, cd_missing, cd_n = [], [], 0
    for r in map(json.loads, open(MANIFESTS / "copydays.jsonl")):
        if not r["accepted"]:
            continue
        cd_n += 1
        p = COPYDAYS_DIR / r["set"] / f"{Path(r['member']).stem}.png"
        if not p.exists():
            cd_missing.append(p.name)
        elif file_sha256(p) != r["out_sha256"]:
            cd_bad.append(p.name)
    report = {"pool_and_excluded_files": len(expected), "pool_mismatched": len(bad), "pool_missing": len(missing),
              "copydays_files": cd_n, "copydays_mismatched": len(cd_bad), "copydays_missing": len(cd_missing)}
    if bad or missing or cd_bad or cd_missing:
        raise RuntimeError(f"data integrity failure: {report}; examples {(bad + missing + cd_bad + cd_missing)[:5]}")
    return report
