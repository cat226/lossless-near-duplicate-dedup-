"""
Phase 3 distractor-pool curation: CONTENT-ONLY exclusion of flat (non-photograph) images.

Rule (decided from the image content, before any Phase 3 method result exists):
  detail energy = mean |gray - bilinear_upsample(16x16 area-resampled gray)|.
  Images whose detail energy lies below the midpoint of the LARGEST GAP in the low tail
  (the gap between the exactly-flat group and the first image that has any detail) are
  excluded. Viewed by hand: the flat group is solid-colour fills; every image just above the
  gap (low-texture moons, sky, fog, dark scenes) is a real photograph and is kept.
No labels, no method outputs, no candidate sets and no pHash are used. Nothing is deleted:
excluded files stay on disk and are listed in the manifest (they form the secondary pool).

Replacements come from the seeded reserve (the candidate keys of the original seed-12345
draw that were never processed), taken in key order, cleaned by the acquisition rules and
the same content rule, until the pool is exactly 10,000 again.

    python phase3/curate_pool.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acquire  # noqa: E402  (sampler and cleaner reused unchanged)

DATA, MANIFESTS = acquire.DATA, acquire.MANIFESTS
POOL_SIZE = 10000
ANALYSIS_BELOW = 1.0  # the gap is searched among detail values below this


def detail_energy(gray: np.ndarray) -> float:
    h, w = gray.shape
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return float(np.abs(gray.astype(np.float32) - up.astype(np.float32)).mean())


def detail_of_file(path: Path) -> float:
    return detail_energy(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))


def detail_of_bytes(data: bytes) -> float:
    return detail_energy(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE))


def main():
    initial_rows = [json.loads(l) for l in open(MANIFESTS / "distractors.jsonl")]
    initial = [r for r in initial_rows if r["accepted"]]
    assert len(initial) == POOL_SIZE

    # 1. the seeded candidate list must reproduce exactly (also proves the bucket is unchanged)
    keys = acquire.sample_candidate_keys()
    processed = [r["key"] for r in initial_rows]
    assert keys[: len(processed)] == processed, "candidate list differs from the acquisition run"
    reserve = keys[len(processed):]
    print(f"candidate keys {len(keys)}; processed {len(processed)}; reserve {len(reserve)}", flush=True)

    # 2. detail energy of every pooled image, and the gap-midpoint cut-off
    root = DATA / "distractors"
    detail = {Path(r["key"]).name: detail_of_file(root / Path(r["key"]).name) for r in initial}
    vals = np.sort(np.array([v for v in detail.values() if v < ANALYSIS_BELOW]))
    gaps = np.diff(vals)
    g = int(np.argmax(gaps))
    gap_lo, gap_hi = float(vals[g]), float(vals[g + 1])
    cutoff = (gap_lo + gap_hi) / 2.0
    flat_names = sorted(n for n, v in detail.items() if v < cutoff)
    print(f"largest gap below {ANALYSIS_BELOW}: {gap_lo:.6f} -> {gap_hi:.6f}; cutoff {cutoff:.6f}; "
          f"flat images {len(flat_names)}", flush=True)

    # 3. replacements from the reserve, in key order
    pool_hashes = {r["out_sha256"] for r in initial}
    need = len(flat_names)
    replacements, reserve_log = [], []
    chunk = 128
    with ThreadPoolExecutor(16) as pool:
        for s in range(0, len(reserve), chunk):
            if len(replacements) >= need:
                break
            for row in pool.map(acquire.clean_one, reserve[s : s + chunk]):
                if len(replacements) >= need:
                    break
                data = row.pop("_data", None)
                if row["accepted"] and row["out_sha256"] in pool_hashes:
                    row.update(accepted=False, reason="exact duplicate (SHA-256)")
                if row["accepted"]:
                    d = detail_of_bytes(data)
                    row["detail_energy"] = d
                    if d < cutoff:
                        row.update(accepted=False, reason="flat image (content rule)")
                if row["accepted"]:
                    pool_hashes.add(row["out_sha256"])
                    (root / Path(row["key"]).name).write_bytes(data)
                    replacements.append(row)
                reserve_log.append({k: v for k, v in row.items() if k != "_data"})
            print(f"  reserve processed {min(s + chunk, len(reserve))}, replacements {len(replacements)}/{need}", flush=True)
    assert len(replacements) == need, "reserve exhausted before restoring the pool"

    # 4. manifests
    flat_set = set(flat_names)
    by_name = {Path(r["key"]).name: r for r in initial}
    final = []
    for n, r in by_name.items():
        if n in flat_set:
            continue
        final.append({"key": r["key"], "name": n, "source": "initial", "out_sha256": r["out_sha256"],
                      "size": r["out_size"], "detail_energy": round(detail[n], 5)})
    for r in replacements:
        final.append({"key": r["key"], "name": Path(r["key"]).name, "source": "reserve",
                      "out_sha256": r["out_sha256"], "size": r["out_size"],
                      "detail_energy": round(r["detail_energy"], 5)})
    final.sort(key=lambda r: r["key"])
    assert len(final) == POOL_SIZE and len({r["out_sha256"] for r in final}) == POOL_SIZE

    excluded = sorted(({"key": by_name[n]["key"], "name": n, "reason": "flat image (content rule)",
                        "detail_energy": round(detail[n], 5),
                        "bytes_per_pixel": round(by_name[n]["out_bytes"] / (by_name[n]["out_size"][0] * by_name[n]["out_size"][1]), 5),
                        "size": by_name[n]["out_size"]} for n in flat_names), key=lambda r: r["key"])

    acquire.write_jsonl(MANIFESTS / "distractor_pool.jsonl", final)
    acquire.write_jsonl(MANIFESTS / "distractor_pool_excluded.jsonl", excluded)
    acquire.write_jsonl(MANIFESTS / "distractor_pool_reserve_log.jsonl", reserve_log)
    summary = {
        "rule": "exclude if detail energy < cutoff; cutoff = midpoint of the largest gap in the low tail",
        "detail_energy_definition": "mean |gray - bilinear_upsample(16x16 area-resampled gray)|, gray = cv2 IMREAD_GRAYSCALE",
        "gap_searched_below": ANALYSIS_BELOW, "gap_lo": gap_lo, "gap_hi": gap_hi, "cutoff": cutoff,
        "content_check": "viewed by hand: a seeded sample of 21 of the flat images are all solid-colour fills; "
                         "the 41 images ranked just above the gap (detail 0.40-2.98) are all real photographs "
                         "(low-texture moons, sky, fog, dark scenes) and are kept",
        "label_free": True, "uses_phash": False, "seed": acquire.SEED,
        "excluded": len(excluded), "replacements_from_reserve": len(replacements),
        "reserve_keys_processed": len(reserve_log),
        "reserve_drops": {},
        "final_pool_size": len(final),
        "secondary_pool": "final pool + the excluded images (distractor_pool_excluded.jsonl)",
        "known_ambiguous_pairs": [],
        "audit_note": "cosine band 0.85-0.90 was not inspected in the initial audit",
        "files_deleted": 0,
    }
    for r in reserve_log:
        if not r["accepted"]:
            k = r["reason"].split(":")[0]
            summary["reserve_drops"][k] = summary["reserve_drops"].get(k, 0) + 1
    (MANIFESTS / "distractor_pool.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("cutoff", "excluded", "replacements_from_reserve", "reserve_drops",
                                               "final_pool_size")}, indent=1))


if __name__ == "__main__":
    main()
