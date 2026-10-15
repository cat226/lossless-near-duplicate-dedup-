"""
Phase 3 acquisition: a CD10K-style bundle (Copydays + 10,000 distractors).

Specification: docs/phase3_dataset_plan.md section 8 (draft). Images are streamed,
cleaned and written one at a time; archives are never stored. Output goes to
phase3/data/ (gitignored); manifests go to phase3/manifests/ (tracked).

    python phase3/acquire.py copydays [--dry-run]
    python phase3/acquire.py distractors --n 10000          # full
    python phase3/acquire.py distractors --n 100 --trial    # small trial, separate dir

Copydays images are INRIA's copyright ("as is"); distractors are Flickr images with
per-image Creative Commons licences that this script cannot see. Local research use
only: do not redistribute the images, publish the manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MANIFESTS = ROOT / "manifests"

SEED = 12345
CAP = 1024  # longest-side cap
MIN_SIDE = 128  # distractors with a shorter side are dropped
JPEG_QUALITY = 95

COPYDAYS_URL = "https://dl.fbaipublicfiles.com/vissl/datasets/copydays_{kind}.tar.gz"
COPYDAYS_KINDS = ("original", "strong")  # jpeg and crop are 403 on this mirror

S3 = "https://multimedia-commons.s3-us-west-2.amazonaws.com"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
N_START_POINTS = 2500
KEYS_PER_START = 6
CANDIDATES_DRAWN = 12000


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def http_get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url}: {last}")


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def write_jsonl(path: Path, rows: list[dict]) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


# ---------------------------------------------------------------- Copydays
def stream_tar(kind: str):
    url = COPYDAYS_URL.format(kind=kind)
    with urllib.request.urlopen(url, timeout=300) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                if member.isfile():
                    yield member.name, tar.extractfile(member).read()


def acquire_copydays(dry_run: bool) -> None:
    scales: dict[str, float] = {}
    rows: list[dict] = []
    out_root = DATA / "copydays"
    for kind in COPYDAYS_KINDS:  # originals first: their scale governs the attacked copies
        n_seen = 0
        for name, raw in stream_tar(kind):
            stem, ext = Path(name).stem, Path(name).suffix.lower()
            if ext not in (".jpg", ".jpeg", ".png"):
                continue
            n_seen += 1
            row = {"set": kind, "member": name, "source_sha256": sha256_bytes(raw)}
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:  # noqa: BLE001
                rows.append({**row, "accepted": False, "reason": f"undecodable: {e}"})
                continue
            w, h = img.size
            block = stem[:4]
            if kind == "original":
                scale = min(1.0, CAP / max(w, h))
                scales[block] = scale
            else:
                scale = scales.get(block)
                if scale is None:
                    rows.append({**row, "accepted": False, "reason": "no original for block " + block})
                    continue
            out_w, out_h = max(1, round(w * scale)), max(1, round(h * scale))
            if (out_w, out_h) != (w, h):
                img = img.resize((out_w, out_h), Image.LANCZOS)
            row.update(block=block, query_id=stem[4:6], src_size=[w, h], scale=scale, out_size=[out_w, out_h])
            if dry_run:
                rows.append({**row, "accepted": True, "dry_run": True})
                continue
            out = out_root / kind / f"{stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out, "PNG")
            rows.append({**row, "accepted": True, "out_sha256": sha256_bytes(out.read_bytes()),
                         "out_bytes": out.stat().st_size})
            if n_seen % 50 == 0:
                print(f"  {kind}: {n_seen} images", flush=True)
        print(f"{kind}: {n_seen} images read", flush=True)

    rows.sort(key=lambda r: (r["set"], r["member"]))
    name = "copydays_dryrun.jsonl" if dry_run else "copydays.jsonl"
    write_jsonl(MANIFESTS / name, rows)
    accepted = [r for r in rows if r["accepted"]]
    blocks = {r["block"] for r in accepted}
    print(f"accepted {len(accepted)}/{len(rows)}; distinct blocks {len(blocks)}; "
          f"dropped {[r['reason'] for r in rows if not r['accepted']][:5]}")
    print(f"disk used: {dir_size(out_root) / 1e6:.0f} MB")


# ------------------------------------------------------------- Distractors
def s3_list(start_after: str, max_keys: int) -> list[tuple[str, int]]:
    q = urllib.parse.urlencode(
        {"list-type": "2", "prefix": "data/images/", "start-after": start_after, "max-keys": str(max_keys)}
    )
    root = ET.fromstring(http_get(f"{S3}/?{q}"))
    return [
        (c.find("s3:Key", S3_NS).text, int(c.find("s3:Size", S3_NS).text))
        for c in root.findall("s3:Contents", S3_NS)
    ]


def sample_candidate_keys() -> list[str]:
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, 4096, size=(N_START_POINTS, 2))
    start_keys = [f"data/images/{a:03x}/{b:03x}/" for a, b in starts]
    with ThreadPoolExecutor(16) as pool:
        listings = list(pool.map(lambda k: s3_list(k, KEYS_PER_START), start_keys))
    keys = sorted({k for lst in listings for k, _ in lst if k.lower().endswith(".jpg")})
    rng2 = np.random.default_rng(SEED)
    pick = np.sort(rng2.choice(len(keys), size=min(CANDIDATES_DRAWN, len(keys)), replace=False))
    return [keys[i] for i in pick]


def clean_one(key: str) -> dict:
    row = {"key": key}
    try:
        raw = http_get(f"{S3}/{urllib.parse.quote(key)}")
    except Exception as e:  # noqa: BLE001
        return {**row, "accepted": False, "reason": f"download failed: {e}"}
    row["source_sha256"] = sha256_bytes(raw)
    row["source_bytes"] = len(raw)
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as e:  # noqa: BLE001
        return {**row, "accepted": False, "reason": f"undecodable: {e}"}
    w, h = img.size
    row["src_size"] = [w, h]
    if min(w, h) < MIN_SIDE:
        return {**row, "accepted": False, "reason": f"shorter side < {MIN_SIDE}"}
    if max(w, h) > CAP:
        s = CAP / max(w, h)
        out_size = (max(1, round(w * s)), max(1, round(h * s)))
        buf = io.BytesIO()
        img.convert("RGB").resize(out_size, Image.LANCZOS).save(buf, "JPEG", quality=JPEG_QUALITY)
        data = buf.getvalue()
        row["resized"] = True
        row["out_size"] = list(out_size)
    else:
        data = raw  # original bytes, unmodified
        row["resized"] = False
        row["out_size"] = [w, h]
    row.update(accepted=True, out_sha256=sha256_bytes(data), out_bytes=len(data), _data=data)
    return row


def acquire_distractors(n: int, trial: bool, dry_run: bool) -> None:
    sub = "distractors_trial" if trial else "distractors"
    out_root = DATA / sub
    print("sampling candidate keys ...", flush=True)
    keys = sample_candidate_keys()
    print(f"{len(keys)} candidate keys drawn", flush=True)
    if dry_run:
        print("first keys:", keys[:3])
        return

    out_root.mkdir(parents=True, exist_ok=True)
    rows, seen_hashes, accepted = [], set(), 0
    chunk = 128
    with ThreadPoolExecutor(16) as pool:
        for s in range(0, len(keys), chunk):
            if accepted >= n:
                break
            for row in pool.map(clean_one, keys[s : s + chunk]):
                if accepted >= n:
                    break
                data = row.pop("_data", None)
                if row["accepted"] and row["out_sha256"] in seen_hashes:
                    row.update(accepted=False, reason="exact duplicate (SHA-256)")
                    data = None
                if row["accepted"]:
                    seen_hashes.add(row["out_sha256"])
                    (out_root / Path(row["key"]).name).write_bytes(data)
                    accepted += 1
                rows.append(row)
            print(f"  processed {min(s + chunk, len(keys))} keys, accepted {accepted}", flush=True)

    write_jsonl(MANIFESTS / f"{sub}.jsonl", rows)
    drops: dict[str, int] = {}
    for r in rows:
        if not r["accepted"]:
            drops[r["reason"].split(":")[0]] = drops.get(r["reason"].split(":")[0], 0) + 1
    summary = {"seed": SEED, "cap": CAP, "min_side": MIN_SIDE, "requested": n, "accepted": accepted,
               "candidates_processed": len(rows), "drops": drops, "disk_bytes": dir_size(out_root)}
    (MANIFESTS / f"{sub}.summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("copydays")
    c.add_argument("--dry-run", action="store_true", help="read and measure only; write no images")
    d = sub.add_parser("distractors")
    d.add_argument("--n", type=int, default=10000)
    d.add_argument("--trial", action="store_true", help="write to a separate trial directory")
    d.add_argument("--dry-run", action="store_true", help="only sample candidate keys")
    args = ap.parse_args()
    if args.cmd == "copydays":
        acquire_copydays(args.dry_run)
    else:
        acquire_distractors(args.n, args.trial, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
