"""
Phase 3 per-image cheap features, extracted once per image (docs/phase3_design.md section 12).

One decode gives everything the candidate stage needs: the Phase 2 pHash and 1056-d coarse
signature (the unmodified phase2.signatures building blocks), the Phase 3 dHash, and the SHA-256 of
the file bytes. Every value is a pure function of one image, so a parallel run equals a
single-process run exactly (asserted by tests/test_phase3.py). No label is reachable here.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from PIL import Image

from phase2 import signatures as sig
from phase3 import hashes as hs


def features_of(path: Path) -> tuple[np.uint64, np.ndarray, np.uint64, str]:
    """(pHash, coarse float32 [1056], dHash, file SHA-256) from one decode."""
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        gray = sig.gray32(img)
        dh = hs.dhash64(img)
    coarse = np.concatenate([sig.block_gray(gray), sig.block_color(rgb), sig.block_sobel(gray)])
    return sig.phash64(gray), coarse.astype(np.float32), dh, hs.file_sha256(path)


def _chunk(paths: list[Path]):
    return [features_of(p) for p in paths]


def _pack(rows: list) -> dict:
    return {"phash": np.array([r[0] for r in rows], dtype=np.uint64),
            "coarse": np.stack([r[1] for r in rows]).astype(np.float32),
            "dhash": np.array([r[2] for r in rows], dtype=np.uint64),
            "sha256": np.array([r[3] for r in rows])}


def extract(paths: list[Path], workers: int = 1, chunk: int = 50) -> dict:
    """Features of every path, in path order. workers > 1 uses spawn; the output is identical."""
    if workers <= 1 or len(paths) <= chunk:
        return _pack(_chunk(paths))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    parts = [paths[s : s + chunk] for s in range(0, len(paths), chunk)]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = [r for part in pool.imap(_chunk, parts) for r in part]  # imap keeps order
    return _pack(rows)


def cache_key(paths: list[Path]) -> str:
    return hashlib.sha256("\n".join(f"{p.name}:{p.stat().st_size}" for p in paths).encode()).hexdigest()


def extract_cached(paths: list[Path], cache: Path, workers: int = 1) -> dict:
    key = cache_key(paths)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return {k: z[k] for k in ("phash", "coarse", "dhash", "sha256")}
    f = extract(paths, workers)
    np.savez(cache, key=np.array(key), **f)
    return f
