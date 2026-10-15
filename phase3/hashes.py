"""
Phase 3 baselines (docs/phase3_design.md section 4.1): dHash and the exact file hash.
The definitions are frozen in the design; neither function sees any label.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def dhash64(img: Image.Image) -> np.uint64:
    """Grayscale (PIL 'L'), LANCZOS to 9 columns x 8 rows from the full-resolution image; bit
    (row r, comparison c) = pixel[r, c] > pixel[r, c+1]; bit index = 8*r + c."""
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = g[:, :-1] > g[:, 1:]  # (8 rows, 8 comparisons)
    value = 0
    for r in range(8):
        for c in range(8):
            if bits[r, c]:
                value |= 1 << (8 * r + c)
    return np.uint64(value)


def compute_dhashes(paths: list[Path]) -> np.ndarray:
    out = np.empty(len(paths), dtype=np.uint64)
    for i, p in enumerate(paths):
        with Image.open(p) as img:
            out[i] = dhash64(img)
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exact_candidate_flats(file_hashes: list[str], n_flat_start) -> np.ndarray:
    """Pairs of byte-identical files, as sorted row-major upper-triangle flat indices."""
    n = len(file_hashes)
    groups: dict[str, list[int]] = {}
    for i, h in enumerate(file_hashes):
        groups.setdefault(h, []).append(i)
    flats = []
    for members in groups.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                flats.append(int(n_flat_start[i]) + (j - i - 1))
    return np.sort(np.array(flats, dtype=np.int64))
