"""
Phase 2 — cheap per-image signatures (docs/phase2_design.md sections 6 and 17.3-17.4).

M1: 64-bit pHash. M2: 1056-d coarse signature. Ground truth is never an input
here: a signature is a pure function of one image's pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

GRAY_SIZE = 32
HASH_BLOCK = 8
COARSE_DIM = GRAY_SIZE * GRAY_SIZE + 24 + 8  # 1056


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix D, so the 2D transform of X is D @ X @ D.T."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    d = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2.0 * n))
    d[0, :] = np.sqrt(1.0 / n)
    return d


_DCT = _dct_matrix(GRAY_SIZE)


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def gray32(img: Image.Image) -> np.ndarray:
    """PIL luma, resized to 32x32 with LANCZOS from the full-resolution image."""
    return np.asarray(img.convert("L").resize((GRAY_SIZE, GRAY_SIZE), Image.LANCZOS), dtype=np.float64)


def phash64(gray: np.ndarray) -> np.uint64:
    coeffs = _DCT @ gray @ _DCT.T
    block = coeffs[:HASH_BLOCK, :HASH_BLOCK].ravel()  # row-major index k -> bit k
    median = np.median(block[1:])  # DC (index 0) excluded from the median only
    value = 0
    for k in np.flatnonzero(block > median):
        value |= 1 << int(k)
    return np.uint64(value)


def block_gray(gray: np.ndarray) -> np.ndarray:
    return _l2_normalise(gray.ravel())


def block_color(rgb: np.ndarray) -> np.ndarray:
    hist = np.concatenate(
        [np.bincount((rgb[..., c] // 32).ravel(), minlength=8) for c in range(3)]
    ).astype(np.float64)
    return _l2_normalise(hist)


def block_sobel(gray: np.ndarray) -> np.ndarray:
    g = gray
    gx = (g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[1:-1, :-2] + g[2:, :-2])
    gy = (g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]) - (g[:-2, :-2] + 2 * g[:-2, 1:-1] + g[:-2, 2:])
    angle = np.arctan2(gy, gx)  # signed, in [-pi, pi]
    bins = np.minimum(((angle + np.pi) / (2 * np.pi) * 8).astype(np.int64), 7)
    hist = np.bincount(bins.ravel(), weights=np.hypot(gx, gy).ravel(), minlength=8)
    return _l2_normalise(hist)


def signature_of(path: Path) -> tuple[np.uint64, np.ndarray]:
    """(M1 hash, M2 coarse signature as float32) for one image."""
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        gray = gray32(img)
    coarse = np.concatenate([block_gray(gray), block_color(rgb), block_sobel(gray)])
    return phash64(gray), coarse.astype(np.float32)


def compute_signatures(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    hashes = np.empty(len(paths), dtype=np.uint64)
    coarse = np.empty((len(paths), COARSE_DIM), dtype=np.float32)
    for i, p in enumerate(paths):
        hashes[i], coarse[i] = signature_of(p)
    return hashes, coarse


def hamming_matrix(hashes: np.ndarray, block: int = 512) -> np.ndarray:
    """(n, n) uint8 Hamming distances: popcount of the XOR."""
    n = hashes.shape[0]
    out = np.empty((n, n), dtype=np.uint8)
    for s in range(0, n, block):
        out[s : s + block] = np.bitwise_count(hashes[s : s + block, None] ^ hashes[None, :])
    return out


def l2_matrix(coarse: np.ndarray) -> np.ndarray:
    """(n, n) float64 L2 distances, computed in float64 from the stored float32 values.
    Upper triangle is computed and mirrored so the matrix is exactly symmetric."""
    a = coarse.astype(np.float64)
    sq = (a * a).sum(axis=1)
    d = a @ a.T
    d *= -2.0
    d += sq[:, None]
    d += sq[None, :]
    np.maximum(d, 0.0, out=d)
    np.sqrt(d, out=d)
    d = np.triu(d, 1)
    d += d.T
    return d
