"""
Phase 3 block-wise M2 distances (docs/phase3_design.md section 12).

The M2 distance is Phase 2's (L2 over the 1056-d coarse signature, float64 from the stored float32
values, Gram identity with clipping). Here it is computed one row block at a time so the largest
size never holds an n x n float64 matrix. Candidate sets are returned as sorted row-major
upper-triangle flat indices (the representation phase2.candidates hashes). No label is reachable.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

import numpy as np

from phase2 import candidates as cand

OTSU_BINS = 256
FLAT_OTSU_TOLERANCE = 1e-9
BLOCK = 512


def prepare(coarse32: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = coarse32.astype(np.float64)
    return a, (a * a).sum(axis=1)


def l2_rows(a: np.ndarray, sq: np.ndarray, s: int, e: int) -> np.ndarray:
    """Distances of rows s:e to all rows; the same operation order as phase2.signatures.l2_matrix."""
    d = a[s:e] @ a.T
    d *= -2.0
    d += sq[s:e, None]
    d += sq[None, :]
    np.maximum(d, 0.0, out=d)
    np.sqrt(d, out=d)
    return d


def _upper_mask(s: int, e: int, n: int) -> np.ndarray:
    return np.arange(n)[None, :] > np.arange(s, e)[:, None]


def m2_candidate_flats(coarse32: np.ndarray, tau: float, block: int = BLOCK) -> np.ndarray:
    """Sorted flat indices of pairs (i < j) with L2 distance < tau."""
    n = coarse32.shape[0]
    a, sq = prepare(coarse32)
    starts = cand._row_starts(n)
    out = []
    for s in range(0, n, block):
        e = min(n, s + block)
        d = l2_rows(a, sq, s, e)
        rows, cols = np.nonzero((d < tau) & _upper_mask(s, e, n))
        gi = rows + s
        out.append(starts[gi] + (cols - gi - 1))
    return np.sort(np.concatenate(out)) if out else np.empty(0, dtype=np.int64)


def pair_distances(coarse32: np.ndarray, rows: np.ndarray, cols: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Distances for the listed pairs (i < j), computed with the same row-block Gram path as the
    candidate generation, so threshold comparisons are numerically consistent."""
    n = coarse32.shape[0]
    a, sq = prepare(coarse32)
    out = np.empty(len(rows), dtype=np.float64)
    order = np.argsort(rows, kind="stable")
    r_sorted = rows[order]
    for s in range(0, n, block):
        e = min(n, s + block)
        lo, hi = np.searchsorted(r_sorted, s), np.searchsorted(r_sorted, e)
        if lo == hi:
            continue
        sel = order[lo:hi]
        d = l2_rows(a, sq, s, e)
        out[sel] = d[rows[sel] - s, cols[sel]]
    return out


def otsu_from_histogram(hist: np.ndarray, edges: np.ndarray, vmin: float, vmax: float) -> tuple[float, str]:
    """cc_dedup.graph.compute_otsu_threshold's logic, applied to an already-accumulated histogram."""
    hist = hist.astype(np.float64)
    total = hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2.0
    bcv = np.zeros(OTSU_BINS - 1, dtype=np.float64)
    for t in range(1, OTSU_BINS):
        w0 = hist[:t].sum() / total
        w1 = hist[t:].sum() / total
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (hist[:t] * centers[:t]).sum() / (w0 * total)
        mu1 = (hist[t:] * centers[t:]).sum() / (w1 * total)
        bcv[t - 1] = w0 * w1 * (mu0 - mu1) ** 2
    if bcv.max() - bcv.min() < FLAT_OTSU_TOLERANCE:
        return (float(vmin) + float(vmax)) / 2.0, "range_midpoint_fallback"
    return float(edges[int(np.argmax(bcv)) + 1]), "otsu"


def otsu_blockwise(coarse32: np.ndarray, block: int = BLOCK) -> tuple[float, str]:
    """Per-run Otsu threshold over all-pairs M2 distances without materialising them (two passes)."""
    n = coarse32.shape[0]
    a, sq = prepare(coarse32)
    vmin, vmax = np.inf, -np.inf
    for s in range(0, n - 1, block):
        e = min(n, s + block)
        d = l2_rows(a, sq, s, e)[_upper_mask(s, e, n)]
        if d.size:
            vmin, vmax = min(vmin, float(d.min())), max(vmax, float(d.max()))
    hist = np.zeros(OTSU_BINS, dtype=np.int64)
    edges = None
    for s in range(0, n - 1, block):
        e = min(n, s + block)
        d = l2_rows(a, sq, s, e)[_upper_mask(s, e, n)]
        h, edges = np.histogram(d, bins=OTSU_BINS, range=(vmin, vmax))
        hist += h
    return otsu_from_histogram(hist, edges, vmin, vmax)


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_working_set_mb() -> float:
    """Peak resident set of THIS process in MB (Windows)."""
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    pmc = _PMC()
    pmc.cb = ctypes.sizeof(_PMC)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return pmc.PeakWorkingSetSize / 1e6
