"""
Phase 2 — parallel wrapper around the UNCHANGED Phase 1c pairwise_distance.

Approved exception (docs/phase2_design.md section 10): the one-time n=280
distance-matrix recompute may use multiprocessing. Images are decoded ONCE in
the parent into a single shared-memory block; workers attach to it (Windows has
no fork, so they cannot inherit memory). The distance function itself is
imported from cc_dedup.graph and never reimplemented or edited.

No timing is recorded by this module: no timing claim may be made from a
parallel run.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
from PIL import Image

from cc_dedup.graph import pairwise_distance  # Phase 1c, unchanged

_STATE: dict = {}


def load_rgb(path: Path) -> np.ndarray:
    """Decode exactly as Phase 1c's graph._load_nodes does."""
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _init_worker(shm_name, shapes, offsets, rows, cols):
    shm = shared_memory.SharedMemory(name=shm_name)
    arrays = [
        np.ndarray(shape, dtype=np.uint8, buffer=shm.buf, offset=offset)
        for shape, offset in zip(shapes, offsets)
    ]
    _STATE.update(shm=shm, arrays=arrays, rows=rows, cols=cols)


def _work(task):
    start, stop = task
    arrays, rows, cols = _STATE["arrays"], _STATE["rows"], _STATE["cols"]
    out = np.empty(stop - start, dtype=np.float64)
    for k in range(start, stop):
        out[k - start] = pairwise_distance(arrays[rows[k]], arrays[cols[k]])
    return start, out


def build_distance_matrix(paths: list[Path], workers: int, chunk: int = 200, log=print) -> np.ndarray:
    """Symmetric (n, n) float64 matrix of Phase 1c pairwise distances over
    `paths`, zero diagonal. Pair order and values are independent of scheduling
    (each pair is a pure function of its two images)."""
    n = len(paths)
    arrays = [load_rgb(p) for p in paths]  # decode once, in the parent
    shapes = [a.shape for a in arrays]
    sizes = [a.nbytes for a in arrays]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64).tolist()

    shm = shared_memory.SharedMemory(create=True, size=int(sum(sizes)))
    try:
        for a, off in zip(arrays, offsets):
            np.ndarray(a.shape, dtype=np.uint8, buffer=shm.buf, offset=off)[...] = a
        del arrays

        rows, cols = np.triu_indices(n, k=1)
        rows = rows.astype(np.int32)
        cols = cols.astype(np.int32)
        n_pairs = rows.shape[0]
        tasks = [(s, min(s + chunk, n_pairs)) for s in range(0, n_pairs, chunk)]
        flat = np.empty(n_pairs, dtype=np.float64)

        ctx = mp.get_context("spawn")
        done = 0
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(shm.name, shapes, offsets, rows, cols),
        ) as pool:
            for start, values in pool.imap_unordered(_work, tasks):
                flat[start : start + values.shape[0]] = values
                done += values.shape[0]
                if (done // chunk) % 20 == 0:
                    log(f"  {done}/{n_pairs} pairs")
    finally:
        shm.close()
        shm.unlink()

    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[rows, cols] = flat
    matrix[cols, rows] = flat
    return matrix
