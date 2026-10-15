"""
Phase 3 parallel verifiers (docs/phase3_design.md Addendum A.4).

The primary verifier is parallelised by the unchanged Phase 2 wrapper (images decoded once into
shared memory around the unmodified Phase 1c pairwise_distance). The aligned verifier is
parallelised here. Both are paired with an exactness assertion against single-process calls.
No timing claim may be made from a parallel run.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np

from cc_dedup.graph import pairwise_distance  # Phase 1c, unchanged
from phase2.parallel_matrix import load_rgb
from phase3.aligned_verifier import AlignConfig, aligned_distance, prepare

_W: dict = {}
CACHE_LIMIT = 16


def _init(paths, cfg):
    import cv2

    cv2.setNumThreads(1)  # one thread per worker: bounds memory, and equality with the multi-threaded single-process call is a stronger test
    _W.update(paths=paths, cfg=cfg, cache={})


def _prep(i: int):
    cache = _W["cache"]
    if i not in cache:
        if len(cache) >= CACHE_LIMIT:
            cache.clear()
        cache[i] = prepare(load_rgb(_W["paths"][i]), _W["cfg"])
    return cache[i]


def _work(task):
    start, pairs = task
    cfg = _W["cfg"]
    return start, [aligned_distance(_prep(i), _prep(j), cfg) for i, j in pairs]


def aligned_pairs_single(paths: list[Path], pairs: list[tuple[int, int]], cfg: AlignConfig = AlignConfig()) -> list[dict]:
    _init(paths, cfg)
    return [aligned_distance(_prep(i), _prep(j), cfg) for i, j in pairs]


def aligned_pairs_parallel(paths: list[Path], pairs: list[tuple[int, int]], workers: int,
                           cfg: AlignConfig = AlignConfig(), chunk: int = 10) -> list[dict]:
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"  # inherited by the spawned workers
    tasks = [(s, pairs[s : s + chunk]) for s in range(0, len(pairs), chunk)]
    out: list = [None] * len(pairs)
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(paths, cfg)) as pool:
        for start, res in pool.imap_unordered(_work, tasks):
            out[start : start + len(res)] = res
    return out


def primary_single(paths: list[Path], pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.array([pairwise_distance(load_rgb(paths[i]), load_rgb(paths[j])) for i, j in pairs])


def primary_pairs_parallel(paths: list[Path], pairs: list[tuple[int, int]], workers: int, chunk: int = 100,
                           max_images: int = 800) -> np.ndarray:
    """Phase 1c distance of the listed pairs on `workers` processes. The unmodified Phase 2 shared-memory
    workers are reused; pairs are processed in batches whose images (decoded once, in the parent) never
    exceed `max_images`, which bounds shared memory. Output order equals `pairs`."""
    import os
    from multiprocessing import shared_memory

    from phase2 import parallel_matrix as pm

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    batches, cur, imgs = [], [], set()
    for k, (i, j) in enumerate(pairs):
        grown = imgs | {i, j}
        if cur and len(grown) > max_images:
            batches.append(cur)
            cur, grown = [], {i, j}
        cur.append(k)
        imgs = grown
    if cur:
        batches.append(cur)
    out = np.empty(len(pairs), dtype=np.float64)
    for b in batches:
        ids = sorted({x for k in b for x in pairs[k]})
        pos = {x: q for q, x in enumerate(ids)}
        arrays = [load_rgb(paths[x]) for x in ids]
        shapes, sizes = [a.shape for a in arrays], [a.nbytes for a in arrays]
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64).tolist()
        shm = shared_memory.SharedMemory(create=True, size=int(sum(sizes)))
        try:
            for a, off in zip(arrays, offsets):
                np.ndarray(a.shape, dtype=np.uint8, buffer=shm.buf, offset=off)[...] = a
            del arrays
            rows = np.array([pos[pairs[k][0]] for k in b], dtype=np.int32)
            cols = np.array([pos[pairs[k][1]] for k in b], dtype=np.int32)
            tasks = [(s, min(s + chunk, len(b))) for s in range(0, len(b), chunk)]
            bidx = np.array(b)
            with mp.get_context("spawn").Pool(workers, initializer=pm._init_worker,
                                              initargs=(shm.name, shapes, offsets, rows, cols)) as pool:
                for start, values in pool.imap_unordered(pm._work, tasks):
                    out[bidx[start : start + values.shape[0]]] = values
        finally:
            shm.close()
            shm.unlink()
    return out


def assert_primary_exact(paths: list[Path], pairs: list[tuple[int, int]], values: np.ndarray, k: int = 500,
                         seed: int = 12345) -> dict:
    """A.4: recompute a seeded sample single-process with the unchanged pairwise_distance; every value must
    be bitwise equal, otherwise the run stops."""
    pick = np.sort(np.random.default_rng(seed).choice(len(pairs), size=min(k, len(pairs)), replace=False))
    single = primary_single(paths, [pairs[q] for q in pick])
    ok = np.array_equal(single, values[pick])
    rep = {"pairs_checked": int(pick.size), "pairs_equal": int((single == values[pick]).sum()),
           "max_abs_difference": float(np.abs(single - values[pick]).max()) if pick.size else 0.0, "bitwise_equal": bool(ok)}
    if not ok:
        raise AssertionError(f"parallel primary verifier differs from single-process: {rep}")
    return rep


def assert_aligned_exact(paths: list[Path], pairs: list[tuple[int, int]], results: list[dict], k: int = 200,
                         seed: int = 12345, cfg: AlignConfig = AlignConfig()) -> dict:
    pick = np.sort(np.random.default_rng(seed).choice(len(pairs), size=min(k, len(pairs)), replace=False))
    single = aligned_pairs_single(paths, [pairs[q] for q in pick], cfg)
    par = [results[q] for q in pick]
    ok = single == par
    rep = {"pairs_checked": int(pick.size), "pairs_equal": int(sum(a == b for a, b in zip(single, par))), "bitwise_equal": bool(ok)}
    if not ok:
        raise AssertionError(f"parallel aligned verifier differs from single-process: {rep}")
    return rep
