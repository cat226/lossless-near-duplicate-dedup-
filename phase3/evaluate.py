"""
Phase 3 evaluation core (docs/phase3_design.md sections 4, 6, 7).

Stage 1, candidate_stage(): images only. Signatures, dHash, file hashes, candidate sets for every
method and operating point, each hashed. No label is reachable here.
Stage 2, score_sets(): only callable with the stage-1 hashes (phase3.labels refuses otherwise).
Results contain no timing and nothing machine-specific, so they are byte-reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from phase2 import candidates as cand
from phase2 import metrics as met
from phase2 import signatures as sig
from phase3 import blockwise as bw
from phase3 import hashes as hs
from phase3.labels import label_arrays, open_labels

R_SET = (4, 6, 8, 10, 12, 14, 16)
R_PRIMARY = 10
M2_FACTORS = (0.5, 0.75, 1.0, 1.25, 1.5)
RECALL_TARGET = 0.97  # tau_B rule (design 4.2)
PRIMARY = {"M1": "M1_R10", "dHash": "dHash_R10", "EXACT": "EXACT", "M2_B": "M2_B_x1.0", "M2_A1": "M2_A1", "M3": "M3"}


def flat_hash(flat: np.ndarray) -> str:
    return hashlib.sha256(np.sort(flat).astype("<i8").tobytes()).hexdigest()


def flats_of(P: np.ndarray) -> np.ndarray:
    return np.sort(cand.pair_flat_indices(P))


def matrix_of(flat: np.ndarray, n: int) -> np.ndarray:
    P = np.zeros((n, n), dtype=bool)
    rows, cols = cand.flat_to_pair(flat, n)
    P[rows, cols] = True
    return P


def signatures(paths: list[Path], cache: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    key = hashlib.sha256("\n".join(f"{p.name}:{p.stat().st_size}" for p in paths).encode()).hexdigest()
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return z["hashes"], z["coarse"]
    hashes, coarse = sig.compute_signatures(paths)
    if cache is not None:
        np.savez(cache, key=np.array(key), hashes=hashes, coarse=coarse)
    return hashes, coarse


def candidate_stage(paths: list[Path], tau_b: float, *, sig_cache: Path | None = None, curves: bool = True,
                    features: dict | None = None) -> dict:
    """Label-free. Returns the candidate sets (sorted flat indices) and their SHA-256 hashes.
    `features` (phase3.features.extract output, rows aligned with `paths`) replaces the per-image
    extraction; the values are identical to the ones computed here."""
    n = len(paths)
    if features is not None:
        p_hashes, coarse, d_hashes = features["phash"], features["coarse"], features["dhash"]
        file_hashes = [str(h) for h in features["sha256"]]
    else:
        p_hashes, coarse = signatures(paths, sig_cache)
        d_hashes = hs.compute_dhashes(paths)
        file_hashes = [hs.file_sha256(p) for p in paths]
    starts = cand._row_starts(n)
    radii = R_SET if curves else (R_PRIMARY,)
    sets: dict[str, np.ndarray] = {}

    for prefix, h in (("M1", p_hashes), ("dHash", d_hashes)):
        ham = sig.hamming_matrix(h)
        for R in radii:
            sets[f"{prefix}_R{R}"] = flats_of(np.triu(ham <= R, 1))
        del ham
    sets["EXACT"] = hs.exact_candidate_flats(file_hashes, starts)

    factors = M2_FACTORS if curves else (1.0,)
    for f in factors:
        sets[f"M2_B_x{f}"] = bw.m2_candidate_flats(coarse, tau_b * f)
    tau_a1, method_a1 = bw.otsu_blockwise(coarse)
    sets["M2_A1"] = bw.m2_candidate_flats(coarse, tau_a1)
    sets["M3"] = np.union1d(sets[PRIMARY["M1"]], sets[PRIMARY["M2_B"]])

    return {"n": n, "sets": sets, "hashes": {k: flat_hash(v) for k, v in sets.items()},
            "tau": {"tau_b": tau_b, "a1": tau_a1, "a1_method": method_a1}}


def score_sets(stage: dict, labels_path: Path, paths: list[Path], draw_seed: int) -> dict:
    """Stage 2. Opens the labels (allowed only because stage-1 hashes exist) and scores every set."""
    labels = open_labels(labels_path, stage["hashes"])
    family, transform = label_arrays(labels, paths)
    n = stage["n"]
    truth = met.truth_matrix(family)
    t_rows, t_cols = np.nonzero(truth)
    true_flats = np.sort(cand._row_starts(n)[t_rows] + (t_cols - t_rows - 1))
    out = {}
    for name, flat in stage["sets"].items():
        P = matrix_of(flat, n)
        s = met.score(P, truth, transform)
        b1 = met.score(cand.random_control(n, int(flat.size), seed=draw_seed), truth, transform)
        entry = {k: s[k] for k in ("candidate_pairs", "true_pairs", "recall", "reduction", "candidates_per_true_pair",
                                    "recall_original_transform", "recall_transform_transform",
                                    "recall_per_transform", "min_recall_per_transform")}
        entry["gate"] = met.passes_gate(s)
        entry["candidate_set_sha256"] = stage["hashes"][name]
        entry["b1_recall"] = b1["recall"]
        entry["above_b1"] = bool(s["recall"] > b1["recall"])
        entry["membership_recall"] = float(np.isin(true_flats, flat).mean()) if true_flats.size else float("nan")
        out[name] = entry
        del P
    return {"n": n, "true_pairs": int(true_flats.size), "tau": stage["tau"], "sets": out}


def calibrate_tau_b(paths: list[Path], labels_path: Path, *, sig_cache: Path | None = None) -> dict:
    """Protocol B (design 4.2): tau_B = nextafter(q, +inf), q = the 0.97-quantile (method='higher') of
    the DEV true-pair M2 distances. Labels are read here under the scoped D3 relaxation (dev only)."""
    if not labels_path.parent.name.endswith("_dev"):
        raise PermissionError("tau_B may only be calibrated from a DEV set (directory name must end in '_dev')")
    _, coarse = signatures(paths, sig_cache)
    labels = json.loads(Path(labels_path).read_text())["images"]
    family, _ = label_arrays(labels, paths)
    truth = met.truth_matrix(family)
    rows, cols = np.nonzero(truth)
    d = bw.pair_distances(coarse, rows, cols)
    q = float(np.quantile(d, RECALL_TARGET, method="higher"))
    tau = float(np.nextafter(q, np.inf))
    return {"tau_b": tau, "q": q, "dev_images": len(paths), "dev_true_pairs": int(d.size),
            "dev_recall_at_tau": float((d < tau).mean()), "true_distance_min_median_max": [float(d.min()), float(np.median(d)), float(d.max())]}
