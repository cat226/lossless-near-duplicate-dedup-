"""
Phase 2 — n=280 distance-matrix recompute (docs/phase2_design.md sections 10, 11).

Runs the unchanged Phase 1c pairwise_distance over the regenerated 280-image
dataset in parallel (approved exception), persists the matrix, and runs the
section-4 consistency check. No timing is recorded or claimed from this run.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import PIL

from phase2.consistency import check_against_phase1c
from phase2.generate import DEFAULT_DATASET_DIR, node_paths, sha256_file
from phase2.parallel_matrix import build_distance_matrix

N_FAMILIES = 40
OUT_MATRIX = REPO_ROOT / "results" / "phase2_n280_distance_matrix.npy"
OUT_META = REPO_ROOT / "results" / "phase2_n280_distance_matrix.meta.json"


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else max(1, (os.cpu_count() or 2) - 2)
    paths = node_paths(N_FAMILIES)
    assert len(paths) == 280 and all(p.exists() for p in paths), "dataset missing or wrong size"

    print(f"building 280x280 matrix with {workers} workers", flush=True)
    matrix = build_distance_matrix(paths, workers=workers, log=lambda m: print(m, flush=True))
    assert matrix.shape == (280, 280) and np.isfinite(matrix).all()
    assert np.array_equal(matrix, matrix.T)

    # Byte-for-byte comparison with the Phase 1c images, when they are on disk.
    phase0_dir = REPO_ROOT / "phase0" / "data"
    same_as_phase0 = None
    if phase0_dir.exists():
        same_as_phase0 = True
        for p in paths:
            sub = "originals" if p.parent.name == "originals" else "transformed"
            q = phase0_dir / sub / p.name
            if not q.exists() or sha256_file(p) != sha256_file(q):
                same_as_phase0 = False
                break

    consistency = check_against_phase1c(matrix)

    OUT_MATRIX.parent.mkdir(exist_ok=True)
    tmp = OUT_MATRIX.with_name("phase2_n280_distance_matrix.tmp.npy")
    np.save(tmp, matrix)
    tmp.replace(OUT_MATRIX)

    meta = {
        "nodes": [p.name for p in paths],
        "n_pairs": 280 * 279 // 2,
        "workers": workers,
        "timing_note": "parallel run: no timing claim is made from it",
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "python": platform.python_version(),
        "images_byte_identical_to_phase0_data": same_as_phase0,
        "consistency_with_phase1c": consistency,
    }
    OUT_META.write_text(json.dumps(meta, indent=1))

    print(f"images byte-identical to phase0/data: {same_as_phase0}")
    print(f"consistency with Phase 1c: all_match={consistency['all_match']}")
    print(json.dumps(consistency["matches"], indent=1))
    print(f"tau={consistency['tau']!r} (ref {consistency['phase1c_reference']['tau']!r}), "
          f"E+={consistency['positive_edge_count']}, "
          f"D_kwik={consistency['kwik_disagreement']!r}, D_base={consistency['baseline_disagreement']!r}")


if __name__ == "__main__":
    main()
