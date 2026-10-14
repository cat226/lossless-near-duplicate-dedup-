"""
Phase 3 ground truth access (docs/phase3_design.md sections 3.5 and 6).

Labels live in a separate labels.json and may be opened ONLY after the candidate sets have been
produced and hashed: open_labels() refuses to run without the candidate-set hashes. This module is
the only reader of labels.json; signature and candidate code never import it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LabelAccessBeforeCandidates(RuntimeError):
    """Labels were requested before any candidate-set hash existed."""


def open_labels(labels_path: Path, candidate_hashes: dict) -> dict:
    if not candidate_hashes:
        raise LabelAccessBeforeCandidates("candidate sets must be produced and hashed before labels are opened")
    return json.loads(Path(labels_path).read_text())["images"]


def label_arrays(labels: dict, paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """(family id per node, transformation per node) aligned with `paths` (node order).
    Singletons get unique negative family ids so they form no true pair."""
    fam_index: dict[str, int] = {}
    family = np.empty(len(paths), dtype=np.int64)
    transform = []
    for k, p in enumerate(paths):
        lab = labels[p.stem]
        if lab["family"] is None:
            family[k] = -1 - k
        else:
            family[k] = fam_index.setdefault(lab["family"], len(fam_index))
        transform.append(lab["transform"])
    return family, np.array(transform)
