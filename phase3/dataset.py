"""
Phase 3 dataset builders (docs/phase3_design.md sections 3.2, 3.3, 3.5).

D3a: each family source is one cleaned-pool photograph plus the six frozen Phase 0 transformations,
applied by importing the UNMODIFIED make_transformations. D3b: Copydays blocks. Images are written
under neutral names i<rank>.<ext> (rank from a seeded permutation); labels go to a separate
labels.json that only the scoring stage may open (see phase3.labels).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from phase2.generate import _phase0_generator

REPO = Path(__file__).resolve().parent.parent
POOL_DIR = REPO / "phase3" / "data" / "distractors"
COPYDAYS_DIR = REPO / "phase3" / "data" / "copydays"
TRANSFORM_ORDER = ("resize", "jpegq50", "crop", "bright", "rotate", "png")


def _inputs_hash(kind: str, groups: list, singles: list[str], seed: int) -> str:
    return hashlib.sha256(json.dumps([kind, groups, singles, seed]).encode()).hexdigest()


def _place(src: Path, dst: Path) -> None:
    """Hard link on the same volume (no extra disk, identical bytes); a copy if linking is impossible."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _finish(out_dir: Path, records: list[dict], seed: int, meta_hash: str, membership: dict | None = None) -> dict:
    """records: dicts with src (Path), source, family (str|None), transform, and optionally group
    ("s" family source / "g" singleton) and idx (position in its list). Writes neutral images,
    labels.json, build_meta.json and, when `membership` {n: (F, S)} is given, membership.json (the
    neutral names of the images of the first F sources and first S singletons, for every size n;
    built from the records in memory, never from labels.json). Returns {'paths', 'labels_path', 'n'}."""
    records = sorted(records, key=lambda r: (r["source"], r["transform"]))
    order = np.random.default_rng([seed, 1]).permutation(len(records))
    images = out_dir / "images"
    images.mkdir(parents=True)
    labels = {}
    member_names: dict[int, list[str]] = {n: [] for n in (membership or {})}
    for rank, k in enumerate(order):
        r = records[k]
        name = f"i{rank:05d}{r['src'].suffix.lower()}"
        _place(r["src"], images / name)
        labels[Path(name).stem] = {"source": r["source"], "family": r["family"], "transform": r["transform"]}
        for n, (f, sg) in (membership or {}).items():
            if (r["group"] == "s" and r["idx"] < f) or (r["group"] == "g" and r["idx"] < sg):
                member_names[n].append(name)
    (out_dir / "labels.json").write_text(json.dumps({"images": labels}, sort_keys=True))
    if membership:
        (out_dir / "membership.json").write_text(json.dumps({str(n): sorted(v) for n, v in member_names.items()}))
    (out_dir / "build_meta.json").write_text(json.dumps({"inputs_sha256": meta_hash, "n": len(records)}))
    return _load(out_dir)


def _load(out_dir: Path) -> dict:
    paths = sorted((out_dir / "images").glob("i*"))
    return {"paths": paths, "labels_path": out_dir / "labels.json", "n": len(paths)}


def _reusable(out_dir: Path, meta_hash: str) -> bool:
    meta = out_dir / "build_meta.json"
    return meta.exists() and json.loads(meta.read_text()).get("inputs_sha256") == meta_hash


def build_d3a_set(out_dir: Path, source_names: list[str], singleton_names: list[str], seed: int,
                  membership: dict | None = None) -> dict:
    """Family = source photo + six transformed images (7 images, 21 true pairs); singletons are
    photographs with no true pair. A source never doubles as a singleton. `membership` {n: (F, S)}
    additionally records, for each ladder size, which neutral names belong to it (prefix nesting)."""
    assert not set(source_names) & set(singleton_names), "a family source must not also be a singleton"
    meta_hash = _inputs_hash("d3a", source_names, singleton_names, seed)
    if membership:
        meta_hash = hashlib.sha256((meta_hash + json.dumps(sorted(membership.items()))).encode()).hexdigest()
    if _reusable(out_dir, meta_hash):
        return _load(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    raw = out_dir / "_raw"
    raw.mkdir(parents=True)
    m = _phase0_generator()
    records = []
    for idx, name in enumerate(source_names):
        src = POOL_DIR / name
        made = m.make_transformations(src, raw)  # unmodified Phase 0 transformations
        assert set(made) == set(TRANSFORM_ORDER)
        records.append({"src": src, "source": name, "family": name, "transform": "original", "group": "s", "idx": idx})
        for t in TRANSFORM_ORDER:
            records.append({"src": made[t], "source": name, "family": name, "transform": t, "group": "s", "idx": idx})
    for idx, name in enumerate(singleton_names):
        records.append({"src": POOL_DIR / name, "source": name, "family": None, "transform": "singleton", "group": "g", "idx": idx})
    try:
        built = _finish(out_dir, records, seed, meta_hash, membership)
    finally:
        shutil.rmtree(raw, ignore_errors=True)  # raw file names carry labels
    return built


def build_d3b_set(out_dir: Path, blocks: list[str], singleton_names: list[str], seed: int) -> dict:
    """Copydays blocks (a block = original + attacked copies = one family) plus singleton photographs."""
    rows = [json.loads(l) for l in open(REPO / "phase3" / "manifests" / "copydays.jsonl")]
    rows = [r for r in rows if r["accepted"] and r["block"] in set(blocks)]
    meta_hash = _inputs_hash("d3b", sorted(blocks), singleton_names, seed)
    if _reusable(out_dir, meta_hash):
        return _load(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    records = []
    for r in rows:
        stem = Path(r["member"]).stem
        src = COPYDAYS_DIR / r["set"] / f"{stem}.png"
        records.append({"src": src, "source": f"copydays_{stem}", "family": r["block"],
                        "transform": "original" if r["set"] == "original" else "strong"})
    for name in singleton_names:
        records.append({"src": POOL_DIR / name, "source": name, "family": None, "transform": "singleton"})
    return _finish(out_dir, records, seed, meta_hash)
