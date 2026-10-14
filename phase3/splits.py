"""
Phase 3 dev/test split (docs/phase3_design.md Addendum A.2).

The split is fixed by seed and its SHA-256 hashes are recorded in the design document. Every
Phase 3 script calls verify_startup() FIRST and stops (SplitMismatch) on any mismatch, before it
touches any image or label. The test split is structurally locked: its accessors raise
PermissionError unless the user has approved the pilot by creating phase3/.test_split_approved.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MANIFESTS = REPO / "phase3" / "manifests"
DESIGN = REPO / "docs" / "phase3_design.md"
TEST_APPROVAL_FILE = REPO / "phase3" / ".test_split_approved"

DRAW_SEEDS = {1: 12345, 2: 12346, 3: 12347}
POOL_N, DEV_N, DEV_SOURCES = 10000, 1500, 150
HASH_KEYS = ["perm", "dev", "dev_sources", "dev_singletons", "test_sources", "test_singletons",
             "d3b_singletons", "variant_sources", "variant_singletons",
             "ext_perm", "ext_dev", "ext_sources250", "ext_singletons3250"]


class SplitMismatch(RuntimeError):
    """A recomputed split hash differs from the one recorded in the design document."""


def _h(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def _perm_hash(perm: np.ndarray) -> str:
    return hashlib.sha256(perm.astype("<i8").tobytes()).hexdigest()


def load_pool_names() -> tuple[list[str], list[str]]:
    pool = [json.loads(l) for l in open(MANIFESTS / "distractor_pool.jsonl")]
    excl = [json.loads(l) for l in open(MANIFESTS / "distractor_pool_excluded.jsonl")]
    names = [r["name"] for r in pool]
    ext = [r["name"] for r in sorted(pool + excl, key=lambda r: r["key"])]
    return names, ext


def compute_draw(names: list[str], ext: list[str], seed: int) -> dict:
    perm = np.random.default_rng(seed).permutation(len(names))
    pn = [names[i] for i in perm]
    pe = np.random.default_rng(seed).permutation(len(ext))
    pen = [ext[i] for i in pe]
    dev, test = pn[:DEV_N], pn[DEV_N:]
    return {
        "lists": {
            "dev": dev, "dev_sources": pn[:DEV_SOURCES], "dev_singletons": pn[DEV_SOURCES:DEV_N],
            "test_sources": test[:500], "test_singletons": test[500:7000],
            "d3b_singletons": test[:4614], "variant_sources": test[:114], "variant_singletons": test[114:114 + 7202],
            "test": test,
            "ext_dev": pen[:DEV_N], "ext_test": pen[DEV_N:],
            "ext_sources250": pen[1500:1750], "ext_singletons3250": pen[2000:5250],
        },
        "hashes": {
            "perm": _perm_hash(perm), "dev": _h(dev), "dev_sources": _h(pn[:DEV_SOURCES]),
            "dev_singletons": _h(pn[DEV_SOURCES:DEV_N]), "test_sources": _h(test[:500]),
            "test_singletons": _h(test[500:7000]), "d3b_singletons": _h(test[:4614]),
            "variant_sources": _h(test[:114]), "variant_singletons": _h(test[114:114 + 7202]),
            "ext_perm": _perm_hash(pe), "ext_dev": _h(pen[:DEV_N]),
            "ext_sources250": _h(pen[1500:1750]), "ext_singletons3250": _h(pen[2000:5250]),
        },
    }


def recorded_hashes(doc_text: str) -> dict[int, list[str]]:
    out = {}
    for d, seed in DRAW_SEEDS.items():
        start = doc_text.index(f"draw {d} (seed {seed})")
        block = doc_text[start:]
        block = block[: block.index("\n\n")]
        found = re.findall(r"[0-9a-f]{64}", block)
        if len(found) != len(HASH_KEYS):
            raise SplitMismatch(f"draw {d}: expected {len(HASH_KEYS)} recorded hashes, found {len(found)}")
        out[d] = found
    return out


class VerifiedSplit:
    """Holds the verified lists. Dev accessors are open; test accessors are locked."""

    def __init__(self, draws: dict[int, dict], report: dict):
        self._draws = draws
        self.report = report

    def dev_sources(self, draw: int) -> list[str]:
        return self._draws[draw]["lists"]["dev_sources"]

    def dev_singletons(self, draw: int) -> list[str]:
        return self._draws[draw]["lists"]["dev_singletons"]

    def dev(self, draw: int) -> list[str]:
        return self._draws[draw]["lists"]["dev"]

    @staticmethod
    def _require_test_approval() -> None:
        if not TEST_APPROVAL_FILE.exists():
            raise PermissionError(
                "the test split is locked: it opens only after the user approves the pilot "
                "(phase3/.test_split_approved)")

    def test_sources(self, draw: int) -> list[str]:
        self._require_test_approval()
        return self._draws[draw]["lists"]["test_sources"]

    def test_singletons(self, draw: int) -> list[str]:
        self._require_test_approval()
        return self._draws[draw]["lists"]["test_singletons"]

    def test_photos(self, draw: int) -> list[str]:
        self._require_test_approval()
        return self._draws[draw]["lists"]["test"]

    def _locked(self, draw: int, key: str) -> list[str]:
        self._require_test_approval()
        return self._draws[draw]["lists"][key]

    def variant_sources(self, draw: int) -> list[str]:
        return self._locked(draw, "variant_sources")

    def variant_singletons(self, draw: int) -> list[str]:
        return self._locked(draw, "variant_singletons")

    def d3b_singletons(self, draw: int) -> list[str]:
        return self._locked(draw, "d3b_singletons")

    def ext_sources250(self, draw: int) -> list[str]:
        return self._locked(draw, "ext_sources250")

    def ext_singletons3250(self, draw: int) -> list[str]:
        return self._locked(draw, "ext_singletons3250")

    # extended-pool DEV split (open): first 150 are family sources, the other 1,350 singletons, as in the main pool
    def ext_dev_sources(self, draw: int) -> list[str]:
        return self._draws[draw]["lists"]["ext_dev"][:DEV_SOURCES]

    def ext_dev_singletons(self, draw: int) -> list[str]:
        return self._draws[draw]["lists"]["ext_dev"][DEV_SOURCES:]


def verify_startup() -> VerifiedSplit:
    """Recompute every recorded hash; raise SplitMismatch on any difference. Reads only the
    manifests and the design document: no image and no label."""
    doc = DESIGN.read_text(encoding="utf-8")
    names, ext = load_pool_names()
    if len(names) != POOL_N or len(set(names)) != POOL_N:
        raise SplitMismatch("pool is not 10,000 distinct names")
    report = {"manifest_hashes_in_design": {}, "draws": {}}
    for f in ("distractor_pool.jsonl", "distractor_pool_excluded.jsonl", "copydays.jsonl",
              "aligned_verifier_validation.json"):
        actual = hashlib.sha256((MANIFESTS / f).read_bytes()).hexdigest()
        ok = actual in doc
        report["manifest_hashes_in_design"][f] = ok
        if not ok:
            raise SplitMismatch(f"manifest {f} does not match the SHA-256 recorded in the design")
    names_hash = _h(names)
    if names_hash not in doc:
        raise SplitMismatch("pool names hash differs from the one recorded in the design")
    recorded = recorded_hashes(doc)
    draws = {}
    for d, seed in DRAW_SEEDS.items():
        comp = compute_draw(names, ext, seed)
        want = dict(zip(HASH_KEYS, recorded[d]))
        bad = [k for k in HASH_KEYS if comp["hashes"][k] != want[k]]
        if bad:
            raise SplitMismatch(f"draw {d}: recomputed hashes differ from the design for {bad}")
        L = comp["lists"]
        if set(L["dev"]) & set(L["test"]) or set(L["ext_dev"]) & set(L["ext_test"]):
            raise SplitMismatch(f"draw {d}: dev and test overlap")
        report["draws"][d] = {"hashes_checked": len(HASH_KEYS), "dev": len(L["dev"]), "test": len(L["test"]),
                              "disjoint": True}
        draws[d] = comp
    return VerifiedSplit(draws, report)
