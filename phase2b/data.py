"""
Phase 2b data (docs/phase2b_design.md sections 3 and 4).

Fresh families 801..1600 from the UNMODIFIED Phase 0 generator. Because the generator
uses one sequential RNG, generating 800+k originals reproduces families 1..800 exactly;
those are verified against the Phase 2 manifest and then discarded, and only the fresh
families get transformed. Imports phase2 unchanged.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2.generate import _phase0_generator, parse_stem, sha256_file  # noqa: E402

TAU_STAR = 0.6004747353136561  # 0.5 x Phase 2 n=280 Otsu tau_sig (design section 1)
PHASE2_FAMILIES = 800
FIRST_FRESH = PHASE2_FAMILIES + 1
SIZES = (40, 100, 200, 400, 800)
DATA_ROOT = REPO_ROOT / "phase2b" / "data"
DEFAULT_DATASET_DIR = DATA_ROOT / "fresh800"
PHASE2_MANIFEST = REPO_ROOT / "phase2" / "data" / "F800" / "manifest.json"


def node_paths(n_families: int, dataset_dir: Path = DEFAULT_DATASET_DIR) -> list[Path]:
    """Ordered nodes for the first `n_families` fresh families (design 3): originals by
    ascending family id, then transformed images by (family id, filename)."""
    ids = range(FIRST_FRESH, FIRST_FRESH + n_families)
    originals = [dataset_dir / "originals" / f"{k:03d}.jpg" for k in ids]
    wanted = {f"{k:03d}" for k in ids}
    transformed = sorted(
        (p for p in (dataset_dir / "transformed").glob("*") if parse_stem(p.stem)[0] in wanted),
        key=lambda p: (int(parse_stem(p.stem)[0]), p.name),
    )
    return originals + transformed


def generate_fresh(n_fresh: int, out_dir: Path = DEFAULT_DATASET_DIR) -> dict:
    """Generate `n_fresh` fresh families (801..800+n_fresh) under out_dir."""
    m = _phase0_generator()
    total = PHASE2_FAMILIES + n_fresh
    scratch = out_dir / "_originals_all"
    paths = m.generate_synthetic_originals(total, scratch)

    phase2_hashes = json.loads(PHASE2_MANIFEST.read_text())["sha256"]
    mismatched = [
        p.name for p in paths[:PHASE2_FAMILIES]
        if sha256_file(p) != phase2_hashes[f"originals/{p.name}"]
    ]
    if mismatched:
        raise RuntimeError(f"regenerated Phase 2 families differ from the Phase 2 manifest: {mismatched[:5]}")

    originals_dir = out_dir / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)
    for p in paths[PHASE2_FAMILIES:]:
        shutil.move(str(p), str(originals_dir / p.name))
    shutil.rmtree(scratch)

    transformed_dir = out_dir / "transformed"
    transformed_dir.mkdir(parents=True, exist_ok=True)
    for original in sorted(originals_dir.glob("*.jpg")):
        m.make_transformations(original, transformed_dir)

    files = sorted(originals_dir.glob("*")) + sorted(transformed_dir.glob("*"))
    manifest = {
        "n_fresh_families": n_fresh,
        "first_family": FIRST_FRESH,
        "last_family": PHASE2_FAMILIES + n_fresh,
        "phase2_families_verified_against_manifest": PHASE2_FAMILIES,
        "n_images": len(files),
        "sha256": {f"{p.parent.name}/{p.name}": sha256_file(p) for p in files},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DATASET_DIR
    mf = generate_fresh(n, out)
    print(f"fresh families {mf['first_family']}..{mf['last_family']}: {mf['n_images']} images in {out}; "
          f"{mf['phase2_families_verified_against_manifest']} Phase 2 originals matched the Phase 2 manifest")
