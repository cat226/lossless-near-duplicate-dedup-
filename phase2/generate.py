"""
Phase 2 dataset generation (docs/phase2_design.md sections 3 and 17.1).

Reuses the Phase 0 generator, imported unmodified. One dataset is generated at
F=800 families; any smaller F is the first F families of it (prefix nesting).
Output goes to phase2/data/ (gitignored), never to phase0/data/.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PHASE0_DIR = REPO_ROOT / "phase0"
DATA_ROOT = REPO_ROOT / "phase2" / "data"
DEFAULT_DATASET_DIR = DATA_ROOT / "F800"
MAX_FAMILIES = 800
TRANSFORMS = ["resize", "jpegq50", "crop", "bright", "rotate", "png"]


def _phase0_generator():
    if str(PHASE0_DIR) not in sys.path:
        sys.path.insert(0, str(PHASE0_DIR))
    import make_test_pairs

    return make_test_pairs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_stem(stem: str) -> tuple[str, str]:
    """Return (family_id, transform); transform is 'original' for originals."""
    for t in TRANSFORMS:
        marker = f"_{t}"
        if stem.endswith(marker):
            return stem[: -len(marker)], t
    return stem, "original"


def node_paths(n_families: int, dataset_dir: Path = DEFAULT_DATASET_DIR) -> list[Path]:
    """Ordered node list for F families (design doc 17.1): the F originals sorted
    by name, then the transformed images of families 1..F sorted by filename.
    For F=40 this equals Phase 1c's discover_images order."""
    originals = [dataset_dir / "originals" / f"{k:03d}.jpg" for k in range(1, n_families + 1)]
    wanted = {f"{k:03d}" for k in range(1, n_families + 1)}
    transformed = sorted(
        p for p in (dataset_dir / "transformed").glob("*") if parse_stem(p.stem)[0] in wanted
    )
    return originals + transformed


def generate_dataset(n_families: int = MAX_FAMILIES, dataset_dir: Path = DEFAULT_DATASET_DIR) -> dict:
    """Generate originals (Phase 0 sequential RNG, seed 12345) and the six
    deterministic transforms per original, family by family on disk."""
    m = _phase0_generator()
    originals = m.generate_synthetic_originals(n_families, dataset_dir / "originals")
    transformed_dir = dataset_dir / "transformed"
    transformed_dir.mkdir(parents=True, exist_ok=True)
    for original in originals:
        m.make_transformations(original, transformed_dir)

    import numpy
    import PIL

    files = sorted((dataset_dir / "originals").glob("*")) + sorted(transformed_dir.glob("*"))
    manifest = {
        "n_families": n_families,
        "n_images": len(files),
        "rng_seed": m.RNG_SEED,
        "synthetic_size": list(m.SYNTHETIC_SIZE),
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "python": platform.python_version(),
        "sha256": {f"{p.parent.name}/{p.name}": sha256_file(p) for p in files},
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_FAMILIES
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DATASET_DIR
    mf = generate_dataset(n, out)
    print(f"generated {mf['n_images']} images for {mf['n_families']} families in {out}")
