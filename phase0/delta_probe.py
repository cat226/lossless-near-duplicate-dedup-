"""
Phase 0 — delta probe.

For every (original, transformed) pair, computes:

  A. raw_delta_size  — delta on the encoded file bytes (xdelta3 if
     installed, otherwise the bsdiff4 python package; no custom delta
     algorithm is implemented here).

  B. pixel_delta_size — decode both images to RGB pixel buffers with
     Pillow, take the raw byte difference of the pixel buffers, and
     compress that difference with zlib. Any dimension-mismatch
     handling for a pair is reported separately by
     pixel_dimension_note() and recorded in run()'s per-pair results.

Then reports median and p90 of delta_size / original_size, broken down
by transformation type, for both raw and pixel deltas.

Usage:
    python delta_probe.py
    python delta_probe.py --originals data/originals --transformed data/transformed --out results.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_ORIGINALS = Path(__file__).resolve().parent / "data" / "originals"
DEFAULT_TRANSFORMED = Path(__file__).resolve().parent / "data" / "transformed"

TRANSFORM_SUFFIXES = ["resize", "jpegq50", "crop", "bright", "rotate", "png"]


def load_pairs(dir_originals: str, dir_transformed: str) -> list[tuple[Path, Path]]:
    """Pair original and transformed files by matching filename stem.

    Original files are named {stem}.jpg (e.g. 001.jpg).
    Transformed files are named {stem}_{transform}.{ext}
    (e.g. 001_resize.jpg, 001_png.png).

    Returns a list of (original_path, transformed_path) tuples, sorted
    for determinism.
    """
    originals_dir = Path(dir_originals)
    transformed_dir = Path(dir_transformed)

    originals_by_stem = {p.stem: p for p in originals_dir.glob("*") if p.is_file()}

    pairs = []
    for t_path in sorted(transformed_dir.glob("*")):
        if not t_path.is_file():
            continue
        name = t_path.stem  # e.g. "001_resize"
        matched_stem = None
        for suffix in TRANSFORM_SUFFIXES:
            marker = f"_{suffix}"
            if name.endswith(marker):
                matched_stem = name[: -len(marker)]
                break
        if matched_stem is None:
            print(f"WARNING: could not parse transform type from {t_path.name}, skipping", file=sys.stderr)
            continue
        orig_path = originals_by_stem.get(matched_stem)
        if orig_path is None:
            print(f"WARNING: no original found for stem {matched_stem} (from {t_path.name}), skipping", file=sys.stderr)
            continue
        pairs.append((orig_path, t_path))

    return pairs


def _transform_of(transformed_path: Path) -> str:
    name = transformed_path.stem
    for suffix in TRANSFORM_SUFFIXES:
        if name.endswith(f"_{suffix}"):
            return suffix
    return "unknown"


_XDELTA3_AVAILABLE = shutil.which("xdelta3") is not None
_BSDIFF4_AVAILABLE = False
try:
    import bsdiff4  # noqa: F401
    _BSDIFF4_AVAILABLE = True
except ImportError:
    pass


def raw_delta_size(a: Path, b: Path) -> int:
    """Compute delta size on raw encoded file bytes.

    Uses xdelta3 if available on PATH, otherwise the bsdiff4 python
    package (a wrapper around the standard bsdiff/bspatch algorithm).
    Does not reimplement either algorithm.

    Raises RuntimeError if neither tool is available, with a message
    describing exactly what is needed.
    """
    if _XDELTA3_AVAILABLE:
        with tempfile.NamedTemporaryFile(suffix=".xdelta", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            result = subprocess.run(
                ["xdelta3", "-e", "-f", "-s", str(a), str(b), str(tmp_path)],
                capture_output=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"xdelta3 failed on {a} -> {b}: {result.stderr.decode(errors='replace')}"
                )
            return tmp_path.stat().st_size
        finally:
            tmp_path.unlink(missing_ok=True)

    if _BSDIFF4_AVAILABLE:
        patch = bsdiff4.diff(a.read_bytes(), b.read_bytes())
        return len(patch)

    raise RuntimeError(
        "Neither xdelta3 (external binary on PATH) nor bsdiff4 (pip package) is available.\n"
        "Install one of:\n"
        "  xdelta3: e.g. `choco install xdelta3` (Windows, needs admin) or your OS package manager\n"
        "  bsdiff4: `pip install bsdiff4`\n"
        "Raw-byte delta cannot be computed without one of these; refusing to substitute "
        "an unrelated compression method."
    )


def pixel_delta_size(a: Path, b: Path) -> int:
    """Decode both images to raw RGB buffers using Pillow, compute the
    raw byte difference, compress it with zlib, and return the
    compressed size.

    Dimension mismatch handling: this experiment does NOT silently
    resize one image to match the other, since that would inject an
    extra transformation into the measurement and bias the delta
    downward. Instead, when dimensions differ (e.g. resize, crop), the
    two pixel buffers are compared over their overlapping top-left
    region only. Callers that need to know whether/how a mismatch was
    handled for a given pair should use pixel_dimension_note(a, b);
    this function's return type is kept to the planned int-only
    signature.
    """
    img_a = Image.open(a).convert("RGB")
    img_b = Image.open(b).convert("RGB")

    arr_a = np.asarray(img_a, dtype=np.uint8)
    arr_b = np.asarray(img_b, dtype=np.uint8)

    if arr_a.shape != arr_b.shape:
        h = min(arr_a.shape[0], arr_b.shape[0])
        w = min(arr_a.shape[1], arr_b.shape[1])
        arr_a = arr_a[:h, :w, :]
        arr_b = arr_b[:h, :w, :]

    diff = (arr_a.astype(np.int16) - arr_b.astype(np.int16)).astype(np.int8)
    compressed = zlib.compress(diff.tobytes(), level=9)
    return len(compressed)


def pixel_dimension_note(a: Path, b: Path) -> str:
    """Describe how pixel_delta_size handled a dimension mismatch
    between a and b, without redoing the diff/compress work.

    Returns "" if dimensions matched natively. This is a lightweight,
    size-only check (Image.size, no full decode-to-array), kept
    separate from pixel_delta_size so that function's signature and
    return value stay exactly the planned `-> int`; the mismatch
    information is surfaced by the caller (run()) instead.
    """
    with Image.open(a) as img_a, Image.open(b) as img_b:
        size_a = img_a.size  # (width, height)
        size_b = img_b.size

    if size_a == size_b:
        return ""

    h = min(size_a[1], size_b[1])
    w = min(size_a[0], size_b[0])
    return (
        f"dimension mismatch (h,w) {(size_a[1], size_a[0])} vs {(size_b[1], size_b[0])}; "
        f"compared overlapping region {(h, w)} only"
    )


def summarize(results: list[dict]) -> None:
    """Print median and p90 delta_size/original_size ratios for both
    raw and pixel deltas, broken down by transformation type.
    """
    by_transform: dict[str, list[dict]] = {}
    for r in results:
        by_transform.setdefault(r["transform"], []).append(r)

    print()
    print(f"{'Transform':<12} {'Raw Median':>11} {'Raw P90':>9} {'Pixel Median':>13} {'Pixel P90':>10}  {'N':>4}")
    print("-" * 66)

    overall_row = {"transform": "ALL", "raw_ratio": [], "pixel_ratio": []}

    for transform in TRANSFORM_SUFFIXES:
        rows = by_transform.get(transform, [])
        if not rows:
            print(f"{transform:<12} {'(no data)':>11}")
            continue

        raw_ratios = [r["raw_ratio"] for r in rows if r["raw_ratio"] is not None]
        pixel_ratios = [r["pixel_ratio"] for r in rows if r["pixel_ratio"] is not None]

        raw_med = statistics.median(raw_ratios) if raw_ratios else float("nan")
        raw_p90 = _percentile(raw_ratios, 90) if raw_ratios else float("nan")
        pix_med = statistics.median(pixel_ratios) if pixel_ratios else float("nan")
        pix_p90 = _percentile(pixel_ratios, 90) if pixel_ratios else float("nan")

        print(f"{transform:<12} {raw_med:>11.3f} {raw_p90:>9.3f} {pix_med:>13.3f} {pix_p90:>10.3f}  {len(rows):>4}")

        overall_row["raw_ratio"].extend(raw_ratios)
        overall_row["pixel_ratio"].extend(pixel_ratios)

    if overall_row["raw_ratio"] or overall_row["pixel_ratio"]:
        raw_med = statistics.median(overall_row["raw_ratio"]) if overall_row["raw_ratio"] else float("nan")
        raw_p90 = _percentile(overall_row["raw_ratio"], 90) if overall_row["raw_ratio"] else float("nan")
        pix_med = statistics.median(overall_row["pixel_ratio"]) if overall_row["pixel_ratio"] else float("nan")
        pix_p90 = _percentile(overall_row["pixel_ratio"], 90) if overall_row["pixel_ratio"] else float("nan")
        print("-" * 66)
        print(f"{'ALL':<12} {raw_med:>11.3f} {raw_p90:>9.3f} {pix_med:>13.3f} {pix_p90:>10.3f}  {len(results):>4}")

    failed = [r for r in results if r.get("error")]
    if failed:
        print(f"\n{len(failed)} pair(s) failed during delta computation (not excluded from denominators, reported explicitly):")
        for r in failed:
            print(f"  {r['transformed']}: {r['error']}")

    dim_mismatches = [r for r in results if r.get("pixel_note")]
    if dim_mismatches:
        print(f"\n{len(dim_mismatches)} pair(s) had pixel-dimension mismatches (compared on overlapping region only):")
        by_t = {}
        for r in dim_mismatches:
            by_t.setdefault(r["transform"], 0)
            by_t[r["transform"]] += 1
        for t, n in sorted(by_t.items()):
            print(f"  {t}: {n}")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run(dir_originals: Path, dir_transformed: Path, out_path: Path | None) -> list[dict]:
    pairs = load_pairs(str(dir_originals), str(dir_transformed))
    print(f"Loaded {len(pairs)} original/transformed pairs.")
    if not _XDELTA3_AVAILABLE and not _BSDIFF4_AVAILABLE:
        print(
            "WARNING: neither xdelta3 nor bsdiff4 is available. "
            "Raw-byte delta results will be recorded as errors.",
            file=sys.stderr,
        )
    elif not _XDELTA3_AVAILABLE:
        print("xdelta3 not found on PATH; using bsdiff4 (pip package) for raw-byte deltas.")
    else:
        print("Using xdelta3 for raw-byte deltas.")

    results = []
    for orig, trans in pairs:
        transform = _transform_of(trans)
        orig_size = orig.stat().st_size
        trans_size = trans.stat().st_size
        row = {
            "original": str(orig),
            "transformed": str(trans),
            "transform": transform,
            "original_size": orig_size,
            "transformed_size": trans_size,
            "raw_delta_size": None,
            "raw_ratio": None,
            "pixel_delta_size": None,
            "pixel_ratio": None,
            "pixel_note": "",
            "error": "",
        }
        try:
            rd = raw_delta_size(orig, trans)
            row["raw_delta_size"] = rd
            row["raw_ratio"] = rd / orig_size
        except Exception as e:  # noqa: BLE001
            row["error"] = f"raw_delta: {e}"

        try:
            pd = pixel_delta_size(orig, trans)
            row["pixel_delta_size"] = pd
            row["pixel_ratio"] = pd / orig_size
            row["pixel_note"] = pixel_dimension_note(orig, trans)
        except Exception as e:  # noqa: BLE001
            row["error"] = (row["error"] + " | " if row["error"] else "") + f"pixel_delta: {e}"

        results.append(row)

    if out_path:
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nWrote raw results to {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 0 delta probe.")
    parser.add_argument("--originals", type=str, default=str(DEFAULT_ORIGINALS))
    parser.add_argument("--transformed", type=str, default=str(DEFAULT_TRANSFORMED))
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "delta_results.json"))
    args = parser.parse_args()

    results = run(Path(args.originals), Path(args.transformed), Path(args.out) if args.out else None)
    summarize(results)


if __name__ == "__main__":
    main()
