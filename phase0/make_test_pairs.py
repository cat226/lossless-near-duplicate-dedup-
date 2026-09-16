"""
Phase 0 — test pair generator.

Produces ~40 original images and, for each, 6 transformed near-duplicates:
    resize      -> resized to 70% of original dimensions, re-encoded JPEG
    jpegq50     -> re-encoded at JPEG quality 50
    crop        -> center crop retaining ~90% of area
    bright      -> brightness increased by 15%
    rotate      -> rotated by 5 degrees
    png         -> format conversion (JPG <-> PNG)

Originals are written to phase0/data/originals/
Transformed images are written to phase0/data/transformed/ using the
naming convention {stem}_{transform}.{ext} so delta_probe.py can pair
them back up reliably by filename.

Source images:
    By default this script looks for --source-dir. If none is given and
    --synthetic is not passed, it will ask you for a path rather than
    silently downloading or fabricating a dataset.

    --synthetic generates ~40 deterministic procedural images locally
    (gradients / noise / geometric shapes rendered with Pillow + numpy,
    fixed random seed) so no external dataset or network access is
    required. This is an explicit, visible choice, not a silent fallback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

RNG_SEED = 12345
N_ORIGINALS = 40
SYNTHETIC_SIZE = (1024, 768)

ROOT = Path(__file__).resolve().parent
ORIGINALS_DIR = ROOT / "data" / "originals"
TRANSFORMED_DIR = ROOT / "data" / "transformed"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def generate_synthetic_originals(n: int, out_dir: Path) -> list[Path]:
    """Generate n deterministic synthetic photo-like images.

    Each image mixes a smooth gradient, sinusoidal texture, random
    geometric shapes and mild noise so that JPEG compression and the
    various transformations behave roughly like they would on real
    photographs (i.e. not perfectly flat / trivially compressible).
    """
    rng = np.random.default_rng(RNG_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = SYNTHETIC_SIZE
    paths = []

    yy, xx = np.mgrid[0:h, 0:w]

    for i in range(n):
        base_color = rng.integers(30, 220, size=3)
        freq_x = rng.uniform(0.005, 0.03)
        freq_y = rng.uniform(0.005, 0.03)
        phase = rng.uniform(0, 2 * np.pi)

        gradient = (xx / w + yy / h) / 2.0
        texture = 0.5 + 0.5 * np.sin(xx * freq_x + yy * freq_y + phase)

        img = np.zeros((h, w, 3), dtype=np.float64)
        for c in range(3):
            channel = base_color[c] * (0.5 + 0.5 * gradient) * (0.6 + 0.4 * texture)
            img[:, :, c] = channel

        noise = rng.normal(0, 8, size=(h, w, 3))
        img = img + noise

        arr = np.clip(img, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr, mode="RGB")

        n_shapes = rng.integers(3, 8)
        from PIL import ImageDraw

        draw = ImageDraw.Draw(pil_img)
        for _ in range(n_shapes):
            shape_color = tuple(int(v) for v in rng.integers(0, 255, size=3))
            x0, y0 = rng.integers(0, w), rng.integers(0, h)
            sw, sh = rng.integers(20, w // 3), rng.integers(20, h // 3)
            x1, y1 = min(x0 + sw, w - 1), min(y0 + sh, h - 1)
            if rng.random() < 0.5:
                draw.rectangle([x0, y0, x1, y1], fill=shape_color)
            else:
                draw.ellipse([x0, y0, x1, y1], fill=shape_color)

        stem = f"{i + 1:03d}"
        out_path = out_dir / f"{stem}.jpg"
        pil_img.save(out_path, "JPEG", quality=90)
        paths.append(out_path)

    return paths


def collect_source_images(source_dir: Path, n: int) -> list[Path]:
    candidates = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(candidates) < n:
        print(
            f"WARNING: only found {len(candidates)} images in {source_dir}, "
            f"expected at least {n}. Proceeding with what is available.",
            file=sys.stderr,
        )
    return candidates[:n]


def prepare_originals_from_source(source_paths: list[Path], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for i, src in enumerate(source_paths):
        stem = f"{i + 1:03d}"
        img = Image.open(src).convert("RGB")
        out_path = out_dir / f"{stem}.jpg"
        img.save(out_path, "JPEG", quality=90)
        result.append(out_path)
    return result


def make_transformations(original_path: Path, out_dir: Path) -> dict[str, Path]:
    stem = original_path.stem
    img = Image.open(original_path).convert("RGB")
    w, h = img.size
    made = {}

    # resize: 70% of original dimensions, re-encoded JPEG
    resized = img.resize((max(1, int(w * 0.7)), max(1, int(h * 0.7))), Image.LANCZOS)
    p = out_dir / f"{stem}_resize.jpg"
    resized.save(p, "JPEG", quality=90)
    made["resize"] = p

    # jpegq50: same pixels, re-encoded at quality 50
    p = out_dir / f"{stem}_jpegq50.jpg"
    img.save(p, "JPEG", quality=50)
    made["jpegq50"] = p

    # crop: center crop retaining ~90% of area -> scale factor sqrt(0.9) per dimension
    scale = 0.9 ** 0.5
    cw, ch = int(w * scale), int(h * scale)
    left = (w - cw) // 2
    top = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    p = out_dir / f"{stem}_crop.jpg"
    cropped.save(p, "JPEG", quality=90)
    made["crop"] = p

    # bright: brightness +15%
    brightened = ImageEnhance.Brightness(img).enhance(1.15)
    p = out_dir / f"{stem}_bright.jpg"
    brightened.save(p, "JPEG", quality=90)
    made["bright"] = p

    # rotate: 5 degrees, expand=False to keep dimensions comparable,
    # fill introduced corners with edge-replicated content via expand+crop
    rotated = img.rotate(5, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    p = out_dir / f"{stem}_rotate.jpg"
    rotated.save(p, "JPEG", quality=90)
    made["rotate"] = p

    # format: PNG <-> JPG conversion (original is JPG, so convert to PNG)
    p = out_dir / f"{stem}_png.png"
    img.save(p, "PNG")
    made["png"] = p

    return made


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 0 near-duplicate test pairs.")
    parser.add_argument("--source-dir", type=str, default=None,
                         help="Directory of >=40 source images to use as originals.")
    parser.add_argument("--synthetic", action="store_true",
                         help="Generate deterministic synthetic originals instead of using a source directory.")
    parser.add_argument("--n", type=int, default=N_ORIGINALS,
                         help=f"Number of originals to prepare (default {N_ORIGINALS}).")
    args = parser.parse_args()

    if not args.source_dir and not args.synthetic:
        print(
            "No --source-dir given and --synthetic not passed.\n"
            "Please either:\n"
            "  1) re-run with --source-dir <path to a folder of >=40 images>, or\n"
            "  2) re-run with --synthetic to generate deterministic procedural test images.\n"
            "This script will not silently pick a dataset for you.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.source_dir:
        source_dir = Path(args.source_dir)
        if not source_dir.is_dir():
            print(f"ERROR: source dir does not exist: {source_dir}", file=sys.stderr)
            sys.exit(1)
        source_paths = collect_source_images(source_dir, args.n)
        if not source_paths:
            print(f"ERROR: no usable images found in {source_dir}", file=sys.stderr)
            sys.exit(1)
        originals = prepare_originals_from_source(source_paths, ORIGINALS_DIR)
        print(f"Prepared {len(originals)} originals from {source_dir} -> {ORIGINALS_DIR}")
    else:
        originals = generate_synthetic_originals(args.n, ORIGINALS_DIR)
        print(f"Generated {len(originals)} synthetic originals (seed={RNG_SEED}) -> {ORIGINALS_DIR}")

    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    for orig in originals:
        made = make_transformations(orig, TRANSFORMED_DIR)
        total_pairs += len(made)

    print(f"Generated {total_pairs} transformed images across {len(originals)} originals "
          f"({total_pairs / max(1, len(originals)):.1f} transforms/original) -> {TRANSFORMED_DIR}")
    print(f"Target near-duplicate pairs: ~{total_pairs}")


if __name__ == "__main__":
    main()
