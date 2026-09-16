# Phase 0 — Kill Tests

Phase 0 answers exactly two questions before any further implementation work
happens:

1. **Do JPEG near-duplicates delta-compress well enough** to justify the
   ARC-Dedup direction (reference + delta storage)?
2. **Does the cost-adaptive premise** (`U(s) = alpha*S(s) - beta*C(s)`)
   actually produce a meaningful, size-dependent trade-off between
   verification cost and storage saving?

Nothing beyond these two questions is in scope for Phase 0. No storage
system, no AWS/S3, no FastAPI, no dashboard, no correlation clustering
implementation, no CNN/embeddings, no patent material.

## 1. Prepare source images

`make_test_pairs.py` needs ~40 source images. You have two options:

```bash
# Option A: use your own images (>= 40 recommended)
python make_test_pairs.py --source-dir /path/to/your/images

# Option B: generate deterministic synthetic images (no external dataset)
python make_test_pairs.py --synthetic
```

The script will refuse to run (and tell you why) if neither flag is given —
it will not silently download or invent a dataset without telling you.

This produces:

```
phase0/data/originals/       # 001.jpg ... 040.jpg
phase0/data/transformed/     # 001_resize.jpg, 001_jpegq50.jpg,
                              # 001_crop.jpg, 001_bright.jpg,
                              # 001_rotate.jpg, 001_png.png, ...
```

6 transformations per original x 40 originals = ~240 transformed images
(~240 near-duplicate pairs), per the Phase 0 spec:

| Transformation     | Definition                          |
|---------------------|--------------------------------------|
| `resize`            | resized to 70% of original dimensions |
| `jpegq50`           | re-encoded at JPEG quality 50        |
| `crop`              | center crop retaining ~90% of area   |
| `bright`            | brightness +15%                      |
| `rotate`            | rotated 5 degrees                    |
| `png`               | format conversion (JPG -> PNG)       |

## 2. Run the delta experiment

```bash
python delta_probe.py
```

For every (original, transformed) pair this computes two kinds of delta:

- **Raw encoded-byte delta**: delta between the two files' actual encoded
  bytes, using `xdelta3` if it's installed on PATH, otherwise the
  `bsdiff4` pip package (no custom delta algorithm is implemented). If
  neither is available, the script reports the missing dependency
  explicitly instead of substituting an unrelated compression method.
- **Pixel delta**: both images decoded to RGB with Pillow, raw pixel
  buffers subtracted, and the difference compressed with `zlib`. When
  dimensions differ (resize, crop), the comparison is restricted to the
  overlapping top-left region and this is recorded per-pair rather than
  silently resizing one image to match the other.

Output: `phase0/delta_results.json` (full per-pair data) plus a printed
table of median/p90 `delta_size / original_size` ratios broken down by
transformation type, for both raw and pixel deltas.

### Dependencies

- `xdelta3` (preferred, external binary) — e.g. `choco install xdelta3`
  (Windows, requires an admin shell) or your OS package manager.
- `bsdiff4` (fallback, pip package) — `pip install bsdiff4`.
- `Pillow`, `numpy` for image decoding and pixel-delta computation.

## 3. Run the cost-scaling experiment

```bash
python cost_scaling_analysis.py
```

Evaluates `U(s) = alpha*S(s) - beta*C(s)` across a realistic image-size
range (10 KB - 20 MB) and several `beta/alpha` ratios spanning multiple
orders of magnitude (`1e-6` to `1`), using the models:

- `storage_saving_model(s) = s` (naive best-case saving)
- `verification_cost_model(s) = s` (cost approximately linear in
  pixel count / file size — documented assumption, not derived)

Output: `phase0/tradeoff_plot.png` plus a printed report of whether
`U(s)` crosses zero within the tested size range, and at which
`beta/alpha` ratios.

### Dependencies

- `numpy`, `matplotlib`.

## The two kill-gates

**Gate 1 — delta viability.** Look at the *raw* median ratio for `resize`,
`jpegq50`, and `bright` specifically. If **any** of these exceeds 0.60,
ARC-Dedup is rejected as the headline direction and the project pivots to
correlation clustering (A2). All three must be <= 0.60 to continue with
ARC-Dedup.

**Gate 2 — cost-adaptive validity.** If `U(s)` does not cross zero at a
realistic image size for any tested `beta/alpha` ratio, the cost-adaptive
framing is not validated by this model and must be recorded as a
limitation rather than treated as proven.

Neither script makes this decision. The decision belongs in
`phase0/DECISION.md`, written by hand after inspecting the actual
numbers.
