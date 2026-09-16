# Phase 0 Decision

**Date:** 16 September 2026  
**Project:** Cost-Adaptive Near-Duplicate Image Storage  
**Phase:** Phase 0 — Kill-Tests

---

## 1. Objective

Phase 0 was conducted to answer two questions before committing to the
full algorithmic direction:

1. Do JPEG near-duplicates delta-compress well enough to justify the
   ARC-Dedup / delta-storage approach?
2. Does the proposed cost-adaptive verification premise produce a
   meaningful image-size-dependent trade-off between verification cost
   and storage savings?

The predefined Phase 0 decision gate was used without changing its
thresholds after observing the results.

---

## 2. Experimental Setup

The experiment used:

- 40 source images
- 6 transformations per source image
- 240 transformed image pairs in total
- Synthetic JPEG source images generated deterministically with seed 12345
- `bsdiff4` for encoded-byte delta computation because `xdelta3` was
  unavailable in the environment
- Pillow, NumPy, and zlib for pixel-level delta analysis

The transformations were:

- resize
- JPEG quality 50 re-encoding
- center crop
- brightness adjustment
- 5-degree rotation
- PNG/JPG format conversion

The experiment completed successfully with:

**240/240 pairs processed and 0 errors.**

---

## 3. Delta Compression Results

The measured median and p90 delta-size-to-original-size ratios were:

| Transformation | Raw Median | Raw P90 | Pixel Median | Pixel P90 |
|---|---:|---:|---:|---:|
| Resize | 0.362 | 0.371 | 4.586 | 4.899 |
| JPEG quality 50 | 0.178 | 0.188 | 6.418 | 6.487 |
| Crop | 0.884 | 0.891 | 7.520 | 7.803 |
| Brightness | 0.938 | 0.943 | 3.456 | 3.575 |
| Rotation | 0.909 | 0.915 | 8.212 | 8.440 |
| Format conversion | 5.840 | 5.874 | 0.012 | 0.012 |

The predefined gate evaluates the raw encoded-byte delta ratios for
resize, JPEG compression, and brightness.

The relevant median values are:

- Resize: **0.362 ≤ 0.60**
- JPEG quality 50: **0.178 ≤ 0.60**
- Brightness: **0.938 > 0.60**

Therefore, the brightness transformation fails the predefined
delta-compression threshold.

---

## 4. Cost-Scaling Result

The cost-adaptive utility was evaluated using:

    U(size) = alpha * S(size) - beta * C(size)

with:

    S(size) = size
    C(size) ≈ size

under the specified linear cost model.

The experiment tested beta/alpha ratios from approximately `1e-6`
through `1e0` over image sizes from 10 KB to 20 MB.

No image-size-dependent zero-crossing was observed.

Because both storage saving and verification cost were modeled as
linear in image size, the utility simplifies to:

    U(size) = (alpha - beta) * size

Therefore, the sign of U depends on the beta/alpha ratio rather than
on image size.

The experiment did not identify a regime in which verification becomes
worthwhile for large images but not worthwhile for small images, or
vice versa.

This means that the original cost-adaptive framing is not independently
validated by this model and will be documented as a limitation rather
than treated as a demonstrated algorithmic contribution.

---

## 5. Decision

### DECISION: PIVOT TO A2 — CORRELATION CLUSTERING

According to the predefined Phase 0 gate, the project will **pivot from
ARC-Dedup / delta-based storage to the correlation-clustering direction
(A2)**.

The primary reason is that the median raw delta ratio for brightness
transformations was **0.938**, which exceeds the predefined threshold
of **0.60**.

Although resize and JPEG quality-50 transformations produced ratios
below the threshold, the failure on brightness means that the
delta-storage approach does not satisfy the predefined viability gate
for the selected common transformation set.

This decision is based on the measured Phase 0 results rather than an
assumption that ARC-Dedup will perform well.

---

## 6. Important Experimental Limitation

The source images used in this Phase 0 experiment were synthetically
generated rather than sourced from a real photographic dataset.

They consisted of procedurally generated gradients, sinusoidal
textures, shapes, and noise and were encoded as JPEG images.

Therefore, the measured delta ratios should not be interpreted as a
general characterization of all real-world photographic JPEG
near-duplicates.

The Phase 0 decision nevertheless follows the predefined gate exactly.
The synthetic-data limitation will be explicitly acknowledged in the
final report.

The raw-byte delta experiment also used `bsdiff4` rather than
`xdelta3`, because `xdelta3` was unavailable in the current environment.
This is a documented experimental limitation.

---

## 7. Next Phase

The next implementation direction is:

**A2 — Correlation Clustering**

The next phase will investigate whether treating pairwise near-duplicate
relationships as a signed weighted graph and minimizing clustering
disagreement provides a meaningful algorithmic improvement over
threshold-based chaining.

The Phase 1c gate will determine whether this formulation actually
improves the clustering objective before proceeding further.

No claim of novelty or patentability is made at this checkpoint.

---

**Phase 0 status: COMPLETE**

**Selected path: A2 — Correlation Clustering**
