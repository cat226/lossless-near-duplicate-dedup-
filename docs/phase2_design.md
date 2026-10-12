# Phase 2 Experimental Design — Candidate Generation

Status: **FROZEN** (approved by the user on 2026-10-12 per `project_handoff.md`
section 7, including the preflight amendments). The gate, M1 R = 10, the M2 Otsu
operating point, the M3 union and reuse of the fixed Phase 1c tau were approved
as written. Section 17 fixes implementation details that the handoff left
unspecified; they were decided before any Phase 2 result existed.

Source: `project_handoff.md` section 7, with the preflight results in section 14.
Frozen and untouchable: `phase0/`, `results/phase1c_results.json`,
`docs/phase1c_design.md`, and Phase 1c code in `cc_dedup/`, `baselines/`,
`experiments/` (imported, never edited).

**Legend:** **[R]** required by project rules · **[D]** design decision made
here (adopted default, needs user confirmation) · **[L]** limitation.

## 1. Research question [R]

Can a cheap per-image signature reduce the number of expensive Phase 1c
distance computations needed to build the signed graph, while retaining high
recall of true near-duplicate pairs?

Pipeline: all images → cheap signature → candidate pairs → expensive pixel
verification (candidates only) → signed graph → correlation clustering.

## 2. Scope [R][L]

Phase 2a cuts the **count of expensive comparisons**. The cheap stage may
still be exhaustive O(n²) over tiny vectorised signatures, so this is a
constant-factor attack and is reported as such. Sub-quadratic indexing (LSH,
multi-index hashing) is Phase 4, only if Phase 2 passes. No novelty claim is
made from pHash or correlation clustering alone.

## 3. Dataset [D]

- Reuse the Phase 0 sequential-RNG generator (`np.random.default_rng(12345)`,
  1024x768, JPEG q90, six deterministic transforms). Import or copy into
  `phase2/`; never edit `phase0/`; write to a new output directory, not
  `phase0/data/`.
- Datasets are **nested by prefix**: the first 40 families of any run are the
  Phase 0 images. Per-family `(12345, k)` seeding is rejected because it would
  not reproduce the Phase 1c images.
- Stream family by family; never hold all full-resolution images in RAM.
  Record numpy and Pillow versions (JPEG encoder differences could change pixels).
- **Preflight evidence (section 14):** nestedness and byte-for-byte
  reproduction of Phase 1c originals 001-003 were both verified.

| F | n = 7F | all pairs | true pairs (21F) | perfect-filter reduction |
|---|---|---|---|---|
| 40 | 280 | 39,060 | 840 | 0.9785 |
| 100 | 700 | 244,650 | 2,100 | 0.9914 |
| 200 | 1,400 | 979,300 | 4,200 | 0.9957 |
| 400 | 2,800 | 3,918,600 | 8,400 | 0.9979 |
| 800 | 5,600 | 15,677,200 | 16,800 | 0.9989 |

Extrapolated all-pairs cost at 0.1300 s/pair (the Phase 1c mean): 8.8 h, 35 h,
5.9 d, 23.6 d for F = 100..800. These are **extrapolated**, never measured, and
are always labelled so. Storage is about 0.28 MB/image.

## 4. Consistency check [D]

Hash matching is impossible (no hashes were recorded in Phase 1c). Instead,
rebuild the Phase 1c graph on regenerated F=40 data and compare against
tau = 0.5805796715948317 and 10,791 positive edges. A mismatch means n=280 is
reported as a **new dataset**, not as Phase 1c.

## 5. Ground truth [R]

A pair is a true near-duplicate iff both images derive from the same source
(21 pairs per family: 6 original-transform and 15 transform-transform). Ground
truth is used **only** to score candidate sets after they are produced.

## 6. Methods (frozen once approved; no tuning) [D]

- **M1 pHash:** grayscale -> 32x32 -> 2D DCT -> top-left 8x8 -> bit = coefficient
  > median (DC excluded from the median); 64-bit hash. Candidate iff Hamming
  distance <= R. **Primary R = 10.**
- **M2 coarse signature:** concatenation of three L2-normalised, equally
  weighted blocks: 32x32 grayscale (1024-d), 8-bin-per-channel colour histogram
  (24-d), 8-bin Sobel gradient-orientation histogram (8-d). L2 distance.
  Primary operating point = **Otsu over the all-pairs signature-distance
  histogram** (same protocol as Phase 1c: 256 bins, flatness tolerance 1e-9,
  fallback (min+max)/2). Candidate iff distance < tau_sig.
- **M3 hybrid:** M1 union M2.
- **Rejected:** per-image kNN with fixed k (k >= 7 encodes the known family size,
  a ground-truth leak).
- **Secondary operating curves** (reported, never used to re-pick the primary):
  M1 with R in {4,6,8,10,12,14,16}; M2 with tau_sig x {0.5,0.75,1.0,1.25,1.5}.

## 7. Metrics [D]

- **Candidate recall** = |P intersect T| / |T| (primary), with original-transform
  vs transform-transform and per-transformation breakdowns. A pair counts toward
  every transformation type among its two images; the original counts toward none.
- **Reduction** = 1 - |P| / (n(n-1)/2); also report |P| and |P|/|T|.

## 8. Non-candidate pairs and the verifier [D][L]

Non-candidate pairs are implicit negatives: never verified, never positive
edges. Missed true pairs are unrecoverable recall loss. Verified pairs use the
**fixed Phase 1c tau = 0.5805796715948317 at all n**.

## 9. Baselines [R]

- **B0** all-pairs: measured at n=280, counted/extrapolated beyond.
- **B1** random-candidate control: uniform random pair set of equal size per
  method and n (seed 12345). A method whose recall is not above B1 has no signal.

## 10. Runtime [D]

Single process, `time.perf_counter`, no parallelism, hardware recorded. Primary
cost metric is the **expensive-comparison count**.

**Approved exception (user, 2026-10-12):** the one-time n=280 distance-matrix
recompute (section 11) may use multiprocessing around the unchanged
`cc_dedup.graph.pairwise_distance`, with images loaded once into shared memory and
the wrapper living in `phase2/`. **No timing claim is made from that run.** The
200-pair verification-cost sample and every other timing are taken single-process,
after the recompute has finished. Verification time at n > 280
is a **projection** from a fixed random sample of 200 *candidate* pairs
(seed 12345). Use 0.1300 s/pair only for the extrapolated all-pairs baseline,
since cost varies strongly by pair type (section 14).

## 11. Secondary analysis at n=280 only [D][L]

Recompute and persist the 280x280 distance matrix once (about 85 minutes; it
doubles as the section 4 check) to `results/phase2_n280_distance_matrix.npy`
(new file). Then report: E+ retention (of 10,791); KwikCluster (seed 12345) on
the candidate graph scored by D on the full graph; and **verifier-confirmed
recall** (true pairs with r < tau).

The verifier compares only the overlapping top-left region, so resize and crop
pairs are misaligned and can be rejected whatever the candidate stage does.
End-to-end quality is therefore **capped by the verifier, not by candidate
generation.**

## 12. Gate (proposed values) [D]

A method passes iff at **every** size: (1) candidate recall >= 0.95, AND
(2) reduction >= 0.90, AND (3) recall for every transformation type >= 0.85.
Phase 2 succeeds iff at least one method passes. No secondary metric rescues a
failed primary gate; all methods are reported regardless.

## 13. Pilot and failure conditions [R]

**Pilot (F=3: 21 images, 210 pairs, 63 true pairs):** determinism across two
runs, nestedness, signature shapes/dtypes, hand-checked candidate/true/reduction
arithmetic, distance output identical to Phase 1c on sampled pairs, Otsu on
signature distances. Pilot output is not evidence and is not committed.

**Failure conditions:** any frozen path changed; any parameter, operating point
or seed changed after seeing results; ground truth used before candidate sets
are fixed; non-determinism; distance mismatch with Phase 1c on the consistency
check; a method's recall not above B1; all methods fail (report the negative
result, no retuning).

## 14. Preflight evidence (this branch, `main` @ d698ee1)

- Repo: both PRs merged; no Phase 2 branch, design doc or PR existed before this one.
- Phase 0 generator imports unmodified (`generate_synthetic_originals`,
  `make_transformations`; seed 12345; 1024x768).
- Frozen paths: `git diff main` empty for `phase0/`, `results/phase1c_results.json`,
  `docs/phase1c_design.md`, `cc_dedup/`, `baselines/`, `experiments/`.
- **Nestedness verified:** the first 3 images of an n=5 run are byte-identical to an n=3 run.
- **Reproduction verified:** regenerated originals 001-003 are byte-identical
  (SHA-256) to the Phase 1c images on disk, with numpy 2.5.1 / Pillow 12.3.0.
- **Size-table arithmetic verified** (pair counts, reductions, extrapolations);
  5079.1731483 s / 39,060 pairs = 0.13004 s/pair.
- Handoff single-sample timings (not a rigorous benchmark): same-size jpegq50 pair
  0.231 s; dimension-mismatch resize pair 0.041 s; different-source same-size pair 0.080 s.

## 15. Decisions (resolved 2026-10-12)

1. Gate (0.95 / 0.90 / 0.85 at every size), M1 R = 10, M2 Otsu operating point,
   M3 union, and reuse of the fixed Phase 1c tau: **approved as written.**
2. The n=280 distance-matrix recompute: **approved**, in parallel (section 10).
3. No Phase 2 steps existed in the repo before this branch.
4. Standing approval: proceed from the freeze commit through implementation, the
   F=3 pilot and the full run without stopping; stop if a pilot validation item
   fails, and stop at the audit for approval. No PR before the audit is approved.

## 16. Known risks (hypotheses, not results) [L]

- pHash may miss rotated and cropped images (64-bit random-pair Hamming centres
  near 32, so R=10 is selective).
- Otsu on signature distances may keep too many pairs: true pairs are ~2% of all
  pairs, so the split may retain far more than 10% and fail the reduction gate.
  Do not retune; report it.
- Reduction is likely optimistic: synthetic families are independent random
  images, so hard negatives are rare.
- Phase 1c distance has low precision (KwikCluster 0.168), so family-truth recall
  is not the same as E+ fidelity.
- Results hold only for this generator and these six transformations. Real-image
  validity is the job of Phase 3 (`docs/phase3_dataset_plan.md`).

## 17. Implementation specifics fixed at freeze (decided before any Phase 2 result)

These fill gaps the handoff left open. None can be changed after results exist.

1. **Data and node order.** One dataset is generated at F=800 into `phase2/data/`
   (gitignored) using the Phase 0 generator imported unmodified; smaller F use the
   first F families (prefix). Family k has original stem `f"{k:03d}"`. For a given F
   the nodes are the F originals sorted by name, then the transformed images of
   families 1..F sorted by filename: for F=40 this is exactly the Phase 1c order
   (`discover_images`). Truth label = family; transformation label = filename suffix.
2. **Decoding.** `Image.open(...).convert("RGB")` as `uint8`, as in Phase 1c.
3. **M1 pHash.** Grayscale = PIL `convert("L")`; resized to 32x32 with
   `Image.LANCZOS` from the full-resolution image; orthonormal 2D DCT-II as
   `D X D^T`; top-left 8x8 block; median taken over the 63 AC coefficients; bit =
   coefficient > median for **all 64** coefficients (the DC coefficient gets a bit
   but is excluded from the median); bit k = row-major index k of a `uint64`.
   Hamming distance = popcount of the XOR; candidate iff distance <= R.
4. **M2 coarse signature.** Block (a): the 32x32 grayscale array (values 0-255, no
   mean-centering) flattened, L2-normalised. Block (b): per-channel 8-bin histogram
   with bin = value // 32 over all full-resolution pixels, R then G then B (24-d),
   L2-normalised. Block (c): Sobel gx, gy over the 30x30 interior of the 32x32
   grayscale, signed angle atan2(gy, gx) in 8 equal bins of [-pi, pi), magnitude
   weighted, L2-normalised. A zero-norm block stays zero. Signature = concatenation
   (1056-d), stored as `float32`; distances are computed in `float64` from the
   stored values. Candidate iff L2 distance < tau_sig.
5. **tau_sig.** `cc_dedup.graph.compute_otsu_threshold` (256 bins, flatness
   tolerance 1e-9, fallback (min+max)/2), imported unmodified, applied separately at
   each n to the all-pairs signature distances; `threshold_method` is recorded.
   Secondary M2 points are multiples of that per-n primary tau_sig.
6. **B1.** For each (method, n, operating point), draw |P| pair indices without
   replacement from the n(n-1)/2 upper-triangle positions with a fresh
   `np.random.default_rng(12345)`.
7. **Metrics.** Truth pairs T: same family. Per-transformation recall uses the
   pairs involving at least one image of that transformation (6 true pairs per
   family per type). Original-transform (OT) pairs have exactly one original;
   transform-transform (TT) pairs have none.
8. **Section 11 definitions (n=280).** V = pairs with r < tau (tau = fixed Phase 1c
   value). *E+ retention* = |P and V| / |V|. *Verifier cap* = |T and V| / |T|.
   *Verifier-confirmed recall* = |P and T and V| / |T|. *Retention of verifiable true
   pairs* = |P and T and V| / |T and V|. KwikCluster (seed 12345) runs on the
   candidate graph: sign +1 for pairs in P and V, -1 otherwise; D is scored on the
   **full** graph (sign +1 iff r < tau, weight |tau - r|). The consistency check also
   reports full-graph KwikCluster and union-find D, which must equal Phase 1c's
   2099.7236160170482 (82 clusters) and 5358.582268054072 (1 cluster).
9. **Timing.** Verification time = wall time of one `pairwise_distance` call on
   already-decoded arrays, `time.perf_counter`, single process, taken after the
   recompute has finished. Sample = `np.random.default_rng(12345).choice(|P|,
   min(200, |P|), replace=False)` of each primary candidate set per (method, n).
   Projection = |P| x sample mean, always labelled a projection. Cheap-stage costs
   (signature extraction, all-pairs signature distance) are also timed
   single-process after the recompute.
10. **Artifacts.** `phase2/data/` and `phase2/artifacts/` (signatures, thresholds,
    candidate-set hashes) are gitignored. Committed on approval:
    `results/phase2_results.json`, `results/phase2_n280_distance_matrix.npy` and its
    metadata file, `docs/phase2_report.md`. Candidate sets are identified by size
    and the SHA-256 of the sorted flat-index array.
11. **Sizes.** F in {40, 100, 200, 400, 800}. Full pipeline plus section 11 at
    n=280; candidate-only (with sampled verification-cost projection) at n >= 700.
