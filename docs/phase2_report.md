# Phase 2 Report — Candidate Generation

Design: `docs/phase2_design.md` (frozen in commit `15f354b`). Results:
`results/phase2_results.json`, `results/phase2_timing.json`,
`results/phase2_n280_distance_matrix.npy` (+ `.meta.json`).

## 1. Verdict

**The frozen Phase 2 gate FAILED for all three methods (M1, M2, M3). Phase 2 did
not succeed. No parameter, operating point or seed was changed after seeing
results.**

Gate (every size): candidate recall >= 0.95 AND reduction >= 0.90 AND
per-transformation recall >= 0.85.

| n | M1 pHash (R=10) | M2 coarse signature (Otsu) | M3 union |
|---|---|---|---|
| 280 | recall 0.7738, reduction 0.9834, min type 0.2917: **fail** | recall 1.0000, reduction 0.7104: **fail** | 1.0000 / 0.7104: **fail** |
| 700 | 0.7833 / 0.9933 / 0.2950: **fail** | 1.0000 / 0.6209: **fail** | 1.0000 / 0.6209: **fail** |
| 1,400 | 0.7619 / 0.9967 / 0.2558: **fail** | 0.9998 / 0.5855: **fail** | 0.9998 / 0.5855: **fail** |
| 2,800 | 0.7715 / 0.9983 / 0.2742: **fail** | 0.9999 / 0.5595: **fail** | 0.9999 / 0.5595: **fail** |
| 5,600 | 0.7770 / 0.9991 / 0.2790: **fail** | 0.9999 / 0.5549: **fail** | 0.9999 / 0.5549: **fail** |

Why each fails (descriptive, from the table above and the per-transformation recalls):

- **M1** meets the reduction criterion at every size but misses recall. Its
  weakness is geometric: rotation recall is 0.26-0.30 and crop recall 0.77-0.81 at
  every size; the other four transformations are at 0.86-0.88. This matches the
  hypothesis recorded in the design (section 16).
- **M2** has near-perfect recall (every transformation >= 0.9992 at n >= 700) but
  keeps too many pairs. Its Otsu operating point retains 29.0% of all pairs at n=280
  and 44.5% at n=5,600, so reduction falls from 0.710 to 0.555 as n grows. This also
  matches the recorded hypothesis.
- **M3** is dominated by M2: the union adds only 41 pairs over M2 at n=2,800 and 172
  at n=5,600 (`|P|` 1,726,247 -> 1,726,288 and 6,977,665 -> 6,977,837).

All three methods are above the random control B1 at every size, so each carries
signal (for example M1 recall 0.7738 vs B1 0.0119 at n=280). Passing B1 is not
sufficient to pass the gate.

## 2. What was run

- Dataset: Phase 0 generator imported unmodified, F=800 families (5,600 images,
  1.6 GB), prefix-nested. Verified in the pilot: an F=3 run equals the first 3
  families of an F=5 run and of the F=800 dataset, byte for byte, and the regenerated
  images equal the Phase 1c images on disk.
- Methods exactly as frozen (design sections 6 and 17); B1 random control, seed 12345.
- Sizes F in {40, 100, 200, 400, 800}.
- Pilot (F=3): all 20 validation items passed; its numbers are not evidence and are
  not committed.
- n=280 distance-matrix recompute: parallel (approved exception), unchanged
  `pairwise_distance`, images loaded once into shared memory. **No timing claim is
  made from it.**

## 3. Consistency with Phase 1c (n=280)

The recomputed graph reproduces Phase 1c exactly: tau = 0.5805796715948317, 10,791
positive edges, KwikCluster seed 12345 -> 82 clusters with D = 2099.7236160170482,
union-find -> 1 cluster with D = 5358.582268054072, and the regenerated images are
byte-identical to the images in `phase0/data`. So n=280 is the Phase 1c dataset.

## 4. Section 11 analysis at n=280 (secondary, not gating)

Verifier cap: only **79.76%** of true pairs have r < tau under the frozen Phase 1c
distance, so end-to-end recall cannot exceed 0.7976 whatever the candidate stage does
(resize and crop pairs are misaligned by the verifier's overlap-only comparison).

| Method | E+ retention (of 10,791) | verifier-confirmed recall | retention of verifiable true pairs | KwikCluster clusters / D on full graph |
|---|---|---|---|---|
| M1 | 0.0542 | 0.6964 | 0.8731 | 83 / 1936.45 |
| M2 | 0.3142 | 0.7976 | 1.0000 | 81 / 1977.70 |
| M3 | 0.3142 | 0.7976 | 1.0000 | 81 / 1977.70 |

Full-graph KwikCluster (Phase 1c reference) has D = 2099.72. The lower D on the
candidate graphs comes from a **single seed** (12345) and the Phase 1c threats-to-validity
note on single-seed variance applies; no improvement claim is made.

## 5. Cost (single-process, run after the parallel job finished)

Hardware: Windows 11, Intel64 Family 6 Model 154, 12 logical CPUs, Python 3.12.10,
numpy 2.5.1, Pillow 12.3.0; BLAS threads pinned to 1.

- Cheap stage: signature extraction 0.0343 s/image (measured on 280 images);
  all-pairs signature distances plus Otsu: 0.03 s at n=280 and 4.13 s at n=5,600.
- Verification time is a **projection**: mean of a fixed sample of 200 candidate-pair
  `pairwise_distance` calls (seed 12345, decode excluded) times `|P|`. The all-pairs
  baseline at n > 280 is **extrapolated** at 0.1300 s/pair; n=280 all-pairs is the
  recorded Phase 1c measurement (5,079 s).

| n | all-pairs (extrapolated) | M1 projected | M2 projected | M2 speedup |
|---|---|---|---|---|
| 280 | 1.41 h (measured 5,079 s) | 0.043 h | 0.476 h | 2.96x |
| 700 | 8.83 h | 0.110 h | 3.59 h | 2.46x |
| 1,400 | 35.4 h | 0.213 h | 17.6 h | 2.02x |
| 2,800 | 141.5 h | 0.481 h | 72.5 h | 1.95x |
| 5,600 | 566 h | 0.995 h | 320 h | 1.77x |

M1's per-pair verification cost (0.24-0.26 s) is higher than M2's (0.14-0.16 s)
because its candidates are mostly true, same-size, expensive pairs, while M2's include
many cheap unrelated pairs. Speedups are projections and assume the sampled mean holds
for the whole candidate set.

## 6. Post-hoc observation (NOT a result, NOT an operating-point choice)

The pre-declared secondary curves (design section 6; reported, never used to re-pick the
primary) show that M2 at 0.5 x tau_sig would have met all three gate thresholds at all
five sizes: recall 0.970-0.980, reduction 0.967-0.983, minimum per-transformation recall
0.896-0.931. At 0.75 x tau_sig it meets them at n <= 700 only (reduction 0.878-0.883 at
n >= 1,400).

This was seen after the results and therefore **cannot** count toward the frozen gate or
be adopted retroactively. It is a hypothesis: a lower M2 threshold may reach the target
trade-off on this generator. Testing it properly needs a new design frozen before it is
run, on data not used to find it (fresh families, and real images in Phase 3). The synthetic
families are independent random images, so hard negatives are rare and this result is
likely optimistic.

## 7. Audit

`experiments/audit_phase2.py`: 13/13 checks passed.

- Headline numbers (candidate sets by SHA-256, recall, reduction, per-transformation
  recall, gate verdicts) recomputed with independent code (bit-unpack matmul Hamming,
  scipy `cdist` L2, separate Otsu, enumerated true pairs) at all five sizes: exact
  agreement; 0 pairs within 1e-9 of tau_sig.
- Section 11 numbers recomputed independently.
- Fresh-process rerun at n=280 byte-identical to the saved results.
- No tracked file modified or deleted versus `main` except `.gitignore`; Phase 0,
  `results/phase1c_results.json`, `docs/phase1c_design.md` and Phase 1c code untouched.
- Signatures and candidate generation import and name no ground-truth field; candidate
  sets are hashed before labels are read.

## 8. Limitations

- One generator, six transformations, independent random families (few hard negatives).
- The verifier is the frozen Phase 1c distance (cap 0.7976 at n=280).
- Verification costs at n > 280 are projections from a 200-pair sample; the all-pairs
  baseline is extrapolated.
- KwikCluster results use one seed.
- Nothing here supports a claim about real photographs; that is Phase 3.

## 9. Not done

Phase 3 (real data), any follow-up design, and the PR. A PR will be opened only after the
audit is approved.
