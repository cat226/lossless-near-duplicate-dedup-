# Phase 2b Report — Fixed-Threshold M2* on Held-Out Families

Design: `docs/phase2b_design.md` (frozen in commit `730cad2`). Results:
`results/phase2b_results.json`, `results/phase2b_timing.json`.

## 1. Verdict

**The pre-registered hypothesis was supported: M2\* met the gate at all five sizes on
held-out families 801-1600.** M2\* is Phase 2's coarse signature with the single fixed,
n-independent threshold tau\* = 0.6004747353136561 (0.5 x the Phase 2 n=280 Otsu
tau_sig), candidate iff L2 distance < tau\*. tau\* was not recomputed, rescaled or tuned,
and nothing was changed after seeing results.

Gate (every size): candidate recall >= 0.95 AND reduction >= 0.90 AND per-transformation
recall >= 0.85.

| n (families) | pairs kept | recall | reduction | min per-transformation recall | random-control recall | gate |
|---|---|---|---|---|---|---|
| 280 (801-840) | 1,212 (3.10%) | 0.9845 | 0.9690 | 0.9458 | 0.0333 | **pass** |
| 700 (801-900) | 4,595 (1.88%) | 0.9762 | 0.9812 | 0.9167 | 0.0148 | **pass** |
| 1,400 (801-1000) | 15,555 (1.59%) | 0.9779 | 0.9841 | 0.9225 | 0.0148 | **pass** |
| 2,800 (801-1200) | 50,783 (1.30%) | 0.9785 | 0.9870 | 0.9246 | 0.0115 | **pass** |
| 5,600 (801-1600) | 187,381 (1.20%) | 0.9820 | 0.9880 | 0.9369 | 0.0107 | **pass** |

Thinnest margins to the thresholds: recall +0.026 (n=700), reduction +0.069 (n=280),
per-transformation recall +0.067 (n=700). Perfect-filter reduction (keeping only the true
pairs) is 0.9785 to 0.9989, so the method sits within 0.0095 to 0.0116 of that bound at
every size.

M1 (pHash) and M3 were not evaluated; M1 stays the reported negative result of Phase 2.

## 2. Recall by transformation

| n | resize | jpegq50 | crop | bright | rotate | png | original-transform | transform-transform |
|---|---|---|---|---|---|---|---|---|
| 280 | 0.9917 | 0.9917 | 0.9917 | 0.9875 | 0.9458 | 0.9917 | 0.9917 | 0.9817 |
| 700 | 0.9867 | 0.9867 | 0.9867 | 0.9833 | 0.9167 | 0.9867 | 0.9867 | 0.9720 |
| 1,400 | 0.9875 | 0.9875 | 0.9883 | 0.9842 | 0.9225 | 0.9875 | 0.9875 | 0.9740 |
| 2,800 | 0.9879 | 0.9883 | 0.9883 | 0.9842 | 0.9246 | 0.9879 | 0.9879 | 0.9747 |
| 5,600 | 0.9902 | 0.9904 | 0.9898 | 0.9865 | 0.9369 | 0.9900 | 0.9900 | 0.9788 |

Rotation (5 degrees) is the weakest transformation at every size (0.917-0.946), as it was in
Phase 2, but clears the 0.85 floor.

## 3. Relation to Phase 2 (post-hoc, descriptive; not part of the gate)

This comparison was made after the Phase 2b results were known and is not a second test.
In Phase 2, M2 at 0.5 x the per-n Otsu tau_sig (itself a post-hoc observation) gave recall
0.970-0.980, reduction 0.967-0.983 and minimum per-transformation recall 0.896-0.931. The
held-out numbers are in the same range (recall 0.976-0.985, reduction 0.969-0.988, minimum
0.917-0.946). The fixed threshold keeps a shrinking share of pairs as n grows (3.1% to 1.2%),
in contrast with Phase 2's Otsu operating point (29% to 44.5%). This is descriptive: the two
experiments used different families and different thresholds, so it is a consistency note, not
a second test.

## 4. Cost (single-process; projections labelled as such)

Hardware: Windows 11, Intel64 Family 6 Model 154, 12 logical CPUs, Python 3.12.10, numpy
2.5.1, Pillow 12.3.0; BLAS threads pinned to 1; run after every other job had finished.
Signature extraction: 0.0221 s per image (measured on 280 images). Cheap stage (all-pairs
signature distances plus the candidate rule): 0.00 s at n=280 (below timer resolution) to
1.62 s at n=5,600.

Verification time is a **projection**: |P| x the mean of a fixed 200-pair sample of candidate
pairs (seed 12345, `pairwise_distance` only, decode excluded). The all-pairs baseline above
n=280 is **extrapolated** at 0.1300 s/pair; at n=280 it is the recorded Phase 1c measurement
(5,079 s).

| n | all-pairs (extrapolated) | sample mean | projected verification | projected speedup |
|---|---|---|---|---|
| 280 | 1.41 h (measured 5,079 s) | 0.1183 s | 0.040 h | 35x |
| 700 | 8.83 h | 0.1172 s | 0.150 h | 59x |
| 1,400 | 35.4 h | 0.1102 s | 0.476 h | 74x |
| 2,800 | 141.5 h | 0.0894 s | 1.26 h | 112x |
| 5,600 | 566 h | 0.0918 s | 4.78 h | 119x |

The sample mean falls from 0.118 s to 0.092 s as n grows. The cause was not investigated (a
change in the mix of sampled pair types is one possibility, not verified), so the speedups carry
that uncertainty on top of being projections.

## 5. Pilot and audit

- F=3 pilot (fresh families 801-803): all 22 items passed (disjointness, regenerated Phase 2
  originals matching the Phase 2 manifest, nestedness against both an F=5 run and the full
  fresh set, determinism, shapes and dtypes, reproduction of Phase 2's cached signatures,
  tau\* provenance, hand-checked arithmetic, random-control size). Pilot output is not evidence
  and is not committed.
- `experiments/audit_phase2b.py`: 14/14 checks passed. Candidate sets recomputed with different
  code (scipy `cdist`) match by SHA-256 at all five sizes; recall, reduction and
  per-transformation recall match to 1e-12; 0 pairs lie within 1e-9 of tau\*; the gate verdict is
  re-derived; the fresh families are exactly 801-1600, disjoint from 1-800, and no fresh image is
  byte-identical to a Phase 2 image; no tracked file from earlier phases changed (versus the
  Phase 2 tip, apart from `.gitignore`); signature and candidate code names no ground truth and
  candidate sets are hashed before labels are read; a fresh-process rerun at n=280 is
  byte-identical.
- `tests`: 12 passed (7 from Phase 2, 5 new).

## 6. What this does and does not show

It shows: on this generator, a single fixed threshold met the pre-registered gate at every size
on one held-out draw of 800 families the threshold was not derived from.

It does not show:

- Anything about real photographs. Phase 3 is the test of that, and tau\* is **not** carried
  into it: the signature-distance distribution will differ and any Phase 3 threshold must come
  from a protocol frozen before Phase 3 results.
- Hard-negative behaviour. The families are independent random images, so unrelated pairs are
  probably easy to reject; the falling retained share (3.1% to 1.2%) may partly reflect that, but
  this was not tested.
- End-to-end quality. The verifier was not re-estimated; the Phase 2 verifier cap (79.76% of true
  pairs pass the frozen Phase 1c distance, at n=280 on families 1-40) is a reference only, and
  end-to-end recall cannot exceed the share of true pairs the verifier accepts.
- Draw-to-draw variance. There is one held-out draw and no interval.
- Generality across transformations. Only the six frozen transformations were tested, and the
  rotation margin (0.917 at its lowest) is the thinnest.

tau\* itself was chosen after seeing Phase 2 curves; the held-out test is what made the claim
falsifiable. The 800 fresh families are now consumed: any further threshold or method change
needs a new design frozen before it is run, on families not used here.

## 7. Not done

The PR (to be opened only after the audit is approved), section-11 analysis on fresh data (not in
the design), and Phase 3.
