# Phase 3 Experimental Design — Real Photographs

Status: **FROZEN.** The user approved decisions D1-D8 on 2026-10-13, exactly as recommended
(Addendum A, which also records the dev/test split by seed and hash). Nothing in this design has
been run on evaluation data and no Phase 3 method result exists. Where the body below still says
[DECISION], the resolution is in Addendum A.

Builds on Phases 0, 1c, 2 and 2b (all merged; `origin/main` was `ac3cb63` at freeze).
Frozen and untouchable: `phase0/`, `results/phase1c_results.json`, `docs/phase1c_design.md`,
Phase 1c code in `cc_dedup/`, `baselines/`, `experiments/run_phase1c.py`,
`experiments/run_pilot.py`, and everything in Phase 2 and 2b (`phase2/`, `phase2b/`, their
designs, results and reports). They are imported or read, never edited.

**Legend:** **[R]** required by project rules or the user's instructions · **[D]** design
decision made here · **[DECISION]** needs a user answer · **[L]** limitation.

## 0. Decisions (all eight approved 2026-10-13 exactly as recommended; see Addendum A)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | D3a prevalence and scale | (a) family-image share 35%, F = n/20 (500 families at n=10,000), as proposed; (b) (a) plus a low-prevalence variant (10% share) at n = 8,000 only (F = 114, 7,202 singletons; 10,000 images at that prevalence would need 9,142 photographs but the test split has 8,500) | (b): (a) alone makes reduction look easy; the variant costs little |
| D2 | Ladder and draws | n in {1,000; 2,500; 5,000; 10,000}; 3 draws (seeds 12345, 12346, 12347) | as proposed |
| D3 | M2 threshold on real data | **A1** per-run Otsu (Phase 2 protocol; reduction decays as n grows); **A2** 0.5 x per-run Otsu (a new rule that 2b does **not** validate: 2b tested a fixed absolute tau\*); **B** calibrate on a labelled dev split, freeze, evaluate once on the locked test split (**relaxes "no ground truth in threshold selection", scoped to the Phase 3 dev split only**) | **B** if you accept the scoped relaxation, with A1 reported as a non-gated comparison; otherwise A1 with its limitation stated. Do **not** reuse tau\* = 0.6004747353136561 or any Phase 2 tau_sig |
| D4 | Threshold for the aligned verifier's distance | **(a)** Otsu over the aligned distances of all candidate pairs of the end-to-end run (label-free, but biased by the candidate mix); **(b)** dev-calibrated so that dev false-accept rate among non-true candidate pairs <= 1% (needs the same scoped relaxation); **(c)** no threshold: report alignment-failure rate, distance distributions and AUC only | **(a)**, plus the AUC from (c) |
| D5 | Gate form | Phase 2 gate on D3a (recall >= 0.95, reduction >= 0.90, per-transformation recall >= 0.85). **Strict:** a method passes iff every (size, draw) passes; **or** judged on the mean over draws | **Strict** (mean and range always reported). D3b: recall >= 0.95 and reduction >= 0.90 in every draw, no per-type criterion |
| D6 | End-to-end clustering size | n = 700 (35 families + 455 singletons; 244,650 pairs) or n = 1,000 (the smallest ladder rung; 499,500 pairs) | **n = 1,000**: the frozen verifier costs about 0.016 s per real-photo pair (exploratory, 40 pairs), so about 2.2 h single-process and well under an hour on 10 workers; candidate sets are then reused from the main run |
| D7 | SSCD comparator | include (only if the maintainers answer the weights-licence question explicitly **before approval**) or drop | **Drop**: the issue has not been answered. If an answer arrives before approval it can be added by addendum |
| D8 | Extra clustering baselines | leave out transitive-closure-plus-greedy-k-cut and Modified Pivot, or include a documented interpretation | **Leave both out** (section 8). Union-find is the published first step (transitive closure) |

## 1. Research question [R]

Do the Phase 2 and 2b findings (candidate recall versus expensive-comparison reduction) hold
on real photographs and on real attacks, and does the resulting candidate graph still support
correlation clustering?

## 2. Frozen inputs [R]

- Cleaned pool: `phase3/manifests/distractor_pool.jsonl`, SHA-256
  `8bd871f592c92b3201d6981965b6ceca942ebd700b357d68a2d3137a1772e76a` (10,000 photographs, 10,000
  distinct SHA-256, content-only exclusion of 236 solid-colour images; excluded list
  `distractor_pool_excluded.jsonl`, SHA-256
  `accbe33e2495861d2fb40e0d5517c447cb99c84ecfce51062bbd08d17ae9a37d`).
- Copydays v1 (original + strong): `phase3/manifests/copydays.jsonl`, SHA-256
  `d0d8e733a89ed6a10019649183e1267d3a65fb70865cb462044b385d3d281495`. 386 images, 157 blocks
  (95 of size 2, 54 of size 3, 7 of size 4, 1 of size 6), longest side capped at 1024 (scale from
  each block's original), PNG RGB. **314 true pairs** (229 original-attacked, 85 attacked-attacked).
- Phase 2 signature code (`phase2.signatures`: M1 64-bit pHash, M2 1056-d coarse signature),
  candidate code (`phase2.candidates`), scoring (`phase2.metrics`) and Otsu
  (`cc_dedup.graph.compute_otsu_threshold`), imported unchanged.
- Primary verifier: `cc_dedup.graph.pairwise_distance` with the fixed Phase 1c
  tau = 0.5805796715948317.
- Aligned verifier configuration (section 5.2).
- Seeds: draws 12345, 12346, 12347; random control and 200-pair samples 12345; clustering seeds
  12345-12364.

## 3. Data

### 3.1 Dev/test split and nesting (every draw d, seed s_d = 12345 + d - 1; recorded in Addendum A.2) [D]

1. Order the pool by key, then `perm = np.random.default_rng(s_d).permutation(10,000)`.
2. **Dev photos** = `perm[0:1500]`; **test photos** = `perm[1500:]` (8,500). The split is always
   made, whichever D3 protocol is chosen, so results are comparable across protocols.
3. Within the test photos, in `perm` order: the first 500 are family sources and the next 6,500
   are singleton distractors. Size n uses the first F(n) = n/20 sources and the first
   S(n) = 0.65 n singletons, so sizes are prefix-nested and a family source never doubles as a
   singleton.

| n | F | singletons | true pairs (21F) | all pairs | perfect-filter reduction |
|---|---|---|---|---|---|
| 1,000 | 50 | 650 | 1,050 | 499,500 | 0.99790 |
| 2,500 | 125 | 1,625 | 2,625 | 3,123,750 | 0.99916 |
| 5,000 | 250 | 3,250 | 5,250 | 12,497,500 | 0.99958 |
| 10,000 | 500 | 6,500 | 10,500 | 49,995,000 | 0.99979 |

(D1b adds, at n = 8,000 only, a low-prevalence variant: F = 114 families and 7,202 singletons, so
798 family images, 9.98% of the images; it needs 7,316 of the 8,500 test photos.) Family images are
35% of the images at every rung of the main ladder, far above realistic prevalence **[L]**. The
largest main rung needs 7,000 test photos (500 sources + 6,500 singletons).

### 3.2 D3a: controlled injection (primary) [D]

Each family source is one cleaned-pool photograph; the **six frozen Phase 0 transformations**
(resize 70%, JPEG q50, ~90%-area centre crop, brightness +15%, 5-degree rotation, JPG to PNG)
are applied by importing the unmodified `make_transformations` from `phase0/make_test_pairs.py`.
A family is the source plus its six transformed images (7 images, 21 true pairs). Generation is
deterministic (no randomness in the transformations).

### 3.3 D3b: native real attacks (sanity check) [D][L]

All 386 Copydays images (157 blocks) plus 4,614 singleton photographs from the test photos
(`perm[1500:]`, first 4,614), so n = 5,000 per draw; only the distractors vary between draws. A
block is one family; true pairs are all pairs within a block (314), the same derivation as
Phase 2 (original-attacked and attacked-attacked). **The Copydays crop and JPEG attack sets are
missing from v1, so D3b is a weaker sanity check that covers only the strong attacks; crop and
JPEG robustness on real images rests on D3a.** D3b is reported separately.

### 3.4 Secondary pool run [D]

The same procedure at n = 5,000 on the **extended pool** (cleaned pool plus the 236 excluded
solid-colour images, 10,236 photographs, permuted by the same rule), three draws. Not gated: it
shows how much the excluded images change the results. That solid-colour images are hard for the
Phase 1c distance (a mostly-zero difference compresses very small) is an inference, not a
measurement.

### 3.5 Neutral names, labels and node order [R]

Evaluation images are written under neutral names `i<rank>.<ext>`, with `rank` from a seeded
permutation of the whole image set of a draw; nodes for a size are the images of its families and
singletons sorted by neutral id. Labels (family id or none, transformation, source photo) are in
a separate `labels_draw<d>.json`, never encoded in file names, and read only by the scoring
stage. Candidate sets are SHA-256 hashed (sorted flat-index array, as Phase 2) before any label
file is opened.

## 4. Methods (definitions frozen here) [R]

### 4.1 Candidate generators

| Method | Definition | Primary operating point | Curve |
|---|---|---|---|
| **M1 pHash** | exactly Phase 2 (`phase2.signatures.phash64`) | Hamming <= 10 | R in {4, 6, 8, 10, 12, 14, 16} |
| **dHash** (new, baseline) | grayscale (PIL `L`), resized with LANCZOS to 9 columns x 8 rows from the full-resolution image; bit (row r, comparison c) = pixel[r,c] > pixel[r,c+1], 64 bits, bit index = 8r + c; Hamming distance | Hamming <= 10 | same R set |
| **Exact hash** (baseline) | SHA-256 of the file bytes; candidate iff equal | n/a | n/a |
| **M2 coarse signature** | exactly Phase 2 (`phase2.signatures`, 1056-d, L2) | threshold per D3 (section 4.2) | multiples {0.5, 0.75, 1.0, 1.25, 1.5} x the primary threshold |
| **M3** (secondary) | M1 union M2 at their primaries | as above | none |
| **B1** random control | for every (method, size, draw): |P| pairs uniformly without replacement, `default_rng(s_d)` | n/a | n/a |
| **SSCD** | dropped (D7 approved); never used for the audit | n/a | n/a |

M1, dHash, the exact hash and B1 need no calibration, so their operating points are fixed here.

### 4.2 M2 threshold protocols (D3 resolved: B is the primary, Addendum A.1)

- **A1:** tau = the Otsu threshold (256 bins, flatness tolerance 1e-9, fallback (min+max)/2;
  `threshold_method` recorded) over the all-pairs M2 distances of that run.
- **A2:** tau = 0.5 x the A1 value. Not validated by Phase 2b.
- **B (dev-calibrated):** for each draw, compute all-pairs M2 distances on the **dev** set (150
  family sources -> 1,050 family images, plus 1,350 singletons; n_dev = 2,400; 3,150 true pairs).
  tau_B = `nextafter(q, +inf)` where q is the 0.97-quantile (`method="higher"`) of the dev
  true-pair distances, i.e. the smallest threshold at which dev candidate recall reaches 0.97. The
  value is frozen before that draw's test split is read and applied once. Labels are used on the
  dev split only. The target 0.97 is set now, above the 0.95 gate, and is not revisited.

**Resolved (Addendum A.1): B is the primary.** A1 is reported as a clearly labelled, non-gated
comparison; A2 is not run. Nothing is re-picked after results.

## 5. Verification

### 5.1 Primary verifier [R]

The frozen Phase 1c distance with tau = 0.5805796715948317. Non-candidate pairs are implicit
negatives and are never verified. Verifier metrics are computed over **true pairs only**
(verifier cap = share of true pairs with r < tau; verifier-confirmed recall = share of true pairs
that are both candidates and verified), plus a false-accept rate measured on a seeded sample of
1,000 non-true candidate pairs per (largest size, draw).

### 5.2 Aligned verifier (reported alongside; never gating) [D]

`phase3/aligned_verifier.py`, configuration C, **frozen**:

| Parameter | Value |
|---|---|
| working size (longest side) | 512 |
| ORB | 1,000 features, FAST threshold 5 |
| matching | brute-force Hamming with cross-check |
| transform | `cv2.estimateAffinePartial2D`, RANSAC, reprojection threshold 3.0 px, 2,000 iterations, confidence 0.99, `cv2.setRNGSeed(12345)` |
| minimum inliers | 12 |
| minimum overlap | 25% of the reference image |
| scale range | [0.25, 4.0] |
| distance | zlib level 9 size of \|A - warped B\| over valid pixels / compared bytes |
| reference image | the lower index |
| failure rule | `status != "ok"` is reported as a failure and counted; **no distance is substituted** |

Validated on synthetic Phase 2 families 1-40 only (regenerated, byte-identical to the Phase 2
manifest; `phase3/manifests/aligned_verifier_validation.json`, SHA-256
`b1e0941ee80dca757ce9969bf74233edfe5670f1287e3203fe049a4b3d1b32e7`): all 840 true pairs align
(no failures in any transformation), 1 of 840 unrelated pairs aligns, AUC 1.000 against 0.857 for
the Phase 1c distance, which passes only 79.8% of true pairs (rotation 20%, crop 75%). This
configuration replaced the draft's working size of 256 (68% aligned) after four configurations
were tried on those synthetic pairs, all reported in the plan. **[L]** The validation images are
smooth synthetic images and the transformations are exact similarity transforms; it is not
evidence about real photographs. **High failure rates on Copydays strong attacks (print-and-scan,
blur, painting) would be a result to report, not something to tune away.** The threshold for its
distance follows Addendum A.3 (D4, option a). Its cost on real images has not been measured.

## 6. Ground truth, labelling policy, metrics [R]

- D3a true pair: same source photograph (21 per family). D3b: same Copydays block. Distractors are
  singletons. "Same scene, next frame" is a **distinct** image under the frozen policy; the
  five-image fixed-camera cluster in the pool (`distractor_pool.summary.json`) is kept as distinct.
- Primary metrics per (dataset, size, draw, method): candidate recall overall; original-transform
  versus transform-transform; per transformation (D3a); reduction = 1 - |P| / (n(n-1)/2); |P|;
  |P|/|T|; the random-control recall.
- Secondary: candidate precision; verifier cap and confirmed recall; alignment-failure rate;
  false-accept rate; projected verification cost.
- Operating curves (recall versus reduction) for every method with a curve, not single points.
- Across draws: mean and range of every primary metric.

## 7. Gate [R] (strict form approved, D5)

Per (method, size, draw) on D3a: recall >= 0.95 AND reduction >= 0.90 AND recall >= 0.85 for every
transformation. **Approved strict form:** a method passes iff every (size, draw) passes. On D3b:
recall >= 0.95 and reduction >= 0.90 in every draw. No secondary metric rescues a failed primary
gate; all methods are reported either way. With most images being singletons the reduction target is
easier than in Phase 2, so recall and per-transformation recall carry the test **[L]**.

## 8. End-to-end clustering (secondary) [D]

At n = 1,000 (D6, approved), every draw: compute the **full** graph with the
frozen verifier (all pairs; parallel workers allowed, no timing claim from that run), with the fixed
tau = 0.5805796715948317 (not a per-run Otsu): sign +1 iff r < tau, weight |tau - r|. Candidate
graph: +1 only for pairs that are in the primary M2 candidate set and have r < tau, -1 otherwise.
KwikCluster (`cc_dedup.correlation_clustering`, unchanged) is run with the **20 frozen seeds
12345-12364** on both graphs, scored by D(C) (`cc_dedup.disagreement`) on the full graph; report
mean +/- sd. Baseline on the same graph: union-find (`baselines.threshold_chaining`). The parallel verifier output must
pass the exactness assertion of Addendum A.4 before the graph is used.

Left out unless the user decides otherwise (D8), with reasons: **transitive closure + greedy k-cut**
(Gusev and Xu, WWW 2020): Algorithm 3 as printed cannot be implemented faithfully (its
cluster-removal and termination steps are ambiguous: `C` never shrinks in the printed loop); the
transitive-closure part is exactly union-find, which is included. **Modified Pivot** (ICML 2025): not
implemented because it cannot be done faithfully from the abstract alone.

## 9. Cost accounting [D]

Primary cost metric: the expensive-comparison count |P|. After every parallel job has finished,
single-process with BLAS pinned to 1: time the cheap stage and the primary verifier. Projected
verification time = |P| x the mean of a fixed 200-pair sample of candidate pairs (seed 12345,
`pairwise_distance` only, decode excluded), labelled a **projection**. The all-pairs baseline uses the
per-pair cost measured on a seeded 200-pair sample of **real** pairs, extrapolated, labelled an
**extrapolation**; the synthetic 0.13 s per pair is not reused (an exploratory 40-pair sample on
real photographs gave about 0.016 s; not a result). Timings stay out of the byte-comparable results
file.

## 10. Pilot [R]

20 families (`perm` first 20 sources), 300 singletons, 20 Copydays blocks, one draw. Items: (1) two
runs give identical signatures, candidate sets and results; (2) label-leak test: signature and
candidate code import and name no label field, and labels are opened only after candidate sets are
hashed; (3) seeded draws reproduce, and dev and test photos are disjoint; (4) hand-checked recall,
reduction and per-type arithmetic on the pilot; (5) neutral names carry no label; (6) the primary and
aligned verifiers reproduce their frozen definitions (the primary equals a direct
`pairwise_distance` call; the aligned verifier reproduces its validation behaviour on a
Phase 2 family) and alignment failures are counted, not substituted; (7) the manifests match the
SHA-256 values in section 2; (8) free disk stays above 1.5 GB. Pilot output is not evidence and is
not committed. Any failed item stops the run.

## 11. Audit [R]

`experiments/audit_phase3.py`: independent recomputation (different code) of candidate sets
(SHA-256), recall, reduction, per-transformation recall and gate verdicts at every size and draw; a
fresh-process rerun byte-identical on at least one configuration; `git diff` on all frozen paths
empty; no label field reachable from signature or candidate code; seeds, splits and thresholds equal
the frozen values; for protocol B, the dev and test photo sets are disjoint and tau_B was written
before the test split was read. The audit result and `docs/phase3_report.md` (with a separate
"post-hoc observations" section that cannot count toward any gate) go to the user before any PR.

## 12. Implementation layout [D]

`phase3/` (dataset builder and labels, dHash and exact hash, split files, aligned verifier),
`experiments/run_phase3_pilot.py`, `run_phase3.py`, `run_phase3_e2e.py`,
`run_phase3_timing.py`, `audit_phase3.py`, tests in `tests/` (metric hand checks, label-leak,
split disjointness, determinism). Signatures and candidate sets are persisted so the audit can
recompute without regenerating images. Kaggle GPU use is restricted to DINOv2 audit embeddings (Addendum A.5);
everything else runs locally on CPU in one deterministic pipeline. Windows: multiprocessing uses spawn, so workers read images
from disk or shared memory; the largest size must compute signature distances **block-wise** (a
10,000 x 10,000 float64 matrix with temporaries would approach the 3.6 GB of free RAM seen at
freeze, 15.7 GB total).

## 13. Failure conditions [R]

Any frozen path changed; any parameter, threshold, exclusion rule, split or seed changed after seeing
results; labels used before candidate sets are hashed; non-determinism; a method not above its
random control (no signal); all methods failing (reported as the negative result, no retuning).

## 14. Limitations [L]

Copydays v1 lacks the crop and JPEG attack sets; distractor licences are unknown (research-local;
manifests only, no images committed); the injected transformations are the same six used in earlier
phases and are exact geometric or photometric operations; the flat-image exclusion is a curation
choice reported both ways; the cosine band below 0.85 in the pool audit was not inspected and only a
24-pair sample of the 0.85-0.90 band was; a small cluster of fixed-camera frames is kept as distinct
images; the Phase 1c verifier is weak on resize and crop pairs (its cap was measured only at n=280 on
synthetic data); family prevalence (35%) is unrealistic; protocol B, if chosen, transfers a
threshold calibrated on injected transformations to Copydays attacks; results hold for these datasets
only and support no claim of generality or novelty.

## 15. Budget [VERIFY at the pilot]

Disk: free space was 6.1 GB at freeze; injected D3a images are estimated at 1-1.5 GB per draw at
the largest size (about one MB per family); stop and ask if free disk would fall below 1.5 GB;
regenerable intermediates may be deleted, **never the Copydays or distractor data**. Compute: the
end-to-end graph is the expensive step (about 2.2 h single-process at n = 1,000, estimated), run in
parallel on 10 workers.

## Addendum A (2026-10-13): approved decisions, recorded split, Kaggle rules

Written after the user answered D1-D8. **Nothing has been run on the test split and no Phase 3
method result exists.** The content-only pool curation and the DINOv2 pool audit looked at pool
photographs before the split below was fixed; they used no labels, families, splits or method
outputs.

### A.1 Decisions (all approved exactly as recommended)

| # | Outcome | Consequence recorded here |
|---|---|---|
| D1 | (b) | main ladder at a 35% family-image share (F = n/20) **plus** the low-prevalence variant at n = 8,000 (F = 114, 7,202 singletons) |
| D2 | as proposed | n in {1,000; 2,500; 5,000; 10,000}; 3 draws, seeds 12345, 12346, 12347 |
| D3 | **B, as a SCOPED relaxation** | see below; A1 is a non-gated comparison; A2 is not run |
| D4 | (a) | procedure in A.3, plus AUC |
| D5 | strict | a method passes iff every (size, draw) passes; D3b: recall >= 0.95 and reduction >= 0.90 in every draw, no per-type criterion |
| D6 | n = 1,000 | procedure and exactness assertion in A.4 |
| D7 | SSCD dropped | no SSCD arm; the licence issue is unanswered |
| D8 | both left out | no transitive-closure-plus-greedy-k-cut, no Modified Pivot; union-find is the baseline |

**D3 scope, in the user's words:** ground-truth labels may be used only on a labelled dev split
that is disjoint from the test split; the split is fixed by seed and recorded in this document
before any test-split run; nothing else in the rule changes. Concretely, labels are used for
exactly one purpose: tau_B (section 4.2), from the dev true-pair distances of that draw. Ground truth
stays prohibited for choosing methods, operating points other than tau_B, seeds, exclusion rules,
and anything on the test split before candidate sets are hashed.

### A.2 Recorded split (fixed by seed; recorded before any test-split run)

Pool = the `name` field of `phase3/manifests/distractor_pool.jsonl` in file order (sorted by key;
file SHA-256 in section 2). SHA-256 of `"\n".join(names)` =
`c5370a836ea52d304d2bbb9f9a2b8e2019d013f0c3312e581e1f184497f95ecb`. For a draw with seed s:
`perm = numpy.random.default_rng(s).permutation(10000)` (numpy 2.5.1); **dev = perm[0:1500]**,
**test = perm[1500:10000]** (8,500 photographs). Dev and test are disjoint (verified for all
three draws, in both pools). Family sources and singletons are index ranges of `perm` as listed
below. The extended pool (cleaned pool plus the 236 excluded images, sorted by key, 10,236
photographs) uses the same rule with its own permutation. Every list is hashed as
`SHA-256("\n".join(names in perm order))`; `perm_sha256` hashes the permutation as int64
little-endian bytes.

```
draw 1 (seed 12345)
  perm_sha256                        8b1d077a7743d700150ced1cad247ac752ae1b196532552194a4c0fad1be18f9
  dev             perm[0:1500]       a446645c31a457d7d61744c2d3e895040a5daa4180aa8ac25f8b445c15c6d9d9
  dev_sources     perm[0:150]        48d057c53dcaf5993bb6eb3262e3add54204d75fe15649c2ce0dbcb7a68c1bc7
  dev_singletons  perm[150:1500]     28c7da1538b5b286b3428eab7f4b3a356975cd31eeb3a3f593c387701a446d3d
  test_sources    perm[1500:2000]    baa44f52f68baaace69ad785ca4141797c27c30c0dc9d9c8702b12aa8a1721ee
  test_singletons perm[2000:8500]    80a3b51c18e442724a792b104df1727855d226ae853e2c25fe6d316a3ec705ee
  d3b_singletons  perm[1500:6114]    9643fde37a287c3f527f65008ec1191a074c2e7276509b25b823d8c511eee7ea
  variant_sources perm[1500:1614]    7bf4a93e2653dbba5ae4e222ae1239d64c00c4dbf0679a250690bf4aeeb4c56a
  variant_singletons perm[1614:8816] ae3e3251e7590d2bee247a6f5649b8a675950841f576ac2400c2487268184c18
  extended pool: ext_perm_sha256     68a0a51abf26ef3a11ff8d01c5a16323c64b957455d53be43b51d81871afda2c
  ext_dev         perm[0:1500]       5261bed6b356d596b5287b3271625735cff52c7210912dce0914dea4bb3a2eb6
  ext_sources250  perm[1500:1750]    6618e7bfc3732b2b1b98b5193947c5a51ab16aec32c93f77586febcb6f194d3f
  ext_singletons3250 perm[2000:5250] bec6f60790799b230c10eec05c2abbdfd6b5ca86c7e149696a0020b758e626e2

draw 2 (seed 12346)
  perm_sha256                        9b73158d4a99a6760d0ec886c4c51f64a764f58afbdb6086550fdf2f7621c793
  dev             perm[0:1500]       92aa401884f7383b0bfd6a19d34b976d489152e662c8f541ca48c5c270eae030
  dev_sources     perm[0:150]        d22be411c36e25dbb247ba58348f99c37571b1b1ec46f4f9eb13b8912d91cf7d
  dev_singletons  perm[150:1500]     6ba2f675d2136187b6d3baadac20c1bad8b0c0527090d7948f123275079e0051
  test_sources    perm[1500:2000]    995fde16a7dc1dc90b0eaf94f1b956e2bc4543273064e7d226ad0fde306f55c8
  test_singletons perm[2000:8500]    7daf652d21ee326e2d450b1469f1d7cf0667fb83fa43f931b640109d991a2682
  d3b_singletons  perm[1500:6114]    b134bdfa3f38f2d3ced1db6f8e9759f0e2c5df28fc77baf9b0af8111c25b0f5b
  variant_sources perm[1500:1614]    53c47d90f39868fbfbd31ab9c1a7d7af4c3e41e051c149abb60596153163b61a
  variant_singletons perm[1614:8816] 7f3d9e9fbe5dd728c9ec6437fab9156b9e35782b47be825dce89379ab0d23f09
  extended pool: ext_perm_sha256     3e7d7b1068d99b7d61ca9e37cbf2737f95ff14f4836779f441f1f3f8e809e1d7
  ext_dev         perm[0:1500]       f0076a8c9af3c1d5db320a064d6d08610443e08b75365ce559e53001dae6bd47
  ext_sources250  perm[1500:1750]    17b159edc7a19af20d88ea9970a181cf6fc676435dae7492d5496c7526e7e6df
  ext_singletons3250 perm[2000:5250] 8d8bfced1adac3f21669d16172fe9b629dbc10f1b552b43924e91bdd56ce6bf3

draw 3 (seed 12347)
  perm_sha256                        5cab64642e82308367641d55fe58d17b0e2dc4c84e426984f063f17f3f63e541
  dev             perm[0:1500]       ffe07a2788337cd9d9d03899b8be17fac295aac2b07cc6faffcb2235a448b565
  dev_sources     perm[0:150]        015be945482185a33cf9c2f199a0e9b963f4b020a8b9ae065e4a9c141760f7b6
  dev_singletons  perm[150:1500]     ba1a0d21ad5c208ac006e8a49fba3b6bbe87e3166cb6ccece2291cab7e27dffa
  test_sources    perm[1500:2000]    114ef3d5dc7ed38a822769ec2c84ed6d2ecb164c9d80e9311b5b71c8a410284e
  test_singletons perm[2000:8500]    f9f87b18daadd3d912f6b48975858737813bf1fcc4afd8288cc5e40386dfa8f2
  d3b_singletons  perm[1500:6114]    32cb03cdffc46dd0a7d66645be05e438fec3d9e13ad208d1b2361e4a9fc2526f
  variant_sources perm[1500:1614]    454ec7d64cb287e2313c3af0339b99559ef388e4ee1e7ac3f40687d22499d577
  variant_singletons perm[1614:8816] fe4283f22778fa0d1f2f43e25d9e47d58b0935df21f5196c3667a1233730ddd6
  extended pool: ext_perm_sha256     da35664a1b7d987b419179b2a0b3cd3a092dde6a9698064d90a1f79a36b8f151
  ext_dev         perm[0:1500]       b5e7b876d34c2d002a850ac075c605272f94b60453e2ff1722af4760e1830c76
  ext_sources250  perm[1500:1750]    9ad5e1307bb1fd760dd4b1be85304e697fd70ea4ce7b9b3f1c2e526ec1b0e274
  ext_singletons3250 perm[2000:5250] 34a3c06c9200289e82101cfea0138ce3f6013bca268b0e5e2ba5c095559c8677
```

**Enforcement.** Before any test-split image is decoded or any test label is read, the
implementation recomputes every hash above from the pool manifest and the seeds and **stops on any
mismatch** (a different numpy version could change `permutation`; the hashes, not the seed, govern).
The split manifests it writes are committed with the code and compared by the audit with this
addendum.

### A.3 Aligned-verifier threshold (D4, option a)

Per draw d, tau_al(d) = the Otsu threshold (256 bins, flatness tolerance 1e-9, fallback
(min+max)/2, `threshold_method` recorded) over the aligned distances of all alignment-ok pairs in the
primary-M2 candidate set at n = 1,000 of that draw (the end-to-end run). Label-free. Alignment
failures are excluded from the Otsu input and counted. tau_al(d) is then frozen and used for that
draw's other sizes and for D3b, where only true pairs and the seeded 1,000-pair false-accept sample are
verified. The AUC (true versus non-true candidate pairs) is reported alongside. The candidate-mix bias
of this threshold is a limitation of the aligned verifier's reported numbers; it never gates.

### A.4 End-to-end run and the exactness assertion (D6)

n = 1,000 per draw (the ladder's first rung: the first 50 test sources and first 650 test singletons),
so the candidate sets of the main run are reused. The verifier runs on 10 workers (images in shared
memory; Windows spawn). **Output must be identical to single-process, and this is asserted, not
assumed:** before the parallel graph is accepted, 500 pairs drawn with `default_rng(12345)` from
all pairs of that run are recomputed single-process with the unchanged `pairwise_distance`, and every
distance must be **bitwise equal**; the same assertion is applied to aligned-verifier distances on a
seeded 200-pair sample of the candidate pairs. Any mismatch stops the run. The number of pairs checked,
the number equal and the maximum absolute difference are written to the results file. No timing is
claimed from the parallel run.

### A.5 Kaggle GPU: DINOv2 audit embeddings only

Allowed use: DINOv2 **inference** to produce audit embeddings, nothing else. All distance,
signature, verifier and clustering work stays on the local CPU in one deterministic pipeline.

- **Credentials** come from the environment or `~/.kaggle` on the user's machine only. They are
  never printed, logged or committed; `.gitignore` and the staged diff are checked before every commit
  (`.gitignore` had no Kaggle entries and gained `kaggle.json` and `.kaggle/` in commit `d31a690`).
- **Status at this addendum:** the Kaggle CLI and Python package are installed, but **no credentials
  were found** (presence-only check of `~/.kaggle/kaggle.json` and the `KAGGLE_*` environment variables;
  contents never read). Until the user provides them locally, any audit embedding runs on local CPU, and
  the report will say so. If Kaggle is unreachable at any time, the same fallback applies.
- **Uploads need the user's prior approval**, for any dataset or any in-notebook fetch of the third-party
  images: Copydays is INRIA's copyright (research use, no redistribution) and the distractors carry
  unknown per-image Flickr licences.
- **Recorded in the manifest for every run:** the DINOv2 variant; the weights revision (the weights file
  SHA-256, which for ViT-S/14 is `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`,
  88,283,115 bytes, checked equal on any new run) and the code revision (pin
  `facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8`, the `main` head on 2026-10-13,
  last committed 2026-06-03 and therefore the code of the 2026-10-12 CPU audit, whose manifest recorded
  `main` without a commit: this equality is inferred, not recorded at the time); the preprocessing; the
  torch version; the GPU type (`torch.cuda.get_device_name`); and the precision.
- **Embeddings are audit-only and never gated.** GPU results can differ from the CPU audit in low-order
  bits and between runs and GPU types (nondeterminism); this is labelled in the manifest, and if a GPU run
  flags different pairs from the CPU audit, the difference is reported, not resolved silently. No audit
  embedding run is currently required: the final-pool CPU audit is complete.

### A.6 What remains before any run

Implementation (section 12), the pilot (stop only if a pilot item fails), then the full run in this order:
D3a at each size and draw, the low-prevalence variant, D3b, the secondary pool, the end-to-end run;
then the audit, `docs/phase3_report.md`, and the user's approval before any PR. Everything else in this
design is unchanged.

## Addendum B (2026-10-14): execution environment and run protocol

Written before any test-split run. **No Phase 3 method result exists and nothing in sections 1-14 or
Addendum A is changed**; this addendum records where the run executes and how the implementation
realises the design, including two interpretations the user can veto at the readiness report.

### B.1 Execution environment (the run executes from D:)

- On 2026-10-14 the system drive had 0 bytes free. The user moved `phase3/data` to
  `D:\losless data\data`, and the run executes from a copy of the repository at
  `D:\losless data\repo`. The copy was made with `robocopy` (code, manifests and git history; no
  image data); its 943 files were verified **SHA-256 identical** to the working tree on C: and its
  `HEAD` equals the design commit `624283f`. `phase3\data` in the copy is a junction to the same
  data directory. Commits for this stage are made from the D: repository; the C: working tree is not
  edited during the run.
- The run process sets `TEMP` and `TMP` to `D:\losless data\tmp` (process environment only) and
  `PYTHONDONTWRITEBYTECODE=1`; nothing is written to C: except the interpreter's own reads.
- Start thresholds (the user's): free space C: >= 2 GB and D: >= 6 GB; they replace the 1.5 GB floor of
  section 15 and pilot item 8. A runtime guard stops the run if C: falls below 1.5 GB or D: below 3 GB;
  every part is a resumable file, so a stopped run continues without recomputation.
- Data integrity after the move: every pool and Copydays file is re-hashed against its manifest
  (`phase3/integrity.py`) at every start, in addition to the 39 split hashes and 4 manifest hashes.

### B.2 Ladder construction

For each draw one image set is built at the largest rung (500 sources + 6,500 singletons = 10,000
images) with neutral names ranked by one seeded permutation over the whole set (section 3.5). A smaller
rung is the subset given by `membership.json` (the first F(n) sources' images and the first S(n)
singletons), written from the in-memory build records, never from `labels.json`; nodes are sorted by
neutral id, so rungs are node-order-preserving subsets. Singleton images are hard links to the pool
files (identical bytes, no copy). Features of the whole set (pHash, 1056-d signature, dHash, file
SHA-256) are extracted once per image by `phase3/features.py`, from a single decode with the unmodified
Phase 2 building blocks and the Phase 3 dHash; a parallel extraction equals a single-process one
(tested).

### B.3 Parts, order, rehearsal

A run is a set of resumable parts under `phase3/artifacts/run_<mode>/`: tau_B per draw (main and
extended pool, from the dev lists only), then per draw the ladder and its verifier metrics, then the
low-prevalence variant, D3b, the secondary pool, and the end-to-end run (the order of A.6, grouped by
stage). Part files carry no timestamps or timings and are byte-reproducible; wall-clock events go to
`events.jsonl`, outside the results. For every evaluated set the candidate sets are hashed and the hashes
are logged before the labels are opened. The secondary pool run uses the extended-pool dev split recorded
in A.2 (`ext_dev`: its first 150 names are family sources, the other 1,350 singletons, the rule of the
main pool) for its own tau_B, applied once to `ext_sources250` and `ext_singletons3250`.
A **rehearsal** (`--mode rehearsal`) runs exactly the same code on the DEV split only, with smaller
rungs (400, 1,000 and 2,000 images; low-prevalence set 1,400 images; D3b with 300 singletons; secondary
pool 500 images; end-to-end at 400). It validates the code, produces no evidence, and its audit must pass
before the full run can start. The full run additionally requires: the dev-only pilot report with no
failed item, the setup verification, a clean working tree (no modified tracked file), no change to any
frozen path, and an explicit `--go`; only then is `phase3/.test_split_approved` written.

### B.4 Gate interpretation (to be confirmed at the readiness report)

Per cell the gate of section 7 is unchanged. D1 (b) adds the low-prevalence variant to D3a, so the
**headline strict verdict covers the 12 ladder cells and the 3 variant cells (n = 8,000)**; the verdict on
the 12 ladder cells alone is also reported. D3b is gated as in D5 (recall >= 0.95 and reduction >= 0.90 in
every draw); the secondary pool run and the A1 threshold (M2_A1) are never gated. A method is
gated at its primary operating point (M1 R <= 10, dHash R <= 10, exact hash, M2 at tau_B, M3). A method that
does not exceed its random control in a cell is flagged.

### B.5 Verifier metrics (secondary, never gating)

- Verifier distances are pure functions of an image pair, so the true pairs of the **largest rung** of a
  draw are verified once (primary verifier on 10 workers in batches of at most 800 shared-memory images;
  aligned verifier on 10 workers) and every smaller rung reads its true pairs' values from them.
- Primary: verifier cap = share of true pairs with r < tau (tau = 0.5805796715948317); confirmed recall =
  share of true pairs that are in the candidate set and have r < tau; false-accept rate on a seeded sample
  (`default_rng(12345)`, without replacement) of up to 1,000 non-true candidate pairs per primary method at the
  largest rung of each draw (and once per variant, D3b and secondary-pool set).
- Aligned (A.3): tau_al(d) = Otsu over the alignment-ok aligned distances of all primary-M2 candidate pairs
  at the smallest rung (n = 1,000); alignment-failure rate, aligned cap, confirmed recall and false-accept rate
  as above, with failures counted as not accepted. AUC (true versus non-true candidate pairs at n = 1,000) is
  reported in two declared variants: over alignment-ok pairs only, and with each failure given +infinity
  (worst) distance. The same AUC is reported for the primary distance on the end-to-end graph.
- Every parallel batch is accompanied by the exactness assertion of A.4 (500 primary pairs and 200 aligned
  pairs for the true-pair batches and the tau_al batch; 100 and 50 for each false-accept sample); a mismatch
  stops the run.

### B.6 End-to-end additions

At n = 1,000 per draw, besides the A.4 / section 8 comparison: the counts of positive edges among true and
among non-true pairs, the primary-distance AUC over the candidate pairs and over all pairs.

### B.7 Cost and audit scripts

`experiments/run_phase3_timing.py` follows section 9 (single process, BLAS pinned to one thread, after all
parallel work; writes `results/phase3_timing.json`, outside the byte-comparable results).
`experiments/audit_phase3.py` follows section 11 with separate code (lookup-table popcount, scipy
`cdist`, plain index arithmetic) for: all primary candidate sets at every rung, draw and secondary set
(every operating point at the smallest rung), recall, reduction, per-transformation recall and gate
verdicts; tau_B from the dev files; the split hashes; a fresh-process byte-identical re-run of draw 1 at the
smallest rung; the frozen-path diff; the label-leak scan; label-after-hash and tau_B-before-test event
order. The verifier metrics of the parallel runs are covered by the recorded bitwise exactness
assertions and are not recomputed by the audit **[L]**.
