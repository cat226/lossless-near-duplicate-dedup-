# Phase 3 Experimental Design — Real Photographs

Status: **FROZEN by the commit that adds this file, EXCEPT the items marked [DECISION]
(D1-D8), which need the user's explicit answers.** Answers will be recorded in an addendum
commit before anything is run on evaluation data. Nothing in this design has been run on
evaluation data: no Phase 3 method result exists.

Builds on Phases 0, 1c, 2 and 2b (all merged; `origin/main` was `ac3cb63` at freeze).
Frozen and untouchable: `phase0/`, `results/phase1c_results.json`, `docs/phase1c_design.md`,
Phase 1c code in `cc_dedup/`, `baselines/`, `experiments/run_phase1c.py`,
`experiments/run_pilot.py`, and everything in Phase 2 and 2b (`phase2/`, `phase2b/`, their
designs, results and reports). They are imported or read, never edited.

**Legend:** **[R]** required by project rules or the user's instructions · **[D]** design
decision made here · **[DECISION]** needs a user answer · **[L]** limitation.

## 0. Decisions needed from the user

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

### 3.1 Dev/test split and nesting (every draw d, seed s_d = 12345 + d - 1) [D]

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
| **SSCD** | per D7 (recommended: dropped; never used for the audit) | n/a | n/a |

M1, dHash, the exact hash and B1 need no calibration, so their operating points are fixed here.

### 4.2 M2 threshold protocols (the user picks one in D3) [DECISION]

- **A1:** tau = the Otsu threshold (256 bins, flatness tolerance 1e-9, fallback (min+max)/2;
  `threshold_method` recorded) over the all-pairs M2 distances of that run.
- **A2:** tau = 0.5 x the A1 value. Not validated by Phase 2b.
- **B (dev-calibrated):** for each draw, compute all-pairs M2 distances on the **dev** set (150
  family sources -> 1,050 family images, plus 1,350 singletons; n_dev = 2,400; 3,150 true pairs).
  tau_B = `nextafter(q, +inf)` where q is the 0.97-quantile (`method="higher"`) of the dev
  true-pair distances, i.e. the smallest threshold at which dev candidate recall reaches 0.97. The
  value is frozen before that draw's test split is read and applied once. Labels are used on the
  dev split only. The target 0.97 is set now, above the 0.95 gate, and is not revisited.

Whichever is chosen is the **primary**; the other two are reported as clearly labelled
comparisons and are never re-picked.

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
distance follows D4. Its cost on real images has not been measured.

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

## 7. Gate [R] (form per D5)

Per (method, size, draw) on D3a: recall >= 0.95 AND reduction >= 0.90 AND recall >= 0.85 for every
transformation. **Recommended strict form:** a method passes iff every (size, draw) passes. On D3b:
recall >= 0.95 and reduction >= 0.90 in every draw. No secondary metric rescues a failed primary
gate; all methods are reported either way. With most images being singletons the reduction target is
easier than in Phase 2, so recall and per-transformation recall carry the test **[L]**.

## 8. End-to-end clustering (secondary) [D]

At the size chosen in D6 (recommended n = 1,000), every draw: compute the **full** graph with the
frozen verifier (all pairs; parallel workers allowed, no timing claim from that run), with the fixed
tau = 0.5805796715948317 (not a per-run Otsu): sign +1 iff r < tau, weight |tau - r|. Candidate
graph: +1 only for pairs that are in the primary M2 candidate set and have r < tau, -1 otherwise.
KwikCluster (`cc_dedup.correlation_clustering`, unchanged) is run with the **20 frozen seeds
12345-12364** on both graphs, scored by D(C) (`cc_dedup.disagreement`) on the full graph; report
mean +/- sd. Baseline on the same graph: union-find (`baselines.threshold_chaining`).

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
recompute without regenerating images. Windows: multiprocessing uses spawn, so workers read images
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
