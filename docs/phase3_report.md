# Phase 3 report: real photographs

**Status:** experiment complete and audited (all checks A0–A10 passed on attempt 2); this report is for review before merge. Branch `phase-3-real-data`. Every number below is generated from the committed JSON files by `experiments/make_phase3_report.py`, which also re-checks the finished text against the JSON.

Results hold for these datasets only (a 10,000-photograph Flickr-derived pool with injected transformations, and Copydays v1 strong attacks). They support no claim of generality or novelty.

## 0. Provenance

<!-- table:prov_commits -->
| Item | Commit / value |
|---|---|
| Code used for the test-split run (all parts) | `47f1b879c51808fec6f9b7b40e30c8ee18932c54` |
| Audit script (counts and scope statements added after the run; no check logic changed) | `32e9447` |
| Pre-registered outputs | `5f0839a` |
| Test-only change after the test split opened (lock tests independent of the local approval file) | `fad25e1` |
| Pool curation (content-only exclusion of flat images) and DINOv2 audit | `ebf94f9` |
| Post-hoc script (committed before its first run) | `002af36`, SHA-256 `2ecb0390aac3a6134516d3eba3e49994dd383c390cfefe7f5a6c60e5db7cc57d` |
| Post-hoc output | `da3c9d3` |
| Design (frozen before any test-split run) | `docs/phase3_design.md` with Addenda A and B |
<!-- /table:prov_commits -->

<!-- table:prov_hashes -->
| File | SHA-256 |
|---|---|
| `results/phase3_results.json` | fdf9bad4b274c95a3da028ff71a4c0a1892410e9d395a3ab21c5769ee6b034fe |
| `results/phase3_timing.json` | 7190fe3a382b8579fa156bf07318673b34e0ad31bd5f41a0033a5a04f44a1683 |
| `results/phase3_audit.json` | 9990dd2bc6d25f5b033aa5ac009d956e54962ee20e92a155532efe67650bead3 |
| `results/phase3_audit_attempt1_A10_diskguard.json` | 0ff2020b80d024d264f8d258c44ab30d595e430e375dd3e8edb519d21f64bee5 |
| `results/phase3_posthoc.json` | 202936006495c2e662bf17396b4bd90ca16546896a2d2a536b77733a50576cbc |
| `phase3/artifacts/run_full/launch_manifest.json` | dfec746af38d4afed4765444f1c1e695bdd15bcd67e07df47d672d30ccc86ad2 |
| `phase3/artifacts/run_full/resume_1.json` | 3212f217a94f5a80bbc6d5184be32acbfe61b753c97ab78626ae16ab55f594d7 |
| `phase3/artifacts/run_full/resume_2.json` | a61b41927966bbee918e5caa1397bea4e1e2b4bbc34ff48cd73fb7d522f2af6a |
| `phase3/manifests/distractor_audit.json` | 23bf08f8089f8f1bc6b05f8a767dc2bc17cc1bc5ddc1bca6b1c84f8ce9ed799e |
<!-- /table:prov_hashes -->

Run launch (UTC): 2026-10-14T14:59:50Z; commit recorded in the launch manifest: `47f1b87`; the test split was unlocked after all preflight gates passed, and 3 preflight passes are in the event log (launch plus two resumes, all on the same commit).

## 1. Pre-registered verdicts

Verdicts are per method and per dataset and are never pooled. **D3a** (injected transformations, the primary dataset) and **D3b** (native Copydays strong attacks, a weaker sanity check) are reported with equal prominence. Gates: recall ≥ 0.95, reduction ≥ 0.90, and recall ≥ 0.85 for every transformation, in every cell. D3a headline = the 12 ladder cells (n = 1,000, 2,500, 5,000, 10,000 × 3 draws) plus the 3 low-prevalence cells (n = 8,000); the ladder-only verdict is reported alongside. D3b: recall ≥ 0.95 and reduction ≥ 0.90 in every draw. M2_A1 is a non-gated comparison.

<!-- table:verdicts -->
| Method | D3a ladder-only | D3a headline (ladder + n=8,000) | D3a recall (12 ladder cells) | D3a reduction | D3a min per-transformation recall | D3b | D3b recall (3 draws) | D3b reduction |
|---|---|---|---|---|---|---|---|---|
| Exact hash | FAIL | FAIL | 0.000–0.000 | 1.0000–1.0000 | 0.000–0.000 | FAIL | 0.000–0.000 | 1.0000–1.0000 |
| M1 pHash (R ≤ 10) | FAIL | FAIL | 0.806–0.843 | 0.9982–0.9998 | 0.347–0.453 | FAIL | 0.038–0.038 | 1.0000–1.0000 |
| dHash (R ≤ 10) | FAIL | FAIL | 0.854–0.887 | 0.9982–0.9998 | 0.507–0.619 | FAIL | 0.051–0.051 | 0.9999–0.9999 |
| M2 at tau_B | PASS | PASS | 0.959–0.982 | 0.9960–0.9990 | 0.882–0.940 | FAIL | 0.207–0.239 | 0.9975–0.9990 |
| M3 (M1 ∪ M2) | PASS | PASS | 0.982–0.992 | 0.9959–0.9990 | 0.939–0.973 | FAIL | 0.229–0.261 | 0.9975–0.9990 |
| M2_A1 (per-run Otsu; NON-GATED comparison) | FAIL | FAIL | 0.999–1.000 | 0.4338–0.4570 | 0.997–1.000 | FAIL | 0.911–0.911 | 0.4361–0.4380 |
<!-- /table:verdicts -->

Reading: on D3a, M2 at tau_B and M3 pass; M1, dHash and the exact hash fail. On D3b every method fails, including M2 at tau_B and M3. M2_A1 (not gated) fails D3a on reduction only (its recall is 0.999–1.000 and its reduction 0.434–0.457); on D3b it fails both recall (0.911–0.911) and reduction.

**How "true" and "non-true" pairs are defined.** In D3a a pair is true iff both images come from the same source photograph (the source and its six transformed copies: 21 true pairs per family); in D3b iff both images are in the same Copydays block. **Every other pair is non-true by construction**, including every pair that involves a distractor photograph. The distractor pool was curated on content only (commit `ebf94f9`: 236 solid-colour images were excluded by a rule that uses no labels and no method output) and was **not** filtered for natural near-duplicates. It was audited for hidden near-duplicates with DINOv2 ViT-S/14 embeddings (flag rule: cosine ≥ 0.90; `phase3/manifests/distractor_audit.json`). In the final pool 13 pairs reach 0.90 (12 distractor–distractor, 1 between two different Copydays originals) and 0 reach 0.95; the flagged pairs and the 30 closest pairs were viewed by hand and none was a duplicate, and a seeded sample of 24 of the 94 pairs in the 0.85–0.90 band contained no duplicate. The other pairs in that band and everything below cosine 0.85 were not inspected, and a DINOv2 embedding would not necessarily flag every kind of near-duplicate. A small cluster of five fixed-camera frames is kept as distinct images under the frozen labelling policy ("same scene, next frame" is a distinct image). Some non-true candidate pairs could therefore be unlabelled true duplicates; this would affect precision-type quantities (false-accept rates, cluster precision, candidates per true pair) and would not change recall, which counts labelled true pairs only.

<!-- table:fail_counts -->
| Method | D3a cells failing recall (of 15) | failing reduction (of 15) | failing per-transformation recall (of 15) | recall above random control in every cell |
|---|---|---|---|---|
| Exact hash | 15 | 0 | 15 | no |
| M1 pHash (R ≤ 10) | 15 | 0 | 15 | yes |
| dHash (R ≤ 10) | 15 | 0 | 15 | yes |
| M2 at tau_B | 0 | 0 | 0 | yes |
| M3 (M1 ∪ M2) | 0 | 0 | 0 | yes |
| M2_A1 (per-run Otsu; NON-GATED comparison) | 0 | 15 | 0 | yes |
<!-- /table:fail_counts -->

### 1.1 D3a by ladder size (mean over the three draws; min–max in brackets)

<!-- table:by_size -->
| Method | n | Recall: mean (min–max) | Reduction (mean) | Min per-transformation recall: mean (min–max) | Candidate pairs (mean) |
|---|---|---|---|---|---|
| M1 pHash (R ≤ 10) | 1,000 | 0.828 (0.812–0.843) | 0.99826 | 0.409 (0.353–0.453) | 870 |
|  | 2,500 | 0.815 (0.806–0.823) | 0.99931 | 0.384 (0.347–0.404) | 2,142 |
|  | 5,000 | 0.816 (0.809–0.821) | 0.99966 | 0.390 (0.367–0.407) | 4,291 |
|  | 10,000 | 0.818 (0.810–0.823) | 0.99983 | 0.397 (0.366–0.413) | 8,619 |
| dHash (R ≤ 10) | 1,000 | 0.860 (0.854–0.867) | 0.99817 | 0.523 (0.507–0.533) | 916 |
|  | 2,500 | 0.870 (0.861–0.887) | 0.99921 | 0.568 (0.540–0.619) | 2,474 |
|  | 5,000 | 0.870 (0.859–0.883) | 0.99958 | 0.570 (0.533–0.605) | 5,293 |
|  | 10,000 | 0.872 (0.869–0.875) | 0.99976 | 0.573 (0.565–0.578) | 12,040 |
| M2 at tau_B | 1,000 | 0.972 (0.965–0.982) | 0.99656 | 0.926 (0.913–0.940) | 1,720 |
|  | 2,500 | 0.966 (0.961–0.969) | 0.99774 | 0.902 (0.885–0.912) | 7,048 |
|  | 5,000 | 0.964 (0.959–0.967) | 0.99824 | 0.902 (0.897–0.909) | 21,974 |
|  | 10,000 | 0.962 (0.959–0.964) | 0.99844 | 0.889 (0.882–0.896) | 78,018 |
| M3 (M1 ∪ M2) | 1,000 | 0.989 (0.988–0.992) | 0.99652 | 0.962 (0.957–0.973) | 1,739 |
|  | 2,500 | 0.988 (0.986–0.989) | 0.99772 | 0.958 (0.949–0.964) | 7,107 |
|  | 5,000 | 0.986 (0.982–0.989) | 0.99823 | 0.952 (0.939–0.960) | 22,100 |
|  | 10,000 | 0.987 (0.985–0.989) | 0.99843 | 0.956 (0.950–0.964) | 78,317 |
<!-- /table:by_size -->

### 1.2 D3a per-transformation recall at n = 10,000 (mean over the three draws)

<!-- table:per_transform -->
| Method | resize | JPEG q50 | crop | brightness | rotate | PNG |
|---|---|---|---|---|---|---|
| Exact hash | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| M1 pHash (R ≤ 10) | 0.896 | 0.896 | 0.852 | 0.892 | 0.397 | 0.896 |
| dHash (R ≤ 10) | 0.930 | 0.929 | 0.893 | 0.924 | 0.573 | 0.929 |
| M2 at tau_B | 0.979 | 0.981 | 0.979 | 0.889 | 0.942 | 0.981 |
| M3 (M1 ∪ M2) | 0.996 | 0.997 | 0.995 | 0.973 | 0.956 | 0.997 |
| M2_A1 (per-run Otsu; NON-GATED comparison) | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 1.000 |
<!-- /table:per_transform -->

### 1.3 D3a low-prevalence variant (n = 8,000, 114 families, about 10% family images)

<!-- table:variant -->
| Method | Recall (3 draws) | Reduction | Min per-transformation recall | Cells passing the gate |
|---|---|---|---|---|
| Exact hash | 0.000–0.000 | 1.0000–1.0000 | 0.000–0.000 | 0/3 |
| M1 pHash (R ≤ 10) | 0.811–0.824 | 0.9999–0.9999 | 0.361–0.415 | 0/3 |
| dHash (R ≤ 10) | 0.860–0.885 | 0.9999–0.9999 | 0.535–0.611 | 0/3 |
| M2 at tau_B | 0.958–0.967 | 0.9978–0.9991 | 0.879–0.908 | 3/3 |
| M3 (M1 ∪ M2) | 0.985–0.989 | 0.9978–0.9991 | 0.947–0.961 | 3/3 |
| M2_A1 (per-run Otsu; NON-GATED comparison) | 1.000–1.000 | 0.4371–0.4403 | 1.000–1.000 | 0/3 |
<!-- /table:variant -->

### 1.4 D3b Copydays strong attacks (314 true pairs, n = 5,000 per draw)

<!-- table:d3b -->
| Method | Recall d1 | Recall d2 | Recall d3 | Reduction d1 | Reduction d2 | Reduction d3 | D3b verdict |
|---|---|---|---|---|---|---|---|
| Exact hash | 0.000 | 0.000 | 0.000 | 1.0000 | 1.0000 | 1.0000 | FAIL |
| M1 pHash (R ≤ 10) | 0.038 | 0.038 | 0.038 | 1.0000 | 1.0000 | 1.0000 | FAIL |
| dHash (R ≤ 10) | 0.051 | 0.051 | 0.051 | 0.9999 | 0.9999 | 0.9999 | FAIL |
| M2 at tau_B | 0.207 | 0.223 | 0.239 | 0.9990 | 0.9985 | 0.9975 | FAIL |
| M3 (M1 ∪ M2) | 0.229 | 0.245 | 0.261 | 0.9990 | 0.9985 | 0.9975 | FAIL |
| M2_A1 (per-run Otsu; NON-GATED comparison) | 0.911 | 0.911 | 0.911 | 0.4369 | 0.4361 | 0.4380 | FAIL |
<!-- /table:d3b -->

The Copydays crop and JPEG attack sets are missing from v1, so D3b covers only strong attacks (print-and-scan, blur, painting and similar); crop and JPEG robustness on real images rests on D3a.

### 1.5 Secondary pool (extended pool, n = 5,000; NOT gated)

<!-- table:secondary -->
| Method | Recall: mean (min) | Reduction (mean) | Min per-transformation recall (min over draws) |
|---|---|---|---|
| M1 pHash (R ≤ 10) | 0.800 (min 0.794) | 0.9995 | 0.354 |
| dHash (R ≤ 10) | 0.859 (min 0.856) | 0.9989 | 0.517 |
| M2 at tau_B | 0.966 (min 0.953) | 0.9893 | 0.879 |
| M3 (M1 ∪ M2) | 0.982 (min 0.974) | 0.9892 | 0.934 |
<!-- /table:secondary -->

## 2. The tau_B protocol, its margins, and the drift with n

tau_B is a dev-calibrated quantile: for each draw it is the smallest threshold at which dev candidate recall reaches the pre-set target of 0.97, computed from the dev split's true-pair distances only. If dev and test true-pair distances are exchangeable draws from one distribution, test recall at tau_B should therefore sit near 0.97, with sampling variation. D3a is consistent with that: M2 at tau_B has recall 0.959–0.982 across the 12 ladder cells. D3b shows the assumption breaking: the same thresholds give recall 0.207–0.239 on Copydays strong attacks, whose distance distribution is not that of the injected transformations. A threshold calibrated on one attack family should not be assumed to transfer to another.

<!-- table:tau_b -->
| Draw | tau_B (main pool) | tau_B (extended pool) | tau_al (aligned verifier, see 4) |
|---|---|---|---|
| 1 | 0.623641 | 0.740951 | 0.341451 |
| 2 | 0.643620 | 0.681823 | 0.003769 |
| 3 | 0.666494 | 0.789198 | 0.346895 |
<!-- /table:tau_b -->

**Margins of M2 at tau_B against the D3a gate** (minimum over the 12 ladder cells unless stated):

<!-- table:margins -->
| Quantity | Minimum observed | Gate | Margin | Set |
|---|---|---|---|---|
| Recall | 0.95914 | 0.95 | 0.00914 | 12 ladder cells (gated) |
| Min per-transformation recall | 0.88200 | 0.85 | 0.03200 | 12 ladder cells (gated) |
| Min per-transformation recall | 0.87865 | 0.85 | 0.02865 | 3 variant cells (gated) |
| Min per-transformation recall | 0.87867 | 0.85 | 0.02867 | secondary pool (NOT gated) |
<!-- /table:margins -->

The weakest transformation for M2 at tau_B is brightness. The post-hoc bootstrap (section 9b) shows that at n = 1,000 and 2,500 the per-draw confidence intervals for overall recall and for brightness recall extend below the gate lines, so the small-n passes are not statistically separated from the gates. No gate verdict is changed by this.

**Recall differences across n are not a scale effect.** tau_B is fixed per draw, and whether a pair is an M2 candidate depends only on the distance between its two images. The rungs are nested prefixes of the same families (n = 1,000 is the first 50 families, n = 10,000 the first 500). The difference in recall between rungs is therefore sampling variation among families, not an effect of pool size. Mean M2 recall over draws is 0.972, 0.966, 0.964 and 0.962 at n = 1,000, 2,500, 5,000 and 10,000; the bootstrap 95% intervals of the mean (0.959–0.984 at n = 1,000 and 0.957–0.966 at n = 10,000) overlap. The rungs are nested, so these estimates are not independent of each other.

## 3. Verifier results on real photographs (secondary; never gating)

The primary verifier is the frozen Phase 1c distance at its fixed tau (0.5805796715948317); the aligned verifier is configuration C, frozen in the design. Non-candidate pairs are never verified.

<!-- table:primary_verifier -->
| Method | n | Verifier cap (share of true pairs with r < tau; mean over draws) | Confirmed recall (candidate and verified; mean over draws) |
|---|---|---|---|
| M1 pHash (R ≤ 10) | 1,000 | 0.573 | 0.519 |
|  | 10,000 | 0.570 | 0.513 |
| dHash (R ≤ 10) | 1,000 | 0.573 | 0.532 |
|  | 10,000 | 0.570 | 0.528 |
| M2 at tau_B | 1,000 | 0.573 | 0.557 |
|  | 10,000 | 0.570 | 0.546 |
| M3 (M1 ∪ M2) | 1,000 | 0.573 | 0.569 |
|  | 10,000 | 0.570 | 0.565 |
<!-- /table:primary_verifier -->

False-accept rate: share of a seeded sample (up to 1,000, without replacement) of non-true candidate pairs at the largest rung that the verifier would accept.

<!-- table:false_accept -->
| Method | Draw | Sampled non-true candidate pairs | Primary false-accept rate | Aligned: alignment-failure rate | Aligned false-accept rate |
|---|---|---|---|---|---|
| M1 pHash (R ≤ 10) | 1 | 31 | 0.323 | 0.935 | 0.032 |
|  | 2 | 30 | 0.467 | 1.000 | 0.000 |
|  | 3 | 38 | 0.342 | 0.974 | 0.000 |
| dHash (R ≤ 10) | 1 | 1,000 | 0.333 | 0.996 | 0.000 |
|  | 2 | 1,000 | 0.390 | 0.999 | 0.000 |
|  | 3 | 1,000 | 0.480 | 0.987 | 0.000 |
| M2 at tau_B | 1 | 1,000 | 0.238 | 1.000 | 0.000 |
|  | 2 | 1,000 | 0.225 | 1.000 | 0.000 |
|  | 3 | 1,000 | 0.221 | 1.000 | 0.000 |
| M3 (M1 ∪ M2) | 1 | 1,000 | 0.227 | 0.998 | 0.001 |
|  | 2 | 1,000 | 0.250 | 1.000 | 0.000 |
|  | 3 | 1,000 | 0.245 | 1.000 | 0.000 |
<!-- /table:false_accept -->

<!-- table:aligned_true -->
| Draw | True pairs (n = 10,000) | Alignment status counts on all true pairs | Alignment-failure rate on true pairs | Aligned cap at tau_al (M2 at n = 10,000) | Aligned confirmed recall |
|---|---|---|---|---|---|
| 1 | 10,500 | {"ok": 10494, "too_few_matches": 6} | 0.0006 | 0.159 | 0.153 |
| 2 | 10,500 | {"ok": 10500} | 0.0000 | 0.048 | 0.048 |
| 3 | 10,500 | {"ok": 10500} | 0.0000 | 0.164 | 0.159 |
<!-- /table:aligned_true -->

On Copydays strong attacks the aligner fails far more often:

<!-- table:d3b_verifier -->
| Draw | Alignment status counts on all 314 true pairs | Aligned: alignment-failure rate on true pairs | Primary verifier cap (share of true pairs with r < tau) | Primary confirmed recall (M2 at tau_B) |
|---|---|---|---|---|
| 1 | {"alignment_failed": 78, "ok": 210, "small_overlap": 23, "too_few_features": 3} | 0.331 | 0.959 | 0.194 |
| 2 | {"alignment_failed": 82, "ok": 204, "small_overlap": 25, "too_few_features": 3} | 0.350 | 0.959 | 0.210 |
| 3 | {"alignment_failed": 81, "ok": 202, "small_overlap": 28, "too_few_features": 3} | 0.357 | 0.959 | 0.226 |
<!-- /table:d3b_verifier -->

Exactness: the results file records 126 parallel-versus-single-process assertions; 126 of them report bitwise equality (a mismatch would have stopped the run).

## 4. The aligned verifier's threshold as measured (design decision D4, option a)

Protocol as pre-registered (Addendum A.3): tau_al(d) is the Otsu threshold over the aligned distances of all alignment-ok primary-M2 candidate pairs at n = 1,000 of draw d; alignment failures are excluded from the Otsu input and counted. This is a protocol result; no rule is changed.

<!-- table:d4 -->
| Draw | tau_al | Candidate pairs | Alignment ok | Status counts | AUC among aligned pairs | AUC with failures counted as worst | Non-true candidate pairs that aligned |
|---|---|---|---|---|---|---|---|
| 1 | 0.3415 | 1,410 | 1,013 | {"alignment_failed": 391, "ok": 1013, "too_few_matches": 6} | undefined (NaN) | 0.9971 | 0 |
| 2 | 0.0038 | 1,730 | 1,031 | {"alignment_failed": 699, "ok": 1031} | undefined (NaN) | 1.0000 | 0 |
| 3 | 0.3469 | 2,021 | 1,013 | {"alignment_failed": 1008, "ok": 1013} | undefined (NaN) | 1.0000 | 0 |
<!-- /table:d4 -->

Reading: tau_al is unstable across draws (the middle draw gives a threshold two orders of magnitude smaller). The AUC among aligned pairs is undefined because no non-true candidate pair aligned at all. The AUC with failures counted as worst is near 1 mostly because it separates pairs that align from pairs that do not, not because the distance separates them. The Otsu input therefore contains only true pairs (no non-true candidate pair aligned), which is the candidate-mix bias the design anticipated; as a result the aligned cap on true pairs (section 3) is low and varies by draw although alignment succeeds on almost every true pair.

## 5. End-to-end clustering at n = 1,000

Setup (section 8 of the design): full graph from the frozen Phase 1c verifier at its fixed tau (sign +1 iff r < tau, weight |tau − r|); candidate graph = +1 only for pairs in the primary M2 candidate set with r < tau; KwikCluster with 20 frozen seeds and union-find on both. **D(C) is scored on the full Phase 1c graph.** That graph is dense in positive edges and almost none of them are true (table below), so D(C) is a disagreement count with respect to that graph, not a measure of quality against ground truth. The ideal cluster count is 700 (650 singletons plus 50 families).

<!-- table:e2e_graph -->
| Draw | Positive edges, full graph | Share of all 499,500 pairs | of which true pairs | Share of positive edges that are true | True pairs in total | Positive edges, candidate graph | Primary AUC, all pairs | Primary AUC, candidate pairs |
|---|---|---|---|---|---|---|---|---|
| 1 | 132,195 | 26.5% | 694 | 0.52% | 1,050 | 757 | 0.773 | 0.812 |
| 2 | 116,806 | 23.4% | 531 | 0.45% | 1,050 | 677 | 0.706 | 0.704 |
| 3 | 124,652 | 25.0% | 579 | 0.46% | 1,050 | 820 | 0.711 | 0.700 |
<!-- /table:e2e_graph -->

<!-- table:e2e_D -->
| Draw | Kwik, full graph: D (mean ± sd, 20 seeds) | clusters | Kwik, candidate graph: D | clusters | Union-find, full graph: D | clusters | Union-find, candidate graph: D | clusters |
|---|---|---|---|---|---|---|---|---|
| 1 | 20,621 ± 8,638 | 384.6 | 13,332 ± 5 | 748.6 | 64,138 | 1 | 13,433 | 643 |
| 2 | 18,487 ± 13,548 | 490.2 | 12,957 ± 7 | 782.3 | 68,487 | 1 | 13,025 | 620 |
| 3 | 20,241 ± 13,339 | 429.4 | 13,727 ± 9 | 758.2 | 64,419 | 1 | 14,274 | 588 |
<!-- /table:e2e_D -->

Observation over three draws at n = 1,000, not a general claim: on the full Phase 1c graph, union-find merges everything into one cluster; KwikCluster returns a few hundred clusters with large seed-to-seed spread; on the candidate graph KwikCluster is much less seed-dependent than on the full graph (standard deviation of D over the 20 seeds in draws 1, 2 and 3: 5, 7 and 9 on the candidate graph, against 8,638, 13,548 and 13,339 on the full graph). Because D(C) is measured against the full graph, a lower D on the candidate graph says that the candidate graph disagrees with the full graph less, which here mostly reflects the full graph's many false positive edges. It is not a statement that the candidate graph is closer to ground truth; the post-hoc section 9a scores the candidate-graph clusterings against ground truth directly.

## 6. Cost: timing, projections and extrapolations

Timings come from a separate single-process run (BLAS pinned to one thread) on a laptop; they are machine-specific and noisy, and are not part of the byte-comparable results. **Everything below marked projection or extrapolation is not a measurement of a full verification.**

<!-- table:timing -->
| Quantity | Value |
|---|---|
| Primary verifier, seconds per pair (mean of a 200-pair sample of candidate pairs; range over the 12 draw × size cells) | 0.042–0.092 |
| Primary verifier, seconds per pair (mean of a 200-pair sample of random real pairs; range over cells) | 0.021–0.050 |
| Aligned verifier, seconds per pair (mean over 100 candidate pairs; range over cells) | 0.020–0.711 |
| Cheap stage at n = 10,000, feature extraction including decode (seconds; range over draws) | 239–259 |
| Cheap stage at n = 10,000, candidate generation for one operating point of each method (seconds; range over draws) | 99–102 |
| PROJECTION: verifying all M2 candidate pairs at n = 10,000 with the primary verifier (hours; number of candidate pairs × candidate-sample mean; range over draws) | 0.81–1.49 |
| PROJECTION: the same for M3 (hours; range over draws) | 0.82–1.50 |
| EXTRAPOLATION: verifying all 49,995,000 pairs at n = 10,000 with the primary verifier (hours; all pairs × random-pair mean; range over draws) | 583–637 |
<!-- /table:timing -->

The per-pair cost differs between cells by roughly a factor of two for the primary verifier and by more than an order of magnitude for the aligned verifier (whose cost depends on whether alignment succeeds), so any single hour figure should be read as an order of magnitude only.

## 7. Audit

`experiments/audit_phase3.py` recomputes the candidate sets and scores with different code (lookup-table popcount instead of `np.bitwise_count`, scipy `cdist` instead of the Gram identity, plain index arithmetic instead of the Phase 2 modules), and checks the pre-registration rules. **The run's own computation (block-wise Gram identity, as pre-registered) is the official one;** the audit's route is an independent cross-check and never replaces the run's results. **Limitation:** the verifier metrics of the parallel runs (true-pair verification, false-accept samples, tau_al and the end-to-end graph) are not recomputed by the audit; they are covered only by the recorded bitwise exactness assertions (parallel versus single-process on seeded samples).

<!-- table:audit_attempts -->
| Attempt | File | All checks passed | Failed checks | Note |
|---|---|---|---|---|
| 1 | `results/phase3_audit_attempt1_A10_diskguard.json` | no | A10 | phase3.runner.DiskGuard: free disk fell below the run guard: C: 0.90 GB (min 1.5), D: 201.62 GB (min 3.0) |
| 2 | `results/phase3_audit.json` | yes | none | A10 reproduced tau_B and the smallest-rung results of draw 1 byte for byte |
<!-- /table:audit_attempts -->

<!-- table:audit_checks -->
| Check (attempt 2) | Result |
|---|---|
| A0 results file and every part exist | PASS |
| A1 split hashes recomputed with separate code equal the design; dev and test disjoint | PASS |
| A10 a fresh process reproduces tau_B and the smallest-rung results of draw 1 byte for byte | PASS |
| A2 candidate-set SHA-256 recomputed independently equals the run's for every audited set | PASS |
| A2 persisted candidate sets equal the independent ones (differences only for pairs within 1e-9 of the set's threshold) | PASS |
| A2 recall, reduction, original/transform splits, per-transformation recall and gate verdicts equal | PASS |
| A3 aggregated strict ladder verdicts equal the conjunction of the per-cell verdicts | PASS |
| A4 tau_B recomputed from the dev files equals the stored value and reaches dev recall 0.97 | PASS |
| A5 every labels_opened event is preceded by its candidate_hashes_final event | PASS |
| A5 tau_B (all draws, both pools) was written before the first ladder image was built and after the unlock | PASS |
| A5 the approval file exists and records the head commit | PASS |
| A6 every photograph in the evaluated sets lies in the intended split (test for full, dev for rehearsal) | PASS |
| A7 seeds, tau, aligned configuration and rung tables equal the frozen values | PASS |
| A8 git diff against the design commit is empty for every frozen path | PASS |
| A9 no label field is reachable from signature, feature, hash, block-wise or candidate-stage code | PASS |
<!-- /table:audit_checks -->

Cells audited: 21 (12 ladder cells, 3 low-prevalence, 3 D3b and 3 secondary-pool sets). Recomputed independently and equal to the run: the SHA-256 of every candidate set, recall, reduction, original/transform splits, per-transformation recall and gate verdicts. Pairs within 1e-9 of a threshold may fall on either side between the two distance routines; the audit accepts a difference only for such pairs. **Near-threshold pairs and flips per audited cell:**

<!-- table:audit_near -->
| Cell | Pairs within 1e-9 of the M2 tau_B threshold | of the M2_A1 threshold | of the other M2 multiples (smallest rung only) | Flips versus the run |
|---|---|---|---|---|
| ladder_d1_n1000 | 0 | 0 | 0 | 0 |
| ladder_d1_n2500 | 0 | 0 | 0 | 0 |
| ladder_d1_n5000 | 0 | 0 | 0 | 0 |
| ladder_d1_n10000 | 0 | 2 | 0 | 0 |
| variant_d1 | 0 | 1 | 0 | 0 |
| d3b_d1 | 0 | 0 | 0 | 0 |
| ext_d1 | 0 | 0 | 0 | 0 |
| ladder_d2_n1000 | 0 | 0 | 0 | 0 |
| ladder_d2_n2500 | 0 | 0 | 0 | 0 |
| ladder_d2_n5000 | 0 | 0 | 0 | 0 |
| ladder_d2_n10000 | 0 | 0 | 0 | 0 |
| variant_d2 | 0 | 0 | 0 | 0 |
| d3b_d2 | 0 | 1 | 0 | 0 |
| ext_d2 | 0 | 0 | 0 | 0 |
| ladder_d3_n1000 | 0 | 0 | 0 | 0 |
| ladder_d3_n2500 | 0 | 0 | 0 | 0 |
| ladder_d3_n5000 | 0 | 0 | 0 | 0 |
| ladder_d3_n10000 | 0 | 1 | 0 | 0 |
| variant_d3 | 0 | 1 | 0 | 0 |
| d3b_d3 | 0 | 0 | 0 | 0 |
| ext_d3 | 0 | 0 | 0 | 0 |
<!-- /table:audit_near -->

## 8. Protocol deviations and interruption log

No run code, parameter, threshold, seed or split was changed after the test split was opened; the commit `47f1b87` recorded in the launch manifest is the commit of every part. (The audit script later gained reporting only, and the post-hoc script was added; both are listed below.) The events in this table interrupted execution, not the method.

<!-- table:interruptions -->
| # | When | Cause | Parts kept | Parts recomputed |
|---|---|---|---|---|
| 1 | run stopped about 16:17Z (21:47 IST) on 2026-10-14; resumed 2026-10-14T16:17:56Z | The run process stopped. The error log begins with a `^C` marker and orphaned workers (`OSError` in worker start-up), consistent with a console control signal delivered to the process group. **Cause not confirmed.** (`resume_1.json` records it as a console Ctrl-C; that is the interpretation at the time, not a confirmed cause.) | tau_B (6 parts), draw 1 ladder and verifier metrics, draw 2 ladder | draw 2 verifier metrics (no output existed) |
| 2 | run stopped during the night of 2026-10-14/15 (IST); resumed 2026-10-15T05:05:30Z | Environmental: Windows Update (TrustedInstaller, MoUsoCoreWorker) forced reboots at 04:47–04:51 IST on 2026-10-15; the run process died during the draw-3 end-to-end graph. Workers logged `BrokenPipeError` as a consequence. No Python error and no assertion failed. Windows also logged Modern Standby entries (lid and idle) during the run. | tau_b x6, ladder_d1-3, verify_d1-3, variant_d1-3, d3b_d1-3, ext_d1-3, e2e_d1, e2e_d2 | e2e_d3 (no output existed; recomputed from scratch) |
| 3 | audit attempt 1, 2026-10-15 | The C: free-space guard (minimum 1.5 GB) stopped the audit's own fresh-process rerun (check A10): the guard reported 0.90 GB free on C: (see the attempt-1 file). C: had fallen during the audit although nothing of the run writes to C:; the cause of the drop is unknown. All other checks passed in attempt 1. | all run results (unchanged) | the whole audit was rerun unchanged as attempt 2 (same script, same commit), which passed all checks |
<!-- /table:interruptions -->

Other departures from the plan as originally written, all recorded in `docs/phase3_design.md` Addendum B or the commit history:

- The run executed from a verified copy of the repository on D: (943 files, SHA-256 identical to the C: tree), because C: had no free space; the start thresholds were C: ≥ 2 GB and D: ≥ 6 GB, with a runtime guard at 1.5 GB and 3 GB.
- Before the test split was opened, a dev-split-only rehearsal and its audit exposed two issues in the audit script itself: a tolerance for pairs within 1e-9 of a threshold (different rounding between the two distance routines) and a rounding tolerance in the tau_B recomputation. Both were fixed in the audit script, not in the run code.
- After the run, the audit script gained the per-cell near-threshold counts and two scope statements (commit `32e9447`); no check logic changed.
- Commit `fad25e1`, made after the test split was opened, changed three unit tests so that they no longer assume the approval file is absent (test-only; no run, audit or analysis code was touched).
- The first rehearsal launch was stopped and restarted to add persistence of every operating point at the smallest rung, before any test-split access.
- Two files in the run record (`launch_manifest.json`, `resume_*.json`) were written with a UTF-8 byte-order mark; the generator reads them with `utf-8-sig`.

## 9. Post-hoc analyses (NOT PRE-REGISTERED)

**These analyses were added after the pre-registered results had been reviewed. They cannot change any gate verdict, and none is used to re-evaluate one.** Script `experiments/posthoc_phase3.py` (committed before its first run, SHA-256 in section 0); output `results/phase3_posthoc.json`, which carries `NOT_PRE_REGISTERED: true`. All three analyses read the committed run's persisted candidate sets; the full-graph KwikCluster was not rerun.

### 9a. Candidate-graph clustering against ground truth (n = 1,000)

Graph: +1 iff the pair is in the primary M2 candidate set and r < tau; all other pairs −1. Scores are pairwise precision, recall, F1 and ARI against **all** true pairs (1,050 per draw, including true pairs that are not in the candidate set); nodes that are not candidates, and isolated nodes, count as singletons. KwikCluster: mean ± sd over the 20 frozen seeds. The ideal cluster count is 700. This is an observation over three draws at n = 1,000, not a general claim.

<!-- table:posthoc_a -->
| Draw | Method | Positive edges | Precision | Recall | F1 | ARI | Clusters (ideal 700) | Giant cluster |
|---|---|---|---|---|---|---|---|---|
| 1 | KwikCluster | 757 | 0.909 ± 0.025 | 0.582 ± 0.028 | 0.709 ± 0.019 | 0.709 ± 0.019 | 748.6 ± 6.7 | 9.2 ± 1.9 |
|  | Union-find | 757 | 0.542 | 0.977 | 0.697 | 0.697 | 643 | 35 |
| 2 | KwikCluster | 677 | 0.768 ± 0.042 | 0.389 ± 0.035 | 0.515 ± 0.032 | 0.514 ± 0.032 | 782.3 ± 8.5 | 10.5 ± 1.8 |
|  | Union-find | 677 | 0.337 | 0.994 | 0.503 | 0.501 | 620 | 45 |
| 3 | KwikCluster | 820 | 0.711 ± 0.050 | 0.454 ± 0.038 | 0.552 ± 0.026 | 0.551 ± 0.026 | 758.2 ± 11.4 | 12.8 ± 3.2 |
|  | Union-find | 820 | 0.120 | 0.972 | 0.213 | 0.210 | 588 | 95 |
<!-- /table:posthoc_a -->

**Candidate-set recall is an edge-level ceiling only.** It bounds how many true pairs can be positive edges; pairwise clustering recall can exceed it through transitive closure, because two images in the same family can end in one cluster through a third image even when their own pair is not a candidate. The table shows this:

<!-- table:posthoc_ceiling -->
| Draw | Edge-level ceiling: share of true pairs that are candidate pairs | Union-find pairwise recall | Pairwise recall exceeds the edge ceiling |
|---|---|---|---|
| 1 | 0.970 | 0.977 | yes |
| 2 | 0.982 | 0.994 | yes |
| 3 | 0.965 | 0.972 | yes |
<!-- /table:posthoc_ceiling -->

Observation: union-find reaches high recall but with low precision (it joins distinct photographs through the Phase 1c verifier's false positives, giving giant clusters), while KwikCluster is more precise but with lower recall and a different cluster count; neither recovers the ideal 700 clusters. With three draws at one size this is descriptive only.

### 9b. Family-level bootstrap confidence intervals for recall

Unit: the family (source photograph). 2,000 resamples, seed 12345, percentile 95% intervals, M2 at tau_B and M3 at their primary operating points. Point estimates equal the pre-registered values (asserted to 1e-12 by the script before reporting). **Resampling:** each draw is resampled independently (a separate set of family indices per draw) and the "mean" rows average the three per-resample recalls; the draws are not resampled jointly. The draws do overlap in families (different permutations of one pool): the number of source photographs shared between two draws, among the first F families, is shown below, so the independence assumption is slightly optimistic. The bootstrap holds tau_B fixed, so the intervals do not include the variance of the dev-split calibration.

<!-- table:overlap -->
| n | Families per draw (F) | Shared by draws 1 and 2 | 1 and 3 | 2 and 3 | Shared by all three |
|---|---|---|---|---|---|
| 1,000 | 50 | 0 | 1 | 0 | 0 |
| 2,500 | 125 | 0 | 1 | 0 | 0 |
| 5,000 | 250 | 5 | 8 | 5 | 0 |
| 10,000 | 500 | 25 | 24 | 23 | 2 |
<!-- /table:overlap -->

**M2 at tau_B:** recall with 95% CI (point estimate, then the interval); "mean" rows are the mean over the three draws.

<!-- table:boot_M2_B -->
| n | Draw | Overall | resize | JPEG q50 | crop | brightness | rotate | PNG |
|---|---|---|---|---|---|---|---|---|
| 1,000 | 1 | 0.970 (0.945–0.991) | 0.983 (0.967–0.997) | 0.987 (0.973–0.997) | 0.987 (0.973–0.997) | 0.923 (0.853–0.983) | 0.947 (0.897–0.987) | 0.983 (0.967–0.997) |
|  | 2 | 0.982 (0.964–0.995) | 0.993 (0.983–1.000) | 0.990 (0.980–1.000) | 0.990 (0.977–1.000) | 0.940 (0.877–0.987) | 0.973 (0.953–0.990) | 0.993 (0.983–1.000) |
|  | 3 | 0.965 (0.939–0.987) | 0.980 (0.963–0.997) | 0.983 (0.970–0.997) | 0.983 (0.970–0.997) | 0.913 (0.837–0.973) | 0.927 (0.870–0.970) | 0.983 (0.970–0.997) |
|  | mean | **0.972 (0.959–0.984)** | **0.986 (0.977–0.993)** | **0.987 (0.979–0.993)** | **0.987 (0.980–0.993)** | **0.926 (0.887–0.959)** | **0.949 (0.923–0.971)** | **0.987 (0.979–0.994)** |
| 2,500 | 1 | 0.969 (0.953–0.983) | 0.984 (0.975–0.992) | 0.985 (0.976–0.993) | 0.984 (0.975–0.992) | 0.909 (0.860–0.951) | 0.953 (0.925–0.975) | 0.984 (0.975–0.992) |
|  | 2 | 0.961 (0.943–0.976) | 0.977 (0.965–0.987) | 0.977 (0.965–0.987) | 0.981 (0.972–0.989) | 0.885 (0.831–0.931) | 0.941 (0.909–0.967) | 0.981 (0.972–0.991) |
|  | 3 | 0.969 (0.954–0.982) | 0.983 (0.973–0.992) | 0.984 (0.975–0.992) | 0.984 (0.975–0.992) | 0.912 (0.865–0.953) | 0.948 (0.919–0.973) | 0.985 (0.976–0.993) |
|  | mean | **0.966 (0.957–0.975)** | **0.981 (0.975–0.987)** | **0.982 (0.976–0.987)** | **0.983 (0.978–0.988)** | **0.902 (0.873–0.927)** | **0.948 (0.931–0.962)** | **0.984 (0.978–0.988)** |
| 5,000 | 1 | 0.967 (0.956–0.978) | 0.983 (0.975–0.989) | 0.985 (0.978–0.991) | 0.981 (0.974–0.988) | 0.909 (0.877–0.940) | 0.946 (0.924–0.965) | 0.984 (0.977–0.990) |
|  | 2 | 0.964 (0.952–0.976) | 0.980 (0.972–0.987) | 0.981 (0.973–0.987) | 0.982 (0.975–0.989) | 0.897 (0.861–0.929) | 0.945 (0.924–0.964) | 0.983 (0.975–0.989) |
|  | 3 | 0.959 (0.946–0.972) | 0.977 (0.969–0.985) | 0.978 (0.970–0.985) | 0.977 (0.969–0.985) | 0.898 (0.862–0.931) | 0.928 (0.901–0.952) | 0.979 (0.971–0.986) |
|  | mean | **0.964 (0.957–0.970)** | **0.980 (0.976–0.984)** | **0.981 (0.977–0.985)** | **0.980 (0.976–0.984)** | **0.902 (0.882–0.919)** | **0.940 (0.926–0.953)** | **0.982 (0.978–0.986)** |
| 10,000 | 1 | 0.959 (0.950–0.968) | 0.977 (0.972–0.983) | 0.980 (0.975–0.985) | 0.977 (0.971–0.982) | 0.889 (0.865–0.913) | 0.933 (0.916–0.949) | 0.979 (0.974–0.984) |
|  | 2 | 0.962 (0.954–0.970) | 0.980 (0.974–0.984) | 0.981 (0.976–0.986) | 0.980 (0.974–0.984) | 0.882 (0.857–0.907) | 0.950 (0.937–0.961) | 0.982 (0.977–0.986) |
|  | 3 | 0.964 (0.955–0.971) | 0.981 (0.975–0.985) | 0.981 (0.976–0.986) | 0.980 (0.975–0.985) | 0.896 (0.870–0.920) | 0.944 (0.929–0.958) | 0.981 (0.976–0.986) |
|  | mean | **0.962 (0.957–0.966)** | **0.979 (0.976–0.982)** | **0.981 (0.978–0.983)** | **0.979 (0.976–0.982)** | **0.889 (0.875–0.903)** | **0.942 (0.934–0.951)** | **0.981 (0.978–0.983)** |
<!-- /table:boot_M2_B -->

**M3:** recall with 95% CI (point estimate, then the interval); "mean" rows are the mean over the three draws.

<!-- table:boot_M3 -->
| n | Draw | Overall | resize | JPEG q50 | crop | brightness | rotate | PNG |
|---|---|---|---|---|---|---|---|---|
| 1,000 | 1 | 0.988 (0.973–0.998) | 0.993 (0.983–1.000) | 0.997 (0.990–1.000) | 0.997 (0.990–1.000) | 0.983 (0.970–0.997) | 0.957 (0.907–0.993) | 0.993 (0.983–1.000) |
|  | 2 | 0.992 (0.987–0.997) | 1.000 (1.000–1.000) | 0.997 (0.990–1.000) | 1.000 (1.000–1.000) | 0.977 (0.960–0.990) | 0.973 (0.953–0.990) | 1.000 (1.000–1.000) |
|  | 3 | 0.988 (0.976–0.996) | 0.993 (0.983–1.000) | 0.997 (0.990–1.000) | 1.000 (1.000–1.000) | 0.973 (0.957–0.987) | 0.957 (0.917–0.987) | 0.997 (0.990–1.000) |
|  | mean | **0.989 (0.983–0.995)** | **0.996 (0.991–0.999)** | **0.997 (0.992–1.000)** | **0.999 (0.997–1.000)** | **0.978 (0.969–0.986)** | **0.962 (0.941–0.981)** | **0.997 (0.993–1.000)** |
| 2,500 | 1 | 0.989 (0.982–0.996) | 0.996 (0.991–1.000) | 0.997 (0.993–1.000) | 0.996 (0.991–1.000) | 0.980 (0.969–0.989) | 0.964 (0.937–0.985) | 0.996 (0.991–1.000) |
|  | 2 | 0.986 (0.977–0.993) | 0.995 (0.989–0.999) | 0.993 (0.987–0.999) | 0.997 (0.993–1.000) | 0.972 (0.961–0.983) | 0.949 (0.919–0.975) | 0.996 (0.991–1.000) |
|  | 3 | 0.989 (0.981–0.994) | 0.996 (0.991–1.000) | 0.997 (0.993–1.000) | 0.999 (0.996–1.000) | 0.973 (0.961–0.983) | 0.960 (0.935–0.980) | 0.997 (0.993–1.000) |
|  | mean | **0.988 (0.984–0.992)** | **0.996 (0.993–0.998)** | **0.996 (0.993–0.998)** | **0.997 (0.995–0.999)** | **0.975 (0.969–0.981)** | **0.958 (0.944–0.972)** | **0.996 (0.994–0.999)** |
| 5,000 | 1 | 0.988 (0.982–0.993) | 0.996 (0.993–0.999) | 0.997 (0.993–0.999) | 0.995 (0.991–0.998) | 0.978 (0.971–0.985) | 0.959 (0.939–0.975) | 0.996 (0.993–0.999) |
|  | 2 | 0.989 (0.983–0.993) | 0.997 (0.993–0.999) | 0.996 (0.993–0.999) | 0.997 (0.995–0.999) | 0.975 (0.968–0.982) | 0.960 (0.941–0.975) | 0.997 (0.995–0.999) |
|  | 3 | 0.982 (0.975–0.989) | 0.993 (0.989–0.997) | 0.993 (0.989–0.997) | 0.991 (0.986–0.996) | 0.973 (0.964–0.981) | 0.939 (0.915–0.962) | 0.993 (0.989–0.997) |
|  | mean | **0.986 (0.983–0.990)** | **0.995 (0.993–0.997)** | **0.995 (0.993–0.997)** | **0.994 (0.992–0.997)** | **0.975 (0.971–0.980)** | **0.952 (0.941–0.964)** | **0.996 (0.994–0.997)** |
| 10,000 | 1 | 0.985 (0.981–0.989) | 0.995 (0.992–0.997) | 0.996 (0.994–0.998) | 0.993 (0.990–0.996) | 0.973 (0.968–0.979) | 0.950 (0.934–0.963) | 0.995 (0.993–0.998) |
|  | 2 | 0.989 (0.986–0.992) | 0.998 (0.996–0.999) | 0.998 (0.996–0.999) | 0.997 (0.994–0.999) | 0.972 (0.966–0.978) | 0.964 (0.953–0.974) | 0.998 (0.997–1.000) |
|  | 3 | 0.987 (0.983–0.991) | 0.996 (0.994–0.998) | 0.996 (0.994–0.998) | 0.995 (0.992–0.998) | 0.973 (0.967–0.979) | 0.956 (0.942–0.968) | 0.997 (0.994–0.998) |
|  | mean | **0.987 (0.985–0.989)** | **0.996 (0.995–0.998)** | **0.997 (0.995–0.998)** | **0.995 (0.994–0.996)** | **0.973 (0.970–0.976)** | **0.956 (0.949–0.964)** | **0.997 (0.996–0.998)** |
<!-- /table:boot_M3 -->

### 9c. EXPLORATORY: alignment success as an edge rule (injected transformations only)

Rule: a candidate pair is a positive edge iff the aligner reports status `ok`, using the frozen configuration C exactly as in the run (`AlignConfig()` defaults, no parameter changed; the tau_al distance threshold is not used). Same clustering metrics as 9a, same true-pair denominator. **This is exploratory, measured on injected exact geometric and photometric transformations only, and motivates follow-up. No gate or verdict is re-evaluated with it.** On Copydays strong attacks the aligner failed on about a third of pairs (section 3), so these numbers say nothing about native attacks.

<!-- table:posthoc_c -->
| Draw | Method | Aligned-ok candidate pairs (positive edges) | of which non-true | Precision | Recall | F1 | ARI | Clusters (ideal 700) | Giant cluster |
|---|---|---|---|---|---|---|---|---|---|
| 1 | KwikCluster | 1,013 | 0 | 1.000 ± 0.000 | 0.963 ± 0.005 | 0.981 ± 0.002 | 0.981 ± 0.002 | 706.5 ± 0.8 | 7.0 ± 0.0 |
|  | Union-find |  |  | 1.000 | 0.971 | 0.986 | 0.985 | 705 | 7 |
| 2 | KwikCluster | 1,031 | 0 | 1.000 ± 0.000 | 0.979 ± 0.007 | 0.989 ± 0.004 | 0.989 ± 0.004 | 703.6 ± 1.1 | 7.0 ± 0.0 |
|  | Union-find |  |  | 1.000 | 0.989 | 0.994 | 0.994 | 702 | 7 |
| 3 | KwikCluster | 1,013 | 0 | 1.000 ± 0.000 | 0.957 ± 0.009 | 0.978 ± 0.005 | 0.978 ± 0.005 | 707.5 ± 1.5 | 7.0 ± 0.0 |
|  | Union-find |  |  | 1.000 | 0.977 | 0.988 | 0.988 | 704 | 7 |
<!-- /table:posthoc_c -->

The family size is 7, so a giant cluster of 7 is a complete family. The perfect precision in this table only restates that no non-true candidate pair aligned (section 4); it is not evidence that the rule would separate near-duplicates from look-alikes in general.

## 10. Limitations

- Two datasets only: an injected-transformation set built from one Flickr-derived pool of unknown per-image licences, and Copydays v1 (strong attacks only; the crop and JPEG sets are missing). D3b is a weak sanity check.
- The injected transformations are the six used in earlier phases and are exact geometric or photometric operations; family prevalence (35%, and 10% in the variant) is far above realistic prevalence.
- tau_B is calibrated on injected transformations of a dev split; the failure on Copydays shows it does not transfer to a different attack family. The 0.97 target is a pre-set design constant.
- M2's pass of the per-transformation gate at small n is not statistically separated from the gate line (brightness; section 2 and 9b); the nested rungs are not independent.
- The Phase 1c verifier is weak on real photographs (it accepts only about 0.57 of true pairs, and falsely accepts about 0.23 of sampled non-true M2 candidate pairs); end-to-end results are conditional on it, and D(C) against the full graph is not a quality-versus-truth measure (section 5).
- The aligned verifier's threshold protocol (D4a) is unstable, its AUC is not informative, and its results on native attacks show many alignment failures; it never gated.
- Timing is from one laptop, single process, with noisy per-pair costs; projections and extrapolations are order-of-magnitude only.
- The post-hoc clustering analysis (section 9a and 9c) covers n = 1,000, three draws, and the candidate graph only; the full-graph KwikCluster was not rerun, so there is no ground-truth clustering comparison on the full graph.
- Non-true pairs are defined by construction (section 1). The distractor pool was audited, not filtered, for natural near-duplicates, and the audit was only partly inspected by hand, so some non-true candidate pairs could be unlabelled true duplicates (this would affect false-accept rates, cluster precision and candidates per true pair, not recall).
- The audit does not recompute the parallel verifier metrics (section 7); the bootstrap holds tau_B fixed and treats the three draws as independent although they share some families; the post-hoc analyses are exploratory.
- Nothing here supports a claim of generality or novelty.

## 11. Reproduction

`python experiments/make_phase3_report.py` regenerates this file from the committed JSON and re-checks it; `--check` only re-checks. The run, audit, timing and post-hoc steps are `experiments/run_phase3.py`, `audit_phase3.py`, `run_phase3_timing.py` and `posthoc_phase3.py`.
