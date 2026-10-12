# Phase 2b Experimental Design — Fixed-Threshold M2 on Held-Out Families

Status: **FROZEN by the commit that adds this file** ("Freeze Phase 2b design").
Nothing in this design has been run, and the fresh families it uses (801-1600)
had not been generated or examined at freeze time. Implementation starts only
after the user reviews this commit.

Builds on Phase 2 (`docs/phase2_report.md`, PR #3). Frozen and untouchable:
`phase0/`, `results/phase1c_results.json`, `docs/phase1c_design.md`, Phase 1c code,
and everything in `phase2/`, `docs/phase2_design.md` and `results/phase2_*` (imported
or read, never edited).

**Legend:** **[R]** required by project rules · **[D]** design decision made here ·
**[L]** limitation.

## 1. Single hypothesis [R]

> A coarse-signature candidate generator (Phase 2's M2) with ONE fixed,
> n-independent threshold tau* = 0.6004747353136561 meets the Phase 2 gate at every
> size of the ladder on families that were not used in Phase 2.

tau* = 0.5 x 1.2009494706273123, where 1.2009494706273123 is the Otsu tau_sig that
Phase 2 computed at n=280 (`results/phase2_results.json`, sizes["40"]["tau_sig"]).
tau* is a **constant of this design**: it is never recomputed, never rescaled with n
and never fitted to the fresh data.

## 2. Origin and status of the hypothesis [R][L]

Phase 2's pre-declared secondary curve showed, after the results were known, that M2
at 0.5 x tau_sig(n) would have met the gate at all five sizes. That is a **post-hoc
observation** (report section 6) and was not a Phase 2 result. Phase 2b exists only to
test it properly: the constant is fixed from Phase 2 data and evaluated on data it was
not derived from. One thing differs from the curve: Phase 2 used 0.5 x the per-n Otsu
value (1.2009 to 1.2681); Phase 2b uses a single constant, so at large n the effective
factor is below 0.5 x tau_sig(n). Recall at tau* for n > 280 has **not been examined**
on Phase 2 data; it is unknown.

## 3. Data [D]

- **Fresh families 801-1600**, from the unmodified Phase 0 generator (sequential RNG,
  seed 12345). Because the RNG is sequential, `generate_synthetic_originals(1600, ...)`
  reproduces families 1-800 exactly and then yields 801-1600. Families 1-800 are
  discarded after their regenerated originals are verified against the SHA-256 values
  in the Phase 2 manifest (`phase2/data/F800/manifest.json`); transforms are generated
  only for families 801-1600.
- **Ladder:** F in {40, 100, 200, 400, 800}; size F uses fresh families 801..800+F, so
  n = 7F in {280, 700, 1400, 2800, 5600} and the datasets are prefix-nested.
- **No overlap** with Phase 2 families (1-800) at any size.
- Family ids >= 1000 have four-digit names. **Node order** for a size: the F originals
  by ascending family id, then the transformed images by (family id, filename).
- Ground truth, true pairs (21 per family) and transformation labels are defined exactly
  as in Phase 2 section 5 and 17.7, and are used only to score candidate sets after they
  are fixed.
- Disk: about 0.35 GB of temporary originals plus about 1.6 GB of fresh data
  (gitignored); Phase 2's data is regenerable and may be deleted if space is needed.

## 4. Method [D]

M2* = Phase 2's M2 coarse signature (1056-d, `phase2.signatures`, imported unchanged;
design 17.4) with candidate rule **L2 distance < tau***. There is no Otsu step, no
per-n computation and no secondary operating curve, so there is nothing to re-pick.
M1 and M3 are not evaluated: M1 stays a **reported negative result** from Phase 2.

## 5. Control and metrics [R]

Random control B1: for each n, |P| pairs drawn uniformly without replacement with a
fresh `np.random.default_rng(12345)`, as in Phase 2. Metrics are those of Phase 2
section 7 (recall, reduction, |P|, |P|/|T|, original-transform vs transform-transform,
per-transformation recall). A method set is identified by its size and the SHA-256 of
the sorted flat-index array.

## 6. Gate [R]

M2* passes iff at **every** size: candidate recall >= 0.95 AND reduction >= 0.90 AND
recall for every transformation >= 0.85. **The hypothesis is supported iff M2* passes at
all five sizes**; otherwise it is rejected and reported as a negative result. No
secondary metric rescues a failed gate. The 800 fresh families are consumed by this
test: any further attempt needs a new design frozen before it is run, on families not
used here.

## 7. Cost [D]

As Phase 2 section 17.9: single-process, BLAS threads pinned to 1, run only after all
heavy jobs have finished. Verification time at each n is a **projection** (|P| x the
mean of a fixed 200-pair sample of candidate pairs, seed 12345, `pairwise_distance`
only); the all-pairs baseline above n=280 is **extrapolated** at 0.1300 s/pair and
labelled so. Cheap-stage costs (signature extraction, all-pairs signature distances)
are timed single-process. Timings are kept out of the byte-comparable results file.

## 8. What is not repeated [D][L]

The n=280 distance-matrix recompute and the section-11 analysis (verifier cap, E+
retention, clustering) are **not** repeated on fresh data. The verifier cap measured in
Phase 2 (79.76% at n=280 on families 1-40) is a reference only and is not re-estimated.
Phase 2b makes a statement about the candidate stage, not end-to-end quality.

## 9. Pilot (before the full run) [R]

F=3 on fresh families 801-803 (21 images, 210 pairs, 63 true pairs). Validation items:
disjointness from Phase 2 families; regenerated families 1-800 match the Phase 2
manifest; nestedness; determinism across two runs; signature shapes and dtypes;
signature code path reproduces the cached Phase 2 signature of a Phase 2 image exactly;
tau* equals 0.5 x the tau_sig in the Phase 2 results file and equals the literal in this
document; hand-checked recall/reduction arithmetic; B1 size equality. Pilot output is not
evidence and is not committed; stop if any item fails.

## 10. Audit [R]

As Phase 2: independent recomputation of candidate sets and headline numbers at every
size (bit/L2 computed by different code), gate verdict recomputation, no tracked file
modified versus the Phase 2 branch tip except additions and `.gitignore`, no ground
truth reachable from signature or candidate code, candidate sets hashed before labels
are read, and a byte-identical fresh-process rerun at n=280. Additionally: tau*
equality check and Phase 2 / Phase 2b family disjointness.

## 11. Implementation layout [D]

New package `phase2b/` (imports `phase2` unchanged), `experiments/run_phase2b_pilot.py`,
`run_phase2b.py`, `run_phase2b_timing.py`, `audit_phase2b.py`, tests in
`tests/test_phase2b.py`, gitignored `phase2b/data/` and `phase2b/artifacts/`, results in
`results/phase2b_results.json` and `results/phase2b_timing.json`, and
`docs/phase2b_report.md`. Seed 12345 everywhere. The results file records no timings
and nothing machine-specific.

## 12. Failure conditions [R]

Any frozen path changed; tau* or any parameter changed after seeing results; ground truth
used before candidate sets are fixed; non-determinism; fresh families overlapping Phase
2's, or regenerated families 1-800 not matching the Phase 2 manifest; M2* recall not above
B1; a failed gate (reported as the negative result, with no retuning).

## 13. Interpretation limits [L]

- Even a full pass says only: on this generator, a fixed threshold met the gate on one
  held-out draw of independent random families. It supports no claim about real
  photographs, no novelty claim, and no claim that tau* transfers to Phase 3, where the
  signature distribution will differ and any threshold must come from a protocol frozen
  before Phase 3 results.
- Independent random families give few hard negatives, so reduction is likely optimistic.
- The rotation margin is thin: Phase 2's per-transformation minimum at the equivalent
  setting was 0.896 against a 0.85 floor.
- One held-out draw gives no estimate of draw-to-draw variance.
- tau* was chosen after seeing Phase 2 curves; the held-out test is what makes the
  hypothesis falsifiable, not the choice of constant.
