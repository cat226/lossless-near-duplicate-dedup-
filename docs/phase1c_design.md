# Phase 1c Experimental Design (Locked)

Status: **DRAFT v3 — final design lock, for review. No code written, no
experiment run.** Phase 0 (`phase0/`) is merged and untouched by this
document.

This revision closes the three remaining open questions from v2
(diagnostic-midpoint labeling, pilot-vs-full-run, flat-Otsu fallback),
tightens the pixel-difference correction's operation order, and adds an
explicit execution-order freeze so no step can implicitly leak into an
earlier one. A final freeze checklist is at the end.

**Legend**, applied to every decision below:
- **A** — explicitly required by the project plan / your instructions
- **B** — an experimentally chosen design decision made in this document
- **C** — a limitation or open question that cannot be resolved without
  running the experiment (not a missing decision — the decision is made;
  the outcome is unknown until measured)

## 1. Research question

Does a practical correlation-clustering heuristic produce strictly lower
weighted disagreement than naive threshold-based chaining, when both
methods run on the identical signed, weighted candidate graph — same
nodes, same similarity values, same threshold, same edge weights — with
that graph built entirely without reference to ground truth? **(A)**

## 2. Dataset, node set, and graph size

Reuse the existing Phase 0 synthetic dataset unmodified
(`phase0/make_test_pairs.py --synthetic`, seed 12345): 40 originals + 240
transformed variants. **(A — Phase 0 generation process unchanged)**

Node set: **all 280 images.** **(A)**

Explicit graph size, stated once here and reused everywhere else in this
document rather than re-derived:

```
n = |V| = 280
|E| = n(n-1)/2 = 280 * 279 / 2 = 39,060 unique undirected pairs
```

This is the concrete complexity baseline for every later section:
`T_graph = O(n^2)`, and both clustering algorithms receive a graph with
`|V| = 280`, `|E| = 39,060`. **(A — explicit calculation requested)**

## 3. Ground truth: scope and anti-leakage rule

| Use | Allowed? |
|---|---|
| Selecting/tuning the positive/negative edge threshold | **No** |
| Selecting which correlation-clustering algorithm to use | **No** |
| Selecting the random seed based on which seed "looks better" | **No** |
| Computing secondary metrics (pairwise P/R/F1) after both methods are frozen and run | **Yes** |
| Computing a diagnostic-only same/different distance midpoint, reported but never fed back in | **Yes** |

**(A)** — see §15 for the exact execution order that makes this
structurally enforced, not just a stated intention.

## 4. Similarity representation — corrected pixel distance

**(A: must explicitly choose and justify; B: the specific choice)**

`raw_delta_size` (bsdiff4 encoded-byte delta) is **rejected**: Phase 0's
own measured medians show it misclassifies 4 of 6 transform types as
dissimilar (crop 0.884, bright 0.938, rotate 0.909, png 5.840 — all far
above any reasonable "similar" cutoff despite being genuine
near-duplicates). It measures byte-stream compressibility, not visual
similarity, and is not resurrected here.

### 4a. Correction to Phase 0's pixel-delta computation

Phase 0's `pixel_delta_size` (`phase0/delta_probe.py`) computes:

```python
diff = (a.astype(np.int16) - b.astype(np.int16)).astype(np.int8)
```

`int8` holds only `[-128, 127]`. A difference of magnitude >127 (plausible
at rotation's black-fill borders or high-contrast regions) silently wraps
around instead of staying correct. **This is a real defect in the
existing Phase 0 code**, documented here explicitly as a Phase 1c
correction — **not** applied retroactively to `phase0/`, which remains
untouched and whose historical results are not altered.

**Corrected computation, operation order made explicit and unambiguous**
(per review — subtraction must happen in a signed, wide-enough type,
*then* `abs()`, *then* narrow to `uint8`; doing the subtraction directly
in `uint8` would wrap before the absolute value is ever taken):

```python
diff = np.abs(a.astype(np.int16) - b.astype(np.int16))  # still int16, safe: range [-255, 255] before abs, [0, 255] after
diff = diff.astype(np.uint8)                              # safe to narrow now: abs value of two uint8 pixels is always in [0, 255]
```

Two statements, deliberately not chained into one expression, so the
order (subtract in `int16` → `abs` → narrow to `uint8`) can't be
misread. This is the same computation proposed in v2 of this document
(the one-line chained version there already applied `abs()` before the
`uint8` cast); it's now written unambiguously to close off any doubt.

### 4b. Behavior under each transform (expected, not yet measured)

| Transform | Expected `r_ij` | Why |
|---|---|---|
| jpegq50 | low | same dimensions, only quantization noise |
| png (format) | ~0 | pixel-identical (matches Phase 0's own pixel median of 0.012) |
| crop | low | overlap-region comparison excludes the cropped-out border |
| resize | low–moderate | interpolation changes pixel values within the overlap region |
| bright | low–moderate | roughly constant per-pixel offset is a structured, compressible diff |
| rotate | moderate–high | 5° rotation perturbs nearly every pixel via interpolation — least favorable case |

**(C)** — not yet measured; an empirical question the full run answers.

### 4c. Why appropriate for Phase 1c

Directly measures decoded visual content difference; immune to
encoding/container artifacts (unlike `raw_delta_size`); symmetric (see
§5); a deterministic classical pixel-space + compression computation,
not a learned model — the "no CNN/embedding unless justified" constraint
is satisfied by not needing one at all. **(A, satisfied; B, the specific
signal)**

## 5. Normalization

```
r_ij = len(zlib.compress(diff.tobytes(), level=9)) / max(w_i*h_i*3, w_j*h_j*3)
```

Symmetric (max of the two images' raw pixel-buffer sizes), replacing
Phase 0's asymmetric encoded-file-size denominator, since a complete
undirected graph has no privileged "original" per pair. **(B)**

## 6. Unsupervised threshold-selection rule

**(A: deterministic, unsupervised, justified before implementation, no
label access — B: the specific algorithm and fallback)**

### 6a. Primary rule: Otsu's method

1. Histogram `{r_ij}` over all 39,060 pairs, fixed **256 bins** (Otsu's
   standard convention, chosen independent of this dataset).
2. For each candidate split `t`: `σ²_between(t) = w0(t)·w1(t)·(μ0(t)−μ1(t))²`.
3. `τ = argmax_t σ²_between(t)`.

Input: only the 39,060 `r_ij` values — no label is consultable.
Complexity: `O(n_pairs) + O(256) ≈ O(n_pairs)`, negligible next to the
`O(n²)` distance computation.

**Why Otsu over a fixed percentile:** each node has exactly 6 true
same-cluster peers out of 279 possible edges (~2.15% positive rate by
construction) — a median split would force ~50% of edges positive,
badly wrong for this graph. Otsu adapts to the actual distribution shape
instead of assuming a base rate.

### 6b. Deterministic fallback: flat Otsu curve

**Flatness criterion (fixed now, not tuned after seeing data):** let
`range = max(σ²_between) - min(σ²_between)` over all 256 candidate
splits. If `range < 1e-9` (an absolute tolerance, chosen because
`σ²_between` values here are products of fractions in `[0,1]` and
distances on a comparable small scale — fixed before implementation, not
adjusted afterward), Otsu is declared **uninformative** and the fallback
applies:

```
τ = (min(r_ij) + max(r_ij)) / 2
```

**Recorded in every experiment output**, never silently substituted:

```
threshold_method = "otsu"                    # normal case
threshold_method = "range_midpoint_fallback"  # flatness criterion triggered
```

This also correctly covers the degenerate case where every `r_ij` is
identical (no spread at all): `range = 0 < 1e-9`, fallback triggers,
`τ` equals that common value, and the tie rule in §7 sends all such
edges negative (weight 0, so it doesn't distort the objective).

## 7. Signed-edge rule

`sign(i,j) = "+"` if `r_ij < τ`, else `"−"`. Exact tie (`r_ij == τ`) is
negative with weight 0, so the convention is inert. **(B, consequence of
§6)**

## 8. Edge weights

`w_ij = |τ − r_ij|` — non-negative, larger further from threshold, 0 at
the threshold. Identical `(sign, w_ij)` computed once, supplied
unmodified to both methods. **(A — same weight to both methods required;
B — the formula)**

## 9. Correlation-clustering algorithm

**(A: known practical heuristic, not invented — B: specific choice)**

**KwikCluster / Pivot** (Ailon, Charikar, Newman, 2005).

- **Procedure:** while unclustered vertices remain, pick a uniformly
  random unclustered pivot `v`; cluster `v` with every remaining
  unclustered vertex having a positive edge to `v`; remove that cluster;
  repeat.
- **Objective:** heuristic minimization of §11's disagreement objective
  — no exact-optimality claim (NP-hard in general).
- **Deterministic/random:** randomized pivot choice.
- **Seed:** `12345` (Phase 0's dataset seed, reused for continuity, fixed
  before any Phase 1c distance was computed — not chosen post-hoc).
- **Time complexity:** `O(|V| + |E|)` amortized = `O(n²)` on this
  complete graph.
- **Approximation caveat (C):** the classical 3-approximation bound
  assumes probability-calibrated weights in `[0,1]`; `w_ij` here is a
  practical construction, not probability-calibrated — used purely as a
  heuristic.

## 10. Baseline

`G+ = (V, E+)` → connected components via union-find (path compression +
union by rank). Same graph, signs, weights as §7–8. Deterministic, no
randomness. `O(|E|·α(|V|)) ≈ O(n²)`. **(A — fully specified, nothing
open)**

## 11. Disagreement objective (primary)

```
D(C) = Σ_{(i,j) ∈ E+, C_i ≠ C_j} w_ij  +  Σ_{(i,j) ∈ E-, C_i = C_j} w_ij
```

Computed identically for both methods' output clusterings. **(A)**

## 12. Primary gate

```
D(correlation clustering) < D(threshold chaining)
```

Strict inequality, identical graph, decided before any run. Not
overridden by runtime, cluster count, or secondary metrics. **(A)**

## 13. Runtime measurement

- Graph construction (similarity + threshold + weights) timed once,
  separately — shared infrastructure, not attributable to either method.
- Each clustering method timed independently via `time.perf_counter()`,
  starting from the already-built graph object.
**(B — measurement protocol, since only "report runtime" was specified)**

## 14. Secondary evaluation (ground-truth-based, post-hoc only)

Computed **only after both methods are frozen and have produced final
clusterings** (see §15's ordering). Never used to select threshold,
algorithm, or seed. **(A — required as secondary, non-gating)**

Standard pairwise clustering evaluation:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

using `same_gt = 1` iff `i,j` share a source original, `same_pred = 1`
iff `C_i = C_j`. Plus number of clusters per method. **(A — metric
family required; B — exact formulas)**

**Diagnostic midpoint (labeling closed per review):**

```
label_midpoint_threshold = (mean(r_ij | same-original) + mean(r_ij | different-original)) / 2
```

Computed at this stage (§14), i.e. *after* the primary graph, threshold,
edge signs/weights, and both clusterings are already frozen (§15) — it
cannot causally influence anything upstream of it. Reported in results
as a clearly separate, explicitly labeled field:

```
label_midpoint_threshold: <value>   # diagnostic/reference only —
                                     # computed from ground-truth labels,
                                     # never used to threshold, weight,
                                     # or cluster. Useful only for
                                     # explaining Otsu's behavior after
                                     # the fact.
```

Reported numerically in the results JSON; any visualization (e.g.
marking it on a histogram of `r_ij`) is a reporting/analysis-stage
choice, not a methodology gate. **(A — diagnostic use explicitly
permitted; B — presentation resolved as "numeric field now, plot later
if useful")**

## 15. Execution order (threshold/config freeze, made explicit)

This ordering is what makes §3's anti-leakage table structurally true,
not just asserted:

1. Load/generate dataset (Phase 0 machinery, unchanged) — **frozen**.
2. Build complete graph: compute `r_ij` for all 39,060 pairs (§4–§5) —
   **frozen**.
3. Compute threshold `τ` via Otsu, or the fallback if the flatness
   criterion fires (§6) — input is *only* the `r_ij` vector — **frozen**.
4. Assign signs and weights (§7–§8) — **frozen**.
5. Run baseline (union-find) on the graph from steps 2–4 — **frozen**.
6. Run KwikCluster, seed `12345`, on the same graph — **frozen**.
7. Compute `D(C)` for both (§11) → **primary gate decision made here**
   (§12).
8. **Only after step 7**, compute secondary metrics: pairwise P/R/F1
   (§14) and the diagnostic label midpoint (§14). These may not alter
   steps 2–6 under any circumstance, including a "disappointing" gate
   result at step 7.

No step may be re-run with different parameters after its output has
been used by a later step in this list. **(A)**

## 16. Pilot validation run

**Purpose: implementation correctness only — never parameter tuning.**
**(A: pilot required per review — B: the specific subset and checks)**

**Deterministic subset:** the first 3 originals by filename order (001,
002, 003) plus their 6 transformed variants each = `3 + 18 = 21` images.
`C(21,2) = 210` pairs — small enough to inspect by hand, large enough to
exercise every code path.

**Checklist the pilot must confirmatively pass:**

- [ ] Graph construction produces exactly `21*20/2 = 210` unique
      undirected pairs (no duplicates, no self-pairs).
- [ ] Every computed weight is finite (no `NaN`/`inf` from a
      zero-size image, degenerate zlib output, etc.).
- [ ] Otsu (or its fallback) returns a value, and `threshold_method` is
      recorded as one of the two allowed strings (§6b).
- [ ] Both positive and negative edges are actually produced (not all
      one sign — a pilot producing zero edges of either sign signals an
      implementation bug, not a real finding, at this tiny scale).
- [ ] Union-find baseline terminates and returns a valid partition of
      all 21 nodes.
- [ ] KwikCluster terminates (bounded by `|V| = 21` iterations) and
      returns a valid partition of all 21 nodes.
- [ ] `D(C)` is computable (finite, non-negative) for both partitions.

**Explicit constraints on the pilot:**

- The pilot's numeric results (its threshold, its `D(C)` values, its
  clusters) are **not interpretable findings** — 210 pairs is too small
  a sample for a meaningful Otsu split — and are **discarded**, not
  folded into or compared against the full-run results.
- The pilot must **not** be used to adjust the 256-bin count, the
  `1e-9` flatness tolerance, the seed, the normalization, or any other
  frozen parameter from §4–§14. If the pilot fails a checklist item, the
  fix is a code-correctness fix, applied uniformly, not a parameter
  change motivated by "what makes the pilot's numbers look better."
- Pilot output is written separately (e.g. `results/pilot_validation.json`
  or console/log output) — never merged into `results/phase1c_results.json`,
  so it can't be mistaken for Phase 1c experimental evidence.

**After the pilot passes all checklist items, the full 280-node,
39,060-pair experiment runs with the exact same code and exact same
frozen parameters — nothing tuned in between.** **(A)**

## 17. Reproducibility

- Dataset generation: deterministic, Phase 0 seed 12345, unchanged.
- Graph construction, threshold (Otsu + fallback), edge weights: fully
  deterministic, no randomness.
- Baseline: fully deterministic.
- KwikCluster: seed `12345`, fixed and reported.
- Pilot subset (first 3 originals by filename) is itself deterministic.
**(A)**

## 18. File structure (proposed, not created)

```
cc_dedup/
    graph.py                   # corrected pixel distance (§4), Otsu +
                                # fallback (§6), edge weights (§7-8) —
                                # single source of truth for both methods
    correlation_clustering.py  # KwikCluster/Pivot (§9)
    disagreement.py            # D(C) objective (§11)
    metrics.py                 # pairwise P/R/F1 (§14) — called only
                                # after both methods are frozen and run

baselines/
    threshold_chaining.py      # union-find over E+ (§10)

experiments/
    run_pilot.py                # §16 — 21-node validation, discarded output
    run_phase1c.py               # §15's ordering, full 280-node run

results/
    pilot_validation.json        # §16 — implementation checks, not evidence
    phase1c_results.json         # full-run metrics, both methods
    phase1c_summary.md           # mirrors phase0/DECISION.md's style

docs/
    phase1c_design.md            # this document
```

`cc_dedup/graph.py` remains the only place the graph is built. `phase0/`
is not modified. **(A)**

## 19. Complexity analysis

```
|V| = 280
|E| = 39,060
T_graph = O(n^2)
```

| Stage | Cost |
|---|---|
| Pairwise similarity (39,060 pairs) | `O(n²)` — dominant cost |
| Otsu threshold (+ fallback check) | `O(n_pairs) + O(256)` ≈ negligible next to above |
| Edge sign/weight assignment | `O(n²)` |
| Baseline (union-find) | `O(n² α(n))` ≈ `O(n²)` |
| KwikCluster | `O(n²)` |
| Disagreement scoring (each method) | `O(n²)` |
| Secondary metrics (each method) | `O(n²)` pairs evaluated |
| Pilot (21 nodes, 210 pairs) | negligible, run once before the full experiment |
| **Total (full run)** | `O(n²)`, bottlenecked by pairwise similarity computation |

## 20. Threats to validity (genuine limitations, not open decisions)

1. **(C)** Otsu assumes a workably bimodal distance distribution; not
   guaranteed given the ~2% true-positive base rate and rotate/resize's
   expected higher distances (§4b). If the flatness fallback (§6b)
   triggers, that is itself a finding about the signal, reported via
   `threshold_method`, not hidden.
2. **(C)** The corrected pixel-distance signal (§4a) fixes a real defect
   but remains an uncalibrated proxy, not a validated perceptual metric.
3. **(C)** No cross-original or cross-transform pairs were ever measured
   in Phase 0 — untested until this graph is actually built.
4. **(C)** Runtime of 39,060 pairwise pixel-diff computations is
   unmeasured before the pilot; the pilot (§16) provides an early
   implementation-correctness check but not a runtime estimate at full
   scale, since 210 pairs won't predictively time 39,060.
5. **(C)** KwikCluster's approximation guarantee doesn't formally hold
   for non-probability-calibrated weights (§9) — heuristic only.
6. **(C)** A single fixed seed is reported (§17) but doesn't
   characterize KwikCluster's variance across pivot orders — a
   multi-seed robustness check is out of scope for this pass.
7. **(C)** Synthetic dataset — inherited from Phase 0 — may not
   generalize to real photographs.

None of these are unresolved *decisions* — every decision they bear on
(§4, §6, §9, §17) is already made and frozen. They are unresolved
*empirical outcomes*, answerable only by running the experiment.

## Final Freeze Checklist

| # | Decision | Locked value | Status |
|---|---|---|---|
| 1 | Phase 0 0.60 threshold | discarded, not reused anywhere | ✅ frozen |
| 2 | Edge threshold rule | Otsu (256 bins) with range-midpoint fallback (§6) | ✅ frozen |
| 3 | Ground truth for threshold/algorithm/seed selection | prohibited (§3, §15) | ✅ frozen |
| 4 | Ground truth for secondary metrics | permitted, post-hoc only (§14, §15 step 8) | ✅ frozen |
| 5 | `raw_delta_size` as similarity signal | rejected (§4) | ✅ frozen |
| 6 | Similarity signal | corrected pixel distance, `abs()` before `uint8` narrowing (§4a) | ✅ frozen |
| 7 | Normalization | symmetric, `max` of raw pixel-buffer sizes (§5) | ✅ frozen |
| 8 | Node set | all 280 images | ✅ frozen |
| 9 | Pair count | 39,060 (§2, restated §19) | ✅ frozen |
| 10 | Edge weights | `w_ij = |τ − r_ij|`, identical to both methods (§8) | ✅ frozen |
| 11 | Correlation-clustering algorithm | KwikCluster/Pivot, seed `12345` (§9) | ✅ frozen |
| 12 | Baseline algorithm | union-find on `E+` (§10) | ✅ frozen |
| 13 | Disagreement objective | as specified (§11) | ✅ frozen |
| 14 | Primary gate | `D(correlation) < D(baseline)`, strict (§12) | ✅ frozen |
| 15 | Diagnostic label midpoint | computed post-hoc, clearly labeled, never fed back (§14) | ✅ frozen |
| 16 | Pilot | 21-node deterministic subset, validation-only, output discarded (§16) | ✅ frozen |
| 17 | Flat-Otsu fallback | `range < 1e-9` → `(min+max)/2`, recorded via `threshold_method` (§6b) | ✅ frozen |
| 18 | Execution order / no post-hoc tuning | §15, steps 1–8, irreversible ordering | ✅ frozen |
| 19 | `phase0/` | untouched | ✅ frozen |
| 20 | Runtime measurement protocol | shared graph time + per-method time (§13) | ✅ frozen |

No unresolved methodological decisions remain. Items in §20 are
measurement outcomes, not open design choices, and will be reported
honestly whichever way they land — including a failed primary gate.
