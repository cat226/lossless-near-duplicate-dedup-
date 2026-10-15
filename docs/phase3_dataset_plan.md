# Phase 3 Dataset Plan (design only)

Status: **DRAFT — design, with local acquisition v1 done (section 8.7). Nothing has been
evaluated and no Phase 3 result exists.** Acquisition and cleaning prep must not feed
Phase 2 or Phase 2b (`project_handoff.md` section 11). Parameters in sections 8.4-8.5 are
frozen only by a Phase 3 design commit, before any Phase 3 result.

Purpose: replace "synthetic only" with real-image evidence, which a conference
reviewer will require. Phase 2 and Phase 1c results hold only for the synthetic
generator and six transformations.

**Evidence labels:** **[V]** verified from the primary page on 2026-10-12 ·
**[U]** unverified, must be checked locally before use.

## 1. Candidate sources

| Source | What it gives | Verified facts | Unverified |
|---|---|---|---|
| **INRIA Copydays** (`lear.inrialpes.fr/~jegou/data.php.html`) | Real copy-detection benchmark with native ground truth: originals plus JPEG (scale 1/16, quality 75..3), crop (10-80% of area removed) and "strong" attacks (print-and-scan, blur, painting) | [V] originals archive 208 MB; crops 2.0 GB; scale+JPEG 77 MB; strong 60 MB (229 strongly attacked images). [V] INRIA is the copyright holder of all images, provided "as is" | [U] number of originals; whether redistribution is allowed (assume **no**); whether a Hugging Face mirror (`randall-lab/INRIA-CopyDays`) carries a different license |
| **DISC21** (Meta Image Similarity Challenge; `ai.meta.com/datasets/disc21-dataset/`) | Standard large-scale copy-detection benchmark: 1M reference images, 50k dev queries, 50k test queries, 1M training images | [V] about 350 GB in 42 zips of 50,000 images each. [V] images come from YFCC100M filtered to the broadest licenses, recognisable people excluded. [V] queries are edited versions of references (edits, collages, re-encoding) | [U] **license wording** (not stated on the landing page; read the download page and the paper); [U] ground-truth file format and location (likely the `facebookresearch/isc2021` repo); [U] which zips hold which reference range |
| **MIRFLICKR-1M + MFND annotations** | Real in-the-wild photos where near-duplicates were later annotated (identical and non-identical) | [V, via search] images are Flickr Creative Commons; Connor et al. annotated near-duplicates in MIRFLICKR-1M (MFND) | [U] where MFND labels are hosted; [U] per-image license handling |
| **NASA Image and Video Library** (`images-api.nasa.gov`) | Large pool of real photographs for distractors, with injected synthetic families | [V] NASA content is "generally not subject to copyright in the United States" but the insignia/logotype are not; third-party copyrighted images belong to their holders; acknowledge NASA as source; identifiable-people and endorsement restrictions apply | [U] whether an API key is required and the rate limits (the pages I fetched did not say) |
| **AI4Mars** (Zenodo 15995036) | Hard-negative stress test (many similar Mars terrain frames) | not re-checked this session | [U] everything; the handoff notes the license is unclear (Hugging Face mirror says CC0, NASA catalog says unspecified) |
| **California-ND** (701 personal photos) | Small set of real *non-identical* near-duplicates | [V, via search] 701 photos, real-world NINDs | [U] license and download |
| **UKBench / Holidays** | Object / scene retrieval groups | not near-duplicate labels by themselves | [U] licenses; pair labels would have to be derived |

## 2. Recommended design

Two tiers, so a negative result on one cannot be hidden by the other.

- **D3a, controlled injection (primary).** Real photographs as distractors plus
  injected families built with the **same six frozen transforms** as Phase 0.
  This isolates "does the method survive real image content" from "does it survive
  new attack types". Distractor pool: a license-clean subset (MIRFLICKR-1M CC subset,
  DISC21 reference subset, or NASA Earth/ISS photos).
- **D3b, native real attacks (sanity check).** INRIA Copydays as-is, mixed into a
  distractor pool from D3a, for crop, heavy JPEG and strong attacks that the
  six-transform generator never produced.
- **D3c, scale (optional).** A DISC21 subset (for example 50k reference images plus
  the dev queries whose ground-truth references fall in the subset). Its ground
  truth is query-to-reference, so all-pairs true pairs must be derived: (query,
  reference) plus queries sharing a reference. Freeze this derivation before looking
  at results.

## 3. The verifier problem (decision needed)

The frozen Phase 1c distance compares only the **overlapping top-left region**
with no resizing or alignment. On Copydays crops (10-80% area removed) and strong
attacks it will reject many true pairs regardless of candidate generation, so
end-to-end quality would be capped by the verifier (as already noted for Phase 2).

Options, to be chosen **before** any real-data run:

1. Keep the frozen verifier and report the cap as a limitation. Honest, but the
   headline numbers on D3b will look poor.
2. Define a **new, separately named and separately frozen Phase 3 verifier**
   (for example resize both images to a common size, then the same zlib distance)
   without touching Phase 1c code. Legitimate, but it is a new component and must
   be justified and reported as such, not as "the Phase 1c result".

## 4. Hygiene rules (from the handoff, restated)

- Freeze the labelling policy first: "same scene, next frame" is a distinct image,
  not a duplicate.
- One rendition per asset ID; drop logos, diagrams, concept art, videos, and
  anything with a copyright credit.
- Convert grayscale to RGB and freeze a resolution rule (the distance normalises
  by w*h*3 and does not align mismatched sizes).
- Audit every distractor pool for hidden duplicates with SHA-256 plus a deep-embedding
  nearest-neighbour pass and manual spot checks. **Do not use pHash for the audit**
  (it is a candidate method). The embedding is used for auditing only.
- Dark astronomy images may look identical under the Phase 1c distance (a mostly-zero
  difference image compresses tiny). This is an inference, not measured. Prefer
  Earth, ISS, Apollo and Mars terrain photos and stratify by category.
- Ground-truth labels live in a separate file; write a manifest (hash, size, source,
  license); lock a dev/test split before any evaluation.
- Do not redistribute Copydays or DISC21 images. Publish manifests and scripts, not images.

## 5. What a conference reviewer will ask for (gaps to close)

- Real-image results (this plan) rather than synthetic-only evidence.
- Comparison against standard baselines beyond the project's own B0/B1: exact hash,
  pHash and dHash at several radii; optionally published copy-detection descriptors
  as a labelled reference upper bound (not part of the method).
- Recall-versus-reduction operating curves per method, not single points.
- Statistics over more than one random draw of distractors (seeded, reported).
- A clear scope statement: the contribution is the measured candidate-recall versus
  expensive-comparison tradeoff feeding correlation clustering, not pHash or
  correlation clustering themselves.

## 6. Next actions (in order)

1. Read the DISC21 download page and paper for license wording; check the
   `facebookresearch/isc2021` repo for ground-truth format.
2. Confirm the Copydays original count and the Hugging Face mirror license.
3. Confirm the NASA API key requirement and rate limits locally.
4. Choose the verifier option in section 3.
5. Only then write the download/cleaning script and `docs/phase3_manifest.md`.

## 7. Literature check (2026-10-12) and the 10-20 GB bundle

Evidence labels as above. Searches were limited (a handful of queries); "I found
no paper doing X" means exactly that, not that none exists.

### 7.1 Datasets reused across papers

| Dataset | Reuse evidence | Size | Fits 10-20 GB? |
|---|---|---|---|
| INRIA Copydays | [V] used by RDCD (arXiv 2405.17928) with 157 originals / 2,000 queries; SSCD reports transfer to it. Sources disagree on counts (another summary says 157 queries / 3,000 gallery): verify locally | [V] about 2.35 GB (208 MB + 2.0 GB + 77 MB + 60 MB) | yes |
| CD10K = Copydays + 10k YFCC100M distractors | [V] RDCD protocol | [U] estimate 1-5 GB for the 10k distractors | yes |
| INRIA Holidays | widely used retrieval benchmark (not re-verified for reuse this session) | [V] about 2.7 GB (1.1 + 1.6 GB archives) | yes |
| DISC21 / NDEC | [V] standard in CopyNCE (arXiv 2602.17484, Feb 2026) and RDCD | [V] about 350 GB; about 8.3 GB per 50k-image zip | **no**: dev-query ground-truth references are drawn from the whole 1M reference set, so a 10-20 GB subset is not self-contained (inference, [U]) |
| MFND (MIRFLICKR-1M near-duplicate labels) | [V] Morra and Lamberti benchmark, ground truth for all pairs | [U] | [U] check where labels and a subset are hosted |
| Pinterest 53k labelled pairs (Gusev and Xu, WWW 2020) | [V] released; 86% non-near-dupe / 14% near-dupe; 5 judges per pair | [U] | [U] only a Pinterest pin link was found, not a download |
| CleanPatrick (Fitzpatrick17k, 2025) | [V] real human near-duplicate annotations | [U] | not recommended: dermatology domain, licence and data link unconfirmed |

**Proposed bundle (about 12-18 GB):** Copydays (2.35) + Holidays (2.7) + a YFCC100M /
MIRFLICKR CC distractor pool of 30-40k images sized to the budget (estimate,
[U]) + D3a injected families using the six frozen transforms. Every component is
either reused across papers or (D3a) directly comparable to Phase 0-2.

### 7.2 Closest 2025-2026 work found

| Work | Relevance |
|---|---|
| CopyNCE / PixTrace, Ant Group, arXiv 2602.17484 (Feb 2026) [V] | Current DISC21 / NDEC state of the art for learned copy detection (ViT-S; DISC21 dev II 88.7% muAP with the matcher, 72.6% with the descriptor). Needs known edit transforms for training; costly inference. Learned, retrieval-style, no storage-cost accounting |
| Modified Pivot, Behnezhad et al., ICML 2025 [V] | Better-than-3 approximation for correlation clustering; makes under 77% of Pivot's mistakes. Theory plus generic experiments, no image dedup |
| CleanPatrick, arXiv 2505.11034 (2025) [V] | Image data-cleaning benchmark with crowd-annotated near-duplicates; self-supervised representations best, hashing weaker |
| IDedupNet, Informatica 2025; semantic-aware dedup, IJIES 2025 [V via search snippets] | Cloud-storage dedup with CNN features; semantic paper reports F1 0.947 on MNIST-Duplicate |
| GPIC and MONET, arXiv 2605.30341 / 2605.21272 (May 2026) [U: full text exceeded fetch limit; claims come from a search summary] | Dataset-curation dedup; MONET reports DCT perceptual hashing misses geometric transforms; GPIC reports threshold choice strongly affects what is removed |
| Gusev and Xu, Pinterest, WWW 2020 (arXiv 2022) [V from full text] | Production system: candidate generation, selection, clustering. States near-duplicate is **not transitive**; clusters by transitive closure over sub-clusters plus greedy k-cut. Not an explicit objective |

### 7.3 Gaps (hypotheses for positioning, to be revisited as results arrive)

1. **No cost accounting.** I found no recent source measuring candidate recall per
   stage against expensive comparisons saved. Benchmarks report retrieval quality
   (muAP, RP90), not verification cost.
2. **Clustering is unprincipled in practice.** Production uses transitive closure
   (plus heuristics) while acknowledging non-transitivity; correlation clustering
   theory advanced in 2024-25, but I found no image-dedup application with real data.
3. **Benchmarks are retrieval-shaped, not all-pairs graph-shaped.** DISC21 ground
   truth is query-to-reference; the graph objective D(C) needs derived pair labels.
4. **Easy negatives.** DISC21 distractors are reported as easy; NDEC hard negatives
   drop all methods substantially.
5. **Weak cloud-dedup evaluations.** Recent storage-dedup papers report accuracy on
   small or toy sets (for example MNIST-Duplicate).
6. **Variance is rarely reported** (RDCD reports no multi-run statistics).

### 7.4 Model decisions proposed (need user approval; item 1 changes a project rule)

| Role | Frozen so far | Proposed addition | Why |
|---|---|---|---|
| Candidate generator | pHash (M1), coarse signature (M2), hybrid (M3) | **Reference arm: SSCD descriptor** (public weights) with a fixed cosine rule, labelled an upper-bound comparator, Phase 3 only | Reviewers will ask whether handcrafted signatures lose to learned descriptors; CleanPatrick and DataComp-style work suggests they may. **Conflicts with the "no CNN/embeddings before gates" rule**, so it needs your explicit approval |
| Verifier | Phase 1c pixel-zlib distance (no alignment) | Optional separately frozen aligned verifier (section 3) | Crops and strong attacks otherwise cap quality |
| Clustering | KwikCluster; union-find baseline | **Modified Pivot** (ICML 2025) as an extra arm; **transitive closure + greedy k-cut** (Pinterest) as a stronger, published baseline | Same graph, same D(C); closes gap 2 |
| Index (Phase 4) | none | multi-index hashing on pHash vs HNSW on embeddings | Only if Phase 2 passes |

## 8. Decisions of 2026-10-12 and acquisition specification (DRAFT, uncommitted)

### 8.1 Decisions from the user

- **First bundle (CD10K-style):** INRIA Copydays plus 10,000 distractors, resolution capped,
  cleaned while streaming, within the ~8 GB free (6.67 GB free at 2026-10-12). Holidays later.
- **SSCD:** approved only as a labelled, **non-gated comparator** in Phase 3; **must not** be
  used for the distractor audit; its weights licence must be checked first (see 8.3).
- **Verifier:** the frozen Phase 1c distance stays **primary**; an **aligned verifier** is
  reported alongside it, with its definition frozen before any Phase 3 result (see 8.5).
- Phase 2 verdict (gate failed for M1/M2/M3) and the Phase 2b hypothesis do not change this
  plan; any Phase 3 threshold must come from a protocol frozen before Phase 3 results, and
  Phase 2b's tau* is not assumed to transfer.

### 8.2 Source status, checked from this machine on 2026-10-12

| Source | Result |
|---|---|
| INRIA host `pascal.inrialpes.fr` (the links on the Copydays page) | **Unreachable**: DNS timeouts and "connection reset" over http and https. The pages on `lear`/`thoth.inrialpes.fr` load (HTTP 200) but no data path was found there |
| Meta mirror `dl.fbaipublicfiles.com/vissl/datasets/` | `copydays_original.tar.gz` 217,895,397 B: **200**; `copydays_strong.tar.gz` 62,884,651 B: **200**; `copydays_jpeg` and `copydays_crop`: **403** |
| Hugging Face `galilai-group/INRIA-CopyDays` | Loader script and README only: **no data files and no licence tag**; the script downloads from the Meta mirror |
| Multimedia Commons S3 (`multimedia-commons`, us-west-2) | Anonymous HTTPS **works**: `?list-type=2&prefix=` listing and object GET both return 200 (sample: `data/images/000/24a/00024a73d1a4c32fb29732d56a2.jpg`, 40,382 B, image/jpeg). Layout `data/images/<3 hex>/<3 hex>/<hash>.jpg` |

Consequences: bundle v1 can contain Copydays **original + strong** only. The JPEG-attack
(77 MB) and crop (2.0 GB) sets wait until the INRIA host is reachable from some network.
Mirror downloads still carry INRIA's copyright ("as is"): local research use only, no
redistribution; publish manifests and scripts, not images.

**D3b is weaker as a result.** Bundle v1 lacks Copydays' **crop** and **JPEG** attack sets,
so D3b covers only the strong attacks (print-and-scan, blur, painting and similar). The
frozen six-transform set of **D3a** includes a 90%-area centre crop and JPEG quality 50, so
D3a covers those transformations on real distractors. Any statement about crop and heavy-JPEG
robustness on real images must therefore rest on D3a, and D3b must not be described as the
full Copydays benchmark.

### 8.3 SSCD weights licence: UNRESOLVED (do not download the weights yet)

The repository's README says "The SSCD codebase uses the MIT license" and the `LICENSE` file
is MIT (Meta Platforms). Neither says anything about the **pretrained weights** hosted at
`dl.fbaipublicfiles.com/sscd-copy-detection/*.torchscript.pt`. MIT on the code does not by
itself license separately distributed weights. Needed before any download: an explicit
statement from the maintainers (for example a GitHub issue) or other Meta terms covering the
weights. **Rule (user, 2026-10-12): if there is no explicit answer by the Phase 3 design
freeze, SSCD is dropped**, not worked around. A draft GitHub issue to the maintainers
exists and awaits the user's posting.

**Audit embedding (chosen, not yet downloaded): DINOv2 ViT-S/14** (21M parameters). It is
neither SSCD nor pHash. Licence, quoted from the facebookresearch/dinov2 README: "DINOv2 code
and model weights are released under the Apache License 2.0." (Separately named models in
that repository, such as XRay-DINO weights under a non-commercial licence and the Cell-DINO
code under CC BY-NC, are not used.) Loading through `torch.hub` needs only PyTorch, which is
installed (2.12.1, CPU). It is used for the distractor-pool audit only, never as a
candidate generator or a comparator.

### 8.4 Acquisition specification (to freeze after review)

- Output: `phase3/data/` (gitignored); manifests in `phase3/manifests/` (tracked): per image
  the source (archive member or S3 key), source SHA-256, original and output sizes, scale,
  output SHA-256, and accept/drop reason. Script: `phase3/acquire.py` (`--dry-run` available).
- **Copydays:** stream each tar (`r|gz`), never store the archive. Images are grouped by
  block (first four digits of the file name; `...00` is the original, `...NN` an attacked
  copy). The scale `s = min(1, 1024 / max(w, h))` is taken from the block's **original** and
  applied identically to its attacked copies, so relative geometry is preserved. LANCZOS,
  saved as lossless PNG RGB.
- **Distractors:** seed 12345. Candidate keys come from 2,500 random start points
  (`start-after=data/images/<h1>/<h2>/`, up to 6 keys each), de-duplicated and sorted; 12,000
  are drawn without replacement and processed in key order; the **first 10,000 that pass**
  are kept. Dropped: undecodable, shorter side < 128 px, exact SHA-256 duplicate. Longest
  side > 1024 is resized (LANCZOS, JPEG q95); otherwise the original bytes are stored
  unmodified. Per-image Flickr licences are **not** available without the YFCC metadata, so
  treat all distractors as research-local.
- **Footprint:** Copydays PNGs are expected to be about 1 GB at most. A 100-image distractor
  trial (all accepted, no drops) averaged 121 KB per image (12.1 MB total), so 10,000
  distractors should take about 1.2 GB. Throughput from the Meta mirror was low (the
  281 MB listing pass alone took about 25 minutes), so the Copydays step is slow, not large.
- **Audit before use:** exact SHA-256, a near-duplicate pass with an embedding model other
  than SSCD and pHash, and manual spot checks of the closest pairs. Labelling policy: "same
  scene, next frame" is a distinct image.

### 8.5 Aligned verifier (draft; must be frozen before any Phase 3 result)

Primary remains `cc_dedup.graph.pairwise_distance`. The aligned verifier is **reported
alongside, never gating**:

1. Work at longest side 256 for both images.
2. Estimate a similarity transform with ORB features and RANSAC (fixed parameters,
   `cv2.setRNGSeed(12345)`).
3. Warp the second image into the first image's frame.
4. Apply the Phase 1c normalised zlib distance over the valid overlap only.
5. If fewer than a fixed number of inliers are found, report "alignment failed" (count the
   failures; do not substitute a distance).

`opencv-python-headless` is **approved and installed** (cv2 5.0.0, numpy 2.5.1 unchanged).
Implementation: `phase3/aligned_verifier.py`; validation: `experiments/validate_aligned_verifier.py`.

**Validation (synthetic Phase 2 families 1-40 only; no Phase 3 data, no families 801-1600).**
Pairs: all 840 true pairs (21 per family) plus 840 different-family pairs (seed 12345). Four
configurations were tried and are all reported; none was tried on anything else.

| Config | true pairs aligned | unrelated pairs aligned | AUC (failures worst) | failure reasons on true pairs |
|---|---|---|---|---|
| default (work 256, FAST 10) | 68.1% | 0.36% (3) | 0.840 | too few features 194, too few matches 58, alignment failed 16 |
| A: FAST 5 | 83.7% | 0.36% (3) | 0.918 | 77 / 23 / 37 |
| B: work 512 | 89.4% | 0.12% (1) | 0.947 | 59 / 21 / 9 |
| **C: work 512, FAST 5** | **100%** | **0.12% (1)** | **1.000** | none |
| frozen Phase 1c distance, same pairs | passes tau on 79.8% (the verifier cap) | n/a | 0.857 | n/a |

C aligns every transformation type (resize, JPEG, crop, brightness, rotation, PNG, and
transform-transform pairs); the Phase 1c verifier passes only 20% of rotation pairs and 75%
of crop pairs. **Proposed frozen configuration (C):** working size 512, ORB 1000 features,
FAST threshold 5, brute-force Hamming matching with cross-check, similarity transform by
RANSAC (reprojection threshold 3.0 px, 2000 iterations, confidence 0.99, `cv2.setRNGSeed(12345)`),
at least 12 inliers, valid overlap at least 25%, scale in [0.25, 4], reference = lower index.
The distance threshold comes from the Otsu protocol on the Phase 3 aligned-distance
distribution, frozen in advance; it cannot be derived from synthetic data. Torch 2.12.1 (CPU)
is installed; torchvision and scikit-image are not.

Caveats, stated up front: the synthetic images are smooth, so ORB finds few features at low
resolution (that is why the working size and FAST threshold mattered); the validation
transformations are exact similarity transforms, which flatters a similarity model, whereas
print-and-scan and painting are not rigid; the cost of the aligned verifier at working size
512 has not been measured; and an AUC of 1.000 on 1,680 synthetic pairs is not evidence about
real photographs. The configuration is frozen only by the Phase 3 design commit.

### 8.6 Next actions

1. Post the SSCD issue (draft ready); if there is no explicit answer by the Phase 3 design
   freeze, drop SSCD.
2. Audit the distractor pool before any use: exact SHA-256 (done at acquisition), then a
   DINOv2 ViT-S/14 near-duplicate pass and manual spot checks of the closest pairs.
3. Freeze 8.4 and 8.5 (including the aligned-verifier configuration) in a Phase 3 design
   commit before any Phase 3 evaluation; define the D3a family injection and labelling policy.
4. Retry the INRIA host later for the JPEG and crop sets; Holidays later.

### 8.7 Acquisition v1 results (local, 2026-10-12)

| Item | Result |
|---|---|
| Copydays original | 157 images, all accepted; 157 blocks |
| Copydays strong | 229 images, all accepted, covering the same 157 blocks (1-5 per block) |
| Copydays resolution cap | every one of the 386 images was scaled (the block's original exceeds 1024 px); saved as PNG; 273 MB |
| Distractors | 10,022 candidate keys processed, **10,000 accepted**, 22 dropped (shorter side < 128 px); no exact duplicates; **none needed resizing** (median 121,662 bytes); 1.2 GB |
| Determinism | the 100-image trial matches the full run on all 100 shared keys (identical output SHA-256) |
| Manifests | `phase3/manifests/copydays.jsonl`, `distractors.jsonl` (+ summary) and the trial files; images are in gitignored `phase3/data/` |

Not yet done: the labelling policy, the D3a injection design and the dev/test split.
Per-image Flickr licences are unknown, so all distractors are research-local.

### 8.8 DINOv2 distractor audit (run 2026-10-12; manifest `phase3/manifests/distractor_audit.json`)

Declared before the run: pool = 10,000 distractors + 157 Copydays originals; DINOv2 ViT-S/14
CLS embedding, 224x224 whole-image resize; exact all-pairs cosine; flag at cosine >= 0.90;
manual check of the 30 closest pairs and the 10 closest distractor-Copydays pairs; nothing deleted.

**Weights licence (recorded in the manifest):** Apache License 2.0. Evidence: the README
"License" section of facebookresearch/dinov2 ("DINOv2 code and model weights are released under
the Apache License 2.0. See LICENSE for additional details.") and the repository `LICENSE`
(Apache License, Version 2.0, January 2004). Weights file 88,283,115 bytes (matches the expected
size), SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.

| Result | Value |
|---|---|
| Pairs with cosine >= 0.80 / 0.85 / 0.90 / 0.95 / 0.98 | 26,983 / 20,627 / 10,786 / 5,155 / 2,055 |
| Flagged (>= 0.90) by kind | 10,785 distractor-distractor, 1 Copydays-Copydays, **0 distractor-Copydays** |
| Closest distractor-Copydays pair | cosine 0.8618: different photographs of similar scenes; no distractor is a copy of a Copydays original |

**Post-hoc finding (not anticipated by the declared rules; thresholds not pre-registered):**
the top 30 pairs were all **flat colour-gradient images, not photographs**. Measured as detail
energy (mean absolute difference from a 16x16-resampled copy), **242 of the 10,000 distractors
(2.4%) fall below 1.0** (277 below 3.0), all with near-identical tiny files (about 0.013
bytes/pixel). They account for 10,773 of the 10,785 flagged pairs. Among real photographs only 12
pairs reach 0.90, and visual inspection after the fact shows different photographs of similar
subjects (the closest, 0.946, is two frames from what appears to be one fixed camera at
different times, a distinct image under the labelling policy). The 0.85-0.90 band was not
inspected.

Consequence: these gradients are degenerate distractors that look alike to any similarity
method. **No file was deleted or edited.** An exclusion rule (for example a detail-energy
cut-off, applied from `phase3/manifests/distractor_audit_low_detail.jsonl`, followed by seeded
resampling to restore 10,000) is a design decision for the Phase 3 design freeze, and the
threshold must be chosen there, not from this exploration alone.

### 8.9 Pool curation (content-only; run 2026-10-12; manifests `distractor_pool*.jsonl`)

The user approved an exclusion rule for the flat images under these conditions: image content
only (no labels, method outputs, candidate sets or pHash); view the images between the flat group
and the 277 below detail 3.0; choose the cut-off from the distribution or a content criterion;
resample from the seeded reserve back to 10,000; report a secondary run that includes the
excluded images.

- **Viewing result.** A seeded sample of 21 of the flattest images were all solid-colour fills.
  The 41 images ranked just above them (detail 0.40-2.98) were all **real photographs**
  (crescent moons, bubbles, a bird against sky, a face in the dark, fog, sunsets, a rainbow swirl,
  the Sydney Opera House, an underwater fish, a stage). So the earlier "242 below 1.0" and "277
  below 3.0" figures would have removed real low-texture photographs and are **not** used.
- **Rule.** Exclude an image iff its detail energy is below the midpoint of the **largest gap in
  the low tail**: the gap between exactly 0.0 and 0.402416, so the cut-off is **0.201208**. This
  excludes **236** images (the exactly-flat group). It depends only on pixel content.
- **Replacement.** The seeded sampler reproduced the original 12,000-key candidate list exactly
  (this also shows the bucket is unchanged); the 1,978 unprocessed keys are the reserve. Taken in
  key order and cleaned by the same acquisition rules plus the same content rule, 236 replacements
  were accepted (3 reserve images were themselves flat and excluded). The final pool is exactly
  **10,000 images with 10,000 distinct SHA-256 values**. No file was deleted; the 236 excluded
  files stay on disk and are listed in `distractor_pool_excluded.jsonl`. They form the
  **secondary pool** (final pool + excluded images).
- **Re-audit of the final pool** (same declared rules, DINOv2 ViT-S/14, 9,921 embeddings reused and
  236 new): pairs at cosine >= 0.80 / 0.85 / 0.90 / 0.95 / 0.98 are 436 / 107 / 13 / 0 / 0; the 13
  at or above 0.90 are 12 distractor pairs (the same 12 viewed earlier, none a duplicate) and the
  one Copydays-internal pair; no distractor-Copydays pair reaches 0.90. The 30 closest pairs were
  viewed: all different photographs of similar subjects.
- **Band 0.85-0.90, previously uninspected:** a seeded sample of 24 of its 94 pairs was viewed: 0
  duplicates. Not exhaustive: the other 70 pairs and everything below 0.85 were not inspected.
- **Known-ambiguous cluster.** Five images appear to be frames from one fixed street camera at
  different times (four pairs in the manifest, cosine 0.946, 0.892, 0.873, 0.866). They are kept as
  **distinct** images under the frozen labelling policy ("same scene, next frame" is a distinct
  image) and are recorded in `distractor_pool.summary.json`. They are the hardest kind of negative
  for a similarity method. One Copydays-internal pair of different blocks (205800 and 207800, two
  views of the Verona arena, cosine 0.8959) is also similar but a different photograph.
