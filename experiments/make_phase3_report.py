"""
Generate docs/phase3_report.md from the committed JSON files, then verify it.

    python experiments/make_phase3_report.py            # write the report, then check it
    python experiments/make_phase3_report.py --check    # only re-check the existing report

Every table cell and every inline number is produced by a function of the JSON files. The check
re-loads the JSON from disk in a fresh dictionary, recomputes every cell and inline number, and
requires the report text to contain exactly the recomputed strings (table cells position by position).
Prose that states no number is hand-written. Constants of the design (gate thresholds, rung sizes) are
written as typed text and are not checked against the JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
OUT = REPO / "docs" / "phase3_report.md"

JSON_FILES = {
    "results": "results/phase3_results.json",
    "timing": "results/phase3_timing.json",
    "audit": "results/phase3_audit.json",
    "audit1": "results/phase3_audit_attempt1_A10_diskguard.json",
    "posthoc": "results/phase3_posthoc.json",
    "launch": "phase3/artifacts/run_full/launch_manifest.json",
    "resume1": "phase3/artifacts/run_full/resume_1.json",
    "resume2": "phase3/artifacts/run_full/resume_2.json",
}
EVENTS = "phase3/artifacts/run_full/events.jsonl"
SIZES = (1000, 2500, 5000, 10000)
DRAWS = (1, 2, 3)
METHODS = [  # label, key in results['methods'], set name
    ("Exact hash", "EXACT", "EXACT"),
    ("M1 pHash (R ≤ 10)", "M1", "M1_R10"),
    ("dHash (R ≤ 10)", "dHash", "dHash_R10"),
    ("M2 at tau_B", "M2_B", "M2_B_x1.0"),
    ("M3 (M1 ∪ M2)", "M3", "M3"),
    ("M2_A1 (per-run Otsu; NON-GATED comparison)", "M2_A1", "M2_A1"),
]
TRANSFORMS = ("resize", "jpegq50", "crop", "bright", "rotate", "png")
TRANSFORM_LABEL = {"resize": "resize", "jpegq50": "JPEG q50", "crop": "crop", "bright": "brightness", "rotate": "rotate", "png": "PNG"}


def load() -> dict:
    def j(p):
        return json.loads((REPO / p).read_text(encoding="utf-8-sig"))

    J = {k: j(p) for k, p in JSON_FILES.items()}
    J["events"] = [json.loads(l) for l in (REPO / EVENTS).read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    return J


# ----------------------------------------------------------------------------- formatting
def isnan(x):
    return isinstance(x, float) and math.isnan(x)


def f2(v): return "n/a" if isnan(v) else f"{v:.2f}"
def f3(v): return "n/a" if isnan(v) else f"{v:.3f}"
def f4(v): return "n/a" if isnan(v) else f"{v:.4f}"
def f5(v): return f"{v:.5f}"
def i0(v): return f"{int(round(v)):,}"
def pct1(v): return f"{100 * v:.1f}%"
def pct2(v): return f"{100 * v:.2f}%"
def verdict(b): return "PASS" if b else "FAIL"
def rng3(t): return f"{t[0]:.3f}–{t[1]:.3f}"
def rng4(t): return f"{t[0]:.4f}–{t[1]:.4f}"
def rng2(t): return f"{t[0]:.2f}–{t[1]:.2f}"
def rng_i(t): return f"{int(t[0]):,}–{int(t[1]):,}"
def ci3(t): return f"{t[0]:.3f} ({t[1]:.3f}–{t[2]:.3f})"
def pm3(t): return f"{t[0]:.3f} ± {t[1]:.3f}"
def pm1(t): return f"{t[0]:.1f} ± {t[1]:.1f}"
def pm0(t): return f"{t[0]:,.0f} ± {t[1]:,.0f}"
def txt(s): return str(s)


class C:
    """A table cell: fn(J) -> value, fmt(value) -> string."""

    def __init__(self, fn, fmt=f3, bold=False):
        self.fn, self.fmt, self.bold = fn, fmt, bold

    def render(self, J) -> str:
        s = self.fmt(self.fn(J))
        return f"**{s}**" if self.bold else s


TABLES: dict[str, tuple[list, list]] = {}
INLINES: list[C] = []
LINES: list[str] = []
J0: dict = {}


def v(fn, fmt=f3) -> str:
    """Inline number: rendered now, re-verified by the checker."""
    c = C(fn, fmt)
    INLINES.append(c)
    return c.render(J0)


def table(tid: str, header: list[str], rows: list[list]) -> None:
    TABLES[tid] = (header, rows)
    LINES.append(f"<!-- table:{tid} -->")
    LINES.append("| " + " | ".join(header) + " |")
    LINES.append("|" + "---|" * len(header))
    for row in rows:
        LINES.append("| " + " | ".join(c.render(J0) if isinstance(c, C) else str(c) for c in row) + " |")
    LINES.append(f"<!-- /table:{tid} -->")
    LINES.append("")


def P(text: str = "") -> None:
    LINES.append(text)
    LINES.append("")


def H(level: int, text: str) -> None:
    LINES.append("#" * level + " " + text)
    LINES.append("")


# ----------------------------------------------------------------------------- data accessors
def parts(J): return J["results"]["parts"]
def R(J): return J["results"]["results"]
def mres(J, m): return R(J)["methods"][m]
def ladder(J, d, n, nm): return parts(J)[f"ladder_d{d}"]["sizes"][str(n)]["score"]["sets"][nm]
def variant(J, d, nm): return parts(J)[f"variant_d{d}"]["score"]["sets"][nm]
def d3b(J, d, nm): return parts(J)[f"d3b_d{d}"]["score"]["sets"][nm]
def ext(J, d, nm): return parts(J)[f"ext_d{d}"]["score"]["sets"][nm]
def mean(xs): return sum(xs) / len(xs)


def lad_vals(J, nm, key): return [ladder(J, d, n, nm)[key] for d in DRAWS for n in SIZES]
def var_vals(J, nm, key): return [variant(J, d, nm)[key] for d in DRAWS]
def mm(xs): return (min(xs), max(xs))


def fail_count(J, nm, crit):
    cells = [ladder(J, d, n, nm) for d in DRAWS for n in SIZES] + [variant(J, d, nm) for d in DRAWS]
    return sum(1 for c in cells if not c["gate"][crit])


def build() -> None:
    J = J0
    # ------------------------------------------------------------------------- header
    H(1, "Phase 3 report: real photographs")
    P("**Status:** experiment complete and audited (all checks A0–A10 passed on attempt 2); this report is for review before merge. "
      "Branch `phase-3-real-data`. Every number below is generated from the committed JSON files by "
      "`experiments/make_phase3_report.py`, which also re-checks the finished text against the JSON.")
    P("Results hold for these datasets only (a 10,000-photograph Flickr-derived pool with injected transformations, and Copydays v1 strong "
      "attacks). They support no claim of generality or novelty.")

    H(2, "0. Provenance")
    table("prov_commits", ["Item", "Commit / value"], [
        ["Code used for the test-split run (all parts)", "`47f1b879c51808fec6f9b7b40e30c8ee18932c54`"],
        ["Audit script (counts and scope statements added after the run; no check logic changed)", "`32e9447`"],
        ["Pre-registered outputs", "`5f0839a`"],
        ["Post-hoc script (committed before its first run)", "`002af36`, SHA-256 `2ecb0390aac3a6134516d3eba3e49994dd383c390cfefe7f5a6c60e5db7cc57d`"],
        ["Post-hoc output", "`da3c9d3`"],
        ["Design (frozen before any test-split run)", "`docs/phase3_design.md` with Addenda A and B"],
    ])
    rows = []
    for k, p in JSON_FILES.items():
        rows.append([f"`{p}`", C(lambda J, p=p: hashlib.sha256((REPO / p).read_bytes()).hexdigest(), txt)])
    table("prov_hashes", ["File", "SHA-256"], rows)
    P(f"Run launch (UTC): {v(lambda J: J['launch']['launch_time_utc'], txt)}; commit recorded in the launch manifest: "
      f"`{v(lambda J: J['launch']['commit_short'], txt)}`; the test split was unlocked after all preflight gates passed, and "
      f"{v(lambda J: sum(1 for e in J['events'] if e['event'] == 'preflight_passed'), i0)} preflight passes are in the event log "
      "(launch plus two resumes, all on the same commit).")

    # ------------------------------------------------------------------------- 1 verdicts
    H(2, "1. Pre-registered verdicts")
    P("Verdicts are per method and per dataset and are never pooled. **D3a** (injected transformations, the primary dataset) and **D3b** "
      "(native Copydays strong attacks, a weaker sanity check) are reported with equal prominence. Gates: recall ≥ 0.95, reduction ≥ 0.90, and "
      "recall ≥ 0.85 for every transformation, in every cell. D3a headline = the 12 ladder cells (n = 1,000, 2,500, 5,000, 10,000 × 3 draws) plus "
      "the 3 low-prevalence cells (n = 8,000); the ladder-only verdict is reported alongside. D3b: recall ≥ 0.95 and reduction ≥ 0.90 in every draw. "
      "M2_A1 is a non-gated comparison.")
    rows = []
    for label, m, nm in METHODS:
        rows.append([label,
                     C(lambda J, m=m: mres(J, m)["strict_pass_ladder"], verdict),
                     C(lambda J, m=m: mres(J, m)["strict_pass_ladder_and_variant"], verdict),
                     C(lambda J, nm=nm: mm(lad_vals(J, nm, "recall")), rng3),
                     C(lambda J, nm=nm: mm(lad_vals(J, nm, "reduction")), rng4),
                     C(lambda J, nm=nm: mm(lad_vals(J, nm, "min_recall_per_transform")), rng3),
                     C(lambda J, m=m: mres(J, m)["d3b"]["strict_pass"], verdict),
                     C(lambda J, nm=nm: mm([d3b(J, d, nm)["recall"] for d in DRAWS]), rng3),
                     C(lambda J, nm=nm: mm([d3b(J, d, nm)["reduction"] for d in DRAWS]), rng4)])
    table("verdicts", ["Method", "D3a ladder-only", "D3a headline (ladder + n=8,000)", "D3a recall (12 ladder cells)", "D3a reduction",
                       "D3a min per-transformation recall", "D3b", "D3b recall (3 draws)", "D3b reduction"], rows)
    P("Reading: on D3a, M2 at tau_B and M3 pass; M1, dHash and the exact hash fail. On D3b every method fails, including M2 at tau_B and M3. "
      f"M2_A1 (not gated) fails D3a on reduction only (its recall is {v(lambda J: mm(lad_vals(J, 'M2_A1', 'recall')), rng3)} and its reduction "
      f"{v(lambda J: mm(lad_vals(J, 'M2_A1', 'reduction')), rng3)}); on D3b it fails both recall ({v(lambda J: mm([d3b(J, d, 'M2_A1')['recall'] for d in DRAWS]), rng3)}) "
      "and reduction.")
    rows = []
    for label, m, nm in METHODS:
        rows.append([label,
                     C(lambda J, nm=nm: fail_count(J, nm, "recall_ge_0.95"), i0),
                     C(lambda J, nm=nm: fail_count(J, nm, "reduction_ge_0.90"), i0),
                     C(lambda J, nm=nm: fail_count(J, nm, "per_transform_recall_ge_0.85"), i0),
                     C(lambda J, m=m: mres(J, m)["above_random_control_every_cell"], lambda b: "yes" if b else "no")])
    table("fail_counts", ["Method", "D3a cells failing recall (of 15)", "failing reduction (of 15)", "failing per-transformation recall (of 15)",
                          "recall above random control in every cell"], rows)

    H(3, "1.1 D3a by ladder size (mean over the three draws; min–max in brackets)")
    rows = []
    for label, m, nm in METHODS[1:5]:
        for n in SIZES:
            rows.append([label if n == SIZES[0] else "", f"{n:,}",
                         C(lambda J, m=m, n=n: (mres(J, m)["by_size"][str(n)]["recall"]["mean"], mres(J, m)["by_size"][str(n)]["recall"]["min"],
                                                mres(J, m)["by_size"][str(n)]["recall"]["max"]), ci3),
                         C(lambda J, m=m, n=n: mres(J, m)["by_size"][str(n)]["reduction"]["mean"], f5),
                         C(lambda J, m=m, n=n: (mres(J, m)["by_size"][str(n)]["min_recall_per_transform"]["mean"],
                                                mres(J, m)["by_size"][str(n)]["min_recall_per_transform"]["min"],
                                                mres(J, m)["by_size"][str(n)]["min_recall_per_transform"]["max"]), ci3),
                         C(lambda J, m=m, n=n: mres(J, m)["by_size"][str(n)]["candidate_pairs"]["mean"], i0)])
    table("by_size", ["Method", "n", "Recall: mean (min–max)", "Reduction (mean)", "Min per-transformation recall: mean (min–max)",
                      "Candidate pairs (mean)"], rows)

    H(3, "1.2 D3a per-transformation recall at n = 10,000 (mean over the three draws)")
    rows = []
    for label, m, nm in METHODS:
        rows.append([label] + [C(lambda J, nm=nm, t=t: mean([ladder(J, d, 10000, nm)["recall_per_transform"][t] for d in DRAWS]), f3) for t in TRANSFORMS])
    table("per_transform", ["Method"] + [TRANSFORM_LABEL[t] for t in TRANSFORMS], rows)

    H(3, "1.3 D3a low-prevalence variant (n = 8,000, 114 families, about 10% family images)")
    rows = []
    for label, m, nm in METHODS:
        rows.append([label, C(lambda J, nm=nm: mm(var_vals(J, nm, "recall")), rng3), C(lambda J, nm=nm: mm(var_vals(J, nm, "reduction")), rng4),
                     C(lambda J, nm=nm: mm(var_vals(J, nm, "min_recall_per_transform")), rng3),
                     C(lambda J, nm=nm: sum(1 for d in DRAWS if variant(J, d, nm)["gate"]["pass"]), lambda k: f"{int(k)}/3")])
    table("variant", ["Method", "Recall (3 draws)", "Reduction", "Min per-transformation recall", "Cells passing the gate"], rows)

    H(3, "1.4 D3b Copydays strong attacks (314 true pairs, n = 5,000 per draw)")
    rows = []
    for label, m, nm in METHODS:
        rows.append([label] + [C(lambda J, nm=nm, d=d: d3b(J, d, nm)["recall"], f3) for d in DRAWS]
                    + [C(lambda J, nm=nm, d=d: d3b(J, d, nm)["reduction"], f4) for d in DRAWS]
                    + [C(lambda J, m=m: mres(J, m)["d3b"]["strict_pass"], verdict)])
    table("d3b", ["Method", "Recall d1", "Recall d2", "Recall d3", "Reduction d1", "Reduction d2", "Reduction d3", "D3b verdict"], rows)
    P("The Copydays crop and JPEG attack sets are missing from v1, so D3b covers only strong attacks (print-and-scan, blur, painting and similar); "
      "crop and JPEG robustness on real images rests on D3a.")

    H(3, "1.5 Secondary pool (extended pool, n = 5,000; NOT gated)")
    rows = []
    for label, m, nm in METHODS[1:5]:
        rows.append([label, C(lambda J, m=m: (mres(J, m)["secondary_pool_not_gated"]["recall"]["mean"], mres(J, m)["secondary_pool_not_gated"]["recall"]["min"]),
                              lambda t: f"{t[0]:.3f} (min {t[1]:.3f})"),
                     C(lambda J, m=m: mres(J, m)["secondary_pool_not_gated"]["reduction"]["mean"], f4),
                     C(lambda J, m=m: mres(J, m)["secondary_pool_not_gated"]["min_recall_per_transform"]["min"], f3)])
    table("secondary", ["Method", "Recall: mean (min)", "Reduction (mean)", "Min per-transformation recall (min over draws)"], rows)

    H(2, "2. The tau_B protocol, its margins, and the drift with n")
    P("tau_B is a dev-calibrated quantile: for each draw it is the smallest threshold at which dev candidate recall reaches the pre-set target of "
      "0.97, computed from the dev split's true-pair distances only. If dev and test true-pair distances are exchangeable draws from one "
      "distribution, test recall at tau_B should therefore sit near 0.97, with sampling variation. D3a is consistent with that: "
      f"M2 at tau_B has recall {v(lambda J: mm(lad_vals(J, 'M2_B_x1.0', 'recall')), rng3)} across the 12 ladder cells. D3b shows the assumption "
      f"breaking: the same thresholds give recall {v(lambda J: mm([d3b(J, d, 'M2_B_x1.0')['recall'] for d in DRAWS]), rng3)} on Copydays strong "
      "attacks, whose distance distribution is not that of the injected transformations. A threshold calibrated on one attack family should not be "
      "assumed to transfer to another.")
    table("tau_b", ["Draw", "tau_B (main pool)", "tau_B (extended pool)", "tau_al (aligned verifier, see 4)"],
          [[str(d), C(lambda J, d=d: R(J)["tau_b"][f"d{d}"], lambda x: f"{x:.6f}"), C(lambda J, d=d: R(J)["tau_b_extended_pool"][f"d{d}"], lambda x: f"{x:.6f}"),
            C(lambda J, d=d: R(J)["tau_al"][f"d{d}"]["tau_al"], lambda x: f"{x:.6f}")] for d in DRAWS])
    P("**Margins of M2 at tau_B against the D3a gate** (minimum over the 12 ladder cells unless stated):")
    table("margins", ["Quantity", "Minimum observed", "Gate", "Margin", "Set"], [
        ["Recall", C(lambda J: min(lad_vals(J, "M2_B_x1.0", "recall")), f5), "0.95", C(lambda J: min(lad_vals(J, "M2_B_x1.0", "recall")) - 0.95, f5), "12 ladder cells (gated)"],
        ["Min per-transformation recall", C(lambda J: min(lad_vals(J, "M2_B_x1.0", "min_recall_per_transform")), f5), "0.85",
         C(lambda J: min(lad_vals(J, "M2_B_x1.0", "min_recall_per_transform")) - 0.85, f5), "12 ladder cells (gated)"],
        ["Min per-transformation recall", C(lambda J: min(var_vals(J, "M2_B_x1.0", "min_recall_per_transform")), f5), "0.85",
         C(lambda J: min(var_vals(J, "M2_B_x1.0", "min_recall_per_transform")) - 0.85, f5), "3 variant cells (gated)"],
        ["Min per-transformation recall", C(lambda J: min(ext(J, d, "M2_B_x1.0")["min_recall_per_transform"] for d in DRAWS), f5), "0.85",
         C(lambda J: min(ext(J, d, "M2_B_x1.0")["min_recall_per_transform"] for d in DRAWS) - 0.85, f5), "secondary pool (NOT gated)"],
    ])
    P("The weakest transformation for M2 at tau_B is brightness. The post-hoc bootstrap (section 9b) shows that at n = 1,000 and 2,500 the per-draw "
      "confidence intervals for overall recall and for brightness recall extend below the gate lines, so the small-n passes are not statistically "
      "separated from the gates. No gate verdict is changed by this.")
    P("**Recall differences across n are not a scale effect.** tau_B is fixed per draw, and whether a pair is an M2 candidate depends only on the "
      "distance between its two images. The rungs are nested prefixes of the same families (n = 1,000 is the first 50 families, n = 10,000 the "
      "first 500). The difference in recall between rungs is therefore sampling variation among families, not an effect of pool size. "
      f"Mean M2 recall over draws is {v(lambda J: mres(J, 'M2_B')['by_size']['1000']['recall']['mean'], f3)}, "
      f"{v(lambda J: mres(J, 'M2_B')['by_size']['2500']['recall']['mean'], f3)}, "
      f"{v(lambda J: mres(J, 'M2_B')['by_size']['5000']['recall']['mean'], f3)} and "
      f"{v(lambda J: mres(J, 'M2_B')['by_size']['10000']['recall']['mean'], f3)} at n = 1,000, 2,500, 5,000 and 10,000; the bootstrap 95% intervals "
      f"of the mean ({v(lambda J: J['posthoc']['b']['M2_B']['1000']['mean_over_draws']['all']['ci95'], rng3)} at n = 1,000 and "
      f"{v(lambda J: J['posthoc']['b']['M2_B']['10000']['mean_over_draws']['all']['ci95'], rng3)} at n = 10,000) overlap. The rungs are nested, "
      "so these estimates are not independent of each other.")

    H(2, "3. Verifier results on real photographs (secondary; never gating)")
    P("The primary verifier is the frozen Phase 1c distance at its fixed tau (0.5805796715948317); the aligned verifier is configuration C, "
      "frozen in the design. Non-candidate pairs are never verified.")
    rows = []
    for label, m, nm in METHODS[1:5]:
        for n in (1000, 10000):
            rows.append([label if n == 1000 else "", f"{n:,}",
                         C(lambda J, m=m, n=n: mean([parts(J)[f"verify_d{d}"]["sizes"][str(n)]["methods"][m]["primary"]["verifier_cap"] for d in DRAWS]), f3),
                         C(lambda J, m=m, n=n: mean([parts(J)[f"verify_d{d}"]["sizes"][str(n)]["methods"][m]["primary"]["confirmed_recall"] for d in DRAWS]), f3)])
    table("primary_verifier", ["Method", "n", "Verifier cap (share of true pairs with r < tau; mean over draws)",
                               "Confirmed recall (candidate and verified; mean over draws)"], rows)
    P("False-accept rate: share of a seeded sample (up to 1,000, without replacement) of non-true candidate pairs at the largest rung that the "
      "verifier would accept.")
    rows = []
    for label, m, nm in METHODS[1:5]:
        for d in DRAWS:
            rows.append([label if d == 1 else "", str(d),
                         C(lambda J, m=m, d=d: parts(J)[f"verify_d{d}"]["false_accept_largest_rung"][m]["primary"]["sampled_non_true_candidate_pairs"], i0),
                         C(lambda J, m=m, d=d: parts(J)[f"verify_d{d}"]["false_accept_largest_rung"][m]["primary"]["false_accept_rate"], f3),
                         C(lambda J, m=m, d=d: parts(J)[f"verify_d{d}"]["false_accept_largest_rung"][m]["aligned"]["alignment_failure_rate"], f3),
                         C(lambda J, m=m, d=d: parts(J)[f"verify_d{d}"]["false_accept_largest_rung"][m]["aligned"]["false_accept_rate"], f3)])
    table("false_accept", ["Method", "Draw", "Sampled non-true candidate pairs", "Primary false-accept rate", "Aligned: alignment-failure rate", "Aligned false-accept rate"], rows)
    rows = []
    for d in DRAWS:
        rows.append([str(d),
                     C(lambda J, d=d: parts(J)[f"verify_d{d}"]["largest_rung_true_pairs"]["true_pairs"], i0),
                     C(lambda J, d=d: json.dumps(parts(J)[f"verify_d{d}"]["largest_rung_true_pairs"]["alignment_status_counts"]).replace("|", "/"), txt),
                     C(lambda J, d=d: parts(J)[f"verify_d{d}"]["sizes"]["10000"]["methods"]["M2_B"]["aligned"]["alignment_failure_rate"], f4),
                     C(lambda J, d=d: parts(J)[f"verify_d{d}"]["sizes"]["10000"]["methods"]["M2_B"]["aligned"]["aligned_cap"], f3),
                     C(lambda J, d=d: parts(J)[f"verify_d{d}"]["sizes"]["10000"]["methods"]["M2_B"]["aligned"]["confirmed_recall"], f3)])
    table("aligned_true", ["Draw", "True pairs (n = 10,000)", "Alignment status counts on all true pairs", "Alignment-failure rate on true pairs",
                           "Aligned cap at tau_al (M2 at n = 10,000)", "Aligned confirmed recall"], rows)
    P("On Copydays strong attacks the aligner fails far more often:")
    rows = []
    for d in DRAWS:
        rows.append([str(d), C(lambda J, d=d: json.dumps(parts(J)[f"d3b_d{d}"]["verifier"]["alignment_status_counts"]), txt),
                     C(lambda J, d=d: parts(J)[f"d3b_d{d}"]["verifier"]["methods"]["M2_B"]["aligned"]["alignment_failure_rate"], f3),
                     C(lambda J, d=d: parts(J)[f"d3b_d{d}"]["verifier"]["methods"]["M2_B"]["primary"]["verifier_cap"], f3),
                     C(lambda J, d=d: parts(J)[f"d3b_d{d}"]["verifier"]["methods"]["M2_B"]["primary"]["confirmed_recall"], f3)])
    table("d3b_verifier", ["Draw", "Alignment status counts on all 314 true pairs", "Aligned: alignment-failure rate on true pairs",
                           "Primary verifier cap (share of true pairs with r < tau)", "Primary confirmed recall (M2 at tau_B)"], rows)
    P(f"Exactness: the results file records {v(lambda J: _count_bitwise(J), i0)} parallel-versus-single-process assertions; "
      f"{v(lambda J: _count_bitwise(J, only_true=True), i0)} of them report bitwise equality (a mismatch would have stopped the run).")

    H(2, "4. The aligned verifier's threshold as measured (design decision D4, option a)")
    P("Protocol as pre-registered (Addendum A.3): tau_al(d) is the Otsu threshold over the aligned distances of all alignment-ok primary-M2 "
      "candidate pairs at n = 1,000 of draw d; alignment failures are excluded from the Otsu input and counted. This is a protocol result; "
      "no rule is changed.")
    rows = []
    for d in DRAWS:
        rows.append([str(d), C(lambda J, d=d: parts(J)[f"verify_d{d}"]["tau_al"]["tau_al"], lambda x: f"{x:.4f}"),
                     C(lambda J, d=d: parts(J)[f"verify_d{d}"]["tau_al"]["candidate_pairs"], i0),
                     C(lambda J, d=d: parts(J)[f"verify_d{d}"]["tau_al"]["alignment_ok"], i0),
                     C(lambda J, d=d: json.dumps(parts(J)[f"verify_d{d}"]["tau_al"]["status_counts"]), txt),
                     C(lambda J, d=d: parts(J)[f"verify_d{d}"]["tau_al"]["auc"]["auc_ok_pairs_only"], lambda x: "undefined (NaN)" if isnan(x) else f"{x:.4f}"),
                     C(lambda J, d=d: parts(J)[f"verify_d{d}"]["tau_al"]["auc"]["auc_failures_as_inf"], f4),
                     C(lambda J, d=d: parts(J)[f"verify_d{d}"]["tau_al"]["auc"]["non_true_ok"], i0)])
    table("d4", ["Draw", "tau_al", "Candidate pairs", "Alignment ok", "Status counts", "AUC among aligned pairs", "AUC with failures counted as worst",
                 "Non-true candidate pairs that aligned"], rows)
    P("Reading: tau_al is unstable across draws (the middle draw gives a threshold two orders of magnitude smaller). The AUC among aligned pairs is "
      "undefined because no non-true candidate pair aligned at all. The AUC with failures counted as worst is near 1 mostly because it separates "
      "pairs that align from pairs that do not, not because the distance separates them. The Otsu input therefore contains only true pairs (no non-true "
      "candidate pair aligned), which is the candidate-mix bias the design anticipated; as a result the aligned cap on true pairs "
      "(section 3) is low and varies by draw although alignment succeeds on almost every true pair.")

    H(2, "5. End-to-end clustering at n = 1,000")
    P("Setup (section 8 of the design): full graph from the frozen Phase 1c verifier at its fixed tau (sign +1 iff r < tau, weight |tau − r|); "
      "candidate graph = +1 only for pairs in the primary M2 candidate set with r < tau; KwikCluster with 20 frozen seeds and union-find on both. "
      "**D(C) is scored on the full Phase 1c graph.** That graph is dense in positive edges and almost none of them are true "
      "(table below), so D(C) is a disagreement count with respect to that graph, not a measure of quality against ground truth. The ideal cluster "
      "count is 700 (650 singletons plus 50 families).")
    rows = []
    for d in DRAWS:
        rows.append([str(d), C(lambda J, d=d: R(J)["end_to_end"][f"d{d}"]["clustering"]["positive_edges_full"], i0),
                     C(lambda J, d=d: R(J)["end_to_end"][f"d{d}"]["clustering"]["positive_edges_full"] / 499500, pct1),
                     C(lambda J, d=d: R(J)["end_to_end"][f"d{d}"]["positive_edges_among_true_pairs"], i0),
                     C(lambda J, d=d: R(J)["end_to_end"][f"d{d}"]["positive_edges_among_true_pairs"] / R(J)["end_to_end"][f"d{d}"]["clustering"]["positive_edges_full"], pct2),
                     C(lambda J, d=d: R(J)["end_to_end"][f"d{d}"]["true_pairs"], i0),
                     C(lambda J, d=d: R(J)["end_to_end"][f"d{d}"]["clustering"]["positive_edges_candidate_graph"], i0),
                     C(lambda J, d=d: R(J)["end_to_end"][f"d{d}"]["primary_auc_all_pairs"], f3),
                     C(lambda J, d=d: R(J)["end_to_end"][f"d{d}"]["primary_auc_on_candidate_pairs"]["auc"], f3)])
    table("e2e_graph", ["Draw", "Positive edges, full graph", "Share of all 499,500 pairs", "of which true pairs", "Share of positive edges that are true",
                        "True pairs in total", "Positive edges, candidate graph", "Primary AUC, all pairs", "Primary AUC, candidate pairs"], rows)
    rows = []
    for d in DRAWS:
        cl = lambda J, d=d: R(J)["end_to_end"][f"d{d}"]["clustering"]
        rows.append([str(d),
                     C(lambda J, d=d: (cl(J, d)["kwik_full_graph"]["D_mean"], cl(J, d)["kwik_full_graph"]["D_sd"]), pm0),
                     C(lambda J, d=d: cl(J, d)["kwik_full_graph"]["clusters_mean"], lambda x: f"{x:.1f}"),
                     C(lambda J, d=d: (cl(J, d)["kwik_candidate_graph"]["D_mean"], cl(J, d)["kwik_candidate_graph"]["D_sd"]), pm0),
                     C(lambda J, d=d: cl(J, d)["kwik_candidate_graph"]["clusters_mean"], lambda x: f"{x:.1f}"),
                     C(lambda J, d=d: cl(J, d)["union_find_full_graph"]["D"], i0), C(lambda J, d=d: cl(J, d)["union_find_full_graph"]["clusters"], i0),
                     C(lambda J, d=d: cl(J, d)["union_find_candidate_graph"]["D"], i0), C(lambda J, d=d: cl(J, d)["union_find_candidate_graph"]["clusters"], i0)])
    table("e2e_D", ["Draw", "Kwik, full graph: D (mean ± sd, 20 seeds)", "clusters", "Kwik, candidate graph: D", "clusters",
                    "Union-find, full graph: D", "clusters", "Union-find, candidate graph: D", "clusters"], rows)
    P("Observation over three draws at n = 1,000, not a general claim: on the full Phase 1c graph, union-find merges everything into one cluster; "
      "KwikCluster returns a few hundred clusters with large seed-to-seed spread; on the candidate graph KwikCluster is nearly seed-independent. "
      "Because D(C) is measured against the full graph, a lower D on the candidate graph says that the candidate graph disagrees with the full graph "
      "less, which here mostly reflects the full graph's many false positive edges. It is not a statement that the candidate graph is closer to "
      "ground truth; the post-hoc section 9a scores the candidate-graph clusterings against ground truth directly.")

    H(2, "6. Cost: timing, projections and extrapolations")
    P("Timings come from a separate single-process run (BLAS pinned to one thread) on a laptop; they are machine-specific and noisy, and are not "
      "part of the byte-comparable results. **Everything below marked projection or extrapolation is not a measurement of a full verification.**")
    cells = lambda J: J["timing"]["cells"]
    rows = [
        ["Primary verifier, seconds per pair (mean of a 200-pair sample of candidate pairs; range over the 12 draw × size cells)",
         C(lambda J: mm([c["primary_verifier_seconds_per_pair"]["candidate_sample_mean"] for c in cells(J).values()]), rng3)],
        ["Primary verifier, seconds per pair (mean of a 200-pair sample of random real pairs; range over cells)",
         C(lambda J: mm([c["primary_verifier_seconds_per_pair"]["random_real_pairs_mean"] for c in cells(J).values()]), rng3)],
        ["Aligned verifier, seconds per pair (mean over 100 candidate pairs; range over cells)",
         C(lambda J: mm([c["aligned_verifier_seconds"]["per_pair_mean"] for c in cells(J).values()]), rng3)],
        ["Cheap stage at n = 10,000, feature extraction including decode (seconds; range over draws)",
         C(lambda J: mm([cells(J)[f"d{d}_n10000"]["cheap_stage_seconds"]["feature_extraction_decode_included"] for d in DRAWS]), lambda t: f"{t[0]:.0f}–{t[1]:.0f}")],
        ["Cheap stage at n = 10,000, candidate generation for one operating point of each method (seconds; range over draws)",
         C(lambda J: mm([cells(J)[f"d{d}_n10000"]["cheap_stage_seconds"]["candidate_generation_one_operating_point_each"] for d in DRAWS]), lambda t: f"{t[0]:.0f}–{t[1]:.0f}")],
        ["PROJECTION: verifying all M2 candidate pairs at n = 10,000 with the primary verifier (hours; number of candidate pairs × candidate-sample mean; range over draws)",
         C(lambda J: mm([cells(J)[f"d{d}_n10000"]["projection_primary_verification_seconds"]["M2_B"] / 3600 for d in DRAWS]), rng2)],
        ["PROJECTION: the same for M3 (hours; range over draws)",
         C(lambda J: mm([cells(J)[f"d{d}_n10000"]["projection_primary_verification_seconds"]["M3"] / 3600 for d in DRAWS]), rng2)],
        ["EXTRAPOLATION: verifying all 49,995,000 pairs at n = 10,000 with the primary verifier (hours; all pairs × random-pair mean; range over draws)",
         C(lambda J: mm([cells(J)[f"d{d}_n10000"]["extrapolation_all_pairs_primary_seconds"] / 3600 for d in DRAWS]), lambda t: f"{t[0]:.0f}–{t[1]:.0f}")],
    ]
    table("timing", ["Quantity", "Value"], rows)
    P("The per-pair cost differs between cells by roughly a factor of two for the primary verifier and by more than an order of magnitude for the "
      "aligned verifier (whose cost depends on whether alignment succeeds), so any single hour figure should be read as an order of magnitude only.")

    H(2, "7. Audit")
    P("`experiments/audit_phase3.py` recomputes the candidate sets and scores with different code (lookup-table popcount instead of "
      "`np.bitwise_count`, scipy `cdist` instead of the Gram identity, plain index arithmetic instead of the Phase 2 modules), and checks the "
      "pre-registration rules. **The run's own computation (block-wise Gram identity, as pre-registered) is the official one;** the audit's route is an "
      "independent cross-check and never replaces the run's results. **Limitation:** the verifier metrics of the parallel runs (true-pair "
      "verification, false-accept samples, tau_al and the end-to-end graph) are not recomputed by the audit; they are covered only by the recorded "
      "bitwise exactness assertions (parallel versus single-process on seeded samples).")
    table("audit_attempts", ["Attempt", "File", "All checks passed", "Failed checks", "Note"], [
        ["1", "`results/phase3_audit_attempt1_A10_diskguard.json`", C(lambda J: J["audit1"]["all_passed"], lambda b: "yes" if b else "no"),
         C(lambda J: ", ".join(k.split(" ")[0] for k, c in J["audit1"]["checks"].items() if not c["pass"]), txt),
         C(lambda J: [c for k, c in J["audit1"]["checks"].items() if k.startswith("A10")][0]["stderr_tail"].strip().splitlines()[-1][:160].replace("|", "/"), txt)],
        ["2", "`results/phase3_audit.json`", C(lambda J: J["audit"]["all_passed"], lambda b: "yes" if b else "no"),
         C(lambda J: ", ".join(k.split(" ")[0] for k, c in J["audit"]["checks"].items() if not c["pass"]) or "none", txt),
         "A10 reproduced tau_B and the smallest-rung results of draw 1 byte for byte"],
    ])
    rows = []
    for idx in range(len(load()["audit"]["checks"])):
        rows.append([C(lambda J, idx=idx: list(J["audit"]["checks"].keys())[idx].replace("|", "/"), txt),
                     C(lambda J, idx=idx: list(J["audit"]["checks"].values())[idx]["pass"], verdict)])
    table("audit_checks", ["Check (attempt 2)", "Result"], rows)
    P(f"Cells audited: {v(lambda J: len(J['audit']['cells_audited']), i0)} (12 ladder cells, 3 low-prevalence, 3 D3b and 3 secondary-pool sets). "
      "Recomputed independently and equal to the run: the SHA-256 of every candidate set, recall, reduction, original/transform splits, "
      "per-transformation recall and gate verdicts. Pairs within 1e-9 of a threshold may fall on either side between the two distance routines; "
      "the audit accepts a difference only for such pairs. **Near-threshold pairs and flips per audited cell:**")
    rows = []
    for idx in range(len(load()["audit"]["cells_audited"])):
        tag = lambda J, idx=idx: J["audit"]["cells_audited"][idx]
        rows.append([C(tag, txt),
                     C(lambda J, idx=idx: J["audit"]["near_threshold_pairs_within_1e-9_per_cell"][tag(J, idx)].get("M2_B_x1.0", 0), i0),
                     C(lambda J, idx=idx: J["audit"]["near_threshold_pairs_within_1e-9_per_cell"][tag(J, idx)].get("M2_A1", 0), i0),
                     C(lambda J, idx=idx: sum(c for k, c in J["audit"]["near_threshold_pairs_within_1e-9_per_cell"][tag(J, idx)].items()
                                              if k not in ("M2_B_x1.0", "M2_A1")), i0),
                     C(lambda J, idx=idx: sum(J["audit"]["near_threshold_flips_per_cell"][tag(J, idx)].values()) if J["audit"]["near_threshold_flips_per_cell"][tag(J, idx)] else 0, i0)])
    table("audit_near", ["Cell", "Pairs within 1e-9 of the M2 tau_B threshold", "of the M2_A1 threshold", "of the other M2 multiples (smallest rung only)",
                         "Flips versus the run"], rows)

    H(2, "8. Protocol deviations and interruption log")
    P("No run code, parameter, threshold, seed or split was changed after the test split was opened; the commit `47f1b87` recorded in the launch "
      "manifest is the commit of every part. (The audit script later gained reporting only, and the post-hoc script was added; both are listed below.) "
      "The events in this table interrupted execution, not the method.")
    table("interruptions", ["#", "When", "Cause", "Parts kept", "Parts recomputed"], [
        ["1", C(lambda J: J["resume1"]["resumed_at_utc"], lambda s: f"run stopped about 16:17Z (21:47 IST) on 2026-10-14; resumed {s}"),
         "The run process stopped. The error log begins with a `^C` marker and orphaned workers (`OSError` in worker start-up), consistent with a console "
         "control signal delivered to the process group. **Cause not confirmed.** (`resume_1.json` records it as a console Ctrl-C; that is the "
         "interpretation at the time, not a confirmed cause.)",
         "tau_B (6 parts), draw 1 ladder and verifier metrics, draw 2 ladder", "draw 2 verifier metrics (no output existed)"],
        ["2", C(lambda J: J["resume2"]["resumed_at_utc"], lambda s: f"run stopped during the night of 2026-10-14/15 (IST); resumed {s}"),
         "Environmental: Windows Update (TrustedInstaller, MoUsoCoreWorker) forced reboots at 04:47–04:51 IST on 2026-10-15; the run process died "
         "during the draw-3 end-to-end graph. Workers logged `BrokenPipeError` as a consequence. No Python error and no assertion failed. "
         "Windows also logged Modern Standby entries (lid and idle) during the run.",
         C(lambda J: J["resume2"]["parts_kept"], txt), C(lambda J: J["resume2"]["parts_recomputed"], txt)],
        ["3", "audit attempt 1, 2026-10-15", "The C: free-space guard (minimum 1.5 GB) stopped the audit's own fresh-process rerun (check A10): the guard reported "
         "0.90 GB free on C: (see the attempt-1 file). C: had fallen during the audit although nothing of the run writes to C:; the cause of the drop is "
         "unknown. All other checks passed in attempt 1.",
         "all run results (unchanged)", "the whole audit was rerun unchanged as attempt 2 (same script, same commit), which passed all checks"],
    ])
    P("Other departures from the plan as originally written, all recorded in `docs/phase3_design.md` Addendum B or the commit history:")
    P("- The run executed from a verified copy of the repository on D: (943 files, SHA-256 identical to the C: tree), because C: had no free space; "
      "the start thresholds were C: ≥ 2 GB and D: ≥ 6 GB, with a runtime guard at 1.5 GB and 3 GB.\n"
      "- Before the test split was opened, a dev-split-only rehearsal and its audit exposed two issues in the audit script itself: a tolerance for pairs "
      "within 1e-9 of a threshold (different rounding between the two distance routines) and a rounding tolerance in the tau_B recomputation. Both "
      "were fixed in the audit script, not in the run code.\n"
      "- After the run, the audit script gained the per-cell near-threshold counts and two scope statements (commit `32e9447`); no check logic changed.\n"
      "- The first rehearsal launch was stopped and restarted to add persistence of every operating point at the smallest rung, "
      "before any test-split access.\n"
      "- Two files in the run record (`launch_manifest.json`, `resume_*.json`) were written with a UTF-8 byte-order mark; the generator reads them with "
      "`utf-8-sig`.")

    H(2, "9. Post-hoc analyses (NOT PRE-REGISTERED)")
    P("**These analyses were added after the pre-registered results had been reviewed. They cannot change any gate verdict, and none is used to "
      "re-evaluate one.** Script `experiments/posthoc_phase3.py` (committed before its first run, SHA-256 in section 0); output "
      "`results/phase3_posthoc.json`, which carries `NOT_PRE_REGISTERED: true`. All three analyses read the committed run's persisted "
      "candidate sets; the full-graph KwikCluster was not rerun.")
    H(3, "9a. Candidate-graph clustering against ground truth (n = 1,000)")
    P("Graph: +1 iff the pair is in the primary M2 candidate set and r < tau; all other pairs −1. Scores are pairwise precision, recall, F1 and ARI "
      "against **all** true pairs "
      f"({v(lambda J: J['posthoc']['a_and_c']['d1']['a_candidate_graph_primary_edges']['union_find']['true_pairs'], i0)} per draw, including true pairs "
      "that are not in the candidate set); nodes that are not candidates, and isolated nodes, count as singletons. KwikCluster: mean ± sd over the 20 "
      "frozen seeds. The ideal cluster count is 700. This is an observation over three draws at n = 1,000, not a general claim.")
    rows = []
    for d in DRAWS:
        ga = lambda J, d=d: J["posthoc"]["a_and_c"][f"d{d}"]["a_candidate_graph_primary_edges"]
        rows.append([str(d), "KwikCluster", C(lambda J, d=d: ga(J, d)["positive_edges"], i0)]
                    + [C(lambda J, d=d, key=key: (ga(J, d)["kwik_20_seeds"][key]["mean"], ga(J, d)["kwik_20_seeds"][key]["sd"]), pm3) for key in ("precision", "recall", "f1", "ari")]
                    + [C(lambda J, d=d: (ga(J, d)["kwik_20_seeds"]["clusters"]["mean"], ga(J, d)["kwik_20_seeds"]["clusters"]["sd"]), pm1),
                       C(lambda J, d=d: (ga(J, d)["kwik_20_seeds"]["giant_cluster"]["mean"], ga(J, d)["kwik_20_seeds"]["giant_cluster"]["sd"]), pm1)])
        rows.append(["", "Union-find", C(lambda J, d=d: ga(J, d)["positive_edges"], i0)]
                    + [C(lambda J, d=d, key=key: ga(J, d)["union_find"][key], f3) for key in ("precision", "recall", "f1", "ari")]
                    + [C(lambda J, d=d: ga(J, d)["union_find"]["clusters"], i0), C(lambda J, d=d: ga(J, d)["union_find"]["giant_cluster"], i0)])
    table("posthoc_a", ["Draw", "Method", "Positive edges", "Precision", "Recall", "F1", "ARI", "Clusters (ideal 700)", "Giant cluster"], rows)
    P("**Candidate-set recall is an edge-level ceiling only.** It bounds how many true pairs can be positive edges; pairwise clustering recall can "
      "exceed it through transitive closure, because two images in the same family can end in one cluster through a third image even when their "
      "own pair is not a candidate. The table shows this:")
    rows = []
    for d in DRAWS:
        rows.append([str(d),
                     C(lambda J, d=d: J["posthoc"]["a_and_c"][f"d{d}"]["true_pairs_in_candidates"] / J["posthoc"]["a_and_c"][f"d{d}"]["a_candidate_graph_primary_edges"]["union_find"]["true_pairs"], f3),
                     C(lambda J, d=d: J["posthoc"]["a_and_c"][f"d{d}"]["a_candidate_graph_primary_edges"]["union_find"]["recall"], f3),
                     C(lambda J, d=d: J["posthoc"]["a_and_c"][f"d{d}"]["a_candidate_graph_primary_edges"]["union_find"]["recall"]
                       > J["posthoc"]["a_and_c"][f"d{d}"]["true_pairs_in_candidates"] / J["posthoc"]["a_and_c"][f"d{d}"]["a_candidate_graph_primary_edges"]["union_find"]["true_pairs"],
                       lambda b: "yes" if b else "no")])
    table("posthoc_ceiling", ["Draw", "Edge-level ceiling: share of true pairs that are candidate pairs", "Union-find pairwise recall", "Pairwise recall exceeds the edge ceiling"], rows)
    P("Observation: union-find reaches high recall but with low precision (it joins distinct photographs through the Phase 1c verifier's false "
      "positives, giving giant clusters), while KwikCluster is more precise but with lower recall and a different cluster count; neither recovers the "
      "ideal 700 clusters. With three draws at one size this is descriptive only.")

    H(3, "9b. Family-level bootstrap confidence intervals for recall")
    P(f"Unit: the family (source photograph). {v(lambda J: J['posthoc']['bootstrap']['resamples'], i0)} resamples, seed "
      f"{v(lambda J: J['posthoc']['bootstrap']['seed'], txt)}, percentile 95% intervals, M2 at tau_B and M3 at their primary operating points. "
      "Point estimates equal the pre-registered values (asserted to 1e-12 by the script before reporting). **Resampling:** each draw is resampled "
      "independently (a separate set of family indices per draw) and the \"mean\" rows average the three per-resample recalls; the draws are not "
      "resampled jointly. The draws do overlap in families (different permutations of one pool): the number of source photographs shared "
      "between two draws, among the first F families, is shown below, so the independence assumption is slightly optimistic. The bootstrap holds tau_B fixed, "
      "so the intervals do not include the variance of the dev-split calibration.")
    from phase3.splits import verify_startup

    def overlaps(J):
        s = verify_startup()
        src = {d: s.test_sources(d) for d in DRAWS}
        out = {}
        for F in (50, 125, 250, 500):
            sets = {d: set(src[d][:F]) for d in DRAWS}
            out[F] = (len(sets[1] & sets[2]), len(sets[1] & sets[3]), len(sets[2] & sets[3]), len(sets[1] & sets[2] & sets[3]))
        return out

    OV: dict = {}

    def ov(J, F, idx):
        if "v" not in OV or OV.get("J") is not J:
            OV["v"], OV["J"] = overlaps(J), J
        return OV["v"][F][idx]

    rows = [[f"{n:,}", str(F)] + [C(lambda J, F=F, i=i: ov(J, F, i), i0) for i in range(4)] for n, F in zip(SIZES, (50, 125, 250, 500))]
    table("overlap", ["n", "Families per draw (F)", "Shared by draws 1 and 2", "1 and 3", "2 and 3", "Shared by all three"], rows)
    for mname, mlabel in (("M2_B", "M2 at tau_B"), ("M3", "M3")):
        P(f"**{mlabel}:** recall with 95% CI (point estimate, then the interval); \"mean\" rows are the mean over the three draws.")
        rows = []
        cols = ["all"] + list(TRANSFORMS)
        for n in SIZES:
            for d in DRAWS:
                rows.append([f"{n:,}" if d == 1 else "", str(d)] + [C(lambda J, mname=mname, n=n, d=d, c=c: (
                    J["posthoc"]["b"][mname][str(n)]["per_draw"][f"d{d}"][c]["point"], *J["posthoc"]["b"][mname][str(n)]["per_draw"][f"d{d}"][c]["ci95"]), ci3) for c in cols])
            rows.append(["", "mean"] + [C(lambda J, mname=mname, n=n, c=c: (
                J["posthoc"]["b"][mname][str(n)]["mean_over_draws"][c]["point"], *J["posthoc"]["b"][mname][str(n)]["mean_over_draws"][c]["ci95"]), ci3, bold=True) for c in cols])
        table(f"boot_{mname}", ["n", "Draw", "Overall"] + [TRANSFORM_LABEL[t] for t in TRANSFORMS], rows)

    H(3, "9c. EXPLORATORY: alignment success as an edge rule (injected transformations only)")
    P("Rule: a candidate pair is a positive edge iff the aligner reports status `ok`, using the frozen configuration C exactly as in the run "
      "(`AlignConfig()` defaults, no parameter changed; the tau_al distance threshold is not used). Same clustering metrics as 9a, same true-pair "
      "denominator. **This is exploratory, measured on injected exact geometric and photometric transformations only, and motivates follow-up. "
      "No gate or verdict is re-evaluated with it.** On Copydays strong attacks the aligner failed on about a third of pairs (section 3), "
      "so these numbers say nothing about native attacks.")
    rows = []
    for d in DRAWS:
        gc = lambda J, d=d: J["posthoc"]["a_and_c"][f"d{d}"]["c_exploratory_alignment_success_edges"]
        rows.append([str(d), "KwikCluster", C(lambda J, d=d: gc(J, d)["aligned_ok"], i0), C(lambda J, d=d: gc(J, d)["aligned_ok_non_true"], i0)]
                    + [C(lambda J, d=d, key=key: (gc(J, d)["kwik_20_seeds"][key]["mean"], gc(J, d)["kwik_20_seeds"][key]["sd"]), pm3) for key in ("precision", "recall", "f1", "ari")]
                    + [C(lambda J, d=d: (gc(J, d)["kwik_20_seeds"]["clusters"]["mean"], gc(J, d)["kwik_20_seeds"]["clusters"]["sd"]), pm1),
                       C(lambda J, d=d: (gc(J, d)["kwik_20_seeds"]["giant_cluster"]["mean"], gc(J, d)["kwik_20_seeds"]["giant_cluster"]["sd"]), pm1)])
        rows.append(["", "Union-find", "", ""] + [C(lambda J, d=d, key=key: gc(J, d)["union_find"][key], f3) for key in ("precision", "recall", "f1", "ari")]
                    + [C(lambda J, d=d: gc(J, d)["union_find"]["clusters"], i0), C(lambda J, d=d: gc(J, d)["union_find"]["giant_cluster"], i0)])
    table("posthoc_c", ["Draw", "Method", "Aligned-ok candidate pairs (positive edges)", "of which non-true", "Precision", "Recall", "F1", "ARI",
                        "Clusters (ideal 700)", "Giant cluster"], rows)
    P("The family size is 7, so a giant cluster of 7 is a complete family. The perfect precision in this table only restates that no non-true "
      "candidate pair aligned (section 4); it is not evidence that the rule would separate near-duplicates from look-alikes in general.")

    H(2, "10. Limitations")
    P("- Two datasets only: an injected-transformation set built from one Flickr-derived pool of unknown per-image licences, and Copydays v1 (strong "
      "attacks only; the crop and JPEG sets are missing). D3b is a weak sanity check.\n"
      "- The injected transformations are the six used in earlier phases and are exact geometric or photometric operations; family prevalence "
      "(35%, and 10% in the variant) is far above realistic prevalence.\n"
      "- tau_B is calibrated on injected transformations of a dev split; the failure on Copydays shows it does not transfer to a different attack "
      "family. The 0.97 target is a pre-set design constant.\n"
      "- M2's pass of the per-transformation gate at small n is not statistically separated from the gate line (brightness; section 2 and 9b); "
      "the nested rungs are not independent.\n"
      "- The Phase 1c verifier is weak on real photographs (it accepts only about "
      f"{v(lambda J: mean([parts(J)[f'verify_d{d}']['sizes']['10000']['methods']['M2_B']['primary']['verifier_cap'] for d in DRAWS]), f2)} of true pairs, and "
      f"falsely accepts about {v(lambda J: mean([parts(J)[f'verify_d{d}']['false_accept_largest_rung']['M2_B']['primary']['false_accept_rate'] for d in DRAWS]), f2)} "
      "of sampled non-true M2 candidate pairs); end-to-end "
      "results are conditional on it, and D(C) against the full graph is not a quality-versus-truth measure (section 5).\n"
      "- The aligned verifier's threshold protocol (D4a) is unstable, its AUC is not informative, and its results on native attacks show many "
      "alignment failures; it never gated.\n"
      "- Timing is from one laptop, single process, with noisy per-pair costs; projections and extrapolations are order-of-magnitude only.\n"
      "- The audit does not recompute the parallel verifier metrics (section 7); the bootstrap holds tau_B fixed and treats the three draws as "
      "independent although they share some families; the post-hoc analyses are exploratory.\n"
      "- Nothing here supports a claim of generality or novelty.")

    H(2, "11. Reproduction")
    P("`python experiments/make_phase3_report.py` regenerates this file from the committed JSON and re-checks it; `--check` only re-checks. "
      "The run, audit, timing and post-hoc steps are `experiments/run_phase3.py`, `audit_phase3.py`, `run_phase3_timing.py` and `posthoc_phase3.py`.")


def _count_bitwise(J, only_true=False) -> int:
    n = 0

    def walk(o):
        nonlocal n
        if isinstance(o, dict):
            if "bitwise_equal" in o and (o["bitwise_equal"] or not only_true):
                n += 1
            for x in o.values():
                walk(x)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(J["results"]["parts"])
    return n


# ----------------------------------------------------------------------------- check
def parse_table(md: str, tid: str) -> list[list[str]]:
    m = re.search(rf"<!-- table:{re.escape(tid)} -->\n(.*?)\n<!-- /table:{re.escape(tid)} -->", md, re.S)
    if not m:
        raise AssertionError(f"table {tid} missing from the report")
    lines = m.group(1).splitlines()
    return [[c.strip() for c in ln.strip()[1:-1].split("|")] for ln in lines[2:]]


def check(md: str) -> tuple[int, int]:
    J2 = load()  # fresh load from disk, independent of the dictionary used to write the text
    n_cells = 0
    for tid, (header, rows) in TABLES.items():
        got = parse_table(md, tid)
        assert len(got) == len(rows), f"{tid}: {len(got)} rows in the report, {len(rows)} expected"
        for ri, (row, parsed) in enumerate(zip(rows, got)):
            assert len(parsed) == len(row), f"{tid} row {ri}: column count"
            for ci, (cell, text) in enumerate(zip(row, parsed)):
                expected = cell.render(J2) if isinstance(cell, C) else str(cell)
                assert text == expected.strip(), f"{tid} row {ri} col {ci}: report has {text!r}, JSON gives {expected!r}"
                n_cells += isinstance(cell, C)
    for c in INLINES:
        s = c.render(J2)
        assert s in md, f"inline number {s!r} not found in the report"
    return n_cells, len(INLINES)


def main() -> None:
    global J0
    J0 = load()
    if "--check" in sys.argv:
        build()  # rebuild tables/inlines in memory only
        md = OUT.read_text(encoding="utf-8")
        regenerated = "\n".join(LINES).rstrip() + "\n"
        assert md == regenerated, "the committed report differs from a fresh generation"
    else:
        build()
        md = "\n".join(LINES).rstrip() + "\n"
        OUT.write_text(md, encoding="utf-8", newline="\n")
        print("wrote", OUT)
    n_cells, n_inline = check(md)
    print(f"CHECK OK: {n_cells} table cells and {n_inline} inline numbers equal the JSON values; {len(TABLES)} tables")


if __name__ == "__main__":
    main()
