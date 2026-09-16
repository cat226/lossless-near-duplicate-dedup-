"""
Phase 0 — cost-scaling analysis.

Tests whether the cost-adaptive premise

    U(s) = alpha * S(s) - beta * C(s)

produces a meaningful (i.e. non-degenerate, zero-crossing) trade-off
between storage saving S(s) and verification cost C(s) as a function
of image size s, across several beta/alpha ratios spanning multiple
orders of magnitude.

This script does not claim the model proves anything beyond what its
stated assumptions support. It only reports whether, under those
assumptions, U(s) crosses zero within a realistic image-size range,
and at which beta/alpha ratios.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_PLOT = __file__.replace("cost_scaling_analysis.py", "tradeoff_plot.png")

# Realistic image size range: 10 KB (thumbnail) to 20 MB (high-res photo).
SIZE_MIN_BYTES = 10 * 1024
SIZE_MAX_BYTES = 20 * 1024 * 1024

# Several orders of magnitude of beta/alpha, per the Phase 0 spec.
BETA_OVER_ALPHA_RATIOS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


def verification_cost_model(image_size_bytes: int) -> float:
    """Model verification cost as approximately linear in pixel count
    / image size.

    ASSUMPTION (explicit, not derived): verifying a candidate
    near-duplicate match requires decoding the image and comparing
    pixel (or feature) data, and both decode time and comparison time
    scale roughly linearly with the number of pixels, which in turn
    scales roughly linearly with encoded file size for images of
    similar content/format. This is a simplification — real decode
    cost has fixed overhead plus sub-linear terms for some codecs —
    but linear-in-size is the model the Phase 0 plan specifies for
    this trade-off check.

    Units are arbitrary "cost units"; only relative scaling across
    sizes and the beta/alpha ratio matter for the zero-crossing
    question, not the absolute scale.
    """
    return float(image_size_bytes)


def storage_saving_model(image_size_bytes: int) -> float:
    """Return the naive storage saving from deduplicating an image.

    ASSUMPTION: if a near-duplicate is successfully matched and
    replaced by a reference + delta, the naive best case saving is
    the full size of the image being deduplicated (i.e. delta cost is
    ignored here — this is the upper bound on saving, not the net
    saving after delta storage, which is what delta_probe.py
    measures separately).
    """
    return float(image_size_bytes)


def does_tradeoff_survive(sizes: list[int]) -> None:
    """Evaluate U(size) = alpha * S(size) - beta * C(size) across
    several beta/alpha ratios spanning multiple orders of magnitude.

    Since S(s) and C(s) are both modeled as linear in s here
    (S(s) = s, C(s) = s), U(s) = (alpha - beta) * s, which is either
    uniformly >= 0, uniformly <= 0, or identically 0 for all s —
    it cannot cross zero at a *specific* size under these two linear
    models alone. This is reported explicitly: the linear-cost /
    linear-saving formulation, as specified for Phase 0, does not by
    itself produce a size-dependent trade-off. A crossing would
    require at least one of S(s) or C(s) to be non-linear in s (e.g.
    fixed per-verification overhead in C(s), or sub-linear S(s) from
    compression). That refinement is out of scope for Phase 0 and is
    recorded as a limitation, not silently patched over.

    Saves phase0/tradeoff_plot.png and prints, for each beta/alpha
    ratio, whether U(s) is positive, negative, or (degenerately)
    exactly zero across the tested size range, and identifies any
    ratio where a genuine sign change occurs.
    """
    sizes_arr = np.array(sizes, dtype=np.float64)
    S = np.array([storage_saving_model(s) for s in sizes])
    C = np.array([verification_cost_model(s) for s in sizes])

    fig, ax = plt.subplots(figsize=(9, 6))

    any_crossing = False
    print("\nCost-scaling analysis: U(s) = alpha * S(s) - beta * C(s)")
    print(f"S(s) = s (storage_saving_model), C(s) = s (verification_cost_model)")
    print(f"Size range tested: {SIZE_MIN_BYTES:,} to {SIZE_MAX_BYTES:,} bytes\n")

    for ratio in BETA_OVER_ALPHA_RATIOS:
        alpha = 1.0
        beta = ratio * alpha
        U = alpha * S - beta * C

        sign_changes = np.where(np.diff(np.sign(U)) != 0)[0]
        crosses = len(sign_changes) > 0

        if crosses:
            any_crossing = True
            crossing_sizes = [(sizes_arr[i], sizes_arr[i + 1]) for i in sign_changes]
            status = f"CROSSES ZERO near size(s): {crossing_sizes}"
        elif np.all(U > 0):
            status = "U(s) > 0 for all tested sizes (saving always dominates cost)"
        elif np.all(U < 0):
            status = "U(s) < 0 for all tested sizes (cost always dominates saving)"
        else:
            status = "U(s) == 0 for all tested sizes (degenerate: alpha == beta with linear models)"

        print(f"beta/alpha = {ratio:>10.0e}: {status}")

        ax.plot(sizes_arr, U, label=f"beta/alpha={ratio:.0e}")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Image size s (bytes, log scale)")
    ax.set_ylabel("U(s) = alpha*S(s) - beta*C(s)")
    ax.set_title("Phase 0: cost-adaptive utility U(s) across beta/alpha ratios")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    print(f"\nSaved plot to {OUT_PLOT}")

    print()
    if any_crossing:
        print("RESULT: U(s) crosses zero within the realistic size range for at least one beta/alpha ratio.")
    else:
        print(
            "RESULT: Under the linear S(s)=s / C(s)=s models specified for Phase 0, U(s) does NOT "
            "cross zero at any particular image size for any tested beta/alpha ratio - it is a fixed "
            "sign (alpha - beta) times s. The sign of U depends only on whether beta/alpha is above or "
            "below 1, not on image size. This means the linear cost-adaptive premise, as modeled here, "
            "does not by itself justify a size-dependent policy; a real zero-crossing would require a "
            "non-linear term (e.g. fixed per-image verification overhead) in C(s) or S(s)."
        )


def main():
    n_points = 200
    sizes = np.unique(
        np.round(
            np.logspace(np.log10(SIZE_MIN_BYTES), np.log10(SIZE_MAX_BYTES), n_points)
        ).astype(int)
    ).tolist()
    does_tradeoff_survive(sizes)


if __name__ == "__main__":
    main()
