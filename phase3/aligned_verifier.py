"""
Phase 3 aligned verifier (DRAFT; parameters are frozen in the Phase 3 design commit,
before any Phase 3 result). docs/phase3_dataset_plan.md section 8.5.

Reported ALONGSIDE the frozen Phase 1c distance, never gating. It is a new component and
must be described as such: it does not modify cc_dedup/.

    1. both images are resized to a working size (longest side = work_size)
    2. ORB features, brute-force Hamming matching with cross-check
    3. a 4-DoF similarity transform (scale, rotation, translation) by RANSAC
    4. image B is warped into image A's frame; only pixels where B is defined count
    5. distance = zlib(level 9) size of |A - warped B| over the valid pixels / number of
       compared bytes (the Phase 1c compression-ratio idea, restricted to the overlap)

Alignment failures are REPORTED as failures (status != "ok"); no distance is invented.
The image with the lower index is the reference A, so a pair has exactly one distance.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AlignConfig:
    # Defaults are configuration C, validated on synthetic Phase 2 families 1-40 only
    # (experiments/validate_aligned_verifier.py); frozen in docs/phase3_design.md.
    work_size: int = 512
    orb_nfeatures: int = 1000
    orb_fast_threshold: int = 5
    ransac_threshold: float = 3.0
    ransac_max_iters: int = 2000
    ransac_confidence: float = 0.99
    min_inliers: int = 12
    min_overlap_fraction: float = 0.25
    scale_range: tuple = (0.25, 4.0)
    seed: int = 12345


def prepare(rgb: np.ndarray, cfg: AlignConfig) -> dict:
    h, w = rgb.shape[:2]
    s = cfg.work_size / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=cfg.orb_nfeatures, fastThreshold=cfg.orb_fast_threshold)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    pts = np.float32([k.pt for k in keypoints]) if keypoints else np.zeros((0, 2), np.float32)
    return {"small": small, "gray": gray, "pts": pts, "des": descriptors}


def _fail(status: str, **extra) -> dict:
    return {"status": status, "distance": None, **extra}


def aligned_distance(a: dict, b: dict, cfg: AlignConfig = AlignConfig()) -> dict:
    """a, b: outputs of prepare(); a is the reference. Returns a dict with status
    ('ok' or a failure reason), distance, inliers and overlap fraction."""
    if a["des"] is None or b["des"] is None or len(a["pts"]) < cfg.min_inliers or len(b["pts"]) < cfg.min_inliers:
        return _fail("too_few_features")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(b["des"], a["des"])  # query = b, train = a
    if len(matches) < cfg.min_inliers:
        return _fail("too_few_matches", matches=len(matches))
    src = np.float32([b["pts"][m.queryIdx] for m in matches])
    dst = np.float32([a["pts"][m.trainIdx] for m in matches])

    cv2.setRNGSeed(cfg.seed)
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_threshold,
        maxIters=cfg.ransac_max_iters, confidence=cfg.ransac_confidence,
    )
    n_in = int(inliers.sum()) if inliers is not None else 0
    if M is None or n_in < cfg.min_inliers:
        return _fail("alignment_failed", matches=len(matches), inliers=n_in)
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    if not (cfg.scale_range[0] <= scale <= cfg.scale_range[1]):
        return _fail("implausible_scale", inliers=n_in, scale=scale)

    h, w = a["gray"].shape
    warped = cv2.warpAffine(b["small"], M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(np.ones(b["gray"].shape, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(bool)
    overlap = float(valid.mean())
    if overlap < cfg.min_overlap_fraction:
        return _fail("small_overlap", inliers=n_in, overlap=overlap)

    diff = np.abs(a["small"].astype(np.int16) - warped.astype(np.int16)).astype(np.uint8)[valid]
    distance = len(zlib.compress(diff.tobytes(), level=9)) / diff.size
    return {"status": "ok", "distance": distance, "inliers": n_in, "overlap": overlap, "scale": scale}
