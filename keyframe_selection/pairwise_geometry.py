"""
Pairwise geometric cues between video frames (two-view geometry proxy).

Uses ORB features + fundamental matrix estimation (RANSAC) on consecutive pairs.
Inlier ratio per edge is a parallax / stability proxy suitable for keyframe spacing
without known intrinsics (F-matrix; E-matrix optional if K known later).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from .config import PairwiseGeometryConfig
from .types import FrameBatch, FrameData

logger = logging.getLogger(__name__)


def compute_consecutive_fundamental_scores(
    frame_batch: FrameBatch,
    config: Optional[PairwiseGeometryConfig] = None,
) -> NDArray[np.float64]:
    """
    For each consecutive pair (i, i+1), return RANSAC inlier ratio for estimated F.

    Shape: (N - 1,). Values in [0, 1]. Low values suggest weak / degenerate geometry.
    """
    cfg = config or PairwiseGeometryConfig()
    if not cfg.enabled:
        n = len(frame_batch)
        return np.ones(max(0, n - 1), dtype=np.float64)

    frames = frame_batch.frames
    if len(frames) < 2:
        return np.array([], dtype=np.float64)

    scores: List[float] = []
    orb = cv2.ORB_create(nfeatures=cfg.n_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(len(frames) - 1):
        s = _pair_inlier_ratio(
            frames[i],
            frames[i + 1],
            orb,
            bf,
            cfg,
        )
        scores.append(s)

    return np.asarray(scores, dtype=np.float64)


def _pair_inlier_ratio(
    a: FrameData,
    b: FrameData,
    orb: cv2.ORB,
    bf: cv2.BFMatcher,
    cfg: PairwiseGeometryConfig,
) -> float:
    gray_a = cv2.cvtColor(a.image, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b.image, cv2.COLOR_BGR2GRAY)
    if cfg.downscale != 1.0:
        h, w = gray_a.shape[:2]
        nw, nh = int(w * cfg.downscale), int(h * cfg.downscale)
        gray_a = cv2.resize(gray_a, (nw, nh))
        gray_b = cv2.resize(gray_b, (nw, nh))

    kp1, des1 = orb.detectAndCompute(gray_a, None)
    kp2, des2 = orb.detectAndCompute(gray_b, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return 0.0

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)
    n_keep = max(8, int(len(matches) * cfg.ratio_test))
    matches = matches[:n_keep]
    if len(matches) < 8:
        return 0.0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        cfg.ransac_threshold,
        cfg.ransac_confidence,
    )
    if F is None or mask is None:
        return 0.0
    inliers = int(mask.ravel().sum())
    return float(inliers) / float(len(matches))


def bottleneck_affinity_matrix(n: int, consecutive_scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Build symmetric affinity A[i,j] in [0,1] as bottleneck(min) of edge scores on path i..j.

    For i < j uses min(consecutive[i:j]); diagonal 1.
    """
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    if n == 1:
        return np.ones((1, 1), dtype=np.float64)
    s = np.asarray(consecutive_scores, dtype=np.float64)
    if len(s) != n - 1:
        raise ValueError(f"Expected {n - 1} consecutive scores, got {len(s)}")

    A = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            lo, hi = i, j
            seg = s[lo:hi]
            A[i, j] = float(np.min(seg)) if len(seg) else 1.0
            A[j, i] = A[i, j]
    return A


def geometric_rbf_kernel(
    affinity: NDArray[np.float64],
    sigma: Optional[float] = None,
) -> NDArray[np.float64]:
    """
    K_g(i,j) = exp( - (1 - A_ij)^2 / sigma^2 ) with PSD repair on diagonal.

    Affinity should be in [0,1] (higher = stronger geometric agreement).
    """
    A = np.clip(affinity.astype(np.float64), 0.0, 1.0)
    d = 1.0 - A
    if sigma is None or sigma <= 0:
        sigma = float(np.median(d[d > 1e-9])) if np.any(d > 1e-9) else 1.0
        sigma = max(sigma, 1e-6)
    K = np.exp(-(d**2) / (sigma**2))
    np.fill_diagonal(K, 1.0)
    return K


def compute_geometry_point_features(consecutive_scores: NDArray[np.float64], n: int) -> NDArray[np.float32]:
    """
    Per-frame geometric scalars for optional fusion with K-means (N, 3).

    Columns: incoming edge score, outgoing edge score, local mean of 3 edges around i.
    """
    s = np.asarray(consecutive_scores, dtype=np.float64)
    feats = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        inc = float(s[i - 1]) if i > 0 else 0.0
        out = float(s[i]) if i < len(s) else 0.0
        lo = max(0, i - 1)
        hi = min(len(s), i + 2)
        local = float(np.mean(s[lo:hi])) if hi > lo else 0.0
        feats[i, 0] = inc
        feats[i, 1] = out
        feats[i, 2] = local
    return feats
