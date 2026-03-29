"""
Photometric and geometry-proxy metrics for keyframe evaluation.

Use these to compare pipelines (semantic vs reconstruction vs geometric_sfm):
- **Photometric:** PSNR / mean absolute error between frames (interpolation / video quality).
- **Geometry proxy:** summary statistics from pairwise geometry (mean inlier ratio), useful
  as a cheap COLMAP/SfM readiness indicator without running full SfM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class PhotometricMetrics:
    """Per-pair or aggregate photometric scores (higher PSNR = better)."""

    psnr_db: float
    mae_l1: float


@dataclass
class GeometryProxySummary:
    """Cheap SfM/COLMAP readiness proxy from pairwise two-view geometry."""

    mean_inlier_ratio: float
    min_inlier_ratio: float
    std_inlier_ratio: float


def mae_l1_uint8(img_a: NDArray[np.uint8], img_b: NDArray[np.uint8]) -> float:
    """Mean absolute error in [0, 255] scale."""
    if img_a.shape != img_b.shape:
        raise ValueError("Images must have the same shape for MAE")
    diff = np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))
    return float(np.mean(diff))


def psnr_uint8(img_a: NDArray[np.uint8], img_b: NDArray[np.uint8], max_val: float = 255.0) -> float:
    """PSNR in dB for uint8 images (identical images -> high value, ~inf for exact match)."""
    if img_a.shape != img_b.shape:
        raise ValueError("Images must have the same shape for PSNR")
    mse = float(np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10((max_val**2) / mse))


def photometric_metrics_pair(
    img_a: NDArray[np.uint8],
    img_b: NDArray[np.uint8],
) -> PhotometricMetrics:
    """MAE and PSNR between two aligned frames."""
    return PhotometricMetrics(
        psnr_db=psnr_uint8(img_a, img_b),
        mae_l1=mae_l1_uint8(img_a, img_b),
    )


def geometry_proxy_summary(consecutive_inlier_ratios: Optional[NDArray[np.floating]]) -> GeometryProxySummary:
    """
    Summarize consecutive pairwise inlier ratios (e.g. from fundamental matrix RANSAC).

    Higher mean/min ratios usually indicate more stable two-view geometry along the chain
    (informative for COLMAP initialization, not a replacement for bundle adjustment).
    """
    if consecutive_inlier_ratios is None or len(consecutive_inlier_ratios) == 0:
        return GeometryProxySummary(mean_inlier_ratio=0.0, min_inlier_ratio=0.0, std_inlier_ratio=0.0)
    s = np.asarray(consecutive_inlier_ratios, dtype=np.float64)
    return GeometryProxySummary(
        mean_inlier_ratio=float(np.mean(s)),
        min_inlier_ratio=float(np.min(s)),
        std_inlier_ratio=float(np.std(s)),
    )


def scanline_psnr_mae_reference(
    images: Sequence[NDArray[np.uint8]],
    reference_index: int = 0,
) -> Tuple[float, float]:
    """
    Aggregate PSNR/MAE vs a single reference frame (simple baseline for a clip).

    Returns mean PSNR (finite only) and mean MAE across all pairs (ref, i).
    """
    if not images:
        return float("nan"), float("nan")
    ref = images[reference_index]
    psnrs: list[float] = []
    maes: list[float] = []
    for i, im in enumerate(images):
        if i == reference_index:
            continue
        m = photometric_metrics_pair(ref, im)
        if np.isfinite(m.psnr_db):
            psnrs.append(m.psnr_db)
        maes.append(m.mae_l1)
    mean_psnr = float(np.mean(psnrs)) if psnrs else float("nan")
    mean_mae = float(np.mean(maes)) if maes else float("nan")
    return mean_psnr, mean_mae
