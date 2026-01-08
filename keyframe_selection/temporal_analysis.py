"""
Temporal change analysis module.

Computes temporal gradients (Δt) between adjacent embeddings to detect
significant changes and segment boundaries in videos.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import TemporalAnalysisConfig
from .types import EmbeddingBatch, TemporalAnalysisResult

logger = logging.getLogger(__name__)


class TemporalDeltaComputer:
    """
    Compute temporal changes between consecutive frame embeddings.
    
    Detects semantic transitions by measuring L2 distances between
    adjacent embeddings and applying percentile-based thresholding.
    """
    
    def __init__(self, config: Optional[TemporalAnalysisConfig] = None):
        """
        Initialize temporal delta computer.
        
        Args:
            config: Analysis configuration. Uses defaults if None.
        """
        self.config = config or TemporalAnalysisConfig()
    
    def compute(
        self,
        embedding_batch: EmbeddingBatch,
    ) -> TemporalAnalysisResult:
        """
        Compute temporal deltas and detect change points.
        
        Args:
            embedding_batch: Batch of frame embeddings.
        
        Returns:
            TemporalAnalysisResult with deltas and change points.
        """
        embeddings = embedding_batch.effective_embeddings
        
        if len(embeddings) < 2:
            return TemporalAnalysisResult(
                deltas=np.array([]),
                threshold=0.0,
                change_points=np.array([], dtype=np.int64),
            )
        
        # Compute L2 deltas between consecutive embeddings
        deltas = self._compute_deltas(embeddings)
        
        # Apply EMA smoothing if configured
        if self.config.use_ema_smoothing:
            deltas = self._apply_ema(deltas, self.config.ema_alpha)
        
        # Compute threshold using percentile
        threshold = np.percentile(deltas, self.config.delta_percentile)
        
        # Find change points above threshold
        change_points = self._find_change_points(deltas, threshold)
        
        # Compute soft segment probabilities
        segment_probs = self._compute_segment_probs(deltas)
        
        logger.info(
            f"Temporal analysis: {len(deltas)} deltas, "
            f"threshold (P{self.config.delta_percentile}): {threshold:.4f}, "
            f"change points: {len(change_points)}"
        )
        
        return TemporalAnalysisResult(
            deltas=deltas,
            threshold=threshold,
            change_points=change_points,
            segment_probs=segment_probs,
        )
    
    def compute_from_embeddings(
        self,
        embeddings: NDArray[np.float32],
    ) -> TemporalAnalysisResult:
        """
        Compute temporal deltas directly from embeddings array.
        
        Args:
            embeddings: Embedding array of shape (N, D).
        
        Returns:
            TemporalAnalysisResult with deltas and change points.
        """
        if len(embeddings) < 2:
            return TemporalAnalysisResult(
                deltas=np.array([]),
                threshold=0.0,
                change_points=np.array([], dtype=np.int64),
            )
        
        deltas = self._compute_deltas(embeddings)
        
        if self.config.use_ema_smoothing:
            deltas = self._apply_ema(deltas, self.config.ema_alpha)
        
        threshold = np.percentile(deltas, self.config.delta_percentile)
        change_points = self._find_change_points(deltas, threshold)
        segment_probs = self._compute_segment_probs(deltas)
        
        return TemporalAnalysisResult(
            deltas=deltas,
            threshold=threshold,
            change_points=change_points,
            segment_probs=segment_probs,
        )
    
    def _compute_deltas(
        self,
        embeddings: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """
        Compute L2 distances between consecutive embeddings.
        
        Δ_t = ||f̃_t − f̃_{t−1}||
        
        Args:
            embeddings: Embedding array of shape (N, D).
        
        Returns:
            Delta array of shape (N-1,).
        """
        # Compute differences
        diffs = embeddings[1:] - embeddings[:-1]
        
        # L2 norm of differences
        deltas = np.linalg.norm(diffs, axis=1)
        
        return deltas.astype(np.float64)
    
    def _apply_ema(
        self,
        deltas: NDArray[np.float64],
        alpha: float,
    ) -> NDArray[np.float64]:
        """
        Apply exponential moving average smoothing.
        
        Args:
            deltas: Raw delta values.
            alpha: EMA decay factor (higher = less smoothing).
        
        Returns:
            Smoothed delta values.
        """
        smoothed = np.zeros_like(deltas)
        smoothed[0] = deltas[0]
        
        for i in range(1, len(deltas)):
            smoothed[i] = alpha * deltas[i] + (1 - alpha) * smoothed[i - 1]
        
        return smoothed
    
    def _find_change_points(
        self,
        deltas: NDArray[np.float64],
        threshold: float,
    ) -> NDArray[np.int64]:
        """
        Find indices where delta exceeds threshold.
        
        Applies minimum segment gap constraint.
        
        Args:
            deltas: Delta values.
            threshold: Threshold for significant change.
        
        Returns:
            Array of change point indices.
        """
        # Find all points above threshold
        above_threshold = np.where(deltas > threshold)[0]
        
        if len(above_threshold) == 0:
            return np.array([], dtype=np.int64)
        
        # Apply minimum segment gap
        min_gap = self.config.min_segment_frames
        filtered = [above_threshold[0]]
        
        for idx in above_threshold[1:]:
            if idx - filtered[-1] >= min_gap:
                filtered.append(idx)
        
        # Shift by 1 because deltas[i] is the change FROM frame i TO frame i+1
        # So change point should be at frame i+1
        return np.array(filtered, dtype=np.int64) + 1
    
    def _compute_segment_probs(
        self,
        deltas: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute soft segment boundary probabilities.
        
        Uses sigmoid-normalized deltas.
        
        Args:
            deltas: Delta values.
        
        Returns:
            Probability array of shape (N-1,).
        """
        if len(deltas) == 0:
            return np.array([])
        
        # Normalize deltas to reasonable range
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas) + 1e-8
        
        normalized = (deltas - mean_delta) / std_delta
        
        # Sigmoid to convert to probabilities
        probs = 1 / (1 + np.exp(-normalized))
        
        return probs


def mark_change_points(
    deltas: NDArray[np.float64],
    percentile: float = 90.0,
    min_gap: int = 3,
) -> Tuple[NDArray[np.int64], float]:
    """
    Utility function to mark change points in a delta sequence.
    
    Args:
        deltas: Array of temporal deltas.
        percentile: Percentile threshold for significant changes.
        min_gap: Minimum gap between change points.
    
    Returns:
        Tuple of (change point indices, threshold value).
    """
    if len(deltas) == 0:
        return np.array([], dtype=np.int64), 0.0
    
    threshold = np.percentile(deltas, percentile)
    above_threshold = np.where(deltas > threshold)[0]
    
    if len(above_threshold) == 0:
        return np.array([], dtype=np.int64), threshold
    
    filtered = [above_threshold[0]]
    for idx in above_threshold[1:]:
        if idx - filtered[-1] >= min_gap:
            filtered.append(idx)
    
    return np.array(filtered, dtype=np.int64) + 1, threshold
