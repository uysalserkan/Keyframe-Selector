"""
Content density estimation via entropy analysis.

Uses Shannon entropy of embedding distribution to estimate information
density and adaptively determine optimal keyframe count.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from .config import EntropyEstimatorConfig
from .types import EmbeddingBatch, EntropyResult, TemporalAnalysisResult

logger = logging.getLogger(__name__)


class EntropyKEstimator:
    """
    Estimate optimal keyframe count based on content entropy.
    
    Uses PCA dimensionality reduction and histogram-based entropy
    estimation to determine video content density.
    
    The adaptive K formula:
        K = β · log(N) · H · mean(Δ)
    
    Where:
        - β: Scaling factor (H6)
        - N: Number of frames
        - H: Normalized Shannon entropy
        - mean(Δ): Average temporal change
    """
    
    def __init__(self, config: Optional[EntropyEstimatorConfig] = None):
        """
        Initialize entropy estimator.
        
        Args:
            config: Estimator configuration. Uses defaults if None.
        """
        self.config = config or EntropyEstimatorConfig()
        self._pca: Optional[PCA] = None
    
    def estimate(
        self,
        embedding_batch: EmbeddingBatch,
        temporal_result: Optional[TemporalAnalysisResult] = None,
    ) -> EntropyResult:
        """
        Estimate optimal keyframe count from embeddings.
        
        Args:
            embedding_batch: Batch of frame embeddings.
            temporal_result: Optional temporal analysis for mean delta.
        
        Returns:
            EntropyResult with entropy and recommended K.
        """
        embeddings = embedding_batch.effective_embeddings
        
        if len(embeddings) < 2:
            return EntropyResult(
                entropy=0.0,
                recommended_k=self.config.k_min,
            )
        
        # Reduce dimensionality with PCA
        reduced = self._reduce_dimensionality(embeddings)
        
        # Compute histogram and entropy
        entropy, histogram = self._compute_entropy(reduced)
        
        # Get mean delta if available
        mean_delta = 1.0
        if temporal_result is not None and len(temporal_result.deltas) > 0:
            mean_delta = temporal_result.mean_delta
            # Normalize mean_delta to reasonable range
            mean_delta = max(0.1, min(mean_delta, 2.0))
        
        # Compute recommended K
        n_frames = len(embeddings)
        k_raw = self.config.beta * np.log(n_frames) * entropy * mean_delta
        
        # Clamp to bounds
        recommended_k = int(np.clip(k_raw, self.config.k_min, self.config.k_max))
        
        # Also ensure K doesn't exceed number of frames
        recommended_k = min(recommended_k, n_frames)
        
        logger.info(
            f"Entropy estimation: H={entropy:.4f}, mean_Δ={mean_delta:.4f}, "
            f"raw_K={k_raw:.2f}, recommended_K={recommended_k}"
        )
        
        return EntropyResult(
            entropy=entropy,
            recommended_k=recommended_k,
            reduced_embeddings=reduced,
            histogram=histogram,
        )
    
    def estimate_from_embeddings(
        self,
        embeddings: NDArray[np.float32],
        mean_delta: float = 1.0,
    ) -> EntropyResult:
        """
        Estimate optimal K directly from embeddings array.
        
        Args:
            embeddings: Embedding array of shape (N, D).
            mean_delta: Average temporal delta (default 1.0).
        
        Returns:
            EntropyResult with entropy and recommended K.
        """
        if len(embeddings) < 2:
            return EntropyResult(
                entropy=0.0,
                recommended_k=self.config.k_min,
            )
        
        reduced = self._reduce_dimensionality(embeddings)
        entropy, histogram = self._compute_entropy(reduced)
        
        mean_delta = max(0.1, min(mean_delta, 2.0))
        
        n_frames = len(embeddings)
        k_raw = self.config.beta * np.log(n_frames) * entropy * mean_delta
        recommended_k = int(np.clip(k_raw, self.config.k_min, self.config.k_max))
        recommended_k = min(recommended_k, n_frames)
        
        return EntropyResult(
            entropy=entropy,
            recommended_k=recommended_k,
            reduced_embeddings=reduced,
            histogram=histogram,
        )
    
    def _reduce_dimensionality(
        self,
        embeddings: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Reduce embedding dimensionality with PCA.
        
        Args:
            embeddings: High-dimensional embeddings.
        
        Returns:
            Reduced embeddings.
        """
        n_samples, n_features = embeddings.shape
        n_components = min(self.config.pca_components, n_samples, n_features)
        
        if n_components < 1:
            return embeddings
        
        self._pca = PCA(n_components=n_components)
        reduced = self._pca.fit_transform(embeddings)
        
        logger.debug(
            f"PCA reduction: {n_features}D -> {n_components}D, "
            f"explained variance: {self._pca.explained_variance_ratio_.sum():.2%}"
        )
        
        return reduced.astype(np.float32)
    
    def _compute_entropy(
        self,
        embeddings: NDArray[np.float32],
    ) -> Tuple[float, NDArray[np.float64]]:
        """
        Compute Shannon entropy from embedding distribution.
        
        Uses multi-dimensional histogram binning.
        
        Args:
            embeddings: Reduced embeddings.
        
        Returns:
            Tuple of (normalized entropy, histogram).
        """
        n_samples, n_dims = embeddings.shape
        n_bins = self.config.num_bins
        eps = self.config.epsilon
        
        # For high-dimensional data, use 1D projection or first few dims
        if n_dims > 3:
            # Use first principal component for histogram
            data_1d = embeddings[:, 0]
            histogram, _ = np.histogram(data_1d, bins=n_bins, density=True)
        elif n_dims == 1:
            histogram, _ = np.histogram(embeddings.flatten(), bins=n_bins, density=True)
        else:
            # For 2-3D, use marginal histograms and average entropy
            entropies = []
            histograms = []
            for d in range(n_dims):
                hist, _ = np.histogram(embeddings[:, d], bins=n_bins, density=True)
                histograms.append(hist)
                
                # Normalize to probability
                hist = hist / (hist.sum() + eps)
                hist = hist[hist > eps]
                
                # Shannon entropy
                h = -np.sum(hist * np.log(hist + eps))
                entropies.append(h)
            
            histogram = np.mean(histograms, axis=0)
            entropy = np.mean(entropies)
            
            # Normalize by max entropy (uniform distribution)
            max_entropy = np.log(n_bins)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return float(normalized_entropy), histogram.astype(np.float64)
        
        # Single dimension case
        histogram = histogram / (histogram.sum() + eps)
        histogram = histogram[histogram > eps]
        
        # Shannon entropy
        entropy = -np.sum(histogram * np.log(histogram + eps))
        
        # Normalize by max entropy
        max_entropy = np.log(n_bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(normalized_entropy), histogram.astype(np.float64)


def compute_adaptive_k(
    n_frames: int,
    entropy: float,
    mean_delta: float = 1.0,
    beta: float = 1.0,
    k_min: int = 3,
    k_max: int = 50,
) -> int:
    """
    Compute adaptive keyframe count.
    
    K = β · log(N) · H · mean(Δ)
    
    Args:
        n_frames: Number of frames.
        entropy: Normalized entropy value.
        mean_delta: Average temporal change.
        beta: Scaling factor.
        k_min: Minimum K.
        k_max: Maximum K.
    
    Returns:
        Recommended number of keyframes.
    """
    if n_frames < 2:
        return k_min
    
    k_raw = beta * np.log(n_frames) * entropy * mean_delta
    k = int(np.clip(k_raw, k_min, k_max))
    k = min(k, n_frames)
    
    return k
