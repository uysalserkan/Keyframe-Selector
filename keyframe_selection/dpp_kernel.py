"""
DPP (Determinantal Point Process) kernel construction.

Builds similarity kernels that encode both semantic similarity
and temporal proximity for diversity-aware subset selection.
"""

import logging
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from .config import DPPKernelConfig
from .types import DPPKernel, EmbeddingBatch

logger = logging.getLogger(__name__)

# Optional GPU support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class DPPKernelBuilder:
    """
    Build DPP kernel matrices for subset selection.
    
    Constructs L-ensemble kernels combining:
        - Feature similarity: K_f(i,j) = exp(−||f_i−f_j||² / σf)
        - Temporal similarity: K_t(i,j) = exp(−|t_i−t_j|² / σt)
    
    Combined via Hadamard product: K = K_f ⊙ K_t
    """
    
    def __init__(self, config: Optional[DPPKernelConfig] = None):
        """
        Initialize kernel builder.
        
        Args:
            config: Kernel configuration. Uses defaults if None.
        """
        self.config = config or DPPKernelConfig()
    
    def build(
        self,
        embedding_batch: EmbeddingBatch,
        video_duration: Optional[float] = None,
        use_temporal: bool = True,
    ) -> DPPKernel:
        """
        Build DPP kernel from embeddings.
        
        Args:
            embedding_batch: Batch of frame embeddings.
            video_duration: Video duration for σt calculation.
            use_temporal: Whether to include temporal kernel.
        
        Returns:
            DPPKernel with combined similarity matrix.
        """
        embeddings = embedding_batch.effective_embeddings
        timestamps = embedding_batch.timestamps
        
        if len(embeddings) == 0:
            return DPPKernel(
                kernel=np.array([[]], dtype=np.float64),
                sigma_f=0.0,
                sigma_t=0.0,
            )
        
        # Determine bandwidth parameters
        sigma_f = self._compute_sigma_f(embeddings)
        
        if video_duration is not None and video_duration > 0:
            sigma_t = self.config.sigma_t_ratio * video_duration
        else:
            # Use timestamp range
            t_range = timestamps.max() - timestamps.min() if len(timestamps) > 1 else 1.0
            sigma_t = self.config.sigma_t_ratio * max(t_range, 1.0)
        
        # Build feature kernel
        feature_kernel = self._build_feature_kernel(embeddings, sigma_f)
        
        # Build temporal kernel if requested
        temporal_kernel = None
        if use_temporal and len(timestamps) > 0:
            temporal_kernel = self._build_temporal_kernel(timestamps, sigma_t)
        
        # Combine kernels
        if temporal_kernel is not None:
            if self.config.combine_method == "hadamard":
                kernel = feature_kernel * temporal_kernel
            else:  # additive
                kernel = 0.5 * (feature_kernel + temporal_kernel)
        else:
            kernel = feature_kernel
        
        # Ensure positive semi-definiteness
        kernel = self._ensure_psd(kernel)
        
        logger.info(
            f"DPP kernel built: shape={kernel.shape}, "
            f"σf={sigma_f:.4f}, σt={sigma_t:.4f}"
        )
        
        return DPPKernel(
            kernel=kernel,
            feature_kernel=feature_kernel,
            temporal_kernel=temporal_kernel,
            sigma_f=sigma_f,
            sigma_t=sigma_t,
        )
    
    def build_from_arrays(
        self,
        embeddings: NDArray[np.float32],
        timestamps: Optional[NDArray[np.float64]] = None,
        video_duration: Optional[float] = None,
        use_temporal: bool = True,
    ) -> DPPKernel:
        """
        Build DPP kernel directly from arrays.
        
        Args:
            embeddings: Embedding array of shape (N, D).
            timestamps: Optional timestamp array of shape (N,).
            video_duration: Video duration for σt calculation.
            use_temporal: Whether to include temporal kernel.
        
        Returns:
            DPPKernel with combined similarity matrix.
        """
        if len(embeddings) == 0:
            return DPPKernel(
                kernel=np.array([[]], dtype=np.float64),
                sigma_f=0.0,
                sigma_t=0.0,
            )
        
        sigma_f = self._compute_sigma_f(embeddings)
        
        if timestamps is not None and len(timestamps) > 0:
            t_range = timestamps.max() - timestamps.min() if len(timestamps) > 1 else 1.0
            sigma_t = self.config.sigma_t_ratio * max(t_range, video_duration or t_range, 1.0)
        else:
            sigma_t = 1.0
            timestamps = np.arange(len(embeddings), dtype=np.float64)
        
        feature_kernel = self._build_feature_kernel(embeddings, sigma_f)
        
        temporal_kernel = None
        if use_temporal:
            temporal_kernel = self._build_temporal_kernel(timestamps, sigma_t)
        
        if temporal_kernel is not None:
            if self.config.combine_method == "hadamard":
                kernel = feature_kernel * temporal_kernel
            else:
                kernel = 0.5 * (feature_kernel + temporal_kernel)
        else:
            kernel = feature_kernel
        
        kernel = self._ensure_psd(kernel)
        
        return DPPKernel(
            kernel=kernel,
            feature_kernel=feature_kernel,
            temporal_kernel=temporal_kernel,
            sigma_f=sigma_f,
            sigma_t=sigma_t,
        )
    
    def _compute_sigma_f(
        self,
        embeddings: NDArray[np.float32],
    ) -> float:
        """
        Compute feature kernel bandwidth.
        
        Uses median heuristic if not specified in config.
        
        Args:
            embeddings: Embedding array.
        
        Returns:
            σf value.
        """
        if self.config.sigma_f is not None:
            return self.config.sigma_f
        
        # Median heuristic: σf = median of pairwise distances
        if len(embeddings) > 1:
            # For efficiency, subsample if too many points
            if len(embeddings) > 500:
                indices = np.random.choice(len(embeddings), 500, replace=False)
                sample = embeddings[indices]
            else:
                sample = embeddings
            
            distances = pdist(sample, metric='euclidean')
            sigma_f = float(np.median(distances))
            sigma_f = max(sigma_f, 1e-6)  # Avoid zero
        else:
            sigma_f = 1.0
        
        return sigma_f
    
    def _build_feature_kernel(
        self,
        embeddings: NDArray[np.float32],
        sigma_f: float,
    ) -> NDArray[np.float64]:
        """
        Build RBF feature similarity kernel.
        
        K_f(i,j) = exp(−||f_i−f_j||² / σf²)
        
        Args:
            embeddings: Embedding array of shape (N, D).
            sigma_f: Kernel bandwidth.
        
        Returns:
            Kernel matrix of shape (N, N).
        """
        if self.config.use_gpu and HAS_TORCH and torch.cuda.is_available():
            return self._build_feature_kernel_gpu(embeddings, sigma_f)
        
        # CPU implementation
        sq_distances = squareform(pdist(embeddings, metric='sqeuclidean'))
        kernel = np.exp(-sq_distances / (sigma_f ** 2))
        
        return kernel.astype(np.float64)
    
    def _build_feature_kernel_gpu(
        self,
        embeddings: NDArray[np.float32],
        sigma_f: float,
    ) -> NDArray[np.float64]:
        """GPU-accelerated feature kernel computation."""
        device = torch.device('cuda')
        X = torch.from_numpy(embeddings).float().to(device)
        
        # Compute squared distances: ||x-y||² = ||x||² + ||y||² - 2<x,y>
        sq_norms = (X ** 2).sum(dim=1, keepdim=True)
        sq_distances = sq_norms + sq_norms.T - 2 * X @ X.T
        sq_distances = torch.clamp(sq_distances, min=0)  # Numerical stability
        
        kernel = torch.exp(-sq_distances / (sigma_f ** 2))
        
        return kernel.cpu().numpy().astype(np.float64)
    
    def _build_temporal_kernel(
        self,
        timestamps: NDArray[np.float64],
        sigma_t: float,
    ) -> NDArray[np.float64]:
        """
        Build RBF temporal similarity kernel.
        
        K_t(i,j) = exp(−|t_i−t_j|² / σt²)
        
        Args:
            timestamps: Timestamp array of shape (N,).
            sigma_t: Kernel bandwidth.
        
        Returns:
            Kernel matrix of shape (N, N).
        """
        t = timestamps.reshape(-1, 1)
        sq_distances = (t - t.T) ** 2
        kernel = np.exp(-sq_distances / (sigma_t ** 2))
        
        return kernel.astype(np.float64)
    
    def _ensure_psd(
        self,
        kernel: NDArray[np.float64],
        eps: float = 1e-6,
    ) -> NDArray[np.float64]:
        """
        Ensure kernel is positive semi-definite.
        
        Adds small value to diagonal if needed.
        
        Args:
            kernel: Kernel matrix.
            eps: Small value for numerical stability.
        
        Returns:
            PSD kernel matrix.
        """
        # Check if already PSD
        try:
            eigvals = np.linalg.eigvalsh(kernel)
            min_eigval = eigvals.min()
            
            if min_eigval < -eps:
                # Add to diagonal to make PSD
                kernel = kernel + (-min_eigval + eps) * np.eye(len(kernel))
                logger.debug(f"Adjusted kernel for PSD: min eigval was {min_eigval}")
        except np.linalg.LinAlgError:
            # Fallback: add small diagonal
            kernel = kernel + eps * np.eye(len(kernel))
        
        return kernel


def build_dpp_kernel(
    embeddings: NDArray[np.float32],
    timestamps: Optional[NDArray[np.float64]] = None,
    sigma_f: Optional[float] = None,
    sigma_t: Optional[float] = None,
    combine: str = "hadamard",
) -> NDArray[np.float64]:
    """
    Convenience function to build DPP kernel.
    
    Args:
        embeddings: Embedding array of shape (N, D).
        timestamps: Optional timestamp array.
        sigma_f: Feature kernel bandwidth (None = median heuristic).
        sigma_t: Temporal kernel bandwidth.
        combine: Combination method ('hadamard' or 'additive').
    
    Returns:
        Combined kernel matrix of shape (N, N).
    """
    config = DPPKernelConfig(
        sigma_f=sigma_f,
        combine_method=combine,
    )
    builder = DPPKernelBuilder(config)
    
    result = builder.build_from_arrays(
        embeddings,
        timestamps,
        use_temporal=timestamps is not None,
    )
    
    return result.kernel
