"""
Diversity-aware subset selection using Determinantal Point Processes.

Implements DPP sampling for selecting diverse, non-redundant keyframes.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import SelectorConfig
from .types import DPPKernel, EmbeddingBatch, KeyframeResult

logger = logging.getLogger(__name__)

# Lazy import dppy
_dpp_available = None


def _check_dppy():
    """Check if dppy is available."""
    global _dpp_available
    if _dpp_available is None:
        try:
            import dppy
            _dpp_available = True
        except ImportError:
            _dpp_available = False
    return _dpp_available


class DPPSelector:
    """
    Select diverse subsets using Determinantal Point Processes.
    
    Uses L-ensemble DPP for sampling subsets that maximize diversity
    while respecting the similarity structure in the kernel.
    """
    
    def __init__(self, config: Optional[SelectorConfig] = None):
        """
        Initialize DPP selector.
        
        Args:
            config: Selector configuration. Uses defaults if None.
        """
        self.config = config or SelectorConfig()
        np.random.seed(self.config.seed)
    
    def select(
        self,
        kernel: DPPKernel,
        k: Optional[int] = None,
        timestamps: Optional[NDArray[np.float64]] = None,
    ) -> KeyframeResult:
        """
        Select diverse subset using DPP.
        
        Args:
            kernel: DPP kernel matrix.
            k: Number of items to select. Uses fixed_k from config if None.
            timestamps: Optional timestamps for selected frames.
        
        Returns:
            KeyframeResult with selected indices.
        """
        if k is None:
            k = self.config.fixed_k
        
        if k is None:
            raise ValueError("k must be specified either in call or config.fixed_k")
        
        n = kernel.size
        k = min(k, n)
        
        if k <= 0:
            return KeyframeResult(
                indices=np.array([], dtype=np.int64),
                timestamps=np.array([], dtype=np.float64),
                metadata={"k": 0, "n": n},
            )
        
        # Select using DPP
        if _check_dppy():
            indices = self._select_dppy(kernel.kernel, k)
        else:
            logger.warning("dppy not available, falling back to greedy selection")
            indices = self._select_greedy(kernel.kernel, k)
        
        # Sort by time order
        indices = np.sort(indices)
        
        # Get timestamps if provided
        if timestamps is not None and len(timestamps) > 0:
            selected_timestamps = timestamps[indices]
        else:
            selected_timestamps = indices.astype(np.float64)
        
        logger.info(f"Selected {len(indices)} keyframes from {n} candidates")
        
        return KeyframeResult(
            indices=indices,
            timestamps=selected_timestamps,
            metadata={
                "k": len(indices),
                "n": n,
                "method": "dpp" if _check_dppy() else "greedy",
                "mode": self.config.mode,
            },
        )
    
    def select_from_embeddings(
        self,
        embedding_batch: EmbeddingBatch,
        kernel: DPPKernel,
        k: int,
    ) -> KeyframeResult:
        """
        Select keyframes with full metadata from embedding batch.
        
        Args:
            embedding_batch: Source embeddings.
            kernel: DPP kernel.
            k: Number of keyframes to select.
        
        Returns:
            KeyframeResult with indices and timestamps.
        """
        result = self.select(kernel, k, embedding_batch.timestamps)
        
        # Add frame indices mapping
        if len(embedding_batch.frame_indices) > 0:
            result.metadata["original_frame_indices"] = embedding_batch.frame_indices[result.indices].tolist()
        
        return result
    
    def _select_dppy(
        self,
        kernel: NDArray[np.float64],
        k: int,
    ) -> NDArray[np.int64]:
        """
        Select using dppy library.
        
        Args:
            kernel: L-ensemble kernel matrix.
            k: Number of items to select.
        
        Returns:
            Selected indices.
        """
        from dppy.finite_dpps import FiniteDPP
        
        # Create L-ensemble DPP
        dpp = FiniteDPP('likelihood', **{'L': kernel})
        
        if self.config.mode == "sample":
            # Stochastic sampling
            dpp.sample_exact_k_dpp(size=k)
            indices = np.array(dpp.list_of_samples[-1], dtype=np.int64)
        else:
            # MAP inference (greedy)
            indices = self._greedy_map(kernel, k)
        
        return indices
    
    def _select_greedy(
        self,
        kernel: NDArray[np.float64],
        k: int,
    ) -> NDArray[np.int64]:
        """
        Greedy selection fallback when dppy is unavailable.
        
        Uses greedy maximization of log-determinant.
        
        Args:
            kernel: Kernel matrix.
            k: Number of items to select.
        
        Returns:
            Selected indices.
        """
        return self._greedy_map(kernel, k)
    
    def _greedy_map(
        self,
        kernel: NDArray[np.float64],
        k: int,
    ) -> NDArray[np.int64]:
        """
        Greedy MAP inference for DPP.
        
        Iteratively selects items that maximize marginal gain in
        log-determinant of the selected subset's kernel.
        
        Args:
            kernel: Kernel matrix.
            k: Number of items to select.
        
        Returns:
            Selected indices.
        """
        n = len(kernel)
        selected: List[int] = []
        remaining = set(range(n))
        
        # Start with highest self-similarity (diagonal)
        first = int(np.argmax(np.diag(kernel)))
        selected.append(first)
        remaining.remove(first)
        
        # Greedy selection
        for _ in range(k - 1):
            if not remaining:
                break
            
            best_idx = -1
            best_gain = -np.inf
            
            for idx in remaining:
                # Compute marginal gain
                test_set = selected + [idx]
                submatrix = kernel[np.ix_(test_set, test_set)]
                
                try:
                    # Log-det as gain
                    _, logdet = np.linalg.slogdet(submatrix)
                    if logdet > best_gain:
                        best_gain = logdet
                        best_idx = idx
                except np.linalg.LinAlgError:
                    continue
            
            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)
        
        return np.array(selected, dtype=np.int64)


def select_keyframes_dpp(
    kernel: NDArray[np.float64],
    k: int,
    mode: str = "sample",
    seed: int = 42,
) -> NDArray[np.int64]:
    """
    Convenience function for DPP keyframe selection.
    
    Args:
        kernel: DPP kernel matrix of shape (N, N).
        k: Number of keyframes to select.
        mode: Selection mode ('sample' or 'map').
        seed: Random seed.
    
    Returns:
        Array of selected indices.
    """
    config = SelectorConfig(mode=mode, seed=seed)
    selector = DPPSelector(config)
    
    dpp_kernel = DPPKernel(kernel=kernel)
    result = selector.select(dpp_kernel, k)
    
    return result.indices


def select_diverse_subset(
    embeddings: NDArray[np.float32],
    k: int,
    timestamps: Optional[NDArray[np.float64]] = None,
    use_temporal: bool = True,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    End-to-end diverse subset selection.
    
    Builds kernel and selects subset in one call.
    
    Args:
        embeddings: Embedding array of shape (N, D).
        k: Number of items to select.
        timestamps: Optional timestamps.
        use_temporal: Whether to use temporal kernel.
    
    Returns:
        Tuple of (selected indices, selected timestamps).
    """
    from .dpp_kernel import DPPKernelBuilder
    
    builder = DPPKernelBuilder()
    kernel = builder.build_from_arrays(embeddings, timestamps, use_temporal=use_temporal)
    
    selector = DPPSelector()
    result = selector.select(kernel, k, timestamps)
    
    return result.indices, result.timestamps
