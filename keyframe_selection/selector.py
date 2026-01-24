"""
Diversity-aware subset selection using Determinantal Point Processes.

Implements DPP sampling for selecting diverse, non-redundant keyframes.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Optional PyTorch for GPU acceleration
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
        change_points: Optional[NDArray[np.int64]] = None,
    ) -> KeyframeResult:
        """
        Select diverse subset using DPP.
        
        Args:
            kernel: DPP kernel matrix.
            k: Number of items to select. Uses fixed_k from config if None.
            timestamps: Optional timestamps for selected frames.
            change_points: Optional change points to prioritize in selection.
        
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
            indices = self._select_greedy(kernel.kernel, k, change_points)
        
        # Sort by time order
        indices = np.sort(indices)
        
        # Get timestamps if provided
        if timestamps is not None and len(timestamps) > 0:
            selected_timestamps = timestamps[indices]
        else:
            selected_timestamps = indices.astype(np.float64)
        
        num_change_points_included = 0
        if change_points is not None and len(change_points) > 0:
            num_change_points_included = len(set(indices) & set(change_points))
        
        logger.info(f"Selected {len(indices)} keyframes from {n} candidates (includes {num_change_points_included} change points)")
        
        return KeyframeResult(
            indices=indices,
            timestamps=selected_timestamps,
            metadata={
                "k": len(indices),
                "n": n,
                "method": "dpp" if _check_dppy() else "greedy",
                "mode": self.config.mode,
                "change_points_included": num_change_points_included,
            },
        )
    
    def select_from_embeddings(
        self,
        embedding_batch: EmbeddingBatch,
        kernel: Optional[DPPKernel] = None,
        k: Optional[int] = None,
        change_points: Optional[NDArray[np.int64]] = None,
    ) -> KeyframeResult:
        """
        Select keyframes with full metadata from embedding batch.
        
        Supports multiple selection methods: DPP, K-means, HDBSCAN.
        
        Args:
            embedding_batch: Source embeddings.
            kernel: DPP kernel (required for DPP, ignored for other methods).
            k: Number of keyframes to select (required for DPP and K-means, ignored for HDBSCAN).
            change_points: Optional array of change point indices to prioritize.
        
        Returns:
            KeyframeResult with indices and timestamps.
        """
        # Dispatch to appropriate selection method
        if self.config.method == "dpp":
            if kernel is None:
                raise ValueError("DPP method requires kernel parameter")
            result = self.select(kernel, k, embedding_batch.timestamps, change_points)
        elif self.config.method == "kmeans":
            result = self._select_kmeans(embedding_batch, k, change_points)
        elif self.config.method == "hdbscan":
            result = self._select_hdbscan(embedding_batch, change_points)
        else:
            raise ValueError(f"Unknown selection method: {self.config.method}")
        
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
            
            # Apply min_gap filter post-hoc for DPP samples
            if self.config.min_frame_gap > 0:
                indices = self._apply_min_gap_filter(indices)
        else:
            # MAP inference (greedy) - min_gap is enforced during selection
            indices = self._greedy_map(kernel, k)
        
        return indices
    
    def _apply_min_gap_filter(self, indices: NDArray[np.int64]) -> NDArray[np.int64]:
        """
        Post-hoc filter to enforce minimum gap between selected indices.
        
        Args:
            indices: Selected indices (may have duplicates in time proximity).
        
        Returns:
            Filtered indices with minimum gap enforced.
        """
        if len(indices) <= 1:
            return indices
        
        sorted_indices = np.sort(indices)
        min_gap = self.config.min_frame_gap
        
        filtered = [sorted_indices[0]]
        for idx in sorted_indices[1:]:
            if idx - filtered[-1] >= min_gap:
                filtered.append(idx)
        
        return np.array(filtered, dtype=np.int64)
    
    def _select_greedy(
        self,
        kernel: NDArray[np.float64],
        k: int,
        change_points: Optional[NDArray[np.int64]] = None,
    ) -> NDArray[np.int64]:
        """
        Greedy selection fallback when dppy is unavailable.
        
        Uses greedy maximization of log-determinant.
        Prioritizes change points if provided.
        
        Args:
            kernel: Kernel matrix.
            k: Number of items to select.
            change_points: Optional change points to include first.
        
        Returns:
            Selected indices.
        """
        return self._greedy_map(kernel, k, change_points)
    
    def _greedy_map(
        self,
        kernel: NDArray[np.float64],
        k: int,
        change_points: Optional[NDArray[np.int64]] = None,
    ) -> NDArray[np.int64]:
        """
        Greedy MAP inference for DPP with minimum gap enforcement.
        
        Iteratively selects items that maximize marginal gain in
        log-determinant of the selected subset's kernel, while
        respecting minimum temporal gap between selections.
        
        Prioritizes change points (scene transitions) first.
        
        Args:
            kernel: Kernel matrix.
            k: Number of items to select.
            change_points: Optional change points to prioritize.
        
        Returns:
            Selected indices.
        """
        # Try GPU acceleration if enabled
        if self.config.dpp_use_gpu and HAS_TORCH and torch.cuda.is_available():
            try:
                return self._greedy_map_gpu(kernel, k, change_points)
            except Exception as e:
                logger.warning(f"GPU greedy MAP failed, falling back to CPU: {e}")
        
        # CPU implementation
        return self._greedy_map_cpu(kernel, k, change_points)
    
    def _greedy_map_gpu(
        self,
        kernel: NDArray[np.float64],
        k: int,
        change_points: Optional[NDArray[np.int64]] = None,
    ) -> NDArray[np.int64]:
        """
        GPU-accelerated greedy MAP for DPP using PyTorch.
        
        Args:
            kernel: Kernel matrix.
            k: Number of items to select.
            change_points: Optional change points to prioritize.
        
        Returns:
            Selected indices.
        """
        device = torch.device('cuda')
        K = torch.from_numpy(kernel).float().to(device)
        n = K.shape[0]
        min_gap = self.config.min_frame_gap
        
        selected: List[int] = []
        mask = torch.ones(n, dtype=torch.bool, device=device)
        
        def mask_neighbors_gpu(idx: int):
            """Mark neighbors within min_gap as unavailable."""
            if min_gap > 0:
                start = max(0, idx - min_gap)
                end = min(n, idx + min_gap + 1)
                mask[start:end] = False
        
        # PHASE 1: Include change points (scene transitions)
        if change_points is not None and len(change_points) > 0:
            valid_cps = [cp for cp in change_points if 0 <= cp < n]
            # Sort by self-similarity
            diag = torch.diag(K)
            cp_scores = [(cp, diag[cp].item()) for cp in valid_cps]
            cp_scores.sort(key=lambda x: x[1], reverse=True)
            
            for cp, _ in cp_scores:
                if len(selected) >= k:
                    break
                if mask[cp]:
                    selected.append(cp)
                    mask[cp] = False
                    mask_neighbors_gpu(cp)
            
            logger.debug(f"Included {len(selected)} change points in selection")
        
        # PHASE 2: Start with highest self-similarity if no change points
        if len(selected) == 0:
            diag = torch.diag(K)
            first_idx = diag.argmax().item()
            selected.append(first_idx)
            mask[first_idx] = False
            mask_neighbors_gpu(first_idx)
        
        # PHASE 3: Greedy selection for remaining slots
        for _ in range(k - len(selected)):
            if not mask.any():
                break
            
            best_idx = -1
            best_gain = -np.inf
            
            # Get available indices
            available = torch.where(mask)[0].cpu().numpy()
            
            for idx in available:
                test_set = selected + [idx]
                # Extract submatrix
                indices_tensor = torch.tensor(test_set, device=device, dtype=torch.long)
                submatrix = K[indices_tensor][:, indices_tensor]
                
                try:
                    # Log-det as gain (more efficient than full computation)
                    _, logdet = torch.linalg.slogdet(submatrix)
                    logdet_val = logdet.item()
                    if logdet_val > best_gain:
                        best_gain = logdet_val
                        best_idx = idx
                except torch.linalg.LinAlgError:
                    continue
            
            if best_idx >= 0:
                selected.append(best_idx)
                mask[best_idx] = False
                mask_neighbors_gpu(best_idx)
        
        return np.array(selected, dtype=np.int64)
    
    def _greedy_map_cpu(
        self,
        kernel: NDArray[np.float64],
        k: int,
        change_points: Optional[NDArray[np.int64]] = None,
    ) -> NDArray[np.int64]:
        """
        CPU-based greedy MAP for DPP.
        
        Args:
            kernel: Kernel matrix.
            k: Number of items to select.
            change_points: Optional change points to prioritize.
        
        Returns:
            Selected indices.
        """
        n = len(kernel)
        selected: List[int] = []
        remaining = set(range(n))
        min_gap = self.config.min_frame_gap
        
        def remove_neighbors(idx: int):
            """Remove neighbors within min_gap from remaining set."""
            if min_gap > 0:
                for neighbor in range(max(0, idx - min_gap), min(n, idx + min_gap + 1)):
                    remaining.discard(neighbor)
        
        # PHASE 1: First include change points (scene transitions)
        if change_points is not None and len(change_points) > 0:
            # Filter valid change points and sort by kernel diagonal (quality)
            valid_cps = [cp for cp in change_points if 0 <= cp < n]
            # Sort by self-similarity (higher = more distinctive)
            valid_cps = sorted(valid_cps, key=lambda x: kernel[x, x], reverse=True)
            
            for cp in valid_cps:
                if len(selected) >= k:
                    break
                if cp in remaining:
                    selected.append(cp)
                    remaining.remove(cp)
                    remove_neighbors(cp)
            
            logger.debug(f"Included {len(selected)} change points in selection")
        
        # PHASE 2: Start with highest self-similarity if no change points added
        if len(selected) == 0:
            first = int(np.argmax(np.diag(kernel)))
            selected.append(first)
            remaining.remove(first)
            remove_neighbors(first)
        
        # PHASE 3: Greedy selection for remaining slots
        for _ in range(k - len(selected)):
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
                remove_neighbors(best_idx)
        
        return np.array(selected, dtype=np.int64)


    def _select_kmeans(
        self,
        embedding_batch: EmbeddingBatch,
        k: Optional[int] = None,
        change_points: Optional[NDArray[np.int64]] = None,
    ) -> KeyframeResult:
        """
        Select keyframes using K-means clustering.
        
        Selects the medoid (frame closest to cluster center) from each cluster.
        
        Args:
            embedding_batch: Batch of frame embeddings.
            k: Number of clusters (keyframes to select). Uses fixed_k from config if None.
            change_points: Optional change points (not used for K-means but kept for interface consistency).
        
        Returns:
            KeyframeResult with selected indices.
        """
        # Try GPU-accelerated version first
        if self.config.kmeans_use_gpu and HAS_TORCH and torch.cuda.is_available():
            try:
                return self._select_kmeans_gpu(embedding_batch, k, change_points)
            except Exception as e:
                logger.warning(f"GPU K-means failed, falling back to sklearn: {e}")
        
        # CPU fallback
        return self._select_kmeans_cpu(embedding_batch, k, change_points)
    
    def _select_kmeans_gpu(
        self,
        embedding_batch: EmbeddingBatch,
        k: Optional[int] = None,
        change_points: Optional[NDArray[np.int64]] = None,
    ) -> KeyframeResult:
        """
        GPU-accelerated K-means selection using torch-kmeans.
        
        Args:
            embedding_batch: Batch of frame embeddings.
            k: Number of clusters (keyframes to select).
            change_points: Optional change points (unused but kept for interface).
        
        Returns:
            KeyframeResult with selected indices.
        """
        try:
            from torch_kmeans import KMeans as TorchKMeans
        except ImportError:
            logger.warning("torch-kmeans not installed, falling back to sklearn")
            return self._select_kmeans_cpu(embedding_batch, k, change_points)
        
        if k is None:
            k = self.config.fixed_k
        
        if k is None:
            raise ValueError("k must be specified either in call or config.fixed_k")
        
        embeddings = embedding_batch.effective_embeddings
        n = len(embeddings)
        k = min(k, n)
        
        if k <= 0 or n == 0:
            return KeyframeResult(
                indices=np.array([], dtype=np.int64),
                timestamps=np.array([], dtype=np.float64),
                metadata={"method": "kmeans_gpu", "k": 0, "n": n},
            )
        
        logger.info(f"Selecting {k} keyframes using GPU-accelerated K-means")
        
        # Prepare embeddings tensor (torch-kmeans expects batch format)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings_tensor = torch.from_numpy(embeddings).float()
        embeddings_tensor = embeddings_tensor.unsqueeze(0).to(device)  # (1, N, D)
        
        # Run GPU K-means
        model = TorchKMeans(
            n_clusters=k,
            init_method='k-means++',
            max_iter=self.config.kmeans_max_iter,
        )
        result = model(embeddings_tensor)
        
        labels = result.labels.squeeze(0).cpu().numpy()
        centers = result.centers.squeeze(0).cpu().numpy()
        
        # Select medoid from each cluster
        selected_indices = []
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            if not np.any(cluster_mask):
                continue
            
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            center = centers[cluster_id]
            
            # Find frame closest to cluster center (medoid)
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            medoid_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(medoid_idx)
        
        selected_indices = np.array(selected_indices, dtype=np.int64)
        
        # Apply min_frame_gap filter if configured
        if self.config.min_frame_gap > 0:
            selected_indices = self._apply_min_gap_filter(selected_indices)
        
        # Sort by time order
        selected_indices = np.sort(selected_indices)
        
        # Get timestamps if provided
        if len(embedding_batch.timestamps) > 0:
            selected_timestamps = embedding_batch.timestamps[selected_indices]
        else:
            selected_timestamps = selected_indices.astype(np.float64)
        
        logger.info(f"GPU K-means selected {len(selected_indices)} keyframes from {n} frames")
        
        return KeyframeResult(
            indices=selected_indices,
            timestamps=selected_timestamps,
            metadata={
                "method": "kmeans_gpu",
                "k": len(selected_indices),
                "n": n,
                "n_clusters_requested": k,
            },
        )
    
    def _select_kmeans_cpu(
        self,
        embedding_batch: EmbeddingBatch,
        k: Optional[int] = None,
        change_points: Optional[NDArray[np.int64]] = None,
    ) -> KeyframeResult:
        """
        CPU-based K-means selection using scikit-learn.
        
        Args:
            embedding_batch: Batch of frame embeddings.
            k: Number of clusters (keyframes to select).
            change_points: Optional change points (unused but kept for interface).
        
        Returns:
            KeyframeResult with selected indices.
        """
        from sklearn.cluster import KMeans
        
        if k is None:
            k = self.config.fixed_k
        
        if k is None:
            raise ValueError("k must be specified either in call or config.fixed_k")
        
        embeddings = embedding_batch.effective_embeddings
        n = len(embeddings)
        k = min(k, n)
        
        if k <= 0 or n == 0:
            return KeyframeResult(
                indices=np.array([], dtype=np.int64),
                timestamps=np.array([], dtype=np.float64),
                metadata={"method": "kmeans", "k": 0, "n": n},
            )
        
        logger.info(f"Selecting {k} keyframes using K-means clustering")
        
        # Fit K-means
        kmeans = KMeans(
            n_clusters=k,
            init=self.config.kmeans_init,
            n_init=self.config.kmeans_n_init,
            max_iter=self.config.kmeans_max_iter,
            random_state=self.config.seed,
        )
        labels = kmeans.fit_predict(embeddings)
        
        # Select medoid from each cluster
        selected_indices = []
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            if not np.any(cluster_mask):
                continue
            
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            center = kmeans.cluster_centers_[cluster_id]
            
            # Find frame closest to cluster center
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            medoid_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(medoid_idx)
        
        selected_indices = np.array(selected_indices, dtype=np.int64)
        
        # Apply min_frame_gap filter if configured
        if self.config.min_frame_gap > 0:
            selected_indices = self._apply_min_gap_filter(selected_indices)
        
        # Sort by time order
        selected_indices = np.sort(selected_indices)
        
        # Get timestamps if provided
        if len(embedding_batch.timestamps) > 0:
            selected_timestamps = embedding_batch.timestamps[selected_indices]
        else:
            selected_timestamps = selected_indices.astype(np.float64)
        
        logger.info(f"K-means selected {len(selected_indices)} keyframes from {n} frames")
        
        return KeyframeResult(
            indices=selected_indices,
            timestamps=selected_timestamps,
            metadata={
                "method": "kmeans",
                "k": len(selected_indices),
                "n": n,
                "n_clusters_requested": k,
            },
        )
    
    def _select_hdbscan(
        self,
        embedding_batch: EmbeddingBatch,
        change_points: Optional[NDArray[np.int64]] = None,
    ) -> KeyframeResult:
        """
        Select keyframes using HDBSCAN density-based clustering.
        
        Automatically determines the number of clusters based on density.
        Selects the medoid from each cluster (including noise).
        
        Args:
            embedding_batch: Batch of frame embeddings.
            change_points: Optional change points (not used for HDBSCAN but kept for interface consistency).
        
        Returns:
            KeyframeResult with selected indices (variable number based on density).
        """
        from sklearn.cluster import HDBSCAN
        
        embeddings = embedding_batch.effective_embeddings
        n = len(embeddings)
        
        if n == 0:
            return KeyframeResult(
                indices=np.array([], dtype=np.int64),
                timestamps=np.array([], dtype=np.float64),
                metadata={"method": "hdbscan", "n_clusters": 0},
            )
        
        logger.info("Selecting keyframes using HDBSCAN density-based clustering")
        
        # Configure HDBSCAN
        min_samples = self.config.hdbscan_min_samples
        if min_samples is None:
            min_samples = self.config.hdbscan_min_cluster_size
        
        hdbscan = HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=self.config.hdbscan_cluster_selection_epsilon,
            cluster_selection_method=self.config.hdbscan_cluster_selection_method,
        )
        labels = hdbscan.fit_predict(embeddings)
        
        # Select medoid from each cluster (including noise cluster -1)
        unique_labels = np.unique(labels)
        selected_indices = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            # Compute medoid (point closest to cluster mean)
            center = np.mean(cluster_embeddings, axis=0)
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            medoid_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(medoid_idx)
        
        selected_indices = np.array(selected_indices, dtype=np.int64)
        
        # Apply min_frame_gap filter if configured
        if self.config.min_frame_gap > 0:
            selected_indices = self._apply_min_gap_filter(selected_indices)
        
        # Sort by time order
        selected_indices = np.sort(selected_indices)
        
        # Get timestamps if provided
        if len(embedding_batch.timestamps) > 0:
            selected_timestamps = embedding_batch.timestamps[selected_indices]
        else:
            selected_timestamps = selected_indices.astype(np.float64)
        
        n_noise = np.sum(labels == -1)
        n_clusters = len(unique_labels)
        
        logger.info(
            f"HDBSCAN found {n_clusters} clusters (including {n_noise} noise points), "
            f"selected {len(selected_indices)} keyframes from {n} frames"
        )
        
        return KeyframeResult(
            indices=selected_indices,
            timestamps=selected_timestamps,
            metadata={
                "method": "hdbscan",
                "k": len(selected_indices),
                "n": n,
                "n_clusters": n_clusters,
                "n_noise": int(n_noise),
            },
        )


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
