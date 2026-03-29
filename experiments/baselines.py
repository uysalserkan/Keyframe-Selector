#!/usr/bin/env python3
"""
Baseline comparison experiments.

Compares the DPP-based method against simpler baselines:
    1. CLIP + KMeans (original method)
    2. Uniform sampling
    3. Random sampling

Produces metrics and visualizations for ablation studies.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from a baseline method."""
    
    method: str
    indices: NDArray[np.int64]
    k: int
    
    # Metrics
    coverage: float = 0.0  # How much of the video is represented
    redundancy: float = 0.0  # How similar selected frames are to each other
    temporal_spread: float = 0.0  # How evenly distributed in time


def uniform_sampling(n_frames: int, k: int) -> NDArray[np.int64]:
    """
    Select frames uniformly distributed across video.
    
    Args:
        n_frames: Total number of frames.
        k: Number of frames to select.
    
    Returns:
        Selected indices.
    """
    if k >= n_frames:
        return np.arange(n_frames, dtype=np.int64)
    
    indices = np.linspace(0, n_frames - 1, k, dtype=np.int64)
    return indices


def random_sampling(n_frames: int, k: int, seed: int = 42) -> NDArray[np.int64]:
    """
    Select frames randomly.
    
    Args:
        n_frames: Total number of frames.
        k: Number of frames to select.
        seed: Random seed.
    
    Returns:
        Selected indices (sorted).
    """
    np.random.seed(seed)
    k = min(k, n_frames)
    indices = np.random.choice(n_frames, k, replace=False)
    return np.sort(indices)


def kmeans_selection(
    embeddings: NDArray[np.float32],
    k: int,
    seed: int = 42,
) -> NDArray[np.int64]:
    """
    Select frames using KMeans clustering.
    
    Selects the frame closest to each cluster center.
    
    Args:
        embeddings: Frame embeddings of shape (N, D).
        k: Number of clusters/frames to select.
        seed: Random seed.
    
    Returns:
        Selected indices (sorted).
    """
    n = len(embeddings)
    k = min(k, n)
    
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    
    indices = []
    for i in range(k):
        cluster_mask = labels == i
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) == 0:
            continue
        
        # Find closest to center
        cluster_embeddings = embeddings[cluster_indices]
        distances = np.linalg.norm(cluster_embeddings - centers[i], axis=1)
        best_local = np.argmin(distances)
        indices.append(cluster_indices[best_local])
    
    return np.sort(np.array(indices, dtype=np.int64))


def compute_coverage(
    embeddings: NDArray[np.float32],
    selected_indices: NDArray[np.int64],
) -> float:
    """
    Compute coverage metric.
    
    Measures how well selected frames represent all frames.
    Higher is better.
    
    Args:
        embeddings: All frame embeddings.
        selected_indices: Indices of selected frames.
    
    Returns:
        Coverage score in [0, 1].
    """
    if len(selected_indices) == 0:
        return 0.0
    
    selected_embeddings = embeddings[selected_indices]
    
    # For each frame, find distance to nearest selected frame
    min_distances = []
    for emb in embeddings:
        distances = np.linalg.norm(selected_embeddings - emb, axis=1)
        min_distances.append(np.min(distances))
    
    # Coverage = 1 - normalized mean distance
    mean_dist = np.mean(min_distances)
    max_possible = np.sqrt(embeddings.shape[1])  # Approximate max distance
    
    coverage = 1.0 - (mean_dist / max_possible)
    return max(0.0, min(1.0, coverage))


def compute_redundancy(
    embeddings: NDArray[np.float32],
    selected_indices: NDArray[np.int64],
) -> float:
    """
    Compute redundancy metric.
    
    Measures how similar selected frames are to each other.
    Lower is better (more diverse selection).
    
    Args:
        embeddings: All frame embeddings.
        selected_indices: Indices of selected frames.
    
    Returns:
        Redundancy score in [0, 1].
    """
    if len(selected_indices) < 2:
        return 0.0
    
    selected_embeddings = embeddings[selected_indices]
    
    # Compute pairwise similarities
    n = len(selected_embeddings)
    similarities = []
    
    for i in range(n):
        for j in range(i + 1, n):
            # Cosine similarity
            sim = np.dot(selected_embeddings[i], selected_embeddings[j])
            sim = sim / (np.linalg.norm(selected_embeddings[i]) * np.linalg.norm(selected_embeddings[j]) + 1e-8)
            similarities.append(sim)
    
    redundancy = np.mean(similarities)
    return max(0.0, min(1.0, (redundancy + 1) / 2))  # Normalize to [0, 1]


def compute_temporal_spread(
    selected_indices: NDArray[np.int64],
    n_frames: int,
) -> float:
    """
    Compute temporal spread metric.
    
    Measures how evenly distributed selected frames are in time.
    Higher is better.
    
    Args:
        selected_indices: Indices of selected frames.
        n_frames: Total number of frames.
    
    Returns:
        Temporal spread score in [0, 1].
    """
    if len(selected_indices) < 2:
        return 1.0 if len(selected_indices) == 1 else 0.0
    
    # Normalize indices to [0, 1]
    normalized = selected_indices / max(n_frames - 1, 1)
    
    # Compute gaps
    gaps = np.diff(np.sort(normalized))
    
    # Ideal gaps would be uniform
    ideal_gap = 1.0 / (len(selected_indices) - 1)
    gap_variance = np.var(gaps)
    
    # Convert variance to spread score (lower variance = higher spread)
    spread = np.exp(-gap_variance * 10)  # Scale factor
    
    return float(spread)


def run_baseline_comparison(
    embeddings: NDArray[np.float32],
    k: int,
    seed: int = 42,
) -> Dict[str, BaselineResult]:
    """
    Run all baseline methods and compute metrics.
    
    Args:
        embeddings: Frame embeddings.
        k: Number of frames to select.
        seed: Random seed.
    
    Returns:
        Dictionary mapping method name to results.
    """
    n_frames = len(embeddings)
    results = {}
    
    # 1. Uniform sampling
    uniform_indices = uniform_sampling(n_frames, k)
    results["uniform"] = BaselineResult(
        method="uniform",
        indices=uniform_indices,
        k=len(uniform_indices),
        coverage=compute_coverage(embeddings, uniform_indices),
        redundancy=compute_redundancy(embeddings, uniform_indices),
        temporal_spread=compute_temporal_spread(uniform_indices, n_frames),
    )
    
    # 2. Random sampling
    random_indices = random_sampling(n_frames, k, seed)
    results["random"] = BaselineResult(
        method="random",
        indices=random_indices,
        k=len(random_indices),
        coverage=compute_coverage(embeddings, random_indices),
        redundancy=compute_redundancy(embeddings, random_indices),
        temporal_spread=compute_temporal_spread(random_indices, n_frames),
    )
    
    # 3. KMeans
    kmeans_indices = kmeans_selection(embeddings, k, seed)
    results["kmeans"] = BaselineResult(
        method="kmeans",
        indices=kmeans_indices,
        k=len(kmeans_indices),
        coverage=compute_coverage(embeddings, kmeans_indices),
        redundancy=compute_redundancy(embeddings, kmeans_indices),
        temporal_spread=compute_temporal_spread(kmeans_indices, n_frames),
    )
    
    # 4. DPP (our method)
    try:
        from keyframe_selection.dpp_kernel import DPPKernelBuilder
        from keyframe_selection.selector import DPPSelector
        
        builder = DPPKernelBuilder()
        kernel = builder.build_from_arrays(embeddings, use_temporal=False)
        
        selector = DPPSelector()
        dpp_result = selector.select(kernel, k=k)
        dpp_indices = dpp_result.indices
        
        results["dpp"] = BaselineResult(
            method="dpp",
            indices=dpp_indices,
            k=len(dpp_indices),
            coverage=compute_coverage(embeddings, dpp_indices),
            redundancy=compute_redundancy(embeddings, dpp_indices),
            temporal_spread=compute_temporal_spread(dpp_indices, n_frames),
        )
    except ImportError:
        logger.warning("DPP modules not available")
    
    # 5. DPP with temporal kernel
    try:
        from keyframe_selection.dpp_kernel import DPPKernelBuilder
        from keyframe_selection.selector import DPPSelector
        
        timestamps = np.linspace(0, 1, n_frames)
        
        builder = DPPKernelBuilder()
        kernel = builder.build_from_arrays(embeddings, timestamps, use_temporal=True)
        
        selector = DPPSelector()
        dpp_t_result = selector.select(kernel, k=k)
        dpp_t_indices = dpp_t_result.indices
        
        results["dpp_temporal"] = BaselineResult(
            method="dpp_temporal",
            indices=dpp_t_indices,
            k=len(dpp_t_indices),
            coverage=compute_coverage(embeddings, dpp_t_indices),
            redundancy=compute_redundancy(embeddings, dpp_t_indices),
            temporal_spread=compute_temporal_spread(dpp_t_indices, n_frames),
        )
    except ImportError:
        pass
    
    return results


def print_comparison_table(results: Dict[str, BaselineResult]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Method':<15} {'K':>5} {'Coverage':>10} {'Redundancy':>12} {'Spread':>10}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{result.method:<15} {result.k:>5} {result.coverage:>10.4f} {result.redundancy:>12.4f} {result.temporal_spread:>10.4f}")
    
    print("=" * 70)
    print("Coverage: Higher is better (how well selection represents all frames)")
    print("Redundancy: Lower is better (how diverse the selection is)")
    print("Spread: Higher is better (temporal distribution)")
    print()


def save_comparison_results(
    results: Dict[str, BaselineResult],
    output_path: Path,
) -> None:
    """Save comparison results to JSON."""
    output = {}
    for name, result in results.items():
        output[name] = {
            "method": result.method,
            "k": int(result.k),
            "indices": result.indices.tolist(),
            "coverage": float(result.coverage),
            "redundancy": float(result.redundancy),
            "temporal_spread": float(result.temporal_spread),
        }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")


def main():
    """Run baseline comparison on sample data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run baseline comparisons")
    parser.add_argument("--frame-dir", type=str, help="Directory with frames")
    parser.add_argument("-k", type=int, default=10, help="Number of keyframes")
    parser.add_argument("--output", type=str, default="baseline_results.json")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    if args.frame_dir:
        # Load real embeddings
        from keyframe_selection.frame_sampling import FrameSampler
        from keyframe_selection.clip_encoder import CLIPTemporalEncoder
        
        sampler = FrameSampler()
        frame_batch = sampler.load_frames_from_directory(args.frame_dir)
        
        encoder = CLIPTemporalEncoder()
        embedding_batch = encoder.encode(frame_batch, add_temporal=False)
        embeddings = embedding_batch.embeddings
        
        logger.info(f"Loaded {len(embeddings)} frames from {args.frame_dir}")
    else:
        # Use synthetic data
        np.random.seed(args.seed)
        embeddings = np.random.randn(100, 512).astype(np.float32)
        logger.info("Using synthetic embeddings (100 x 512)")
    
    # Run comparison
    results = run_baseline_comparison(embeddings, args.k, args.seed)
    
    # Print and save
    print_comparison_table(results)
    save_comparison_results(results, Path(args.output))


if __name__ == "__main__":
    main()
