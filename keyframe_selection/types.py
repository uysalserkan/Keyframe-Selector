"""
Type definitions and data structures for the keyframe selection pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class FrameData:
    """Single frame with metadata."""
    
    image: NDArray[np.uint8]  # BGR format from OpenCV
    timestamp: float  # Seconds from video start
    frame_index: int  # Original frame index in video
    path: Optional[Path] = None  # Path if loaded from disk


@dataclass
class FrameBatch:
    """Collection of frames with metadata."""
    
    frames: List[FrameData]
    video_duration: float  # Total video duration in seconds
    source_fps: float  # Original video FPS
    source_path: Optional[Path] = None
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __getitem__(self, idx: int) -> FrameData:
        return self.frames[idx]
    
    @property
    def images(self) -> List[NDArray[np.uint8]]:
        """Get all images as a list."""
        return [f.image for f in self.frames]
    
    @property
    def timestamps(self) -> NDArray[np.float64]:
        """Get all timestamps as numpy array."""
        return np.array([f.timestamp for f in self.frames], dtype=np.float64)
    
    @property
    def normalized_timestamps(self) -> NDArray[np.float64]:
        """Get timestamps normalized to [0, 1]."""
        if self.video_duration > 0:
            return self.timestamps / self.video_duration
        return self.timestamps


@dataclass
class EmbeddingBatch:
    """CLIP embeddings with temporal information."""
    
    # Core embeddings: shape (N, D) where D is embedding dimension
    embeddings: NDArray[np.float32]
    
    # Temporal-augmented embeddings if enabled: shape (N, D+1) or (N, D+flow_dim)
    temporal_embeddings: Optional[NDArray[np.float32]] = None
    
    # Timestamps for each embedding
    timestamps: NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    
    # Frame indices mapping back to original video
    frame_indices: NDArray[np.int64] = field(default_factory=lambda: np.array([]))
    
    # Optional motion features: shape (N, motion_dim)
    motion_features: Optional[NDArray[np.float32]] = None
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    @property
    def effective_embeddings(self) -> NDArray[np.float32]:
        """Return temporal embeddings if available, otherwise base embeddings."""
        return self.temporal_embeddings if self.temporal_embeddings is not None else self.embeddings
    
    @property
    def embedding_dim(self) -> int:
        """Dimension of base embeddings."""
        return self.embeddings.shape[1] if len(self.embeddings) > 0 else 0


@dataclass
class TemporalAnalysisResult:
    """Results from temporal change analysis."""
    
    # Per-frame temporal deltas: shape (N-1,)
    deltas: NDArray[np.float64]
    
    # Threshold for significant changes
    threshold: float
    
    # Indices of detected change points
    change_points: NDArray[np.int64]
    
    # Soft segment boundaries (probabilities)
    segment_probs: Optional[NDArray[np.float64]] = None
    
    @property
    def num_segments(self) -> int:
        """Estimated number of segments."""
        return len(self.change_points) + 1
    
    @property
    def mean_delta(self) -> float:
        """Average temporal change."""
        return float(np.mean(self.deltas)) if len(self.deltas) > 0 else 0.0


@dataclass
class EntropyResult:
    """Results from entropy-based content analysis."""
    
    # Shannon entropy of embedding distribution
    entropy: float
    
    # Recommended number of keyframes
    recommended_k: int
    
    # PCA-reduced embeddings used for computation
    reduced_embeddings: Optional[NDArray[np.float32]] = None
    
    # Histogram used for entropy calculation
    histogram: Optional[NDArray[np.float64]] = None


@dataclass
class DPPKernel:
    """DPP kernel matrix with metadata."""
    
    # L-ensemble kernel: shape (N, N)
    kernel: NDArray[np.float64]
    
    # Feature similarity kernel
    feature_kernel: Optional[NDArray[np.float64]] = None
    
    # Temporal similarity kernel
    temporal_kernel: Optional[NDArray[np.float64]] = None
    
    # Bandwidth parameters used
    sigma_f: float = 0.0
    sigma_t: float = 0.0
    
    @property
    def size(self) -> int:
        return self.kernel.shape[0]


@dataclass
class KeyframeResult:
    """Final keyframe selection results."""
    
    # Selected frame indices (sorted by time)
    indices: NDArray[np.int64]
    
    # Timestamps of selected frames
    timestamps: NDArray[np.float64]
    
    # Paths to selected frames if available
    paths: Optional[List[Path]] = None
    
    # Selection scores/probabilities
    scores: Optional[NDArray[np.float64]] = None
    
    # Metadata from selection process
    metadata: dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    @property
    def k(self) -> int:
        """Number of selected keyframes."""
        return len(self.indices)


@dataclass
class PipelineResult:
    """Complete results from the keyframe selection pipeline."""
    
    # Final keyframes
    keyframes: KeyframeResult
    
    # Intermediate results for analysis
    frame_batch: Optional[FrameBatch] = None
    embedding_batch: Optional[EmbeddingBatch] = None
    temporal_analysis: Optional[TemporalAnalysisResult] = None
    entropy_result: Optional[EntropyResult] = None
    dpp_kernel: Optional[DPPKernel] = None
    
    # Timing information
    timing: dict = field(default_factory=dict)
    
    # Configuration used
    config: Optional[dict] = None
