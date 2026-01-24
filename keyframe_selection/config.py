"""
Configuration dataclasses for all pipeline hyperparameters (H1-H10).

Hyperparameter Reference:
    H1: CLIP model variant
    H2: Frame sampling FPS
    H3: Temporal change threshold τΔ percentile
    H4: Temporal encoding weight α
    H5: Entropy histogram bin count
    H6: Keyframe count scaling factor β
    H7: Feature kernel bandwidth σf
    H8: Temporal kernel bandwidth σt
    H9: DPP sampling mode
    H10: Motion encoding weight γ
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path
import yaml


@dataclass
class FrameSamplingConfig:
    """Configuration for frame extraction from video."""
    
    # H2: Frames per second to sample
    fps: float = 1.0
    
    # Adaptive sampling based on scene changes
    adaptive: bool = False
    adaptive_threshold: float = 30.0  # Scene change threshold
    
    # Output settings
    output_format: str = "jpg"
    jpeg_quality: int = 95


@dataclass
class CLIPEncoderConfig:
    """Configuration for CLIP embedding extraction."""
    
    # H1: CLIP model variant
    model_name: str = "ViT-L/14"
    
    # H4: Temporal encoding weight α
    temporal_weight: float = 0.1
    
    # Processing settings
    batch_size: int = 32
    use_fp16: bool = True  # Mixed precision for faster inference
    
    # Device selection (auto-detected if None)
    device: Optional[str] = None


@dataclass
class DINOv3EncoderConfig:
    """Configuration for DINOv3 embedding extraction via Hugging Face Transformers."""
    
    # Model identifier on Hugging Face Hub
    # Options include:
    #   - facebook/dinov2-small, facebook/dinov2-base, facebook/dinov2-large
    #   - facebook/dinov2-giant
    model_id: str = "facebook/dinov2-base"
    
    # Model revision for reproducibility (None = latest)
    revision: Optional[str] = None
    
    # H4: Temporal encoding weight α (same semantics as CLIP)
    temporal_weight: float = 0.1
    
    # Pooling strategy for CLS token extraction
    # Options: "cls" (CLS token), "mean" (mean of patch tokens)
    pooling: Literal["cls", "mean"] = "cls"
    
    # Processing settings
    batch_size: int = 32
    use_fp16: bool = True  # Mixed precision for faster inference on CUDA
    
    # Device selection (auto-detected if None)
    device: Optional[str] = None


@dataclass
class TemporalAnalysisConfig:
    """Configuration for temporal change analysis."""
    
    # H3: Percentile threshold for significant changes
    delta_percentile: float = 90.0
    
    # Smoothing for noisy videos
    use_ema_smoothing: bool = False
    ema_alpha: float = 0.3  # EMA decay factor
    
    # Minimum gap between detected change points
    min_segment_frames: int = 3
    
    # GPU acceleration for delta computation
    use_gpu: bool = True


@dataclass
class EntropyEstimatorConfig:
    """Configuration for content density estimation."""
    
    # H5: Number of histogram bins for entropy calculation
    num_bins: int = 50
    
    # H6: Scaling factor β for keyframe count
    beta: float = 1.0
    
    # PCA dimensionality reduction
    pca_components: int = 32
    
    # Bounds for adaptive K
    k_min: int = 3
    k_max: int = 50
    
    # Numerical stability
    epsilon: float = 1e-10
    
    # GPU acceleration for PCA computation
    use_gpu: bool = True


@dataclass
class DPPKernelConfig:
    """Configuration for DPP kernel construction."""
    
    # H7: Feature kernel bandwidth σf (None = use median heuristic)
    sigma_f: Optional[float] = None
    
    # H8: Temporal kernel bandwidth σt as fraction of video length
    sigma_t_ratio: float = 0.2
    
    # Kernel combination method
    combine_method: Literal["hadamard", "additive"] = "hadamard"
    
    # Use GPU for kernel computation
    use_gpu: bool = True


@dataclass
class SelectorConfig:
    """Configuration for keyframe selection."""
    
    # Selection method: dpp, kmeans, or hdbscan
    method: Literal["dpp", "kmeans", "hdbscan"] = "dpp"
    
    # === DPP-specific parameters ===
    # H9: DPP sampling mode (only used when method="dpp")
    mode: Literal["sample", "map"] = "sample"  # sample = stochastic, map = deterministic
    
    # === Common parameters ===
    # Override K from entropy estimator (None = use adaptive K)
    # Not used for HDBSCAN (which determines K automatically)
    fixed_k: Optional[int] = None
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Number of samples for stochastic mode (only used when method="dpp" and mode="sample")
    num_samples: int = 1
    
    # Minimum gap between selected frames (prevents temporal clustering)
    # Set to 0 to disable. Applies to all selection methods.
    min_frame_gap: int = 5
    
    # Include detected change points in selection (only used when method="dpp")
    # Ensures scene transitions are captured
    include_change_points: bool = True
    
    # === K-means specific parameters ===
    # Initialization method for K-means
    kmeans_init: Literal["k-means++", "random"] = "k-means++"
    
    # Number of times K-means is run with different initializations
    kmeans_n_init: int = 10
    
    # Maximum number of iterations for K-means
    kmeans_max_iter: int = 300
    
    # GPU acceleration for K-means
    kmeans_use_gpu: bool = True
    
    # === HDBSCAN specific parameters ===
    # Minimum number of samples in a neighborhood for a point to be considered as a core point
    hdbscan_min_cluster_size: int = 2
    
    # Minimum number of samples in a neighborhood for a point to be considered as part of a dense neighborhood
    # If None, defaults to min_cluster_size
    hdbscan_min_samples: Optional[int] = None
    
    # A distance threshold for cluster formation
    hdbscan_cluster_selection_epsilon: float = 0.0
    
    # Method used to select clusters from the cluster hierarchy
    hdbscan_cluster_selection_method: Literal["eom", "leaf"] = "eom"
    
    # GPU acceleration for DPP greedy MAP
    dpp_use_gpu: bool = True


@dataclass
class MotionConfig:
    """Configuration for optional motion awareness."""
    
    # Enable motion encoding
    enabled: bool = False
    
    # H10: Motion encoding weight γ
    gamma: float = 0.1
    
    # Optical flow method
    method: Literal["farneback", "raft"] = "farneback"
    
    # Flow computation settings
    flow_scale: float = 0.5  # Downscale factor for flow computation


@dataclass
class PipelineConfig:
    """Master configuration combining all pipeline stages."""
    
    # Video input
    video_path: Optional[Path] = None
    frame_dir: Optional[Path] = None
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("output"))
    save_intermediate: bool = False
    
    # Encoder backend selection
    encoder_backend: Literal["clip", "dinov3"] = "clip"
    
    # Global device setting (overrides individual component settings)
    # None = auto-detect, "cuda" = GPU, "cpu" = CPU
    device: Optional[str] = None
    
    # Stage configurations
    frame_sampling: FrameSamplingConfig = field(default_factory=FrameSamplingConfig)
    clip_encoder: CLIPEncoderConfig = field(default_factory=CLIPEncoderConfig)
    dinov3_encoder: DINOv3EncoderConfig = field(default_factory=DINOv3EncoderConfig)
    temporal_analysis: TemporalAnalysisConfig = field(default_factory=TemporalAnalysisConfig)
    entropy_estimator: EntropyEstimatorConfig = field(default_factory=EntropyEstimatorConfig)
    dpp_kernel: DPPKernelConfig = field(default_factory=DPPKernelConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    
    # Ablation toggles
    use_temporal_encoding: bool = True
    use_entropy_k: bool = True
    use_temporal_kernel: bool = True
    
    # Global settings
    random_seed: int = 42
    verbose: bool = True
    
    def __post_init__(self):
        """Convert paths to Path objects if needed."""
        if self.video_path is not None and not isinstance(self.video_path, Path):
            self.video_path = Path(self.video_path)
        if self.frame_dir is not None and not isinstance(self.frame_dir, Path):
            self.frame_dir = Path(self.frame_dir)
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "PipelineConfig":
        """Create config from a dictionary (e.g., loaded from YAML/JSON)."""
        # Extract nested configs
        frame_sampling = FrameSamplingConfig(**config_dict.pop("frame_sampling", {}))
        clip_encoder = CLIPEncoderConfig(**config_dict.pop("clip_encoder", {}))
        dinov3_encoder = DINOv3EncoderConfig(**config_dict.pop("dinov3_encoder", {}))
        temporal_analysis = TemporalAnalysisConfig(**config_dict.pop("temporal_analysis", {}))
        entropy_estimator = EntropyEstimatorConfig(**config_dict.pop("entropy_estimator", {}))
        dpp_kernel = DPPKernelConfig(**config_dict.pop("dpp_kernel", {}))
        selector = SelectorConfig(**config_dict.pop("selector", {}))
        motion = MotionConfig(**config_dict.pop("motion", {}))
        
        return cls(
            frame_sampling=frame_sampling,
            clip_encoder=clip_encoder,
            dinov3_encoder=dinov3_encoder,
            temporal_analysis=temporal_analysis,
            entropy_estimator=entropy_estimator,
            dpp_kernel=dpp_kernel,
            selector=selector,
            motion=motion,
            **config_dict,
        )
    
    def to_dict(self) -> dict:
        """Serialize config to dictionary for logging/saving."""
        from dataclasses import asdict
        return asdict(self)
    
    def __str__(self) -> str:
        """Return string representation of config."""
        return yaml.dump(self.to_dict(), indent=4)
