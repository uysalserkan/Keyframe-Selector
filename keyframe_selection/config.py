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
class TemporalAnalysisConfig:
    """Configuration for temporal change analysis."""
    
    # H3: Percentile threshold for significant changes
    delta_percentile: float = 90.0
    
    # Smoothing for noisy videos
    use_ema_smoothing: bool = False
    ema_alpha: float = 0.3  # EMA decay factor
    
    # Minimum gap between detected change points
    min_segment_frames: int = 3


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
    """Configuration for DPP subset selection."""
    
    # H9: DPP sampling mode
    mode: Literal["sample", "map"] = "sample"  # sample = stochastic, map = deterministic
    
    # Override K from entropy estimator (None = use adaptive K)
    fixed_k: Optional[int] = None
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Number of samples for stochastic mode
    num_samples: int = 1


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
    
    # Stage configurations
    frame_sampling: FrameSamplingConfig = field(default_factory=FrameSamplingConfig)
    clip_encoder: CLIPEncoderConfig = field(default_factory=CLIPEncoderConfig)
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
        temporal_analysis = TemporalAnalysisConfig(**config_dict.pop("temporal_analysis", {}))
        entropy_estimator = EntropyEstimatorConfig(**config_dict.pop("entropy_estimator", {}))
        dpp_kernel = DPPKernelConfig(**config_dict.pop("dpp_kernel", {}))
        selector = SelectorConfig(**config_dict.pop("selector", {}))
        motion = MotionConfig(**config_dict.pop("motion", {}))
        
        return cls(
            frame_sampling=frame_sampling,
            clip_encoder=clip_encoder,
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
