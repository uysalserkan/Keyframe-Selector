#!/usr/bin/env python3
"""
CLI entry point for the keyframe selection pipeline.

Usage:
    # From video file (default: CLIP encoder)
    python run_pipeline.py --video /path/to/video.mp4 --output ./output
    
    # From pre-extracted frames
    python run_pipeline.py --frames ./frames --output ./output
    
    # Use DINOv3 encoder instead of CLIP
    python run_pipeline.py --frames ./frames --encoder dinov3
    
    # Use specific DINOv3 model
    python run_pipeline.py --frames ./frames --encoder dinov3 --dinov3-model-id facebook/dinov2-large
    
    # With custom config file
    python run_pipeline.py --config config.yaml
    
    # Ablation: disable temporal features
    python run_pipeline.py --frames ./frames --no-temporal
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add parent to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from keyframe_selection import PipelineConfig
from keyframe_selection.pipeline import KeyframeSelectionPipeline
from keyframe_selection.utils.io import setup_logging, load_config, save_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Temporal-Aware Keyframe Selection with DPP (CLIP or DINOv3 embeddings)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file (default: CLIP encoder)
  python run_pipeline.py --video video.mp4 --output results/
  
  # Process pre-extracted frames
  python run_pipeline.py --frames ./frames/ --output results/
  
  # Use DINOv3 encoder instead of CLIP
  python run_pipeline.py --frames ./frames/ --encoder dinov3
  
  # Use specific DINOv3 model variant
  python run_pipeline.py --frames ./frames/ --encoder dinov3 --dinov3-model-id facebook/dinov2-large
  
  # Use config file
  python run_pipeline.py --config my_config.yaml
  
  # Fixed number of keyframes (disable adaptive K)
  python run_pipeline.py --frames ./frames/ -k 10 --no-entropy-k
  
  # Ablation: disable temporal encoding
  python run_pipeline.py --frames ./frames/ --no-temporal
        """
    )
    
    # Input options
    input_group = parser.add_argument_group("Input")
    input_group.add_argument(
        "--video", "-v",
        type=str,
        help="Path to input video file",
    )
    input_group.add_argument(
        "--frames", "-f",
        type=str,
        help="Path to directory with pre-extracted frames",
    )
    input_group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML/JSON config file",
    )
    
    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    output_group.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results (embeddings, kernels)",
    )
    
    # Frame sampling options
    sampling_group = parser.add_argument_group("Frame Sampling")
    sampling_group.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frame sampling rate (default: 1.0)",
    )
    sampling_group.add_argument(
        "--adaptive-sampling",
        action="store_true",
        help="Enable adaptive scene-based sampling",
    )
    
    # Encoder backend selection
    encoder_group = parser.add_argument_group("Encoder")
    encoder_group.add_argument(
        "--encoder",
        type=str,
        choices=["clip", "dinov3"],
        help="Encoder backend: 'clip' (OpenAI CLIP) or 'dinov3' (Meta DINOv2/v3) (default: clip)",
    )
    encoder_group.add_argument(
        "--temporal-weight",
        type=float,
        default=0.1,
        help="Temporal encoding weight α (default: 0.1, applies to both encoders)",
    )
    
    # CLIP-specific options
    clip_group = parser.add_argument_group("CLIP Encoder Options (when --encoder clip)")
    clip_group.add_argument(
        "--model",
        type=str,
        default="ViT-L/14",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],
        help="CLIP model variant (default: ViT-L/14)",
    )
    
    # DINOv3-specific options
    dinov3_group = parser.add_argument_group("DINOv3 Encoder Options (when --encoder dinov3)")
    dinov3_group.add_argument(
        "--dinov3-model-id",
        type=str,
        default="facebook/dinov2-base",
        help="Hugging Face model ID for DINOv3 (default: facebook/dinov2-base). "
             "Options: facebook/dinov2-small, facebook/dinov2-base, facebook/dinov2-large, facebook/dinov2-giant",
    )
    dinov3_group.add_argument(
        "--dinov3-revision",
        type=str,
        default=None,
        help="Model revision/commit hash for reproducibility (default: latest)",
    )
    dinov3_group.add_argument(
        "--dinov3-pooling",
        type=str,
        default="cls",
        choices=["cls", "mean"],
        help="Pooling strategy for DINOv3: 'cls' (CLS token) or 'mean' (mean of patches) (default: cls)",
    )
    
    # Selection options
    selection_group = parser.add_argument_group("Selection")
    selection_group.add_argument(
        "--selection-method",
        type=str,
        choices=["dpp", "kmeans", "hdbscan"],
        help="Selection method: 'dpp' (DPP diversity), 'kmeans' (K-means clustering), "
             "'hdbscan' (density-based clustering) (default: dpp)",
    )
    selection_group.add_argument(
        "-k", "--num-keyframes",
        type=int,
        help="Fixed number of keyframes (overrides adaptive K, not used for HDBSCAN)",
    )
    selection_group.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Entropy K scaling factor β (default: 1.0)",
    )
    selection_group.add_argument(
        "--k-min",
        type=int,
        default=3,
        help="Minimum keyframes (default: 3)",
    )
    selection_group.add_argument(
        "--k-max",
        type=int,
        default=250,
        help="Maximum keyframes (default: 50)",
    )
    selection_group.add_argument(
        "--min-gap",
        type=int,
        default=5,
        help="Minimum frame gap between keyframes (default: 5, set 0 to disable)",
    )
    
    # K-means specific options
    kmeans_group = parser.add_argument_group("K-means Options (when --selection-method kmeans)")
    kmeans_group.add_argument(
        "--kmeans-init",
        type=str,
        choices=["k-means++", "random"],
        default="k-means++",
        help="K-means initialization method (default: k-means++)",
    )
    kmeans_group.add_argument(
        "--kmeans-n-init",
        type=int,
        default=10,
        help="Number of K-means initializations (default: 10)",
    )
    kmeans_group.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=300,
        help="Maximum K-means iterations (default: 300)",
    )
    
    # HDBSCAN specific options
    hdbscan_group = parser.add_argument_group("HDBSCAN Options (when --selection-method hdbscan)")
    hdbscan_group.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=2,
        help="HDBSCAN minimum cluster size (default: 2)",
    )
    hdbscan_group.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="HDBSCAN minimum samples (default: None, same as min-cluster-size)",
    )
    hdbscan_group.add_argument(
        "--hdbscan-cluster-selection-epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster selection epsilon (default: 0.0)",
    )
    hdbscan_group.add_argument(
        "--hdbscan-cluster-selection-method",
        type=str,
        choices=["eom", "leaf"],
        default="eom",
        help="HDBSCAN cluster selection method (default: eom)",
    )
    
    # Ablation toggles
    ablation_group = parser.add_argument_group("Ablation Controls")
    ablation_group.add_argument(
        "--no-temporal",
        action="store_true",
        help="Disable temporal encoding",
    )
    ablation_group.add_argument(
        "--no-entropy-k",
        action="store_true",
        help="Disable entropy-based adaptive K",
    )
    ablation_group.add_argument(
        "--no-temporal-kernel",
        action="store_true",
        help="Disable temporal kernel in DPP",
    )
    ablation_group.add_argument(
        "--enable-motion",
        action="store_true",
        help="Enable motion awareness (optical flow)",
    )
    
    # Other options
    other_group = parser.add_argument_group("Other")
    other_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    other_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    other_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output except errors",
    )
    
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build pipeline config from arguments."""
    
    # Start with config file if provided
    if args.config:
        config_dict = load_config(args.config)
        config = PipelineConfig.from_dict(config_dict)
        return config
    else:
        config = PipelineConfig()
    
    # Override with CLI arguments
    if args.video:
        config.video_path = Path(args.video)
    if args.frames:
        config.frame_dir = Path(args.frames)
    
    config.output_dir = Path(args.output)
    config.save_intermediate = args.save_intermediate
    
    # Frame sampling
    config.frame_sampling.fps = args.fps
    config.frame_sampling.adaptive = args.adaptive_sampling
    
    # Encoder backend selection (only override if explicitly specified on CLI)
    if args.encoder is not None:
        config.encoder_backend = args.encoder  # type: ignore
    
    # CLIP encoder settings
    # These use argparse defaults, so they'll override config values.
    # This is acceptable behavior: CLI args take precedence over config file.
    config.clip_encoder.model_name = args.model
    config.clip_encoder.temporal_weight = args.temporal_weight
    
    # DINOv3 encoder settings
    config.dinov3_encoder.model_id = args.dinov3_model_id
    config.dinov3_encoder.temporal_weight = args.temporal_weight
    config.dinov3_encoder.pooling = args.dinov3_pooling  # type: ignore
    if args.dinov3_revision:
        config.dinov3_encoder.revision = args.dinov3_revision
    
    # Selection
    if args.num_keyframes:
        config.selector.fixed_k = args.num_keyframes
    config.selector.min_frame_gap = args.min_gap
    config.entropy_estimator.beta = args.beta
    config.entropy_estimator.k_min = args.k_min
    config.entropy_estimator.k_max = args.k_max
    
    # Selection method
    if args.selection_method:
        config.selector.method = args.selection_method  # type: ignore
    
    # K-means parameters
    if args.kmeans_init:
        config.selector.kmeans_init = args.kmeans_init  # type: ignore
    config.selector.kmeans_n_init = args.kmeans_n_init
    config.selector.kmeans_max_iter = args.kmeans_max_iter
    
    # HDBSCAN parameters
    config.selector.hdbscan_min_cluster_size = args.hdbscan_min_cluster_size
    if args.hdbscan_min_samples is not None:
        config.selector.hdbscan_min_samples = args.hdbscan_min_samples
    config.selector.hdbscan_cluster_selection_epsilon = args.hdbscan_cluster_selection_epsilon
    config.selector.hdbscan_cluster_selection_method = args.hdbscan_cluster_selection_method  # type: ignore
    
    # Ablation toggles
    config.use_temporal_encoding = not args.no_temporal
    config.use_entropy_k = not args.no_entropy_k
    config.use_temporal_kernel = not args.no_temporal_kernel
    config.motion.enabled = args.enable_motion
    
    # Other
    config.random_seed = args.seed
    config.verbose = args.verbose and not args.quiet
    
    return config


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not args.video and not args.frames and not args.config:
        logger.error("Must provide --video, --frames, or --config")
        sys.exit(1)
    
    # Build config
    config = build_config(args)
    
    # Validate we have input
    if config.video_path is None and config.frame_dir is None:
        logger.error("No input source specified. Use --video or --frames.")
        sys.exit(1)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("Keyframe Selection Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input: {config.video_path or config.frame_dir}")
    logger.info(f"Output: {config.output_dir}")
    
    # Log encoder-specific information
    if config.encoder_backend == "dinov3":
        logger.info(f"Encoder: DINOv3 ({config.dinov3_encoder.model_id})")
        logger.info(f"DINOv3 Pooling: {config.dinov3_encoder.pooling}")
        logger.info(f"Temporal Weight: {config.dinov3_encoder.temporal_weight}")
    else:
        logger.info(f"Encoder: CLIP ({config.clip_encoder.model_name})")
        logger.info(f"Temporal Weight: {config.clip_encoder.temporal_weight}")
    
    logger.info(f"FPS: {config.frame_sampling.fps}")
    logger.info(f"Temporal Encoding: {config.use_temporal_encoding}")
    logger.info(f"Entropy-based K: {config.use_entropy_k}")
    logger.info(f"Temporal Kernel: {config.use_temporal_kernel}")
    logger.info(f"Motion Awareness: {config.motion.enabled}")
    logger.info(f"Selection Method: {config.selector.method}")
    if config.selector.method == "dpp":
        logger.info(f"DPP Mode: {config.selector.mode}")
    elif config.selector.method == "kmeans":
        logger.info(f"K-means init: {config.selector.kmeans_init}, n_init: {config.selector.kmeans_n_init}")
    elif config.selector.method == "hdbscan":
        logger.info(f"HDBSCAN min_cluster_size: {config.selector.hdbscan_min_cluster_size}")
    if config.selector.fixed_k:
        logger.info(f"Fixed K: {config.selector.fixed_k}")
    logger.info("=" * 60)
    
    # Run pipeline
    try:
        pipeline = KeyframeSelectionPipeline(config)
        result = pipeline.run()
        
        # Print results
        logger.info("")
        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info(f"Selected {len(result.keyframes)} keyframes")
        logger.info(f"Frame indices: {result.keyframes.indices.tolist()}")
        logger.info(f"Timestamps: {[f'{t:.2f}s' for t in result.keyframes.timestamps]}")
        
        if result.keyframes.paths:
            logger.info(f"Saved to: {config.output_dir / 'keyframes'}")
        
        # Save config used
        save_config(
            config.to_dict(),
            config.output_dir / "config_used.yaml"
        )
        
        logger.info("")
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
