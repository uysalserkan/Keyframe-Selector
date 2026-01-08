#!/usr/bin/env python3
"""
CLI entry point for the keyframe selection pipeline.

Usage:
    # From video file
    python run_pipeline.py --video /path/to/video.mp4 --output ./output
    
    # From pre-extracted frames
    python run_pipeline.py --frames ./frames --output ./output
    
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
        description="Temporal-Aware CLIP + DPP Keyframe Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file
  python run_pipeline.py --video video.mp4 --output results/
  
  # Process pre-extracted frames
  python run_pipeline.py --frames ./frames/ --output results/
  
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
    
    # CLIP options
    clip_group = parser.add_argument_group("CLIP Encoder")
    clip_group.add_argument(
        "--model",
        type=str,
        default="ViT-L/14",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],
        help="CLIP model variant (default: ViT-L/14)",
    )
    clip_group.add_argument(
        "--temporal-weight",
        type=float,
        default=0.1,
        help="Temporal encoding weight α (default: 0.1)",
    )
    
    # Selection options
    selection_group = parser.add_argument_group("Selection")
    selection_group.add_argument(
        "-k", "--num-keyframes",
        type=int,
        help="Fixed number of keyframes (overrides adaptive K)",
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
        default=50,
        help="Maximum keyframes (default: 50)",
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
    
    # CLIP encoder
    config.clip_encoder.model_name = args.model
    config.clip_encoder.temporal_weight = args.temporal_weight
    
    # Selection
    if args.num_keyframes:
        config.selector.fixed_k = args.num_keyframes
    config.entropy_estimator.beta = args.beta
    config.entropy_estimator.k_min = args.k_min
    config.entropy_estimator.k_max = args.k_max
    
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
    logger.info(f"CLIP Model: {config.clip_encoder.model_name}")
    logger.info(f"FPS: {config.frame_sampling.fps}")
    logger.info(f"Temporal Encoding: {config.use_temporal_encoding}")
    logger.info(f"Entropy-based K: {config.use_entropy_k}")
    logger.info(f"Temporal Kernel: {config.use_temporal_kernel}")
    logger.info(f"Motion Awareness: {config.motion.enabled}")
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
