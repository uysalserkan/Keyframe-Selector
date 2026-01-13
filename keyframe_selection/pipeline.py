"""
Main pipeline orchestration.

Combines all modules into a cohesive end-to-end keyframe selection pipeline
with support for ablation studies and configurable stages.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .config import PipelineConfig
from .types import (
    EmbeddingBatch,
    FrameBatch,
    KeyframeResult,
    PipelineResult,
)
from .frame_sampling import FrameSampler
from .clip_encoder import CLIPTemporalEncoder
from .temporal_analysis import TemporalDeltaComputer
from .entropy_estimator import EntropyKEstimator
from .dpp_kernel import DPPKernelBuilder
from .selector import DPPSelector
from .motion import MotionEncoder
from .utils.timing import Timer
from .utils.io import set_global_seed, ensure_dir

logger = logging.getLogger(__name__)


class KeyframeSelectionPipeline:
    """
    End-to-end keyframe selection pipeline.
    
    Orchestrates all stages:
        1. Frame sampling (from video or directory)
        2. CLIP embedding extraction with temporal encoding
        3. Optional motion feature extraction
        4. Temporal change analysis
        5. Entropy-based K estimation
        6. DPP kernel construction
        7. Diversity-aware subset selection
    
    Supports ablation toggles for:
        - Temporal encoding (H4)
        - Entropy-based adaptive K
        - Temporal kernel in DPP
        - Motion awareness (H10)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()
        
        # Set global seed for reproducibility
        set_global_seed(self.config.random_seed)
        
        # Initialize modules lazily
        self._frame_sampler: Optional[FrameSampler] = None
        self._clip_encoder: Optional[CLIPTemporalEncoder] = None
        self._motion_encoder: Optional[MotionEncoder] = None
        self._temporal_analyzer: Optional[TemporalDeltaComputer] = None
        self._entropy_estimator: Optional[EntropyKEstimator] = None
        self._kernel_builder: Optional[DPPKernelBuilder] = None
        self._selector: Optional[DPPSelector] = None
    
    @property
    def frame_sampler(self) -> FrameSampler:
        if self._frame_sampler is None:
            self._frame_sampler = FrameSampler(self.config.frame_sampling)
        return self._frame_sampler
    
    @property
    def clip_encoder(self) -> CLIPTemporalEncoder:
        if self._clip_encoder is None:
            self._clip_encoder = CLIPTemporalEncoder(self.config.clip_encoder)
        return self._clip_encoder
    
    @property
    def motion_encoder(self) -> MotionEncoder:
        if self._motion_encoder is None:
            self._motion_encoder = MotionEncoder(self.config.motion)
        return self._motion_encoder
    
    @property
    def temporal_analyzer(self) -> TemporalDeltaComputer:
        if self._temporal_analyzer is None:
            self._temporal_analyzer = TemporalDeltaComputer(self.config.temporal_analysis)
        return self._temporal_analyzer
    
    @property
    def entropy_estimator(self) -> EntropyKEstimator:
        if self._entropy_estimator is None:
            self._entropy_estimator = EntropyKEstimator(self.config.entropy_estimator)
        return self._entropy_estimator
    
    @property
    def kernel_builder(self) -> DPPKernelBuilder:
        if self._kernel_builder is None:
            self._kernel_builder = DPPKernelBuilder(self.config.dpp_kernel)
        return self._kernel_builder
    
    @property
    def selector(self) -> DPPSelector:
        if self._selector is None:
            self._selector = DPPSelector(self.config.selector)
        return self._selector
    
    def run(
        self,
        video_path: Optional[Union[str, Path]] = None,
        frame_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> PipelineResult:
        """
        Run the complete keyframe selection pipeline.
        
        Args:
            video_path: Path to video file (extracts frames).
            frame_dir: Path to directory with pre-extracted frames.
            output_dir: Directory to save results.
        
        Returns:
            PipelineResult with keyframes and intermediate results.
        
        Raises:
            ValueError: If neither video_path nor frame_dir is provided.
        """
        # Resolve paths from config if not provided
        video_path = video_path or self.config.video_path
        frame_dir = frame_dir or self.config.frame_dir
        output_dir = output_dir or self.config.output_dir
        
        if video_path is None and frame_dir is None:
            raise ValueError("Either video_path or frame_dir must be provided")
        
        timing = {}
        
        # Stage 1: Frame sampling
        with Timer("1_frame_sampling", log=self.config.verbose) as t:
            if video_path is not None:
                frame_batch = self.frame_sampler.sample_video(
                    video_path,
                    output_dir=output_dir / "frames" if output_dir else None,
                )
            else:
                frame_batch = self.frame_sampler.load_frames_from_directory(
                    frame_dir,
                    video_fps=self.config.frame_sampling.fps,
                )
        timing["frame_sampling"] = t.elapsed
        
        logger.info(f"Loaded {len(frame_batch)} frames")
        
        # Stage 2: CLIP encoding with temporal encoding
        with Timer("2_clip_encoding", log=self.config.verbose) as t:
            embedding_batch = self.clip_encoder.encode(
                frame_batch,
                add_temporal=self.config.use_temporal_encoding,
            )
        timing["clip_encoding"] = t.elapsed
        
        # Stage 2b: Optional motion encoding
        if self.config.motion.enabled:
            with Timer("2b_motion_encoding", log=self.config.verbose) as t:
                motion_features = self.motion_encoder.compute_motion_features(frame_batch)
                embedding_batch.motion_features = motion_features
                
                # Augment embeddings with motion if using temporal embeddings
                if embedding_batch.temporal_embeddings is not None:
                    embedding_batch.temporal_embeddings = self.motion_encoder.augment_embeddings(
                        embedding_batch.temporal_embeddings,
                        motion_features,
                    )
                else:
                    embedding_batch.temporal_embeddings = self.motion_encoder.augment_embeddings(
                        embedding_batch.embeddings,
                        motion_features,
                    )
            timing["motion_encoding"] = t.elapsed
        
        # Stage 3: Temporal change analysis
        with Timer("3_temporal_analysis", log=self.config.verbose) as t:
            temporal_result = self.temporal_analyzer.compute(embedding_batch)
        timing["temporal_analysis"] = t.elapsed
        
        # Stage 4: K determination (fixed or entropy-based)
        with Timer("4_entropy_estimation", log=self.config.verbose) as t:
            # If user specified fixed_k, always use it (override entropy)
            if self.config.selector.fixed_k is not None:
                k = self.config.selector.fixed_k
                entropy_result = None
                logger.info(f"Using fixed K={k} (--no-entropy-k not needed when -k is specified)")
            elif self.config.use_entropy_k:
                entropy_result = self.entropy_estimator.estimate(
                    embedding_batch,
                    temporal_result,
                )
                k = entropy_result.recommended_k
            else:
                # Fallback default
                entropy_result = None
                k = 10
        timing["entropy_estimation"] = t.elapsed
        
        logger.info(f"Target keyframe count K={k}")
        
        # Stage 5: DPP kernel construction
        with Timer("5_dpp_kernel", log=self.config.verbose) as t:
            dpp_kernel = self.kernel_builder.build(
                embedding_batch,
                video_duration=frame_batch.video_duration,
                use_temporal=self.config.use_temporal_kernel,
            )
        timing["dpp_kernel"] = t.elapsed
        
        # Stage 6: Keyframe selection
        with Timer("6_selection", log=self.config.verbose) as t:
            # Pass change points if configured to include them
            change_points = None
            if self.config.selector.include_change_points and temporal_result is not None:
                change_points = temporal_result.change_points
            
            keyframe_result = self.selector.select_from_embeddings(
                embedding_batch,
                dpp_kernel,
                k=k,
                change_points=change_points,
            )
        timing["selection"] = t.elapsed
        
        # Add frame paths to result
        if len(frame_batch.frames) > 0 and frame_batch.frames[0].path is not None:
            keyframe_result.paths = [
                frame_batch.frames[i].path for i in keyframe_result.indices
            ]
        
        # Save results if output_dir specified
        if output_dir is not None:
            output_dir = ensure_dir(output_dir)
            self._save_results(keyframe_result, frame_batch, output_dir)
        
        # Build final result
        result = PipelineResult(
            keyframes=keyframe_result,
            timing=timing,
            config=self.config.to_dict() if self.config.save_intermediate else None,
        )
        
        if self.config.save_intermediate:
            result.frame_batch = frame_batch
            result.embedding_batch = embedding_batch
            result.temporal_analysis = temporal_result
            result.entropy_result = entropy_result
            result.dpp_kernel = dpp_kernel
        
        logger.info(f"Pipeline complete. Selected {len(keyframe_result)} keyframes.")
        logger.info(Timer.summary())
        
        return result
    
    def _save_results(
        self,
        keyframe_result: KeyframeResult,
        frame_batch: FrameBatch,
        output_dir: Path,
    ) -> None:
        """Save keyframe results to output directory."""
        import shutil
        
        # Create keyframes directory
        keyframes_dir = ensure_dir(output_dir / "keyframes")
        
        # Copy selected frames
        for i, idx in enumerate(keyframe_result.indices):
            frame_data = frame_batch.frames[idx]
            
            if frame_data.path is not None:
                # Copy from original path
                ext = frame_data.path.suffix
                dst = keyframes_dir / f"keyframe_{i:04d}{ext}"
                shutil.copy(frame_data.path, dst)
            else:
                # Save from memory
                import cv2
                dst = keyframes_dir / f"keyframe_{i:04d}.jpg"
                cv2.imwrite(str(dst), frame_data.image)
        
        # Save metadata
        metadata = {
            "selected_indices": keyframe_result.indices.tolist(),
            "timestamps": keyframe_result.timestamps.tolist(),
            "k": len(keyframe_result),
            "total_frames": len(frame_batch),
            "metadata": keyframe_result.metadata,
        }
        
        import json
        with open(output_dir / "keyframes.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {len(keyframe_result)} keyframes to {keyframes_dir}")


def run_pipeline(
    video_path: Optional[str] = None,
    frame_dir: Optional[str] = None,
    output_dir: str = "output",
    fps: float = 1.0,
    k: Optional[int] = None,
    use_temporal: bool = True,
    use_entropy_k: bool = True,
    use_motion: bool = False,
    model_name: str = "ViT-L/14",
    seed: int = 42,
) -> PipelineResult:
    """
    Convenience function to run the pipeline with common options.
    
    Args:
        video_path: Path to video file.
        frame_dir: Path to frame directory.
        output_dir: Output directory.
        fps: Frame sampling rate.
        k: Fixed keyframe count (None = adaptive).
        use_temporal: Enable temporal encoding.
        use_entropy_k: Enable entropy-based K.
        use_motion: Enable motion features.
        model_name: CLIP model name.
        seed: Random seed.
    
    Returns:
        PipelineResult with selected keyframes.
    """
    from .config import (
        PipelineConfig,
        FrameSamplingConfig,
        CLIPEncoderConfig,
        SelectorConfig,
        MotionConfig,
    )
    
    config = PipelineConfig(
        video_path=Path(video_path) if video_path else None,
        frame_dir=Path(frame_dir) if frame_dir else None,
        output_dir=Path(output_dir),
        frame_sampling=FrameSamplingConfig(fps=fps),
        clip_encoder=CLIPEncoderConfig(model_name=model_name),
        selector=SelectorConfig(fixed_k=k, seed=seed),
        motion=MotionConfig(enabled=use_motion),
        use_temporal_encoding=use_temporal,
        use_entropy_k=use_entropy_k,
        use_temporal_kernel=use_temporal,
        random_seed=seed,
        save_intermediate=False,
    )
    
    pipeline = KeyframeSelectionPipeline(config)
    return pipeline.run()
