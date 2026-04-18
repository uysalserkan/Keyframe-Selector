"""
Frame sampling module for extracting frames from video files.

Supports fixed FPS sampling and adaptive scene-change-based sampling.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .config import FrameSamplingConfig
from .types import FrameBatch, FrameData
from .utils.threading_env import get_last_configured_num_threads
from .utils.timing import Timer

logger = logging.getLogger(__name__)


class FrameSampler:
    """
    Extract frames from video with configurable sampling strategies.
    
    Supports:
        - Fixed FPS sampling
        - Adaptive sampling based on scene changes
        - Lazy iteration for memory efficiency
    """
    
    def __init__(self, config: Optional[FrameSamplingConfig] = None):
        """
        Initialize frame sampler.
        
        Args:
            config: Sampling configuration. Uses defaults if None.
        """
        self.config = config or FrameSamplingConfig()
    
    def sample_video(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> FrameBatch:
        """
        Sample frames from a video file.
        
        Args:
            video_path: Path to input video.
            output_dir: Optional directory to save extracted frames.
        
        Returns:
            FrameBatch containing sampled frames with metadata.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        
        try:
            # Get video metadata
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / source_fps if source_fps > 0 else 0.0
            
            logger.info(
                f"Video: {video_path.name} | "
                f"FPS: {source_fps:.2f} | "
                f"Frames: {total_frames} | "
                f"Duration: {duration:.2f}s"
            )
            
            # Calculate frame interval (pre-compute to avoid repeated operations)
            frame_interval = max(1, int(source_fps / self.config.fps)) if self.config.fps > 0 else 1
            
            logger.info(
                f"Sampling at {self.config.fps} FPS (every {frame_interval} frames)"
            )
            
            # Prepare output directory
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            frames: List[FrameData] = []
            prev_frame = None if self.config.adaptive else False  # False = disabled
            frame_idx = 0
            saved_idx = 0
            inv_source_fps = 1.0 / source_fps if source_fps > 0 else 0.0  # Pre-compute
            
            with Timer("frame_sampling"):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Check if we should sample this frame
                    should_sample = False
                    
                    # Adaptive sampling: detect scene changes (only if enabled)
                    if prev_frame is not False and prev_frame is not None:
                        change_score = self._compute_frame_difference(prev_frame, frame)
                        if change_score > self.config.adaptive_threshold:
                            should_sample = True
                    
                    # Fixed FPS sampling
                    if frame_idx % frame_interval == 0:
                        should_sample = True
                    
                    if should_sample:
                        timestamp = frame_idx * inv_source_fps
                        
                        # Save to disk if requested
                        frame_path = None
                        if output_dir is not None:
                            frame_path = output_dir / f"frame_{saved_idx:04d}.{self.config.output_format}"
                            self._save_frame(frame, frame_path)
                            # Disk-backed frames: do not retain full BGR buffers in RAM
                            # (avoids OOM on long / high-res videos; encoders load in batches).
                            frames.append(
                                FrameData(
                                    image=None,
                                    timestamp=timestamp,
                                    frame_index=frame_idx,
                                    path=frame_path,
                                )
                            )
                        else:
                            frames.append(
                                FrameData(
                                    image=frame,
                                    timestamp=timestamp,
                                    frame_index=frame_idx,
                                    path=None,
                                )
                            )
                        saved_idx += 1
                    
                    # Only store prev_frame if adaptive mode is enabled
                    if prev_frame is not False:
                        prev_frame = frame
                    frame_idx += 1
            
            logger.info(f"Extracted {len(frames)} frames")
            if (
                frames
                and frames[0].path is not None
                and frames[0].image is None
            ):
                logger.info(
                    "Disk-backed frame sampling: %d frames on disk (low RAM); "
                    "encoders load images in batches.",
                    len(frames),
                )

            return FrameBatch(
                frames=frames,
                video_duration=duration,
                source_fps=source_fps,
                source_path=video_path,
            )
        
        finally:
            cap.release()
    
    def sample_frames_lazy(
        self,
        video_path: Union[str, Path],
    ) -> Generator[FrameData, None, None]:
        """
        Lazily yield frames from video without loading all into memory.
        
        Args:
            video_path: Path to input video.
        
        Yields:
            FrameData for each sampled frame.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        
        try:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(source_fps / self.config.fps)) if self.config.fps > 0 else 1
            inv_source_fps = 1.0 / source_fps if source_fps > 0 else 0.0
            
            prev_frame = None if self.config.adaptive else False
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                should_sample = False
                
                # Only compute frame difference if adaptive mode is enabled
                if prev_frame is not False and prev_frame is not None:
                    change_score = self._compute_frame_difference(prev_frame, frame)
                    if change_score > self.config.adaptive_threshold:
                        should_sample = True
                
                if frame_idx % frame_interval == 0:
                    should_sample = True
                
                if should_sample:
                    timestamp = frame_idx * inv_source_fps
                    yield FrameData(
                        image=frame,
                        timestamp=timestamp,
                        frame_index=frame_idx,
                    )
                
                if prev_frame is not False:
                    prev_frame = frame
                frame_idx += 1
        
        finally:
            cap.release()
    
    def load_frames_from_directory(
        self,
        frame_dir: Union[str, Path],
        pattern: str = "*.jpg",
        video_fps: float = 1.0,
        max_workers: Optional[int] = None,
    ) -> FrameBatch:
        """
        Load pre-extracted frames from a directory.
        
        Args:
            frame_dir: Directory containing frame images.
            pattern: Glob pattern for frame files.
            video_fps: Assumed FPS for timestamp calculation.
        
        Returns:
            FrameBatch with loaded frames.
        """
        frame_dir = Path(frame_dir)
        if not frame_dir.exists():
            raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
        
        # Find and sort frame files (optimized: single traversal with fallback)
        extensions = [pattern, "*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.jpg", "*.JPG", "*.bmp"]
        frame_paths = []
        
        for ext in extensions:
            frame_paths = sorted(frame_dir.glob(ext))
            if frame_paths:
                break
        
        if not frame_paths:
            raise ValueError(f"No frames found in {frame_dir}")
        
        logger.info(f"Loading {len(frame_paths)} frames from {frame_dir}")
        
        inv_video_fps = 1.0 / video_fps if video_fps > 0 else 0.0
        cfg = self.config
        workers_cap = max_workers if max_workers is not None else get_last_configured_num_threads()
        workers_cap = max(1, min(workers_cap, 32, len(frame_paths)))
        use_parallel = (
            cfg.parallel_load_frames
            and len(frame_paths) >= cfg.min_frames_for_parallel_load
            and workers_cap > 1
        )

        frames: List[FrameData] = []
        with Timer("load_frames"):
            if use_parallel:
                prev_cv2 = cv2.getNumThreads()
                cv2.setNumThreads(1)
                try:
                    slots: List[Optional[FrameData]] = [None] * len(frame_paths)
                    paths_list = frame_paths

                    def _load_at_index(i: int) -> None:
                        path = paths_list[i]
                        image = cv2.imread(str(path))
                        if image is None:
                            logger.warning(f"Could not load: {path}")
                            return
                        slots[i] = FrameData(
                            image=image,
                            timestamp=i * inv_video_fps,
                            frame_index=i,
                            path=path,
                        )

                    with ThreadPoolExecutor(max_workers=workers_cap) as ex:
                        list(ex.map(_load_at_index, range(len(paths_list))))
                    frames = [f for f in slots if f is not None]
                finally:
                    cv2.setNumThreads(prev_cv2)
                logger.debug(
                    "Parallel frame load: %d files, %d workers",
                    len(frame_paths),
                    workers_cap,
                )
            else:
                for idx, path in enumerate(frame_paths):
                    image = cv2.imread(str(path))
                    if image is None:
                        logger.warning(f"Could not load: {path}")
                        continue
                    frames.append(
                        FrameData(
                            image=image,
                            timestamp=idx * inv_video_fps,
                            frame_index=idx,
                            path=path,
                        )
                    )
        
        duration = len(frames) / video_fps if video_fps > 0 else 0.0
        
        return FrameBatch(
            frames=frames,
            video_duration=duration,
            source_fps=video_fps,
            source_path=frame_dir,
        )
    
    def _compute_frame_difference(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> float:
        """
        Compute difference score between two frames for scene detection.
        
        Uses histogram comparison for efficiency.
        
        Args:
            frame1: First frame (BGR).
            frame2: Second frame (BGR).
        
        Returns:
            Difference score (higher = more different).
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute and normalize histograms in-place
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Normalize in-place (dst=src for in-place operation)
        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Compare histograms (higher = more different)
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
        
        return float(diff)
    
    def _save_frame(
        self,
        frame: np.ndarray,
        path: Path,
    ) -> None:
        """Save a frame to disk."""
        if path.suffix.lower() in (".jpg", ".jpeg"):
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        else:
            cv2.imwrite(str(path), frame)


def extract_frames(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    fps: float = 1.0,
    adaptive: bool = False,
) -> FrameBatch:
    """
    Convenience function for frame extraction.
    
    Args:
        video_path: Path to video file.
        output_dir: Directory to save frames.
        fps: Frames per second to extract.
        adaptive: Enable adaptive scene-change sampling.
    
    Returns:
        FrameBatch with extracted frames.
    """
    config = FrameSamplingConfig(fps=fps, adaptive=adaptive)
    sampler = FrameSampler(config)
    return sampler.sample_video(video_path, output_dir)
