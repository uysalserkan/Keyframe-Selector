"""
Frame sampling module for extracting frames from video files.

Supports fixed FPS sampling and adaptive scene-change-based sampling.
"""

import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .config import FrameSamplingConfig
from .types import FrameBatch, FrameData
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
            
            # Calculate frame interval
            frame_interval = int(source_fps / self.config.fps) if self.config.fps > 0 else 1
            frame_interval = max(1, frame_interval)
            
            logger.info(
                f"Sampling at {self.config.fps} FPS (every {frame_interval} frames)"
            )
            
            # Prepare output directory
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            frames: List[FrameData] = []
            prev_frame = None
            frame_idx = 0
            saved_idx = 0
            
            with Timer("frame_sampling"):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Check if we should sample this frame
                    should_sample = False
                    
                    if self.config.adaptive and prev_frame is not None:
                        # Adaptive sampling: detect scene changes
                        change_score = self._compute_frame_difference(prev_frame, frame)
                        if change_score > self.config.adaptive_threshold:
                            should_sample = True
                    
                    # Fixed FPS sampling
                    if frame_idx % frame_interval == 0:
                        should_sample = True
                    
                    if should_sample:
                        timestamp = frame_idx / source_fps if source_fps > 0 else 0.0
                        
                        # Save to disk if requested
                        frame_path = None
                        if output_dir is not None:
                            frame_path = output_dir / f"frame_{saved_idx:04d}.{self.config.output_format}"
                            self._save_frame(frame, frame_path)
                        
                        frames.append(FrameData(
                            image=frame,
                            timestamp=timestamp,
                            frame_index=frame_idx,
                            path=frame_path,
                        ))
                        saved_idx += 1
                    
                    prev_frame = frame
                    frame_idx += 1
            
            logger.info(f"Extracted {len(frames)} frames")
            
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
            frame_interval = int(source_fps / self.config.fps) if self.config.fps > 0 else 1
            frame_interval = max(1, frame_interval)
            
            prev_frame = None
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                should_sample = False
                
                if self.config.adaptive and prev_frame is not None:
                    change_score = self._compute_frame_difference(prev_frame, frame)
                    if change_score > self.config.adaptive_threshold:
                        should_sample = True
                
                if frame_idx % frame_interval == 0:
                    should_sample = True
                
                if should_sample:
                    timestamp = frame_idx / source_fps if source_fps > 0 else 0.0
                    yield FrameData(
                        image=frame,
                        timestamp=timestamp,
                        frame_index=frame_idx,
                    )
                
                prev_frame = frame
                frame_idx += 1
        
        finally:
            cap.release()
    
    def load_frames_from_directory(
        self,
        frame_dir: Union[str, Path],
        pattern: str = "*.jpg",
        video_fps: float = 1.0,
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
        
        # Find and sort frame files
        frame_paths = sorted(frame_dir.glob(pattern))
        
        if not frame_paths:
            # Try other common extensions
            for ext in ["*.png", "*.jpeg", "*.bmp"]:
                frame_paths = sorted(frame_dir.glob(ext))
                if frame_paths:
                    break
        
        if not frame_paths:
            raise ValueError(f"No frames found in {frame_dir}")
        
        logger.info(f"Loading {len(frame_paths)} frames from {frame_dir}")
        
        frames: List[FrameData] = []
        
        with Timer("load_frames"):
            for idx, path in enumerate(frame_paths):
                # Load image with OpenCV (BGR format)
                image = cv2.imread(str(path))
                if image is None:
                    logger.warning(f"Could not load: {path}")
                    continue
                
                timestamp = idx / video_fps
                
                frames.append(FrameData(
                    image=image,
                    timestamp=timestamp,
                    frame_index=idx,
                    path=path,
                ))
        
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
        
        # Compute histograms
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Normalize
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        
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
