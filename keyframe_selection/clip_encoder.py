"""
CLIP embedding extraction with temporal encoding.

Implements the temporal-aware CLIP encoder that produces semantically rich
embeddings with optional temporal position encoding.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from numpy.typing import NDArray

from .config import CLIPEncoderConfig
from .types import EmbeddingBatch, FrameBatch, FrameData
from .utils.timing import Timer

logger = logging.getLogger(__name__)

# Lazy load CLIP to avoid import errors if not installed
_clip_model = None
_clip_preprocess = None


def _load_clip(model_name: str, device: str) -> Tuple:
    """Lazy load CLIP model."""
    global _clip_model, _clip_preprocess
    
    import clip
    
    logger.info(f"Loading CLIP model: {model_name} on {device}")
    _clip_model, _clip_preprocess = clip.load(model_name, device=device)
    _clip_model.eval()
    
    return _clip_model, _clip_preprocess


class CLIPTemporalEncoder:
    """
    CLIP-based encoder with temporal position encoding.
    
    Extracts semantic embeddings from frames and optionally augments them
    with normalized temporal position information.
    
    Features:
        - Batch processing for efficiency
        - GPU acceleration with mixed precision
        - Temporal encoding with configurable weight (H4)
        - L2 normalization of embeddings
    """
    
    def __init__(self, config: Optional[CLIPEncoderConfig] = None):
        """
        Initialize CLIP encoder.
        
        Args:
            config: Encoder configuration. Uses defaults if None.
        """
        self.config = config or CLIPEncoderConfig()
        
        # Auto-detect device
        if self.config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        
        self.model = None
        self.preprocess = None
        self._embedding_dim: Optional[int] = None
    
    def _ensure_model_loaded(self) -> None:
        """Load model if not already loaded."""
        if self.model is None:
            self.model, self.preprocess = _load_clip(
                self.config.model_name, 
                self.device
            )
            
            # Determine embedding dimension
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                if self.config.use_fp16 and self.device == "cuda":
                    dummy = dummy.half()
                    out = self.model.encode_image(dummy)
                else:
                    out = self.model.encode_image(dummy.float())
                self._embedding_dim = out.shape[1]
            
            logger.info(f"CLIP embedding dimension: {self._embedding_dim}")
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        self._ensure_model_loaded()
        return self._embedding_dim
    
    def encode(
        self,
        frame_batch: FrameBatch,
        add_temporal: bool = True,
    ) -> EmbeddingBatch:
        """
        Encode frames into CLIP embeddings.
        
        Args:
            frame_batch: Batch of frames to encode.
            add_temporal: Whether to add temporal encoding.
        
        Returns:
            EmbeddingBatch with embeddings and metadata.
        """
        self._ensure_model_loaded()
        
        if len(frame_batch) == 0:
            return EmbeddingBatch(
                embeddings=np.array([], dtype=np.float32).reshape(0, self.embedding_dim),
                timestamps=np.array([]),
                frame_indices=np.array([]),
            )
        
        logger.info(f"Encoding {len(frame_batch)} frames with CLIP")
        
        # Extract base embeddings
        with Timer("clip_encoding"):
            embeddings = self._batch_encode(frame_batch.images)
        
        # Gather metadata (optimized: direct array access)
        timestamps = frame_batch.timestamps
        frame_indices = np.array([f.frame_index for f in frame_batch.frames], dtype=np.int64)
        
        # Add temporal encoding if requested
        temporal_embeddings = None
        if add_temporal and self.config.temporal_weight > 0:
            temporal_embeddings = self._add_temporal_encoding(
                embeddings,
                frame_batch.normalized_timestamps,
            )
            logger.info(
                f"Added temporal encoding (α={self.config.temporal_weight}), "
                f"dim: {embeddings.shape[1]} -> {temporal_embeddings.shape[1]}"
            )
        
        return EmbeddingBatch(
            embeddings=embeddings,
            temporal_embeddings=temporal_embeddings,
            timestamps=timestamps,
            frame_indices=frame_indices,
        )
    
    def encode_images(
        self,
        images: List[np.ndarray],
        timestamps: Optional[np.ndarray] = None,
        video_duration: Optional[float] = None,
        add_temporal: bool = True,
    ) -> EmbeddingBatch:
        """
        Encode a list of images directly.
        
        Args:
            images: List of images (BGR format from OpenCV).
            timestamps: Optional timestamps for each image.
            video_duration: Total video duration for normalization.
            add_temporal: Whether to add temporal encoding.
        
        Returns:
            EmbeddingBatch with embeddings.
        """
        self._ensure_model_loaded()
        
        if len(images) == 0:
            return EmbeddingBatch(
                embeddings=np.array([], dtype=np.float32).reshape(0, self.embedding_dim),
                timestamps=np.array([]),
                frame_indices=np.array([]),
            )
        
        # Encode images
        embeddings = self._batch_encode(images)
        
        # Handle timestamps
        if timestamps is None:
            timestamps = np.arange(len(images), dtype=np.float64)
        
        frame_indices = np.arange(len(images), dtype=np.int64)
        
        # Temporal encoding
        temporal_embeddings = None
        if add_temporal and self.config.temporal_weight > 0:
            # Optimized: simplified normalization logic
            if video_duration is not None and video_duration > 0:
                norm_timestamps = timestamps / video_duration
            else:
                max_t = timestamps.max() if len(timestamps) > 0 else 1.0
                norm_timestamps = timestamps / max_t if max_t > 0 else timestamps
            
            temporal_embeddings = self._add_temporal_encoding(embeddings, norm_timestamps)
        
        return EmbeddingBatch(
            embeddings=embeddings,
            temporal_embeddings=temporal_embeddings,
            timestamps=timestamps,
            frame_indices=frame_indices,
        )
    
    def _batch_encode(self, images: List[np.ndarray]) -> NDArray[np.float32]:
        """
        Encode images in batches.
        
        Args:
            images: List of images (BGR format).
        
        Returns:
            Normalized embeddings of shape (N, D).
        """
        all_embeddings = []
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                
                # Preprocess images (optimized: reduced allocations)
                preprocessed = []
                for img in batch_images:
                    # Convert BGR to RGB (optimized: in-place slicing)
                    if img.ndim == 3 and img.shape[2] == 3:
                        # Use negative stride for BGR->RGB without copy
                        img_rgb = np.ascontiguousarray(img[:, :, ::-1])
                    else:
                        img_rgb = img
                    
                    # Convert to PIL and preprocess
                    pil_img = Image.fromarray(img_rgb)
                    tensor = self.preprocess(pil_img)
                    preprocessed.append(tensor)
                
                # Stack into batch
                batch_tensor = torch.stack(preprocessed).to(self.device)
                
                # Mixed precision (optimized: single conditional)
                if self.config.use_fp16 and self.device == "cuda":
                    batch_tensor = batch_tensor.half()
                
                # Encode and normalize in one step
                features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(features.cpu().float().numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)
    
    def _add_temporal_encoding(
        self,
        embeddings: NDArray[np.float32],
        normalized_timestamps: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        """
        Add temporal position encoding to embeddings.
        
        Uses simple concatenation: f̃_t = concat(f_t, α * t_norm)
        
        Args:
            embeddings: Base embeddings of shape (N, D).
            normalized_timestamps: Timestamps in [0, 1] of shape (N,).
        
        Returns:
            Temporal embeddings of shape (N, D+1).
        """
        # Optimized: compute scaled timestamps directly as column vector
        alpha = self.config.temporal_weight
        temporal_component = (alpha * normalized_timestamps.astype(np.float32)).reshape(-1, 1)
        
        return np.hstack([embeddings, temporal_component])
    
    def encode_single(self, image: np.ndarray) -> NDArray[np.float32]:
        """
        Encode a single image.
        
        Args:
            image: Image in BGR format.
        
        Returns:
            Normalized embedding of shape (D,).
        """
        embeddings = self._batch_encode([image])
        return embeddings[0]


def extract_clip_features(
    frame_dir: str,
    model_name: str = "ViT-L/14",
    temporal_weight: float = 0.1,
) -> Tuple[NDArray[np.float32], List[str]]:
    """
    Convenience function for CLIP feature extraction from a directory.
    
    Args:
        frame_dir: Directory containing frame images.
        model_name: CLIP model variant.
        temporal_weight: Temporal encoding weight (H4).
    
    Returns:
        Tuple of (embeddings array, frame paths list).
    """
    from .frame_sampling import FrameSampler
    
    config = CLIPEncoderConfig(model_name=model_name, temporal_weight=temporal_weight)
    encoder = CLIPTemporalEncoder(config)
    
    sampler = FrameSampler()
    frame_batch = sampler.load_frames_from_directory(frame_dir)
    
    embedding_batch = encoder.encode(frame_batch)
    
    # Optimized: single-pass list comprehension
    paths = [str(f.path) for f in frame_batch.frames if f.path is not None]
    
    return embedding_batch.effective_embeddings, paths
