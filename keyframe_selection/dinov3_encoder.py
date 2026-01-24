"""
DINOv3/DINOv2 embedding extraction with temporal encoding.

Implements a temporal-aware DINOv3 encoder that produces semantically rich
embeddings with optional temporal position encoding, mirroring the CLIP encoder API.

Uses Hugging Face Transformers for safe model loading (no trust_remote_code).
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from numpy.typing import NDArray

from .config import DINOv3EncoderConfig
from .types import EmbeddingBatch, FrameBatch, FrameData
from .utils.timing import Timer

logger = logging.getLogger(__name__)

# Lazy load transformers to avoid import errors if not installed
_dinov3_model = None
_dinov3_processor = None


def _load_dinov3(
    model_id: str,
    device: str,
    revision: Optional[str] = None,
) -> Tuple:
    """
    Lazy load DINOv3/DINOv2 model from Hugging Face Hub.
    
    Uses safe loading practices:
    - trust_remote_code=False (no arbitrary code execution)
    - use_safetensors=True when available (safer weight format)
    
    Args:
        model_id: Hugging Face model identifier.
        device: Target device (cuda/cpu).
        revision: Optional model revision for reproducibility.
    
    Returns:
        Tuple of (model, processor).
    """
    global _dinov3_model, _dinov3_processor
    
    from transformers import AutoImageProcessor, AutoModel
    
    logger.info(f"Loading DINOv3 model: {model_id} on {device}")
    
    # Load processor (handles image preprocessing)
    _dinov3_processor = AutoImageProcessor.from_pretrained(
        model_id,
        trust_remote_code=False,
        revision=revision,
    )
    
    # Load model with safe settings
    _dinov3_model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=False,
        revision=revision,
        # Prefer safetensors format when available
        use_safetensors=True,
    )
    
    _dinov3_model.to(device)
    _dinov3_model.eval()
    
    return _dinov3_model, _dinov3_processor


class DINOv3TemporalEncoder:
    """
    DINOv3/DINOv2-based encoder with temporal position encoding.
    
    Extracts semantic embeddings from frames using DINOv3 vision transformer
    and optionally augments them with normalized temporal position information.
    
    Features:
        - Batch processing for efficiency
        - GPU acceleration with mixed precision (fp16)
        - Temporal encoding with configurable weight (H4)
        - L2 normalization of embeddings
        - Safe model loading via Hugging Face Transformers
        - Configurable pooling strategy (CLS token or mean)
    
    Note:
        DINOv2/v3 models produce different embedding dimensions than CLIP:
        - dinov2-small: 384
        - dinov2-base: 768
        - dinov2-large: 1024
        - dinov2-giant: 1536
    """
    
    def __init__(self, config: Optional[DINOv3EncoderConfig] = None):
        """
        Initialize DINOv3 encoder.
        
        Args:
            config: Encoder configuration. Uses defaults if None.
        """
        self.config = config or DINOv3EncoderConfig()
        
        # Auto-detect device
        if self.config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        
        self.model = None
        self.processor = None
        self._embedding_dim: Optional[int] = None
    
    def _ensure_model_loaded(self) -> None:
        """Load model if not already loaded."""
        if self.model is None:
            self.model, self.processor = _load_dinov3(
                self.config.model_id,
                self.device,
                self.config.revision,
            )
            
            # Determine embedding dimension from model config
            self._embedding_dim = self.model.config.hidden_size
            
            logger.info(f"DINOv3 embedding dimension: {self._embedding_dim}")
    
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
        Encode frames into DINOv3 embeddings.
        
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
        
        logger.info(f"Encoding {len(frame_batch)} frames with DINOv3")
        
        # Extract base embeddings
        with Timer("dinov3_encoding"):
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
        
        # Temporal encoding (optimized: simplified normalization)
        temporal_embeddings = None
        if add_temporal and self.config.temporal_weight > 0:
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
        
        with torch.inference_mode():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                
                # Convert BGR to RGB PIL images (optimized: contiguous arrays)
                pil_images = []
                for img in batch_images:
                    if img.ndim == 3 and img.shape[2] == 3:
                        # Use contiguous array for better memory layout
                        img_rgb = np.ascontiguousarray(img[:, :, ::-1])
                    else:
                        img_rgb = img
                    pil_images.append(Image.fromarray(img_rgb))
                
                # Process images using the HF processor
                inputs = self.processor(
                    images=pil_images,
                    return_tensors="pt",
                )
                
                # Move to device (optimized: in-place modification)
                for k in inputs:
                    inputs[k] = inputs[k].to(self.device)
                
                # Mixed precision inference on CUDA
                if self.config.use_fp16 and self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
                
                # Extract features based on pooling strategy
                features = self._extract_features(outputs)
                
                # L2 normalize
                features = features / features.norm(dim=-1, keepdim=True)
                
                all_embeddings.append(features.cpu().float().numpy())
        
        # Optimized: check if vstack result is already float32
        result = np.vstack(all_embeddings)
        return result if result.dtype == np.float32 else result.astype(np.float32)
    
    def _extract_features(self, outputs) -> torch.Tensor:
        """
        Extract features from model outputs based on pooling strategy.
        
        Args:
            outputs: Model outputs from forward pass.
        
        Returns:
            Feature tensor of shape (batch_size, hidden_size).
        """
        if self.config.pooling == "cls":
            # Use CLS token (first token of last hidden state)
            # DINOv2 models always have CLS token at position 0
            features = outputs.last_hidden_state[:, 0]
        elif self.config.pooling == "mean":
            # Mean pooling over patch tokens (excluding CLS token)
            # last_hidden_state shape: (batch, seq_len, hidden_size)
            # Skip CLS token at position 0
            patch_tokens = outputs.last_hidden_state[:, 1:]
            features = patch_tokens.mean(dim=1)
        else:
            # Fallback to CLS token
            features = outputs.last_hidden_state[:, 0]
        
        return features
    
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


def extract_dinov3_features(
    frame_dir: str,
    model_id: str = "facebook/dinov2-base",
    temporal_weight: float = 0.1,
) -> Tuple[NDArray[np.float32], List[str]]:
    """
    Convenience function for DINOv3 feature extraction from a directory.
    
    Args:
        frame_dir: Directory containing frame images.
        model_id: Hugging Face model identifier.
        temporal_weight: Temporal encoding weight (H4).
    
    Returns:
        Tuple of (embeddings array, frame paths list).
    """
    from .frame_sampling import FrameSampler
    
    config = DINOv3EncoderConfig(model_id=model_id, temporal_weight=temporal_weight)
    encoder = DINOv3TemporalEncoder(config)
    
    sampler = FrameSampler()
    frame_batch = sampler.load_frames_from_directory(frame_dir)
    
    embedding_batch = encoder.encode(frame_batch)
    
    # Optimized: single-pass list comprehension
    paths = [str(f.path) for f in frame_batch.frames if f.path is not None]
    
    return embedding_batch.effective_embeddings, paths
