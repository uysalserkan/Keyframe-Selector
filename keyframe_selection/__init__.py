"""
Keyframe Selection with Temporal-Aware CLIP + DPP

A research-grade implementation for intelligent keyframe selection from videos
using semantic embeddings, temporal analysis, and diversity-optimized subset selection.

Example usage:
    ```python
    from keyframe_selection import PipelineConfig
    from keyframe_selection.pipeline import KeyframeSelectionPipeline
    
    config = PipelineConfig(frame_dir="./frames")
    pipeline = KeyframeSelectionPipeline(config)
    result = pipeline.run()
    
    print(f"Selected {len(result.keyframes)} keyframes")
    print(f"Indices: {result.keyframes.indices}")
    ```

For command-line usage:
    ```bash
    python run_pipeline.py --frames ./frames --output ./output
    ```
"""

__version__ = "0.1.0"

from .config import (
    PipelineConfig,
    FrameSamplingConfig,
    CLIPEncoderConfig,
    TemporalAnalysisConfig,
    EntropyEstimatorConfig,
    DPPKernelConfig,
    SelectorConfig,
    MotionConfig,
)
from .types import (
    FrameBatch,
    FrameData,
    EmbeddingBatch,
    KeyframeResult,
    PipelineResult,
    TemporalAnalysisResult,
    EntropyResult,
    DPPKernel,
)

__all__ = [
    # Version
    "__version__",
    # Config classes
    "PipelineConfig",
    "FrameSamplingConfig",
    "CLIPEncoderConfig",
    "TemporalAnalysisConfig",
    "EntropyEstimatorConfig",
    "DPPKernelConfig",
    "SelectorConfig",
    "MotionConfig",
    # Data types
    "FrameBatch",
    "FrameData",
    "EmbeddingBatch",
    "KeyframeResult",
    "PipelineResult",
    "TemporalAnalysisResult",
    "EntropyResult",
    "DPPKernel",
]
