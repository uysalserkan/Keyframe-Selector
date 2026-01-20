"""
Unit tests for individual pipeline modules.
"""

import numpy as np
import pytest
from pathlib import Path

# Test fixtures


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    return np.random.randn(20, 512).astype(np.float32)


@pytest.fixture
def sample_timestamps():
    """Generate sample timestamps."""
    return np.linspace(0, 10, 20)


@pytest.fixture
def sample_images():
    """Generate sample BGR images."""
    np.random.seed(42)
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]


# Config tests


class TestConfig:
    """Test configuration classes."""
    
    def test_pipeline_config_defaults(self):
        from keyframe_selection.config import PipelineConfig
        
        config = PipelineConfig()
        assert config.frame_sampling.fps == 1.0
        assert config.clip_encoder.model_name == "ViT-L/14"
        assert config.use_temporal_encoding is True
        # New: encoder_backend defaults to "clip"
        assert config.encoder_backend == "clip"
    
    def test_pipeline_config_from_dict(self):
        from keyframe_selection.config import PipelineConfig
        
        config_dict = {
            "frame_sampling": {"fps": 2.0},
            "clip_encoder": {"model_name": "ViT-B/32"},
            "random_seed": 123,
        }
        
        config = PipelineConfig.from_dict(config_dict)
        assert config.frame_sampling.fps == 2.0
        assert config.clip_encoder.model_name == "ViT-B/32"
        assert config.random_seed == 123
    
    def test_config_to_dict(self):
        from keyframe_selection.config import PipelineConfig
        
        config = PipelineConfig()
        config_dict = config.to_dict()
        
        assert "frame_sampling" in config_dict
        assert "clip_encoder" in config_dict
        assert config_dict["random_seed"] == 42
    
    def test_dinov3_encoder_config_defaults(self):
        """Test DINOv3 encoder config defaults."""
        from keyframe_selection.config import DINOv3EncoderConfig
        
        config = DINOv3EncoderConfig()
        assert config.model_id == "facebook/dinov2-base"
        assert config.temporal_weight == 0.1
        assert config.pooling == "cls"
        assert config.batch_size == 32
        assert config.use_fp16 is True
        assert config.device is None
        assert config.revision is None
    
    def test_pipeline_config_dinov3_from_dict(self):
        """Test parsing dinov3_encoder and encoder_backend from dict."""
        from keyframe_selection.config import PipelineConfig
        
        config_dict = {
            "encoder_backend": "dinov3",
            "dinov3_encoder": {
                "model_id": "facebook/dinov2-large",
                "pooling": "mean",
                "temporal_weight": 0.2,
            },
        }
        
        config = PipelineConfig.from_dict(config_dict)
        assert config.encoder_backend == "dinov3"
        assert config.dinov3_encoder.model_id == "facebook/dinov2-large"
        assert config.dinov3_encoder.pooling == "mean"
        assert config.dinov3_encoder.temporal_weight == 0.2
    
    def test_backward_compat_old_config_dict(self):
        """Test that old config dicts (without encoder_backend/dinov3) still work."""
        from keyframe_selection.config import PipelineConfig
        
        # Old-style config dict with only clip_encoder
        old_config_dict = {
            "frame_sampling": {"fps": 1.5},
            "clip_encoder": {"model_name": "ViT-B/16"},
            "selector": {"fixed_k": 10},
        }
        
        config = PipelineConfig.from_dict(old_config_dict)
        # Should default to clip backend
        assert config.encoder_backend == "clip"
        assert config.clip_encoder.model_name == "ViT-B/16"
        # dinov3_encoder should have defaults
        assert config.dinov3_encoder.model_id == "facebook/dinov2-base"
    
    def test_config_to_dict_includes_dinov3(self):
        """Test that to_dict includes dinov3_encoder."""
        from keyframe_selection.config import PipelineConfig
        
        config = PipelineConfig(encoder_backend="dinov3")
        config_dict = config.to_dict()
        
        assert "encoder_backend" in config_dict
        assert config_dict["encoder_backend"] == "dinov3"
        assert "dinov3_encoder" in config_dict
        assert config_dict["dinov3_encoder"]["model_id"] == "facebook/dinov2-base"


# Types tests


class TestTypes:
    """Test type definitions."""
    
    def test_frame_batch_properties(self, sample_images):
        from keyframe_selection.types import FrameBatch, FrameData
        
        frames = [
            FrameData(image=img, timestamp=i * 0.5, frame_index=i)
            for i, img in enumerate(sample_images)
        ]
        
        batch = FrameBatch(
            frames=frames,
            video_duration=5.0,
            source_fps=2.0,
        )
        
        assert len(batch) == 10
        assert len(batch.images) == 10
        assert len(batch.timestamps) == 10
        assert batch.normalized_timestamps.max() <= 1.0
    
    def test_embedding_batch_effective(self, sample_embeddings):
        from keyframe_selection.types import EmbeddingBatch
        
        batch = EmbeddingBatch(
            embeddings=sample_embeddings,
            timestamps=np.arange(20, dtype=np.float64),
        )
        
        # Without temporal, effective = base
        assert np.array_equal(batch.effective_embeddings, batch.embeddings)
        
        # With temporal
        temporal = np.hstack([sample_embeddings, np.ones((20, 1), dtype=np.float32)])
        batch.temporal_embeddings = temporal
        assert batch.effective_embeddings.shape[1] == 513


# Temporal analysis tests


class TestTemporalAnalysis:
    """Test temporal change analysis."""
    
    def test_delta_computation(self, sample_embeddings):
        from keyframe_selection.temporal_analysis import TemporalDeltaComputer
        from keyframe_selection.types import EmbeddingBatch
        
        batch = EmbeddingBatch(
            embeddings=sample_embeddings,
            timestamps=np.arange(20, dtype=np.float64),
        )
        
        computer = TemporalDeltaComputer()
        result = computer.compute(batch)
        
        assert len(result.deltas) == 19  # N-1 deltas
        assert result.threshold > 0
        assert len(result.change_points) >= 0
    
    def test_empty_embeddings(self):
        from keyframe_selection.temporal_analysis import TemporalDeltaComputer
        from keyframe_selection.types import EmbeddingBatch
        
        batch = EmbeddingBatch(
            embeddings=np.array([]).reshape(0, 512).astype(np.float32),
            timestamps=np.array([]),
        )
        
        computer = TemporalDeltaComputer()
        result = computer.compute(batch)
        
        assert len(result.deltas) == 0
        assert result.threshold == 0.0


# Entropy estimation tests


class TestEntropyEstimator:
    """Test entropy-based K estimation."""
    
    def test_entropy_computation(self, sample_embeddings):
        from keyframe_selection.entropy_estimator import EntropyKEstimator
        
        estimator = EntropyKEstimator()
        result = estimator.estimate_from_embeddings(sample_embeddings)
        
        assert 0 <= result.entropy <= 1.0
        assert result.recommended_k >= 3
        assert result.recommended_k <= 50
    
    def test_adaptive_k_formula(self):
        from keyframe_selection.entropy_estimator import compute_adaptive_k
        
        # High entropy, high delta -> more keyframes
        k_high = compute_adaptive_k(100, entropy=0.9, mean_delta=1.5, beta=1.0)
        
        # Low entropy, low delta -> fewer keyframes
        k_low = compute_adaptive_k(100, entropy=0.3, mean_delta=0.5, beta=1.0)
        
        assert k_high > k_low


# DPP kernel tests


class TestDPPKernel:
    """Test DPP kernel construction."""
    
    def test_kernel_shape(self, sample_embeddings, sample_timestamps):
        from keyframe_selection.dpp_kernel import DPPKernelBuilder
        
        builder = DPPKernelBuilder()
        kernel = builder.build_from_arrays(
            sample_embeddings,
            sample_timestamps,
            use_temporal=True,
        )
        
        assert kernel.kernel.shape == (20, 20)
        assert kernel.sigma_f > 0
        assert kernel.sigma_t > 0
    
    def test_kernel_psd(self, sample_embeddings):
        from keyframe_selection.dpp_kernel import DPPKernelBuilder
        
        builder = DPPKernelBuilder()
        kernel = builder.build_from_arrays(sample_embeddings, use_temporal=False)
        
        # Check positive semi-definiteness
        eigvals = np.linalg.eigvalsh(kernel.kernel)
        assert eigvals.min() >= -1e-6
    
    def test_hadamard_combination(self, sample_embeddings, sample_timestamps):
        from keyframe_selection.dpp_kernel import DPPKernelBuilder
        from keyframe_selection.config import DPPKernelConfig
        
        config = DPPKernelConfig(combine_method="hadamard")
        builder = DPPKernelBuilder(config)
        
        kernel = builder.build_from_arrays(sample_embeddings, sample_timestamps)
        
        # Hadamard product should be element-wise
        assert kernel.feature_kernel is not None
        assert kernel.temporal_kernel is not None


# Selector tests


class TestSelector:
    """Test DPP-based selection."""
    
    def test_selection_count(self, sample_embeddings):
        from keyframe_selection.dpp_kernel import DPPKernelBuilder
        from keyframe_selection.selector import DPPSelector
        
        builder = DPPKernelBuilder()
        kernel = builder.build_from_arrays(sample_embeddings, use_temporal=False)
        
        selector = DPPSelector()
        result = selector.select(kernel, k=5)
        
        assert len(result.indices) == 5
        assert len(np.unique(result.indices)) == 5  # All unique
    
    def test_sorted_output(self, sample_embeddings, sample_timestamps):
        from keyframe_selection.dpp_kernel import DPPKernelBuilder
        from keyframe_selection.selector import DPPSelector
        
        builder = DPPKernelBuilder()
        kernel = builder.build_from_arrays(sample_embeddings, sample_timestamps)
        
        selector = DPPSelector()
        result = selector.select(kernel, k=5, timestamps=sample_timestamps)
        
        # Should be sorted by time
        assert np.all(np.diff(result.indices) >= 0)
    
    def test_k_larger_than_n(self, sample_embeddings):
        from keyframe_selection.dpp_kernel import DPPKernelBuilder
        from keyframe_selection.selector import DPPSelector
        
        builder = DPPKernelBuilder()
        kernel = builder.build_from_arrays(sample_embeddings[:5], use_temporal=False)
        
        selector = DPPSelector()
        result = selector.select(kernel, k=10)  # k > n
        
        assert len(result.indices) == 5  # Should cap at n


# Utility tests


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        from keyframe_selection.utils.io import set_global_seed
        
        set_global_seed(42)
        a = np.random.rand(10)
        
        set_global_seed(42)
        b = np.random.rand(10)
        
        np.testing.assert_array_equal(a, b)
    
    def test_timer(self):
        from keyframe_selection.utils.timing import Timer
        import time
        
        with Timer("test_op", log=False) as t:
            time.sleep(0.01)
        
        assert t.elapsed >= 0.01
        assert "test_op" in Timer.get_timings()


# Integration test


class TestIntegration:
    """Integration tests for the pipeline."""
    
    def test_embedding_to_keyframes(self, sample_embeddings, sample_timestamps):
        """Test the core flow from embeddings to keyframes."""
        from keyframe_selection.temporal_analysis import TemporalDeltaComputer
        from keyframe_selection.entropy_estimator import EntropyKEstimator
        from keyframe_selection.dpp_kernel import DPPKernelBuilder
        from keyframe_selection.selector import DPPSelector
        from keyframe_selection.types import EmbeddingBatch
        
        # Create embedding batch
        batch = EmbeddingBatch(
            embeddings=sample_embeddings,
            timestamps=sample_timestamps,
            frame_indices=np.arange(20, dtype=np.int64),
        )
        
        # Temporal analysis
        temporal = TemporalDeltaComputer()
        temporal_result = temporal.compute(batch)
        
        # Entropy estimation
        entropy = EntropyKEstimator()
        entropy_result = entropy.estimate(batch, temporal_result)
        
        # Build kernel
        kernel_builder = DPPKernelBuilder()
        kernel = kernel_builder.build_from_arrays(
            sample_embeddings,
            sample_timestamps,
            video_duration=10.0,
        )
        
        # Select
        selector = DPPSelector()
        result = selector.select(kernel, k=entropy_result.recommended_k, timestamps=sample_timestamps)
        
        assert len(result.indices) > 0
        assert len(result.indices) <= 20


# DINOv3 encoder tests (mock-based, no network)


class TestDINOv3Encoder:
    """Test DINOv3 encoder with mocked transformers."""
    
    def test_temporal_encoding_concatenation(self):
        """Test that temporal encoding adds one dimension."""
        from keyframe_selection.dinov3_encoder import DINOv3TemporalEncoder
        from keyframe_selection.config import DINOv3EncoderConfig
        
        # Create encoder with mocked internals
        config = DINOv3EncoderConfig(temporal_weight=0.1)
        encoder = DINOv3TemporalEncoder(config)
        
        # Test the _add_temporal_encoding method directly (no model needed)
        embeddings = np.random.randn(10, 768).astype(np.float32)
        timestamps = np.linspace(0, 1, 10)
        
        temporal_emb = encoder._add_temporal_encoding(embeddings, timestamps)
        
        # Should add one dimension
        assert temporal_emb.shape == (10, 769)
        # Last column should be scaled timestamps
        np.testing.assert_allclose(
            temporal_emb[:, -1],
            0.1 * timestamps,
            rtol=1e-5,
        )
    
    def test_l2_normalization(self, sample_images, monkeypatch):
        """Test that embeddings are L2-normalized (using mock model)."""
        import torch
        from unittest.mock import MagicMock, patch
        
        # Create mock model and processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        
        # Mock model config
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = 768
        
        # Mock model output with random embeddings
        def mock_forward(**kwargs):
            batch_size = kwargs.get("pixel_values", kwargs.get("input_ids")).shape[0] if "pixel_values" in kwargs else 2
            output = MagicMock()
            # last_hidden_state: (batch, seq_len, hidden)
            output.last_hidden_state = torch.randn(batch_size, 197, 768)
            return output
        
        mock_model.__call__ = mock_forward
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        
        # Mock processor to return proper tensor
        def mock_process(images, return_tensors):
            return {"pixel_values": torch.randn(len(images), 3, 224, 224)}
        
        mock_processor.__call__ = mock_process
        
        # Patch the load function
        from keyframe_selection import dinov3_encoder
        
        def mock_load(model_id, device, revision=None):
            return mock_model, mock_processor
        
        monkeypatch.setattr(dinov3_encoder, "_load_dinov3", mock_load)
        
        from keyframe_selection.dinov3_encoder import DINOv3TemporalEncoder
        from keyframe_selection.config import DINOv3EncoderConfig
        
        config = DINOv3EncoderConfig()
        encoder = DINOv3TemporalEncoder(config)
        
        # Encode sample images
        images = sample_images[:3]  # Just use 3 images
        result = encoder.encode_images(images, add_temporal=False)
        
        # Check L2 normalization (each row should have norm ~1)
        norms = np.linalg.norm(result.embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_pooling_strategies(self):
        """Test CLS and mean pooling produce expected shapes."""
        import torch
        from unittest.mock import MagicMock
        
        from keyframe_selection.dinov3_encoder import DINOv3TemporalEncoder
        from keyframe_selection.config import DINOv3EncoderConfig
        
        # Create mock outputs
        batch_size = 4
        seq_len = 197  # 1 CLS + 196 patches (14x14)
        hidden_size = 768
        
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test CLS pooling
        config_cls = DINOv3EncoderConfig(pooling="cls")
        encoder_cls = DINOv3TemporalEncoder(config_cls)
        features_cls = encoder_cls._extract_features(mock_outputs)
        assert features_cls.shape == (batch_size, hidden_size)
        
        # Test mean pooling
        config_mean = DINOv3EncoderConfig(pooling="mean")
        encoder_mean = DINOv3TemporalEncoder(config_mean)
        features_mean = encoder_mean._extract_features(mock_outputs)
        assert features_mean.shape == (batch_size, hidden_size)
        
        # CLS and mean should give different results
        assert not torch.allclose(features_cls, features_mean)
    
    def test_empty_input_handling(self):
        """Test encoder handles empty input gracefully."""
        from keyframe_selection.dinov3_encoder import DINOv3TemporalEncoder
        from keyframe_selection.config import DINOv3EncoderConfig
        from keyframe_selection.types import FrameBatch
        
        config = DINOv3EncoderConfig()
        encoder = DINOv3TemporalEncoder(config)
        
        # Don't load the model, just test the encode method's early return
        encoder._embedding_dim = 768  # Set manually to avoid model load
        
        # Create empty frame batch
        empty_batch = FrameBatch(
            frames=[],
            video_duration=0.0,
            source_fps=1.0,
        )
        
        result = encoder.encode(empty_batch)
        
        assert result.embeddings.shape == (0, 768)
        assert len(result.timestamps) == 0
        assert len(result.frame_indices) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
