# Temporal-Aware CLIP + DPP Keyframe Selection

A research-grade implementation for intelligent keyframe selection from videos using semantic embeddings, temporal analysis, and diversity-optimized subset selection via Determinantal Point Processes (DPP).

## Features

- **Semantic Understanding**: CLIP-based embeddings capture high-level visual semantics
- **Temporal Awareness**: Temporal encoding and change detection for scene understanding
- **Adaptive Selection**: Entropy-based estimation of optimal keyframe count
- **Diversity Optimization**: DPP-based selection ensures non-redundant, representative keyframes
- **Optional Motion Awareness**: Optical flow integration for action-heavy videos
- **Ablation-Friendly**: Modular design with toggleable components for research

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd keyframe-selection-with-clip

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Command Line

```bash
# From pre-extracted frames
python run_pipeline.py --frames ./frames --output ./output

# From video file
python run_pipeline.py --video video.mp4 --output ./output

# With custom settings
python run_pipeline.py --frames ./frames -k 15 --model ViT-B/32 --no-temporal

# Using config file
python run_pipeline.py --config config.example.yaml
```

### Python API

```python
from keyframe_selection import PipelineConfig
from keyframe_selection.pipeline import KeyframeSelectionPipeline

# Basic usage
config = PipelineConfig(frame_dir="./frames")
pipeline = KeyframeSelectionPipeline(config)
result = pipeline.run()

print(f"Selected {len(result.keyframes)} keyframes")
print(f"Indices: {result.keyframes.indices}")
print(f"Timestamps: {result.keyframes.timestamps}")
```

### Modular Usage

```python
from keyframe_selection.frame_sampling import FrameSampler
from keyframe_selection.clip_encoder import CLIPTemporalEncoder
from keyframe_selection.temporal_analysis import TemporalDeltaComputer
from keyframe_selection.entropy_estimator import EntropyKEstimator
from keyframe_selection.dpp_kernel import DPPKernelBuilder
from keyframe_selection.selector import DPPSelector

# Load frames
sampler = FrameSampler()
frame_batch = sampler.load_frames_from_directory("./frames")

# Extract CLIP embeddings
encoder = CLIPTemporalEncoder()
embedding_batch = encoder.encode(frame_batch)

# Temporal analysis
temporal = TemporalDeltaComputer()
temporal_result = temporal.compute(embedding_batch)

# Estimate optimal K
entropy = EntropyKEstimator()
entropy_result = entropy.estimate(embedding_batch, temporal_result)

# Build DPP kernel
kernel_builder = DPPKernelBuilder()
kernel = kernel_builder.build(embedding_batch, frame_batch.video_duration)

# Select diverse keyframes
selector = DPPSelector()
result = selector.select(kernel, k=entropy_result.recommended_k)

print(f"Selected frames: {result.indices}")
```

## Pipeline Overview

```
Video → Frame Sampling → CLIP Encoding → Temporal Analysis
                              ↓
                        Motion (optional)
                              ↓
                    Entropy K Estimation
                              ↓
                    DPP Kernel Construction
                              ↓
                    Diversity-Aware Selection
                              ↓
                        Keyframes
```

## Hyperparameters

| ID | Parameter | Default | Description |
|----|-----------|---------|-------------|
| H1 | `model_name` | ViT-L/14 | CLIP model variant |
| H2 | `fps` | 1.0 | Frame sampling rate |
| H3 | `delta_percentile` | 90 | Percentile threshold for changes |
| H4 | `temporal_weight` | 0.1 | Temporal encoding weight α |
| H5 | `num_bins` | 50 | Entropy histogram bins |
| H6 | `beta` | 1.0 | K scaling factor β |
| H7 | `sigma_f` | auto | Feature kernel bandwidth |
| H8 | `sigma_t_ratio` | 0.2 | Temporal kernel bandwidth ratio |
| H9 | `mode` | sample | DPP sampling mode |
| H10 | `gamma` | 0.1 | Motion encoding weight γ |

## Ablation Studies

Disable specific components for ablation:

```bash
# Disable temporal encoding
python run_pipeline.py --frames ./frames --no-temporal

# Disable entropy-based K (use fixed K)
python run_pipeline.py --frames ./frames -k 10 --no-entropy-k

# Disable temporal kernel in DPP
python run_pipeline.py --frames ./frames --no-temporal-kernel

# Enable motion features
python run_pipeline.py --frames ./frames --enable-motion
```

## Baseline Comparison

```bash
# Run baseline experiments
python experiments/baselines.py --frame-dir ./frames -k 10 --output results.json
```

This compares:
- Uniform sampling
- Random sampling
- CLIP + KMeans
- DPP (ours)
- DPP + Temporal (ours)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=keyframe_selection --cov-report=html
```

## Project Structure

```
keyframe-selection-with-clip/
├── keyframe_selection/          # Main package
│   ├── __init__.py
│   ├── config.py               # Configuration dataclasses
│   ├── types.py                # Type definitions
│   ├── frame_sampling.py       # Frame extraction
│   ├── clip_encoder.py         # CLIP embeddings
│   ├── temporal_analysis.py    # Temporal change detection
│   ├── entropy_estimator.py    # Adaptive K estimation
│   ├── dpp_kernel.py           # DPP kernel construction
│   ├── selector.py             # DPP-based selection
│   ├── motion.py               # Optical flow features
│   ├── pipeline.py             # End-to-end orchestration
│   └── utils/                  # Utilities
├── experiments/                 # Experiment scripts
│   └── baselines.py            # Baseline comparisons
├── tests/                       # Unit tests
├── run_pipeline.py             # CLI entry point
├── config.example.yaml         # Example configuration
└── requirements.txt
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{keyframe_selection_clip_dpp,
  title={Temporal-Aware CLIP + DPP Keyframe Selection},
  year={2024},
  url={https://github.com/your-repo}
}
```

## License

MIT License
