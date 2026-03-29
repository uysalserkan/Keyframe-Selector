# Geometric diversity and SfM-oriented keyframe selection

This document describes the **semantic vs geometric** objective, the modules that implement **geometry-aware** keyframe selection, and how to configure and evaluate them.

## Overview

### What this feature does

The default pipeline optimizes **semantic diversity**: CLIP or DINO embeddings are combined into a DPP kernel so that selected frames spread out in a high-level visual feature space. That is ideal for **summarization** and **browsing**, but it does not directly optimize **two-view geometry** (baseline, parallax, stable matches) needed for **structure-from-motion (SfM)**, **COLMAP-style reconstruction**, or **Gaussian splatting** inputs.

The geometric path adds:

1. **Pairwise geometry scores** on consecutive frames (ORB features + fundamental matrix RANSAC inlier ratio as a cheap proxy).
2. A **geometric similarity kernel** \(K_g\) derived from those scores (bottleneck affinity along the time chain, then an RBF).
3. Optional **fusion** with the semantic kernel for DPP: `feature_source: fused` with `alpha_semantic`.
4. **Sequential keyframe selection** (`sequential_geometric`) that advances along time when a minimum “parallax proxy” score is met.
5. Optional **K boost** from mean geometry score when `use_geometry_k` is enabled on the entropy estimator.
6. **Evaluation helpers** in `metrics.py` (photometric PSNR/MAE and geometry proxy summaries).

### Why it exists

- **Semantic diversity** \(\neq\) **geometric diversity**: different scenes can be semantically far apart but pairwise geometry can be weak or degenerate (e.g. pure rotation, textureless regions).
- **Pixel representativeness** (see `PSNR_L1_ANALYSIS.md` and `configs/config.reconstruction.yaml`) addresses reconstruction error via K-means and denser sampling; **geometric diversity** adds explicit **multi-view constraints** proxies without running full COLMAP inside this repo.

### Terminology

| Term | Meaning |
|------|--------|
| **Semantic kernel** \(K_f\) | RBF on CLIP/DINO (or temporal-augmented) embeddings. |
| **Temporal kernel** \(K_t\) | RBF on timestamp differences; combined with \(K_f\) via Hadamard or additive. |
| **Consecutive geometry score** | Per edge \((i,i+1)\): inlier ratio from `findFundamentalMat` (RANSAC) after ORB matching. |
| **Bottleneck affinity** | For frames \(i,j\), affinity is the minimum edge score on the path \(i\ldots j\) (weakest link along the chain). |
| **Geometric kernel** \(K_g\) | RBF on \(1 - A_{ij}\) where \(A\) is bottleneck affinity (values in \([0,1]\)). |
| **Geometry proxy** | Cheap statistics (mean/min inlier ratio) correlating with match stability—not a replacement for bundle adjustment. |

---

## Architecture

```text
Frame sampling → Image encoder (CLIP/DINO)
      → [optional] Motion augmentation
      → [optional] Pairwise geometry (ORB + F-matrix) → scores on EmbeddingBatch
      → Temporal analysis
      → Entropy K (+ optional geometry K boost)
      → DPP kernel (semantic / geometric / fused) × temporal
      → Selection (DPP / K-means / HDBSCAN / sequential_geometric)
      → Keyframes
```

Important files:

| Module | Role |
|--------|------|
| [`pairwise_geometry.py`](../keyframe_selection/pairwise_geometry.py) | Consecutive F-matrix scores, bottleneck matrix, \(K_g\), point features for K-means fusion. |
| [`dpp_kernel.py`](../keyframe_selection/dpp_kernel.py) | `_build_feature_kernels`: `semantic` \| `geometric` \| `fused`. |
| [`entropy_estimator.py`](../keyframe_selection/entropy_estimator.py) | Optional `use_geometry_k` boost using mean consecutive score. |
| [`selector.py`](../keyframe_selection/selector.py) | `sequential_geometric`; K-means with `kmeans_fuse_geometry_features`. |
| [`pipeline.py`](../keyframe_selection/pipeline.py) | `pipeline_objective: geometric_sfm` enables pairwise block when `pairwise_geometry` was off. |
| [`metrics.py`](../keyframe_selection/metrics.py) | PSNR/MAE and `geometry_proxy_summary`. |

### Design decisions

- **F-matrix on consecutive pairs** keeps cost linear in \(N\) and avoids an \(O(N^2)\) dense two-view loop in the default path.
- **Bottleneck affinity** encodes “how strong is the weakest link” between two frames on the timeline—reasonable for video where geometry is propagated along a chain.
- **Fused kernel** uses a **convex combination** \( \alpha K_{\text{sem}} + (1-\alpha) K_{\text{geo}} \) so PSD structure is preserved before combining with the temporal kernel (same as additive blend of PSD matrices).
- **Sequential selection** does not use DPP; it uses stored consecutive scores and thresholds—good when you want **SfM-style spacing** without changing the global subset objective.

---

## Configuration

### `PipelineConfig`

- **`pipeline_objective`**: `semantic` (default) | `reconstruction` | `geometric_sfm`.  
  For `geometric_sfm`, the pipeline turns on pairwise geometry if it was disabled, and can enable geometry-aware entropy (see below).
- **`pairwise_geometry`**: [`PairwiseGeometryConfig`](#pairwisegeometryconfig).

### `PairwiseGeometryConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `enabled` | `false` | Compute consecutive geometry scores. |
| `n_features` | `500` | ORB keypoint budget. |
| `ratio_test` | `0.75` | Fraction of best matches kept before F estimation. |
| `ransac_threshold` | `1.0` | RANSAC reprojection threshold (pixels, downscaled space). |
| `ransac_confidence` | `0.99` | OpenCV RANSAC confidence. |
| `downscale` | `0.5` | Grayscale resize factor before ORB (speed vs accuracy). |

### `DPPKernelConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `feature_source` | `semantic` | `semantic` \| `geometric` \| `fused`. |
| `alpha_semantic` | `0.5` | Weight on semantic kernel in `fused` mode. |
| `sigma_geometric` | `null` | Bandwidth for geometric RBF; `null` uses median heuristic. |

`feature_source: geometric` **requires** `EmbeddingBatch.geometry_consecutive_scores` with length \(N-1\).

### `EntropyEstimatorConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `use_geometry_k` | `false` | Multiply raw \(K\) estimate by \(1 + \text{geometry\_k\_weight} \cdot \text{mean score}\). |
| `geometry_k_weight` | `0.35` | Strength of that boost. |

### `SelectorConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `method` | `dpp` | Add `sequential_geometric` for chain-based selection. |
| `sequential_min_score` | `0.12` | Minimum bottleneck segment score to accept a jump. |
| `sequential_max_span` | `45` | Max frame index gap when threshold is not met (force advance). |
| `kmeans_fuse_geometry_features` | `false` | Concatenate `geometry_point_features` to embeddings for K-means. |

### Preset YAML

See [`configs/config.geometric_sfm.yaml`](../configs/config.geometric_sfm.yaml) for a full example (`pipeline_objective: geometric_sfm`, fused DPP, pairwise geometry enabled).

---

## Data types

### `EmbeddingBatch` ([`types.py`](../keyframe_selection/types.py))

- **`geometry_consecutive_scores`**: `Optional[NDArray]` shape `(N-1,)`, values in \([0,1]\).
- **`geometry_point_features`**: `Optional[NDArray]` shape `(N, 3)` — incoming edge, outgoing edge, local mean (for K-means fusion).

### `EntropyResult`

- **`mean_geometry_score`**: Mean of consecutive scores when geometry-aware K was used; else `None`.

### `DPPKernel`

- **`geometric_kernel`**: Raw \(K_g\) when computed.
- **`sigma_g`**: Diagnostic bandwidth scale for the geometric block.

---

## API reference

### `pairwise_geometry`

```python
def compute_consecutive_fundamental_scores(
    frame_batch: FrameBatch,
    config: Optional[PairwiseGeometryConfig] = None,
) -> NDArray[np.float64]:
    """Shape (N-1,). Zeros or disabled config yields ones or empty as appropriate."""

def bottleneck_affinity_matrix(n: int, consecutive_scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetric A[i,j] in [0,1]."""

def geometric_rbf_kernel(
    affinity: NDArray[np.float64],
    sigma: Optional[float] = None,
) -> NDArray[np.float64]:
    """PSD-friendly similarity from affinity."""

def compute_geometry_point_features(
    consecutive_scores: NDArray[np.float64],
    n: int,
) -> NDArray[np.float32]:
    """Per-frame 3-D features for optional K-means fusion."""
```

**Errors / edge cases**

- Fewer than two frames: empty consecutive score array; geometric-only DPP will raise from `DPPKernelBuilder` if `feature_source` requires scores.
- Very few matches: consecutive score `0.0` for that edge (degenerate pair).

### `metrics`

```python
def photometric_metrics_pair(img_a, img_b) -> PhotometricMetrics:
    """PSNR (dB) and MAE for aligned uint8 BGR images."""

def geometry_proxy_summary(consecutive_inlier_ratios) -> GeometryProxySummary:
    """Mean, min, std; empty input yields zeros."""
```

### `DPPKernelBuilder.build`

Uses `embedding_batch.effective_embeddings` and optional `geometry_consecutive_scores` according to `DPPKernelConfig.feature_source`.

**Raises**

- `ValueError` if `feature_source == "geometric"` and scores are missing or wrong length.

### `DPPSelector.select_from_embeddings`

**Raises**

- `ValueError` if `method == "sequential_geometric"` and `geometry_consecutive_scores` is missing.

---

## Examples

### CLI with geometric preset

```bash
python run_pipeline.py --config configs/config.geometric_sfm.yaml --frames ./frames --output ./out_geo
```

### Python: fused kernel + adaptive K

```python
from dataclasses import replace
from pathlib import Path

from keyframe_selection import PipelineConfig
from keyframe_selection.config import DPPKernelConfig, SelectorConfig
from keyframe_selection.pipeline import KeyframeSelectionPipeline

config = PipelineConfig(
    frame_dir=Path("./frames"),
    output_dir=Path("./out"),
    pipeline_objective="geometric_sfm",
    dpp_kernel=DPPKernelConfig(feature_source="fused", alpha_semantic=0.45),
    selector=SelectorConfig(method="dpp", mode="map"),
)
result = KeyframeSelectionPipeline(config).run()
```

Use `replace(config.dpp_kernel, ...)` when you only need to override a few fields on an existing config loaded from YAML.

### Sequential geometric selection

Set `selector.method: sequential_geometric` and supply `fixed_k` or entropy-based `k`. Pairwise geometry must run (enable `pairwise_geometry` or use `pipeline_objective: geometric_sfm`).

### Evaluate proxies

```python
from keyframe_selection.metrics import geometry_proxy_summary, photometric_metrics_pair

summary = geometry_proxy_summary(embedding_batch.geometry_consecutive_scores)
# summary.mean_inlier_ratio, .min_inlier_ratio

# Aligned same-size frames only:
m = photometric_metrics_pair(frame_a, frame_b)
# m.psnr_db, m.mae_l1
```

---

## Best practices

1. **Use higher FPS** for SfM-bound workflows than for pure summarization (see `config.reconstruction.yaml` vs `config.geometric_sfm.yaml`).
2. **Start with `fused`** if you still want semantic scene coverage; use **`geometric`** only when you trust the geometry proxy more than global appearance.
3. **Tune `sequential_min_score` and `sequential_max_span`** on a short validation clip before long runs.
4. **Report both** geometry proxy summaries and (when applicable) photometric metrics from `metrics.py` alongside downstream COLMAP or splatting metrics.

---

## Common pitfalls

| Pitfall | Mitigation |
|---------|------------|
| Expecting true metric 3D parallax without calibration | F-matrix inlier ratio is a **proxy**; use known intrinsics + E-matrix or SLAM if you need metric baseline. |
| `feature_source: geometric` without scores | Enable `pairwise_geometry` or set `pipeline_objective: geometric_sfm`. |
| Degenerate motion (rotation-only, planar scenes) | Low scores; increase FPS, tune ORB/`downscale`, or combine with `fused` and semantic \(K_f\). |
| Confusing reconstruction K-means with geometric diversity | `config.reconstruction.yaml` targets **pixel representativeness**; geometric SFM adds **explicit two-view cues**—objectives differ. |
| Very large `sequential_max_span` | Can skip important baseline growth; reduce span or lower `sequential_min_score` slightly. |

---

## Related documentation

- [`PSNR_L1_ANALYSIS.md`](../PSNR_L1_ANALYSIS.md) — semantic vs pixel representativeness.
- [`configs/config.reconstruction.yaml`](../configs/config.reconstruction.yaml) — reconstruction-oriented preset.
- [`CONFIG_FILES_README.md`](../CONFIG_FILES_README.md) — other YAML presets.

---

## See also

- [AliceVision keyframe documentation](https://alicevision.readthedocs.io/en/doc-fixes/md__home_docs_checkouts_readthedocs_8org_user_builds_alicevision_checkouts_doc-fixes_src_aliceVision_keyframe_README.html) — industry framing for motion, sharpness, and geometric suitability (external).
