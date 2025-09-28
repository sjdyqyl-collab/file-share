# Phase 3: Experiments and Results - Compact Attention Evaluation

## Experimental Setup

### Models and Hardware
- **Models**: Wan2.1 (14B parameters) and Hunyuan video generation models
- **Hardware**: Single H800 GPU
- **Video Specifications**:
  - Wan2.1: 81 frames at 768×1280 resolution
  - Hunyuan: 129 frames at 768×1280 resolution

### Evaluation Metrics
#### Quality Metrics
- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio
- **MSE**: Mean Squared Error
- **VBench Metrics**:
  - Subject Consistency
  - Background Consistency
  - Aesthetic Quality
- **Text-Video Alignment**:
  - CLIPSIM
  - CLIP-T (CLIP-Text)

#### Performance Metrics
- **Sparsity Rate**: Percentage of attention computation skipped
- **Latency**: End-to-end generation time (seconds)
- **Speedup**: Relative to full attention baseline

### Baseline Methods
1. **Full Attention**: Standard dense attention computation
2. **STA (Sliding Tile Attention)**: Spatio-temporal locality with cubic windows
3. **Sparse VideoGen**: Static sparse patterns
4. **Sparge Attention**: Dynamic block sparsity based on cosine similarity

## Main Results

### Wan2.1 Model Results (80K tokens)
| Method | Sparsity | SSIM↑ | PSNR↑ | MSE↓ | Latency(s) | Speedup |
|--------|----------|--------|--------|--------|------------|---------|
| Full Attention | 0% | - | - | - | 1092.168 | 1.00x |
| Sparse VideoGen | 32.08% | 0.529 | 15.9564 | 1894.3672 | 1200.148 | 0.91x |
| SpargeAttn | 32.27% | 0.6102 | 20.5163 | 676.0723 | 1065.796 | 1.02x |
| **Compact Attention** | 33.99% | 0.7754 | 23.7297 | 351.6015 | **663.824** | **1.65x** |
| **Compact Attention** | 24.66% | 0.8147 | 25.2664 | 254.1789 | **758.176** | **1.44x** |

### Hunyuan Model Results (127K tokens)
| Method | Sparsity | SSIM↑ | PSNR↑ | MSE↓ | Latency(s) | Speedup |
|--------|----------|--------|--------|--------|------------|---------|
| Full Attention | 0% | - | - | - | 1370.658 | 1.00x |
| Sparse VideoGen | 50.35% | 0.7254 | 20.4297 | 822.8567 | 1117.767 | 1.23x |
| SpargeAttn | 47.77% | 0.7794 | 23.5889 | 369.3112 | 1148.628 | 1.19x |
| **Compact Attention** | 62.36% | 0.9040 | 30.0822 | 105.1957 | **546.504** | **2.51x** |
| **Compact Attention** | 52.90% | 0.9452 | 34.5506 | 35.1307 | **750.201** | **1.83x** |

## Quality Preservation Analysis

### VBench Metrics Comparison
#### Wan2.1 Results
| Method | Subject Consistency↑ | Background Consistency↑ | Aesthetic Quality↑ | CLIPSIM↑ | CLIP-T↑ |
|--------|---------------------|------------------------|-------------------|----------|---------|
| Full Attention | 0.9681 | 0.9616 | 0.6486 | 0.2118 | 0.9985 |
| Sparse VideoGen | 0.9547 | 0.9565 | 0.6380 | 0.2116 | 0.9987 |
| SpargeAttn | 0.9357 | 0.9500 | 0.5320 | 0.2064 | 0.9982 |
| **Compact Attention** | **0.9659** | **0.9650** | **0.6480** | **0.2121** | **0.9985** |

#### Hunyuan Results
| Method | Subject Consistency↑ | Background Consistency↑ | Aesthetic Quality↑ | CLIPSIM↑ | CLIP-T↑ |
|--------|---------------------|------------------------|-------------------|----------|---------|
| Full Attention | 0.9736 | 0.9735 | 0.6542 | 0.2181 | 0.9995 |
| Sparse VideoGen | 0.9701 | 0.9722 | 0.6638 | 0.2014 | 0.9995 |
| SpargeAttn | 0.9664 | 0.9731 | 0.5794 | 0.2112 | 0.9995 |
| **Compact Attention** | **0.9716** | **0.9693** | **0.6531** | **0.2184** | **0.9995** |

## Ablation Studies

### Sparse Pattern Effectiveness
**Impact of Individual Components**:
- **Dual Attention Windows**: Increases cross pattern sparsity by ~10%
- **Frame-group-wise patterns**: Additional 3% improvement for temporal variation
- **Combined approach**: Achieves 9.8% more sparsity than cubic window baseline

### Pattern-wise Sparsity Analysis
| Pattern Type | Cubic Window | Frame-group-wise | + Dual Windows |
|--------------|--------------|------------------|----------------|
| Local Patterns | 0.726 | 0.758 | 0.766 |
| Cross Patterns | 0.385 | 0.406 | 0.516 |
| Global Patterns | 0.078 | 0.085 | 0.099 |
| Time-Variant | 0.441 | 0.472 | 0.567 |
| Time-Invariant | 0.306 | 0.317 | 0.385 |
| **Overall** | **0.361** | **0.370** | **0.459** |

## Sensitivity Analysis

### Parameter Sensitivity
- **Recall Threshold (τ)**: Lower thresholds increase sparsity but may compromise quality
- **Cost Threshold (λ)**: Determines upper bound of achievable sparsity
- **Optimal Configuration**: τ=0.9, λ=0.011 for Wan; τ=0.9, λ=0.04 for Hunyuan

### Temporal Sensitivity
- **Early Denoising Steps**: Full attention crucial for first 15 steps
- **Quality Impact**: 1.02dB PSNR drop when sparse attention applied too early
- **Acceleration Strategy**: Full attention for steps 0-15, sparse attention for remaining steps

### Robustness Analysis
- **Input Stability**: PSNR values stable across different prompts and seeds
- **Configuration Consistency**: Narrow interquartile range indicates reliable performance
- **Cross-prompt Generalization**: Union merging strategy ensures robustness

## Visual Quality Comparison

### Key Findings
- **Compact Attention** maintains better visual quality than baselines at higher sparsity levels
- **PSNR Improvements**: 
  - Hunyuan: 30.08 vs 20.43 (Sparse VideoGen) at 62.36% sparsity
  - Wan2.1: 23.73 vs 15.96 (Sparse VideoGen) at 33.99% sparsity
- **Artifact Reduction**: Fewer visual artifacts compared to STA and Sparge Attention
- **Detail Preservation**: Better preservation of fine-grained details and temporal consistency