# Compact Attention: Experiments

## Experimental Setup

### 1. Models and Configurations
- **Primary Models**: 
  - Wan2.1 (14B parameters)
  - Hunyuan Video
- **GPU**: Single H800 GPU
- **Video Specifications**:
  - Wan2.1: 81 frames at 768×1280 resolution
  - Hunyuan: 129 frames at 768×1280 resolution
- **Implementation**: Based on ThunderKittens framework with FlashAttention-2

### 2. Evaluation Metrics
- **Quality Metrics**:
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)
  - MSE (Mean Squared Error)
  - VBench metrics: Subject Consistency, Background Consistency, Aesthetic Quality
  - CLIPSIM and CLIP-Temp (CLIP-T) for text-video alignment
- **Performance Metrics**:
  - Attention sparsity rate (%)
  - Attention latency (seconds)
  - End-to-end speedup (×)

### 3. Baseline Comparisons
- **Full Attention**: Original dense attention baseline
- **STA (Sliding Tile Attention)**: Local spatio-temporal attention with cubic windows
- **Sparse VideoGen**: Static sparse patterns
- **Sparge Attention**: Dynamic sparse attention with cosine similarity thresholds

## Quantitative Results

### 1. Acceleration Performance and Quality Preservation

#### Wan2.1 Model (80K tokens)
| Method | Sparsity | SSIM ↑ | PSNR ↑ | MSE ↓ | Latency (s) | Speedup |
|--------|----------|---------|---------|--------|-------------|---------|
| Full Attention | 0% | - | - | - | 1092.168 | 1.00× |
| Sparse VideoGen | 32.08% | 0.529 | 15.9564 | 1894.3672 | 1200.148 | 0.91× |
| SpargeAttn | 32.27% | 0.6102 | 20.5163 | 676.0723 | 1065.796 | 1.02× |
| **Compact Attention** | 33.99% | 0.7754 | 23.7297 | 351.6015 | 663.824 | 1.65× |
| **Compact Attention** | 24.66% | 0.8147 | 25.2664 | 254.1789 | 758.176 | 1.44× |

#### Hunyuan Model (127K tokens)
| Method | Sparsity | SSIM ↑ | PSNR ↑ | MSE ↓ | Latency (s) | Speedup |
|--------|----------|---------|---------|--------|-------------|---------|
| Full Attention | 0% | - | - | - | 1370.658 | 1.00× |
| Sparse VideoGen | 50.35% | 0.7254 | 20.4297 | 822.8567 | 1117.767 | 1.23× |
| SpargeAttn | 47.77% | 0.7794 | 23.5889 | 369.3112 | 1148.628 | 1.19× |
| **Compact Attention** | 62.36% | 0.9040 | 30.0822 | 105.1957 | 546.504 | 2.51× |
| **Compact Attention** | 52.90% | 0.9452 | 34.5506 | 35.1307 | 750.201 | 1.83× |

### 2. Visual Quality Comparison (VBench)

#### Wan2.1 Results
| Method | Sparsity | Subject Consistency | Background Consistency | Aesthetic Quality | CLIPSIM | CLIP-T |
|--------|----------|---------------------|------------------------|-------------------|---------|--------|
| Full Attention | 0% | 0.9681 | 0.9616 | 0.6486 | 0.2118 | 0.9985 |
| Sparse VideoGen | 32.08% | 0.9547 | 0.9565 | 0.6380 | 0.2116 | 0.9987 |
| SpargeAttn | 32.27% | 0.9357 | 0.9500 | 0.5320 | 0.2064 | 0.9982 |
| **Compact Attention** | 33.99% | 0.9659 | 0.9650 | 0.6480 | 0.2121 | 0.9985 |
| **Compact Attention** | 24.66% | 0.9674 | 0.9638 | 0.6459 | 0.2122 | 0.9986 |

#### Hunyuan Results
| Method | Sparsity | Subject Consistency | Background Consistency | Aesthetic Quality | CLIPSIM | CLIP-T |
|--------|----------|---------------------|------------------------|-------------------|---------|--------|
| Full Attention | 0% | 0.9736 | 0.9735 | 0.6542 | 0.2181 | 0.9995 |
| Sparse VideoGen | 50.35% | 0.9701 | 0.9722 | 0.6638 | 0.2014 | 0.9995 |
| SpargeAttn | 47.77% | 0.9664 | 0.9731 | 0.5794 | 0.2112 | 0.9995 |
| **Compact Attention** | 62.36% | 0.9716 | 0.9693 | 0.6531 | 0.2184 | 0.9995 |
| **Compact Attention** | 52.90% | 0.9723 | 0.9735 | 0.6536 | 0.2184 | 0.9995 |

## Ablation Studies

### 1. Sparse Pattern Effectiveness
- **Dual Attention Windows**: Increases cross pattern sparsity by ~10%
- **Frame-group-wise Patterns**: Contributes additional 3% improvement
- **Combined Approach**: Achieves 9.8% more sparsity than cubic window baseline

### 2. Pattern Analysis by Type
| Pattern Type | Cubic Window | Frame-group-wise | + Dual Windows |
|--------------|--------------|------------------|----------------|
| Local Patterns | 0.726 | 0.758 | 0.766 |
| Cross Patterns | 0.385 | 0.406 | 0.516 |
| Global Patterns | 0.078 | 0.085 | 0.099 |
| Time-Variant | 0.441 | 0.472 | 0.567 |
| Time-Invariant | 0.306 | 0.317 | 0.385 |
| **Overall** | 0.361 | 0.370 | 0.459 |

### 3. Sensitivity Analysis

#### 3.1 Recall Threshold Impact
- **Observation**: Sparsity converges to upper bound determined by cost constraint
- **Finding**: Hunyuan achieves higher sparsity than Wan2.1 due to smaller model size
- **Trade-off**: Lower recall threshold → higher sparsity but potential quality degradation

#### 3.2 Temporal Sensitivity
- **Early Steps**: Most sensitive during initial denoising (high-noise inputs)
- **Quality Impact**: 1.02dB PSNR drop when full attention applied only in final 15 steps vs first 15
- **Recommendation**: Maintain full attention for first 15 denoising steps, then apply sparsity

#### 3.3 Input Robustness
- **Stability**: Compact Attention shows highest median PSNR with narrow interquartile range
- **Consistency**: Stable performance across diverse text prompts and random seeds
- **Parameter Sensitivity**: (τ=0.9, λ=0.011) configuration shows optimal balance

## Visual Results

### 1. Qualitative Comparisons
- **Hunyuan Model**: Compact Attention achieves PSNR=24.24 at 62.36% sparsity vs full attention
- **STA Comparison**: Compact Attention maintains better quality at higher sparsity (62.36% vs 58.37%)
- **Baseline Performance**: Significantly outperforms Sparse VideoGen and Sparge Attention in visual quality

### 2. Generation Examples
- **Video Length**: 117-129 frames for consistent evaluation
- **Resolution**: 768×1280 HD video
- **Visual Fidelity**: Maintained details and temporal consistency despite high sparsity

## Implementation Details

### 1. Baseline Configurations
- **STA**: Uses 69-frame or 117-frame videos due to resolution constraints
- **Sparge Attention**: Default hyperparameters (simthreshd1=0.1, cdfthreshd=0.9, pvthreshd=20)
- **Sparse VideoGen**: Adjusted sparsity to match other baselines

### 2. Parameter Settings
- **Recall Threshold (τ)**: 0.9 for quality preservation
- **Cost Threshold (λ)**: 0.011 (Wan2.1), 0.04 (Hunyuan)
- **Mask Reuse**: Across n consecutive denoising steps
- **Tile Size**: Optimized for GPU architecture and model characteristics

## Key Findings

### 1. Performance Summary
- **Best Speedup**: 2.51× on Hunyuan at 62.36% sparsity
- **Quality Preservation**: PSNR > 30 at high sparsity levels
- **Consistency**: Stable across different inputs and configurations

### 2. Advantages Over Baselines
- **Higher Sparsity**: Achieves 62.36% vs 50.35% (Sparse VideoGen) and 47.77% (SpargeAttn)
- **Better Quality**: Maintains higher PSNR at comparable sparsity levels
- **No Training**: Training-free approach vs distillation-based methods
- **Hardware Efficiency**: Optimized for GPU acceleration

### 3. Practical Impact
- **Real-world Deployment**: Single GPU acceleration for long-form video generation
- **Quality Guarantee**: Maintains visual fidelity while achieving significant speedup
- **Model Agnostic**: Framework applicable to different video diffusion models