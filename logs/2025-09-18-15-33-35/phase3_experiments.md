# Phase 3: Experiments Extraction - Compact Attention

## 1. Experimental Setup

### 1.1 Models and Architecture
- **Primary Models**: 
  - Wan2.1 (14B parameters)
  - Hunyuan
- **Hardware**: Single H800 GPU
- **Implementation**: ThunderKittens framework with FlashAttention-2

### 1.2 Video Specifications
- **Wan2.1**: 81 frames at 768×1280 resolution
- **Hunyuan**: 129 frames at 768×1280 resolution
- **Token counts**: 
  - Wan2.1: ~80K tokens
  - Hunyuan: ~127K tokens

### 1.3 Evaluation Metrics
- **Quality Metrics**:
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)
  - MSE (Mean Squared Error)
  - VBench metrics: Subject Consistency, Background Consistency, Aesthetic Quality
  - CLIPSIM & CLIP-T (text-video alignment)
- **Performance Metrics**:
  - Attention sparsity rate (%)
  - Attention latency (seconds)
  - End-to-end speedup (×)

### 1.4 Baseline Methods
1. **Full Attention**: Standard dense attention computation
2. **STA (Sliding Tile Attention)**: Spatio-temporal locality with cubic windows
3. **Sparse VideoGen**: Static sparse patterns
4. **Sparge Attention**: Dynamic sparse patterns with cosine similarity thresholds

## 2. Main Results

### 2.1 Wan2.1 Results (80K tokens)
| Method | Sparsity | SSIM↑ | PSNR↑ | MSE↓ | Latency(s) | Speedup |
|--------|----------|--------|--------|--------|------------|---------|
| Full Attention | 0% | - | - | - | 1092.168 | 1.00× |
| Sparse VideoGen | 32.08% | 0.529 | 15.9564 | 1894.3672 | 1200.148 | 0.91× |
| SpargeAttn | 32.27% | 0.6102 | 20.5163 | 676.0723 | 1065.796 | 1.02× |
| **Compact Attention** | 33.99% | 0.7754 | 23.7297 | 351.6015 | 663.824 | **1.65×** |
| **Compact Attention** | 24.66% | 0.8147 | 25.2664 | 254.1789 | 758.176 | **1.44×** |

### 2.2 Hunyuan Results (127K tokens)
| Method | Sparsity | SSIM↑ | PSNR↑ | MSE↓ | Latency(s) | Speedup |
|--------|----------|--------|--------|--------|------------|---------|
| Full Attention | 0% | - | - | - | 1370.658 | 1.00× |
| Sparse VideoGen | 50.35% | 0.7254 | 20.4297 | 822.8567 | 1117.767 | 1.23× |
| SpargeAttn | 47.77% | 0.7794 | 23.5889 | 369.3112 | 1148.628 | 1.19× |
| **Compact Attention** | 62.36% | 0.9040 | 30.0822 | 105.1957 | 546.504 | **2.51×** |
| **Compact Attention** | 52.90% | 0.9452 | 34.5506 | 35.1307 | 750.201 | **1.83×** |

### 2.3 Quality Preservation Analysis

#### 2.3.1 VBench Evaluation - Wan2.1
| Method | Sparsity | Subject Consistency | Background Consistency | Aesthetic Quality | CLIPSIM | CLIP-T |
|--------|----------|---------------------|------------------------|-------------------|---------|--------|
| Full Attention | 0% | 0.9681 | 0.9616 | 0.6486 | 0.2118 | 0.9985 |
| Sparse VideoGen | 32.08% | 0.9547 | 0.9565 | 0.6380 | 0.2116 | 0.9987 |
| SpargeAttn | 32.27% | 0.9357 | 0.9500 | 0.5320 | 0.2064 | 0.9982 |
| **Compact Attention** | 33.99% | **0.9659** | **0.9650** | **0.6480** | **0.2121** | **0.9985** |

#### 2.3.2 VBench Evaluation - Hunyuan
| Method | Sparsity | Subject Consistency | Background Consistency | Aesthetic Quality | CLIPSIM | CLIP-T |
|--------|----------|---------------------|------------------------|-------------------|---------|--------|
| Full Attention | 0% | 0.9736 | 0.9735 | 0.6542 | 0.2181 | 0.9995 |
| Sparse VideoGen | 50.35% | 0.9701 | 0.9722 | 0.6638 | 0.2014 | 0.9995 |
| SpargeAttn | 47.77% | 0.9664 | 0.9731 | 0.5794 | 0.2112 | 0.9995 |
| **Compact Attention** | 62.36% | **0.9716** | **0.9693** | **0.6531** | **0.2184** | **0.9995** |

## 3. Ablation Studies

### 3.1 Sparse Pattern Effectiveness
**Analysis of individual pattern contributions:**

| Pattern Type | Baseline | +Frame-group-wise | +Dual Windows | Improvement |
|--------------|----------|-------------------|---------------|-------------|
| Locality | 0.726 | 0.758 | 0.766 | +5.5% |
| Cross | 0.385 | 0.406 | 0.516 | +34.0% |
| Global | 0.078 | 0.085 | 0.099 | +27.0% |
| Time-Variant | 0.441 | 0.472 | 0.567 | +28.6% |
| Time-Invariant | 0.306 | 0.317 | 0.385 | +25.8% |
| **Overall** | **0.361** | **0.370** | **0.459** | **+27.1%** |

**Key findings:**
- Dual Attention Windows contribute most to cross pattern improvement (+10%)
- Frame-group-wise patterns add ~3% additional improvement
- Combined approach achieves 9.8% more sparsity than cubic window baseline

### 3.2 Parameter Sensitivity Analysis

#### 3.2.1 Recall Threshold Impact
- **Observation**: Sparsity converges to upper bound determined by cost constraint as recall threshold decreases
- **Trade-off**: Lower recall threshold → higher sparsity but potential quality degradation
- **Model difference**: Hunyuan (smaller) achieves higher sparsity than Wan2.1 under same thresholds

#### 3.2.2 Early Denoising Sensitivity
**Experimental setup**: Apply Compact Attention from different denoising steps

| Start Step | PSNR | Latency(s) | Quality Impact |
|------------|------|------------|----------------|
| Step 0 | 11.29 | 640.97 | Severe degradation |
| Step 5 | 13.44 | 646.78 | High degradation |
| Step 10 | 15.87 | 655.72 | Moderate degradation |
| Step 15 | 19.17 | 663.82 | Acceptable quality |
| Step 20 | 22.49 | 674.64 | Minimal degradation |
| Full Attention | ~25.3 | 1544.00 | Baseline |

**Key insight**: Full attention needed for first 15 denoising steps to preserve quality

## 4. Robustness Analysis

### 4.1 Stability Across Input Conditions
- **PSNR distribution analysis** across different prompts and seeds
- **Compact_Attention (τ=0.9, λ=0.011)** shows:
  - Highest median and mean PSNR values
  - Narrow interquartile range (limited variance)
  - Consistent performance across stochastic perturbations

### 4.2 Visual Quality Comparison
- **Hunyuan 117-frame comparison**:
  - STA: PSNR = 25.30, MSE = 123.40, Sparsity = 58.37%
  - Compact Attention: PSNR = 27.27, MSE = 193.39, Sparsity = 62.36%
- **Key observation**: Compact Attention achieves better quality at higher sparsity

## 5. Computational Complexity Analysis

### 5.1 Attention Complexity Comparison
- **Full Attention**: O(n²d) where n = sequence length, d = hidden dimension
- **Compact Attention**: O(s·n²d) where s = sparsity rate (s < 1)
- **Practical reduction**: 
  - Wan2.1: 33.99% sparsity → ~66% computation reduction
  - Hunyuan: 62.36% sparsity → ~38% computation remaining

### 5.2 Memory Efficiency
- **Tile-based processing**: Reduces memory footprint through block computation
- **Sparsity benefit**: Memory usage scales with sparsity rate
- **Hardware optimization**: Compatible with FlashAttention-2 memory-efficient attention