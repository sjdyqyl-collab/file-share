# Phase 3: Experiments and Results

## Experimental Setup

### Models and Datasets
- **Primary Models**: Wan2.1 (14B parameters) and Hunyuan
- **Video Specifications**: 
  - Wan2.1: 81 frames at 768×1280 resolution
  - Hunyuan: 129 frames at 768×1280 resolution
- **Hardware**: Single H800 GPU
- **Implementation**: ThunderKittens framework with reference to STA

### Evaluation Metrics
**Quality Metrics:**
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- MSE (Mean Squared Error)
- VBench metrics: Subject Consistency, Background Consistency, Aesthetic Quality
- CLIPSIM and CLIP-Temp (CLIP-T) for text-video alignment

**Performance Metrics:**
- Attention sparsity rate (%)
- Attention latency (seconds)
- End-to-end speedup ratio

### Baseline Methods
1. **Full Attention**: Standard dense attention baseline
2. **STA (Sliding Tile Attention)**: Spatio-temporal locality with cubic windows
3. **Sparse VideoGen**: Static sparse patterns
4. **SpargeAttn**: Dynamic block sparsity based on cosine similarity

## Quantitative Results

### Wan2.1 Model Results (80K tokens)
| Method | Sparsity | SSIM↑ | PSNR↑ | MSE↓ | Latency(s) | Speedup |
|--------|----------|--------|--------|--------|------------|---------|
| Full Attention | 0% | - | - | - | 1092.168 | 1.00x |
| Sparse VideoGen | 32.08% | 0.529 | 15.9564 | 1894.3672 | 1200.148 | 0.91x |
| SpargeAttn | 32.27% | 0.6102 | 20.5163 | 676.0723 | 1065.796 | 1.02x |
| Compact Attention | 33.99% | 0.7754 | 23.7297 | 351.6015 | 663.824 | 1.65x |
| Compact Attention | 24.66% | 0.8147 | 25.2664 | 254.1789 | 758.176 | 1.44x |

### Hunyuan Model Results (127K tokens)
| Method | Sparsity | SSIM↑ | PSNR↑ | MSE↓ | Latency(s) | Speedup |
|--------|----------|--------|--------|--------|------------|---------|
| Full Attention | 0% | - | - | - | 1370.658 | 1.00x |
| Sparse VideoGen | 50.35% | 0.7254 | 20.4297 | 822.8567 | 1117.767 | 1.23x |
| SpargeAttn | 47.77% | 0.7794 | 23.5889 | 369.3112 | 1148.628 | 1.19x |
| Compact Attention | 62.36% | 0.9040 | 30.0822 | 105.1957 | 546.504 | 2.51x |
| Compact Attention | 52.90% | 0.9452 | 34.5506 | 35.1307 | 750.201 | 1.83x |

## Quality Analysis (VBench Metrics)

### Wan2.1 Quality Results
| Method | Sparsity | Subject Consistency | Background Consistency | Aesthetic Quality | CLIPSIM | CLIP-T |
|--------|----------|---------------------|------------------------|-------------------|---------|--------|
| Full Attention | 0% | 0.9681 | 0.9616 | 0.6486 | 0.2118 | 0.9985 |
| Sparse VideoGen | 32.08% | 0.9547 | 0.9565 | 0.6380 | 0.2116 | 0.9987 |
| SpargeAttn | 32.27% | 0.9357 | 0.9500 | 0.5320 | 0.2064 | 0.9982 |
| Compact Attention | 33.99% | 0.9659 | 0.9650 | 0.6480 | 0.2121 | 0.9985 |

### Hunyuan Quality Results
| Method | Sparsity | Subject Consistency | Background Consistency | Aesthetic Quality | CLIPSIM | CLIP-T |
|--------|----------|---------------------|------------------------|-------------------|---------|--------|
| Full Attention | 0% | 0.9736 | 0.9735 | 0.6542 | 0.2181 | 0.9995 |
| Sparse VideoGen | 50.35% | 0.9701 | 0.9722 | 0.6638 | 0.2014 | 0.9995 |
| SpargeAttn | 47.77% | 0.9664 | 0.9731 | 0.5794 | 0.2112 | 0.9995 |
| Compact Attention | 62.36% | 0.9716 | 0.9693 | 0.6531 | 0.2184 | 0.9995 |

## Ablation Studies

### Sparse Pattern Effectiveness
**Improvements by Pattern Type:**
- **Cross Patterns**: Dual Attention Windows increase sparsity by ~10%
- **Temporal Patterns**: Frame-group-wise patterns add ~3% improvement
- **Overall Sparsity Gain**: 9.8% more sparsity with τ=0.9, λ=0.011

### Parameter Sensitivity Analysis
**Recall Threshold Impact:**
- Lower recall thresholds increase sparsity but converge to upper bound determined by cost constraints
- Hunyuan (smaller model) achieves higher sparsity than Wan2.1
- Trade-off between acceleration (higher sparsity) and generation quality (lower recall)

### Early Denoising Sensitivity
- **Critical Period**: First 15 denoising steps require full attention
- **Quality Impact**: 1.02dB PSNR drop when applying sparse attention from step 0 vs step 15
- **Optimal Strategy**: Full attention for initial 15 steps, sparse attention for remaining steps

## Visual Performance Analysis

### Comparative Visual Results
- **Hunyuan**: Compact Attention achieves PSNR=24.24 at 62.36% sparsity vs PSNR=18.33 for Sparse VideoGen at 50.35% sparsity
- **Wan2.1**: Maintains quality with 33.99% sparsity achieving 1.65× speedup

### Robustness Analysis
- **Input Stability**: Consistent performance across diverse prompts and seeds
- **Parameter Robustness**: Compact_Attention (τ=0.9, λ=0.011) shows highest median PSNR with narrow variance

## Runtime Complexity Comparison

### Baseline Full Attention
- **Computation**: [N, N, d] where N = f×h×w (100K+ tokens for 128-frame 720p)
- **Memory**: O(N²) attention storage
- **Time**: O(N²d) complexity

### Compact Attention
- **Sparsity**: 33.99% (Wan2.1) to 62.36% (Hunyuan)
- **Computation**: [N, k, d] where k = (1-sparsity)×N
- **Memory**: O(Nk) sparse storage
- **Time**: O(Nkd) with 1.65-2.51× speedup
- **Communication**: Offline mask generation (one-time cost), negligible runtime overhead

## Key Findings
1. **Superior Trade-off**: Compact Attention achieves higher sparsity with better quality preservation
2. **Model Scaling**: Benefits more pronounced on larger models (Hunyuan shows 2.51× vs Wan2.1's 1.65×)
3. **Pattern-Specific Gains**: Cross-shaped and temporal patterns show highest sparsity improvements
4. **Practical Deployment**: Stable across inputs with offline optimization making it deployment-ready