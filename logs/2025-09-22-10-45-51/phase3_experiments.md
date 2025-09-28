# Phase 3: Experiments Extraction - Compact Attention

## Experimental Setup

### Models and Datasets
**Primary Models:**
- Wan2.1 (14B parameters) - 81 frames at 768×1280 resolution
- Hunyuan - 129 frames at 768×1280 resolution

**Hardware Configuration:**
- Single H800 GPU
- ThunderKittens framework implementation
- FlashAttention-2 integration

**Evaluation Metrics:**
- **Quality Metrics**: SSIM, PSNR, MSE
- **VBench Metrics**: Subject Consistency, Background Consistency, Aesthetic Quality
- **Text-Video Alignment**: CLIPSIM, CLIP-Temp (CLIP-T)
- **Computational**: Attention sparsity rate, attention latency
- **Benchmark**: Open-Sora benchmark

### Baseline Methods
1. **STA (Sliding Tile Attention)**: Spatio-temporal locality with cubic windows
2. **Sparse VideoGen**: Static sparse patterns
3. **Sparge Attention**: Dynamic sparse patterns with cosine similarity thresholds

## Quantitative Results

### Performance Comparison Table

#### Wan2.1 Model Results (80K tokens)
| Method | Sparsity | SSIM ↑ | PSNR ↑ | MSE ↓ | Latency (s) | Speedup |
|--------|----------|---------|---------|--------|-------------|---------|
| Full Attention | 0% | - | - | - | 1092.168 | 1.00x |
| Sparse VideoGen | 32.08% | 0.529 | 15.9564 | 1894.3672 | 1200.148 | 0.91x |
| SpargeAttn | 32.27% | 0.6102 | 20.5163 | 676.0723 | 1065.796 | 1.02x |
| Compact Attention | 33.99% | 0.7754 | 23.7297 | 351.6015 | 663.824 | 1.65x |
| Compact Attention | 24.66% | 0.8147 | 25.2664 | 254.1789 | 758.176 | 1.44x |

#### Hunyuan Model Results (127K tokens)
| Method | Sparsity | SSIM ↑ | PSNR ↑ | MSE ↓ | Latency (s) | Speedup |
|--------|----------|---------|---------|--------|-------------|---------|
| Full Attention | 0% | - | - | - | 1370.658 | 1.00x |
| Sparse VideoGen | 50.35% | 0.7254 | 20.4297 | 822.8567 | 1117.767 | 1.23x |
| SpargeAttn | 47.77% | 0.7794 | 23.5889 | 369.3112 | 1148.628 | 1.19x |
| Compact Attention | 62.36% | 0.9040 | 30.0822 | 105.1957 | 546.504 | 2.51x |
| Compact Attention | 52.90% | 0.9452 | 34.5506 | 35.1307 | 750.201 | 1.83x |

### VBench Quality Metrics

#### Wan2.1 VBench Results
| Method | Sparsity | Subject Consistency | Background Consistency | Aesthetic Quality | CLIPSIM | CLIP-T |
|--------|----------|-------------------|----------------------|-------------------|---------|--------|
| Full Attention | 0% | 0.9681 | 0.9616 | 0.6486 | 0.2118 | 0.9985 |
| Sparse VideoGen | 32.08% | 0.9547 | 0.9565 | 0.6380 | 0.2116 | 0.9987 |
| SpargeAttn | 32.27% | 0.9357 | 0.9500 | 0.5320 | 0.2064 | 0.9982 |
| Compact Attention | 33.99% | 0.9659 | 0.9650 | 0.6480 | 0.2121 | 0.9985 |
| Compact Attention | 24.66% | 0.9674 | 0.9638 | 0.6459 | 0.2122 | 0.9986 |

#### Hunyuan VBench Results
| Method | Sparsity | Subject Consistency | Background Consistency | Aesthetic Quality | CLIPSIM | CLIP-T |
|--------|----------|-------------------|----------------------|-------------------|---------|--------|
| Full Attention | 0% | 0.9736 | 0.9735 | 0.6542 | 0.2181 | 0.9995 |
| Sparse VideoGen | 50.35% | 0.9701 | 0.9722 | 0.6638 | 0.2014 | 0.9995 |
| SpargeAttn | 47.77% | 0.9664 | 0.9731 | 0.5794 | 0.2112 | 0.9995 |
| Compact Attention | 62.36% | 0.9716 | 0.9693 | 0.6531 | 0.2184 | 0.9995 |
| Compact Attention | 52.90% | 0.9723 | 0.9735 | 0.6536 | 0.2184 | 0.9995 |

## Ablation Studies

### Sparse Pattern Effectiveness
**Experimental Setup**: Categorized attention heads from Wan2.1 inference phase into pattern groups

**Results**:
- **Dual Attention Windows**: Increased cross-pattern sparsity by ~10%
- **Frame-group-wise patterns**: Additional 3% improvement through temporal variation
- **Overall improvement**: 9.8% more sparsity with τ=0.9, λ=0.011

**Pattern-Specific Improvements**:
| Pattern Type | Cubic Window | Frame-group-wise | Dual Windows |
|--------------|---------------|------------------|---------------|
| Local Patterns | 0.726 | 0.758 | 0.766 |
| Cross Patterns | 0.385 | 0.406 | 0.516 |
| Global Patterns | 0.078 | 0.085 | 0.099 |
| Time-Variant | 0.441 | 0.472 | 0.567 |
| Time-Invariant | 0.306 | 0.317 | 0.385 |
| **Overall** | **0.361** | **0.370** | **0.459** |

### Sensitivity Analysis

#### Recall Threshold Sensitivity
**Setup**: Fixed cost thresholds (λ=0.011 for Wan, λ=0.04 for Hunyuan)
**Findings**:
- Hunyuan achieves higher sparsity than Wan (smaller model)
- Sparsity converges to upper bound determined by cost constraint
- Need to balance acceleration (higher sparsity) vs quality (lower recall threshold)

#### Early Denoising Sensitivity
**Setup**: Applied sparse attention at different denoising steps
**Results**:
- **1.02dB PSNR drop** when full attention applied only in final 15 steps vs first 15
- **Critical finding**: Full attention essential in early timesteps for quality preservation
- **Optimal strategy**: Full attention for initial 15 steps, sparse for remaining steps

**Visual Performance Progression**:
| Starting Step | PSNR | Latency (s) |
|---------------|------|-------------|
| Step 0 | 11.29 | 640.97 |
| Step 5 | 13.44 | 646.78 |
| Step 10 | 15.87 | 655.72 |
| Step 15 | 19.17 | 663.82 |
| Step 20 | 22.49 | 674.64 |
| Full Attention | - | 1544.00 |

## Robustness Analysis

### Input Condition Stability
**Setup**: Tested across diverse text prompts and random seeds
**Metric**: PSNR distribution analysis
**Finding**: Compact Attention (τ=0.9, λ=0.011) shows:
- Highest median and mean PSNR values
- Narrow interquartile range
- Consistent high-quality outputs with limited variance

### Cross-Model Generalization
**Wan2.1**: 1.65× speedup at 33.99% sparsity
**Hunyuan**: 2.51× speedup at 62.36% sparsity
**Quality Preservation**: Both models maintain comparable visual quality to full attention baselines

## Visual Quality Assessment

### Comparative Visual Performance
- **STA**: Notable quality degradation despite speed improvement
- **Sparse VideoGen**: Substantial quality degradation due to uniform sparsity
- **SpargeAttn**: Limited stability with adaptive top-k selection
- **Compact Attention**: Superior quality preservation at higher sparsity levels

### Key Visual Findings
1. **Detail Preservation**: Compact Attention maintains fine-grained details better than baselines
2. **Temporal Consistency**: Superior performance in maintaining temporal coherence
3. **Text-Video Alignment**: Maintains or improves CLIP scores compared to full attention
4. **Aesthetic Quality**: Preserves aesthetic scores while achieving higher sparsity

## Computational Efficiency

### Attention Computation Time
- **Hunyuan**: 68-72% of total generation time in full attention
- **Wan2.1**: Similar attention dominance in computation
- **Compact Attention**: Reduces attention time by 60-75% while preserving quality

### Memory Efficiency
- Tile-based processing reduces memory footprint
- Block-wise computation enables larger sequence processing
- Offline mask computation eliminates runtime overhead

## Experimental Conclusions

1. **Superior Acceleration**: Compact Attention achieves highest speedup (2.51×) among sparse methods
2. **Quality Preservation**: Maintains or improves quality metrics at higher sparsity levels
3. **Pattern Effectiveness**: Dual windows and frame-group patterns significantly improve sparsity
4. **Robustness**: Stable performance across diverse inputs and conditions
5. **Optimal Timing**: Early denoising steps require full attention for quality preservation