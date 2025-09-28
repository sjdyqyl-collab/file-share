# Phase 3: Experiments and Results - DraftAttention

## Experimental Setup

### Models and Configurations
| Model | Resolution | Frames | Latent Size | Attention Type |
|-------|------------|--------|-------------|----------------|
| Wan2.1-T2V | 512p | 80 | 32×48 | Full + Sparse |
| Wan2.1-T2V | 768p | 80 | 48×80 | Full + Sparse |
| HunyuanVideo-T2V | 768p | 128 | 48×80 | Full + Sparse |

### Implementation Details
- **Hardware**: NVIDIA H100 GPU
- **Pooling Kernel**: 8×16 with stride 8×16 (128× reduction)
- **Sparsity Ratios**: 55%, 60%, 75%, 80%, 90%
- **Baseline**: Full attention for first 25% denoising steps
- **Framework**: Block Sparse Attention implementation

## Main Results

### 1. Quantitative Performance Comparison

#### Wan2.1 Model Results
**512p Resolution:**
| Method | Sparsity | PSNR↑ | SSIM↑ | LPIPS↓ | Img.Quality | Subject Cons. | Background Cons. | Dynamic Deg. | Aesthetic |
|--------|----------|-------|-------|--------|-------------|---------------|------------------|--------------|-----------|
| SVG | 0% | - | - | - | 65.1% | 95.0% | 95.9% | 44.7% | 58.9% |
| SVG | 55% | 25.61 | 83.63 | 10.42 | 65.2% | 94.8% | 95.9% | 45.2% | 58.9% |
| SVG | 75% | 23.66 | 78.80 | 15.05 | 64.7% | 94.5% | 95.7% | 45.7% | 58.6% |
| **Ours** | 0% | - | - | - | **69.3%** | **95.5%** | **96.7%** | **47.6%** | **61.5%** |
| **Ours** | 55% | **25.13** | **84.77** | **8.43** | **69.2%** | **95.5%** | **96.6%** | **47.6%** | **61.5%** |
| **Ours** | 75% | **23.10** | **79.07** | **12.37** | **69.0%** | **95.4%** | **96.5%** | **46.9%** | **61.5%** |

**768p Resolution:**
| Method | Sparsity | PSNR↑ | SSIM↑ | LPIPS↓ | Img.Quality | Subject Cons. | Background Cons. | Dynamic Deg. | Aesthetic |
|--------|----------|-------|-------|--------|-------------|---------------|------------------|--------------|-----------|
| SVG | 0% | - | - | - | 67.7% | 95.3% | 96.4% | 43.4% | 60.4% |
| SVG | 55% | 26.01 | 84.81 | 10.89 | 67.9% | 95.1% | 96.3% | 42.1% | 60.0% |
| SVG | 75% | 23.62 | 79.05 | 17.57 | 67.5% | 94.8% | 96.1% | 42.1% | 58.8% |
| **Ours** | 0% | - | - | - | 67.5% | **95.7%** | **97.1%** | 37.7% | **60.8%** |
| **Ours** | 55% | **29.22** | **92.16** | **5.82** | **67.4%** | **95.6%** | **97.0%** | 37.2% | **60.8%** |
| **Ours** | 75% | **27.17** | **88.97** | **8.71** | **67.2%** | **95.6%** | **97.0%** | **38.6%** | **60.7%** |

#### HunyuanVideo Model Results
| Method | Sparsity | PSNR↑ | SSIM↑ | LPIPS↓ | Img.Quality | Subject Cons. | Background Cons. | Dynamic Deg. | Aesthetic |
|--------|----------|-------|-------|--------|-------------|---------------|------------------|--------------|-----------|
| Dense | 0% | - | - | - | 66.4% | 96.0% | 97.0% | 36.4% | 58.6% |
| SVG | 60% | 25.80 | 84.46 | 14.20 | 66.4% | 95.9% | 97.0% | 36.6% | 58.2% |
| SVG | 80% | 24.70 | 81.90 | 17.55 | 66.0% | 95.7% | 96.9% | 33.9% | 58.1% |
| SVG | 90% | 23.48 | 78.57 | 22.60 | 65.1% | 95.4% | 96.7% | 32.8% | 57.5% |
| **Ours** | 60% | **32.08** | **93.21** | **5.58** | **66.4%** | **95.9%** | **97.0%** | **35.9%** | **58.5%** |
| **Ours** | 80% | **29.19** | **89.32** | **9.19** | **66.2%** | **95.8%** | **97.0%** | **35.7%** | **58.2%** |
| **Ours** | 90% | **24.22** | **79.90** | **18.12** | **65.9%** | **95.7%** | **96.9%** | **36.6%** | **57.8%** |

### 2. Computational Cost Analysis

#### FLOPs Comparison
| Model | Method | Sparsity | PFLOPs↓ | Reduction |
|-------|--------|----------|---------|-----------|
| Wan2.1 (512p) | Dense | 0% | 145.65 | - |
| Wan2.1 (512p) | SVG/Ours | 55% | 99.26 | 31.8% |
| Wan2.1 (512p) | SVG/Ours | 75% | 91.12 | 37.5% |
| Wan2.1 (768p) | Dense | 0% | 609.52 | - |
| Wan2.1 (768p) | SVG/Ours | 55% | 354.69 | 41.8% |
| Wan2.1 (768p) | SVG/Ours | 75% | 309.95 | 49.1% |
| Hunyuan (768p) | Dense | 0% | 682.67 | - |
| Hunyuan (768p) | SVG/Ours | 60% | 343.73 | 49.6% |
| Hunyuan (768p) | SVG/Ours | 80% | 295.31 | 56.7% |
| Hunyuan (768p) | SVG/Ours | 90% | 283.20 | 58.5% |

### 3. Latency Results

#### GPU Performance (H100)
**Speedup Achieved:**
- **Wan2.1 (768p)**: 1.22× at 55% sparsity, 1.42× at 75% sparsity
- **Hunyuan (768p)**: 1.31× at 60% sparsity, 1.58× at 80% sparsity, **1.75× at 90% sparsity**

#### Runtime Analysis
- **Baseline Runtime**: ~2000s for dense attention (Hunyuan 768p)
- **Optimized Runtime**: ~1150s at 90% sparsity (1.75× speedup)
- **Matrix Representation**: 
  - Baseline: [614K, 6144, 614K] (Q×K^T computation)
  - DraftAttention: [614K, 6144, 61.4K] (90% sparsity → 10% active)

### 4. Ablation Studies

#### Pooling Strategy Comparison
**Average Pooling vs Max Pooling** (90% sparsity):
- **Average Pooling**: Better background quality, smoother transitions
- **Max Pooling**: Sharper edges but artifacts in smooth regions
- **Conclusion**: Average pooling superior for video generation quality

#### Sparsity Pattern Analysis
- **Static vs Dynamic**: Dynamic patterns outperform static approaches
- **Block Size Impact**: 8×16 optimal balance between efficiency and quality
- **Temporal Consistency**: Maintained across frames due to pooling strategy

### 5. Visualization Results

#### Quality Comparison (90% sparsity)
- **SVG**: Noticeable blurriness, especially in background regions
- **DraftAttention**: Maintains sharpness and temporal consistency
- **Dense Baseline**: Highest quality but computationally expensive

#### Key Visual Findings
1. **Background Preservation**: DraftAttention maintains background details better than SVG
2. **Temporal Coherence**: Smooth transitions across frames
3. **Edge Sharpness**: Better preservation of object boundaries
4. **Color Consistency**: Maintained across sparsity levels

## Experimental Insights

### Performance Patterns
1. **Quality vs Sparsity**: Gradual degradation with increasing sparsity, but DraftAttention maintains better quality than SVG
2. **Resolution Scaling**: Benefits increase with higher resolutions (768p shows better speedups)
3. **Model Agnostic**: Works across different model architectures (Wan2.1, HunyuanVideo)

### Critical Observations
1. **Sparsity Sweet Spot**: 75-80% sparsity offers good balance of quality and speed
2. **Pooling Effectiveness**: 8×16 pooling optimal for video characteristics
3. **Hardware Efficiency**: Reordering crucial for achieving theoretical speedups
4. **Training-Free**: No quality degradation from lack of fine-tuning