# Phase 3: Experiments and Results

## Experiment Setup

### Models Tested
1. **HunyuanVideo-T2V**
   - Resolution: 768p
   - Frames: 128
   - Latent size: 48×80 (divisible by 8×16 kernel)

2. **Wan2.1-T2V**
   - Resolutions: 512p and 768p
   - Frames: 80
   - Latent sizes: 32×48 (512p) and 48×80 (768p)

### Implementation Details
- **Pooling Kernel**: 8×16 with stride=kernel size
- **Token Reduction**: 128× (from n to n/128)
- **Sparsity Ratios**: 55%, 60%, 75%, 80%, 90%
- **Hardware**: H100 GPU
- **Attention Framework**: Block Sparse Attention
- **Baseline Comparison**: Sparse VideoGen (SVG)

### Evaluation Metrics
- **Quality Metrics** (VBench):
  - Image Quality
  - Subject Consistency
  - Background Consistency
  - Dynamic Degree
  - Aesthetic Quality
- **Similarity Metrics**:
  - Peak Signal-to-Noise Ratio (PSNR) ↑
  - Structural Similarity Index Measure (SSIM) ↑
  - Learned Perceptual Image Patch Similarity (LPIPS) ↓
- **Computational Metrics**:
  - PFLOPs (total computation cost)
  - End-to-end latency (seconds)

## Main Results

### Wan2.1 Model Results

#### 512p Resolution
| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img. Qual. | Sub. Cons. | Bakg. Cons. | Dyn. Deg. | Aes. Qual. | PFLOPs ↓ |
|--------|--------------|--------|--------|---------|------------|------------|-------------|-----------|------------|----------|
| SVG | 0% | - | - | - | 65.1% | 95.0% | 95.9% | 44.7% | 58.9% | 145.65 |
| SVG | 55% | 25.61 | 83.63 | 10.42 | 65.2% | 94.8% | 95.9% | 45.2% | 58.9% | 99.26 |
| SVG | 75% | 23.66 | 78.80 | 15.05 | 64.7% | 94.5% | 95.7% | 45.7% | 58.6% | 91.12 |
| **Ours** | 0% | - | - | - | **69.3%** | **95.5%** | **96.7%** | **47.6%** | **61.5%** | 145.65 |
| **Ours** | 55% | 25.13 | 84.77 | **8.43** | **69.2%** | **95.5%** | **96.6%** | **47.6%** | **61.5%** | 99.26 |
| **Ours** | 75% | 23.10 | 79.07 | **12.37** | **69.0%** | **95.4%** | **96.5%** | **46.9%** | **61.5%** | 91.12 |

#### 768p Resolution
| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img. Qual. | Sub. Cons. | Bakg. Cons. | Dyn. Deg. | Aes. Qual. | PFLOPs ↓ |
|--------|--------------|--------|--------|---------|------------|------------|-------------|-----------|------------|----------|
| SVG | 0% | - | - | - | 67.7% | 95.3% | 96.4% | 43.4% | 60.4% | 609.52 |
| SVG | 55% | 26.01 | 84.81 | 10.89 | 67.9% | 95.1% | 96.3% | 42.1% | 60.0% | 354.68 |
| SVG | 75% | 23.62 | 79.05 | 17.57 | 67.5% | 94.8% | 96.1% | 42.1% | 58.8% | 309.95 |
| **Ours** | 0% | - | - | - | 67.5% | 95.7% | **97.1%** | 37.7% | 60.8% | 609.52 |
| **Ours** | 55% | **29.22** | **92.16** | **5.82** | **67.4%** | **95.6%** | **97.0%** | 37.2% | **60.8%** | 354.69 |
| **Ours** | 75% | **27.17** | **88.97** | **8.71** | **67.2%** | **95.6%** | **97.0%** | **38.6%** | **60.7%** | 309.95 |

### Hunyuan Model Results (768p)
| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img. Qual. | Sub. Cons. | Bakg. Cons. | Dyn. Deg. | Aes. Qual. | PFLOPs ↓ |
|--------|--------------|--------|--------|---------|------------|------------|-------------|-----------|------------|----------|
| Dense | 0% | - | - | - | 66.4% | 96.0% | 97.0% | 36.4% | 58.6% | 682.67 |
| SVG | 60% | 25.80 | 84.46 | 14.20 | 66.4% | 95.9% | 97.0% | 36.6% | 58.2% | 343.72 |
| SVG | 80% | 24.70 | 81.90 | 17.55 | 66.0% | 95.7% | 96.9% | 33.9% | 58.1% | 295.30 |
| SVG | 90% | 23.48 | 78.57 | 22.60 | 65.1% | 95.4% | 96.7% | 32.8% | 57.5% | 283.20 |
| **Ours** | 60% | **32.08** | **93.21** | **5.58** | **66.4%** | **95.9%** | **97.0%** | **35.9%** | **58.5%** | 343.73 |
| **Ours** | 80% | **29.19** | **89.32** | **9.19** | **66.2%** | **95.8%** | **97.0%** | **35.7%** | **58.2%** | 295.31 |
| **Ours** | 90% | 24.22 | 79.90 | **18.12** | **65.9%** | **95.7%** | **96.9%** | **36.6%** | **57.8%** | 283.20 |

## Latency Results

### Speedup Achieved (H100 GPU, 768p)
- **Hunyuan Model:**
  - 60% sparsity: 1.31× speedup
  - 80% sparsity: 1.58× speedup
  - 90% sparsity: **1.75× speedup**

- **Wan2.1 Model:**
  - 55% sparsity: 1.22× speedup
  - 75% sparsity: 1.42× speedup

## Visualization Results
- **Quality Preservation:** DraftAttention maintains better visual quality compared to SVG
- **Blur Reduction:** Significantly reduces blurry artifacts at high sparsity ratios (90%)
- **Background Consistency:** Better preservation of background details

## Ablation Study
- **Average vs Max Pooling:**
  - Average pooling achieves better generation quality
  - Particularly noticeable in background preservation
  - Max pooling leads to more artifacts and quality degradation

## Runtime Analysis
- **Baseline (Full Attention):** O(n²d) = [n, n, d]
- **DraftAttention:**
  - Draft computation: O((n/128)²d) = [n/128, n/128, d]
  - Sparse attention: O(rn²d) = [n, n, d] with sparsity ratio r
- **Communication Overhead:** Minimal due to deterministic reordering

## Key Findings
1. **Superior Quality:** DraftAttention outperforms SVG across all similarity metrics
2. **Better Sparsity Tolerance:** Maintains quality even at 90% sparsity
3. **Consistent Speedup:** Achieves 1.75× speedup at 90% sparsity
4. **No Training Required:** Plug-and-play integration with existing models