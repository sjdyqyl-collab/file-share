# DraftAttention Experiments - Detailed Analysis

## 4.1 Experiment Setup

### 4.1.1 Model Configurations
**HunyuanVideo-T2V**:
- Resolution: 768p
- Frames: 128
- Latent size: 48×80 (perfectly divisible by 8×16 kernel)

**Wan2.1-T2V**:
- Resolutions: 512p and 768p
- Frames: 80
- Latent sizes: 32×48 (512p) and 48×80 (768p)

### 4.1.2 Implementation Details
- **Full attention retention**: First 25% of denoising steps
- **Pooling kernel**: 8×16 with stride=kernel size
- **Token reduction**: 128× factor
- **Framework**: Block Sparse Attention [18]
- **Baseline comparison**: Sparse VideoGen (SVG) [16]

### 4.1.3 Evaluation Metrics
**Quality Metrics**:
- VBench scores: Image quality, Subject consistency, Background consistency, Dynamic degree, Aesthetic quality
- Similarity metrics: PSNR, SSIM, LPIPS

**Efficiency Metrics**:
- PFLOPs (including main diffusion transformer models)
- End-to-end latency on H100 GPU

**Prompts**: Penguin Video Benchmark [5] released by HunyuanVideo

## 4.2 Main Results

### 4.2.1 Wan2.1 Model Results

**512p Resolution**:
| Method | Sparse Ratio | PSNR | SSIM | LPIPS | Img. Qual. | Sub. Cons. | Bakg. Cons. | Dyn. Deg. | Aes. Qual. | PFLOPs |
|--------|--------------|------|------|-------|------------|------------|-------------|-----------|------------|--------|
| SVG    | 0%           | -    | -    | -     | 65.1%      | 95.0%      | 95.9%       | 44.7%     | 58.9%      | 145.65 |
| SVG    | 55%          | 25.61| 83.63| 10.42 | 65.2%      | 94.8%      | 95.9%       | 45.2%     | 58.9%      | 99.26  |
| SVG    | 75%          | 23.66| 78.80| 15.05 | 64.7%      | 94.5%      | 95.7%       | 45.7%     | 58.6%      | 91.12  |
| Ours   | 0%           | -    | -    | -     | 69.3%      | 95.5%      | 96.7%       | 47.6%     | 61.5%      | 145.65 |
| Ours   | 55%          | 25.13| 84.77| 8.43  | 69.2%      | 95.5%      | 96.6%       | 47.6%     | 61.5%      | 99.26  |
| Ours   | 75%          | 23.10| 79.07| 12.37 | 69.0%      | 95.4%      | 96.5%       | 46.9%     | 61.5%      | 91.12  |

**768p Resolution**:
| Method | Sparse Ratio | PSNR | SSIM | LPIPS | Img. Qual. | Sub. Cons. | Bakg. Cons. | Dyn. Deg. | Aes. Qual. | PFLOPs |
|--------|--------------|------|------|-------|------------|------------|-------------|-----------|------------|--------|
| SVG    | 0%           | -    | -    | -     | 67.7%      | 95.3%      | 96.4%       | 43.4%     | 60.4%      | 609.52 |
| SVG    | 55%          | 26.01| 84.81| 10.89 | 67.9%      | 95.1%      | 96.3%       | 42.1%     | 60.0%      | 354.68 |
| SVG    | 75%          | 23.62| 79.05| 17.57 | 67.5%      | 94.8%      | 96.1%       | 42.1%     | 58.8%      | 309.95 |
| Ours   | 0%           | -    | -    | -     | 67.5%      | 95.7%      | 97.1%       | 37.7%     | 60.8%      | 609.52 |
| Ours   | 55%          | 29.22| 92.16| 5.82  | 67.4%      | 95.6%      | 97.0%       | 37.2%     | 60.8%      | 354.69 |
| Ours   | 75%          | 27.17| 88.97| 8.71  | 67.2%      | 95.6%      | 97.0%       | 38.6%     | 60.7%      | 309.95 |

### 4.2.2 Hunyuan Model Results (768p)

| Method | Sparse Ratio | PSNR | SSIM | LPIPS | Img. Qual. | Sub. Cons. | Bakg. Cons. | Dyn. Deg. | Aes. Qual. | PFLOPs |
|--------|--------------|------|------|-------|------------|------------|-------------|-----------|------------|--------|
| Dense  | 0%           | -    | -    | -     | 66.4%      | 96.0%      | 97.0%       | 36.4%     | 58.6%      | 682.67 |
| SVG    | 60%          | 25.80| 84.46| 14.20 | 66.4%      | 95.9%      | 97.0%       | 36.6%     | 58.2%      | 343.72 |
| SVG    | 80%          | 24.70| 81.90| 17.55 | 66.0%      | 95.7%      | 96.9%       | 33.9%     | 58.1%      | 295.30 |
| SVG    | 90%          | 23.48| 78.57| 22.60 | 65.1%      | 95.4%      | 96.7%       | 32.8%     | 57.5%      | 283.20 |
| Ours   | 60%          | 32.08| 93.21| 5.58  | 66.4%      | 95.9%      | 97.0%       | 35.9%     | 58.5%      | 343.73 |
| Ours   | 80%          | 29.19| 89.32| 9.19  | 66.2%      | 95.8%      | 97.0%       | 35.7%     | 58.2%      | 295.31 |
| Ours   | 90%          | 24.22| 79.90| 18.12 | 65.9%      | 95.7%      | 96.9%       | 36.6%     | 57.8%      | 283.20 |

## 4.3 Performance Analysis

### 4.3.1 Speedup Results (768p, H100 GPU)
- **Hunyuan**: 1.75× speedup at 90% sparsity
- **Wan2.1**: 1.42× speedup at 75% sparsity
- **Wan2.1**: 1.22× speedup at 55% sparsity

### 4.3.2 Quality Preservation
**Key observations**:
1. **Less degradation**: DraftAttention maintains better quality at same sparsity levels
2. **Similarity metrics**: Consistently higher PSNR and SSIM, lower LPIPS
3. **Visual quality**: Better preservation of fine details and background consistency
4. **Perceptual metrics**: Maintained VBench scores across sparsity levels

### 4.3.3 Comparative Analysis
**Wan2.1 (768p) - 75% sparsity**:
- **PSNR improvement**: 27.17 vs 23.62 (+3.55 dB)
- **SSIM improvement**: 88.97 vs 79.05 (+9.92)
- **LPIPS improvement**: 8.71 vs 17.57 (-8.86)

**Hunyuan (768p) - 90% sparsity**:
- **PSNR improvement**: 24.22 vs 23.48 (+0.74 dB)
- **SSIM improvement**: 79.90 vs 78.57 (+1.33)
- **LPIPS improvement**: 18.12 vs 22.60 (-4.48)

## 4.4 Ablation Study

### 4.4.1 Pooling Strategy Comparison
**Average pooling vs Max pooling** (90% sparsity):
- **Average pooling**: Better generation quality, especially for background preservation
- **Max pooling**: Noticeable quality degradation with blurry pixels
- **Conclusion**: Average pooling provides more stable high-level representations

### 4.4.2 Visual Analysis
**Key visual observations**:
1. **SVG artifacts**: Blurry pixels and degraded fine details at 90% sparsity
2. **DraftAttention**: Maintained sharpness and detail preservation
3. **Background consistency**: Better maintained with average pooling
4. **Temporal coherence**: Improved stability across frames

## 4.5 Computational Efficiency

### 4.5.1 FLOPs Analysis
- **Linear reduction**: PFLOPs scale approximately linearly with sparsity ratio
- **Overhead**: Minimal additional computation for draft attention and reordering
- **Memory efficiency**: 128× reduction in draft attention computation

### 4.5.2 Hardware Utilization
- **GPU acceleration**: Optimized for H100 architecture
- **Memory access**: Coalesced through deterministic reordering
- **Kernel efficiency**: Compatible with FlashAttention and Block Sparse Attention

## 4.6 Limitations and Future Work

### 4.6.1 Current Limitations
1. **Fixed pooling kernel**: 8×16 may not be optimal for all resolutions
2. **Sparsity ratio**: Requires manual tuning for different models
3. **First 25% steps**: Full attention required for quality preservation
4. **Resolution dependency**: Padding needed for non-divisible latent sizes

### 4.6.2 Future Directions
1. **Adaptive pooling**: Dynamic kernel size selection
2. **Learned sparsity**: Data-driven sparsity ratio determination
3. **Quantization integration**: Further acceleration through weight quantization
4. **Multi-GPU scaling**: Distributed implementation for longer videos