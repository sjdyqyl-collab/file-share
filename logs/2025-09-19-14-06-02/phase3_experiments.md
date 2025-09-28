# Phase 3: Experiments Extraction - DraftAttention Paper

## 4.1 Experiment Setup

### Models Tested
1. **HunyuanVideo-T2V** [5]
   - Resolution: 768p
   - Frames: 128
   - Latent size: 48×80 (divisible by 8×16 kernel)

2. **Wan2.1-T2V** [6]
   - Resolutions: 512p and 768p
   - Frames: 80
   - Latent sizes: 32×48 (512p) and 48×80 (768p)

### Technical Configuration
- **Pooling kernel**: 8×16 with stride=kernel size
- **Full attention retention**: First 25% of denoising steps
- **Implementation**: Block Sparse Attention [18]
- **Baseline comparison**: Sparse VideoGen (SVG) [16]
- **Hardware**: H100 GPU
- **Evaluation**: VBench [33] + PSNR/SSIM/LPIPS metrics

### Prompt Dataset
- **Source**: Penguin Video Benchmark [5] from HunyuanVideo
- **Evaluation metrics**:
  - Image quality
  - Subject consistency
  - Background consistency
  - Dynamic degree
  - Aesthetic quality

## 4.2 Main Results

### Quality Comparison (Table 1 Summary)

#### Wan2.1 (512p)
| Method | Sparsity | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img Qual | Sub Cons | Bakg Cons | PFLOPs ↓ |
|--------|----------|--------|--------|---------|----------|----------|-----------|----------|
| SVG    | 55%      | 25.61  | 83.63  | 10.42   | 65.2%    | 94.8%    | 95.9%     | 99.26    |
| Ours   | 55%      | 25.13  | 84.77  | 8.43    | 69.2%    | 95.5%    | 96.6%     | 99.26    |
| SVG    | 75%      | 23.66  | 78.80  | 15.05   | 64.7%    | 94.5%    | 95.7%     | 91.12    |
| Ours   | 75%      | 23.10  | 79.07  | 12.37   | 69.0%    | 95.4%    | 96.5%     | 91.12    |

#### Wan2.1 (768p)
| Method | Sparsity | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img Qual | PFLOPs ↓ |
|--------|----------|--------|--------|---------|----------|----------|
| SVG    | 55%      | 26.01  | 84.81  | 10.89   | 67.9%    | 354.68   |
| Ours   | 55%      | 29.22  | 92.16  | 5.82    | 67.4%    | 354.69   |
| SVG    | 75%      | 23.62  | 79.05  | 17.57   | 67.5%    | 309.95   |
| Ours   | 75%      | 27.17  | 88.97  | 8.71    | 67.2%    | 309.95   |

#### Hunyuan (768p)
| Method | Sparsity | PSNR ↑ | SSIM ↑ | LPIPS ↓ | PFLOPs ↓ |
|--------|----------|--------|--------|---------|----------|
| SVG    | 60%      | 25.80  | 84.46  | 14.20   | 343.72   |
| Ours   | 60%      | 32.08  | 93.21  | 5.58    | 343.73   |
| SVG    | 80%      | 24.70  | 81.90  | 17.55   | 295.30   |
| Ours   | 80%      | 29.19  | 89.32  | 9.19    | 295.31   |
| SVG    | 90%      | 23.48  | 78.57  | 22.60   | 283.20   |
| Ours   | 90%      | 24.22  | 79.90  | 18.12   | 283.20   |

### Key Quality Observations
- **Wan2.1 (768p)**: Our method achieves 29.22 vs 26.01 PSNR at 55% sparsity
- **LPIPS improvement**: 8.71 vs 17.57 at 75% sparsity (Wan2.1 768p)
- **Hunyuan**: 32.08 vs 25.80 PSNR at 60% sparsity
- **Minimal degradation**: Our 90% sparsity results comparable to SVG 60-80%

### Latency Results (Figure 4)
- **Hunyuan 768p**: 1.75× speedup at 90% sparsity
- **Wan2.1 768p**: 1.42× speedup at 75% sparsity, 1.22× at 55% sparsity
- **Hardware**: H100 GPU testing

## 4.3 Ablation Study

### Pooling Method Comparison (Figure 6)
**Tested**: Average pooling vs Max pooling at 90% sparsity
**Results**: 
- **Average pooling**: Superior generation quality, especially background preservation
- **Max pooling**: Noticeable quality degradation with blurry artifacts

### Visualization Results (Figure 5)
**90% sparsity comparison**:
- **SVG**: Noticeable blurry pixels, quality degradation
- **Ours**: Better maintains generation quality, closer to dense baseline
- **Examples**: Thames river camera movement, kitchen refrigerator scenes, spinning fan, falling dress

## Runtime Analysis

### Computational Cost Representation
- **Baseline full attention**: [n, d, n] matrix multiplication
- **DraftAttention**: 
  - Draft stage: [n/128, d, n/128] (128× reduction)
  - Sparse stage: [n, d, 0.1n] (90% sparsity = 0.1× computation)
  - Reordering: O(n) negligible overhead

### Performance Scaling
- **Linear speedup**: Speedup proportional to sparsity ratio
- **Quality preservation**: Superior to static sparse patterns (SVG)
- **Hardware efficiency**: Structured sparsity enables GPU optimization

## Experimental Limitations
1. **Resolution constraint**: Optimal at 512p/768p due to 8×16 kernel alignment
2. **Model scope**: Limited to HunyuanVideo and Wan2.1 architectures
3. **Hardware**: Results specific to H100 GPU architecture
4. **Prompt diversity**: Limited to Penguin Video Benchmark prompts