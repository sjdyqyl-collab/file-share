# Phase 3: Experiments Extraction - DraftAttention Paper

## 4.1 Experiment Setup

### Model Family
- **HunyuanVideo-T2V**: 768p resolution, 128 frames
- **Wan2.1-T2V**: 512p and 768p resolutions, 80 frames
- **Resolution alignment**: 512p (32×48 latent) and 768p (48×80 latent) perfectly divisible by 8×16 kernel

### Implementation Details
- **Attention framework**: Block Sparse Attention
- **Baseline comparison**: Sparse VideoGen (SVG)
- **Fallback strategy**: Full attention for first 25% of denoising steps
- **Hardware**: H100 GPU for latency testing

### Evaluation Metrics
- **Quality**: VBench metrics (Image quality, Subject consistency, Background consistency, Dynamic degree, Aesthetic quality)
- **Similarity**: PSNR, SSIM, LPIPS
- **Efficiency**: PFLOPs (computation cost), Latency (seconds)
- **Dataset**: Penguin Video Benchmark prompts

## 4.2 Main Results

### Quantitative Results (Table 1)

#### Wan2.1 (512p)
| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img. Qual. | PFLOPs ↓ |
|--------|-------------|--------|--------|---------|------------|----------|
| SVG    | 55%         | 25.61  | 83.63  | 10.42   | 65.2%      | 99.26    |
| Ours   | 55%         | 25.13  | 84.77  | 8.43    | 69.2%      | 99.26    |
| SVG    | 75%         | 23.66  | 78.80  | 15.05   | 64.7%      | 91.12    |
| Ours   | 75%         | 23.10  | 79.07  | 12.37   | 69.0%      | 91.12    |

#### Wan2.1 (768p)
| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img. Qual. | PFLOPs ↓ |
|--------|-------------|--------|--------|---------|------------|----------|
| SVG    | 55%         | 26.01  | 84.81  | 10.89   | 67.9%      | 354.68   |
| Ours   | 55%         | 29.22  | 92.16  | 5.82    | 67.4%      | 354.69   |
| SVG    | 75%         | 23.62  | 79.05  | 17.57   | 67.5%      | 309.95   |
| Ours   | 75%         | 27.17  | 88.97  | 8.71    | 67.2%      | 309.95   |

#### Hunyuan (768p)
| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img. Qual. | PFLOPs ↓ |
|--------|-------------|--------|--------|---------|------------|----------|
| SVG    | 60%         | 25.80  | 84.46  | 14.20   | 66.4%      | 343.72   |
| Ours   | 60%         | 32.08  | 93.21  | 5.58    | 66.4%      | 343.73   |
| SVG    | 80%         | 24.70  | 81.90  | 17.55   | 66.0%      | 295.30   |
| Ours   | 80%         | 29.19  | 89.32  | 9.19    | 66.2%      | 295.31   |
| SVG    | 90%         | 23.48  | 78.57  | 22.60   | 65.1%      | 283.20   |
| Ours   | 90%         | 24.22  | 79.90  | 18.12   | 65.9%      | 283.20   |

### Speedup Results (Figure 4)
- **Hunyuan (768p)**: 1.75× acceleration at 90% sparsity
- **Wan2.1 (768p)**: 1.42× acceleration at 75% sparsity
- **Baseline**: Dense attention (0% sparsity)

## 4.3 Ablation Study

### Pooling Kernel Comparison (Figure 6)
- **Average pooling**: Better generation quality, especially for background
- **Max pooling**: Inferior visual results with artifacts
- **Conclusion**: Average pooling better captures smooth transitions in video content

### Key Findings
1. **Quality preservation**: DraftAttention maintains better similarity metrics (PSNR, SSIM, LPIPS) compared to SVG
2. **Higher sparsity tolerance**: Performance degrades more gracefully at high sparsity ratios
3. **Resolution scalability**: Consistent improvements across 512p and 768p resolutions
4. **Model generalizability**: Effective on both HunyuanVideo and Wan2.1 architectures

## Computational Analysis

### Runtime Complexity
- **Baseline (Dense Attention)**: O(n²) where n = tokens per frame × frames
- **DraftAttention**: O(g² + r·n²) where g = n/128, r = sparsity ratio
- **Memory overhead**: O(n) for reordering indices

### Practical Performance
- **Token reduction**: 128× reduction in draft attention computation
- **Block processing**: 128 tokens per kernel for efficient GPU utilization
- **Memory alignment**: Contiguous blocks enable coalesced access
- **Kernel efficiency**: Compatible with FlashAttention and Block Sparse Attention

## Limitations Identified
1. **Resolution dependency**: Optimal with resolutions divisible by 8×16 kernel
2. **Padding overhead**: Non-divisible resolutions require padding
3. **Sparsity threshold**: Performance degrades beyond 90% sparsity
4. **Model-specific tuning**: Different optimal sparsity ratios across models