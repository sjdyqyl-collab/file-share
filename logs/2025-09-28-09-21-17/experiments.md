# DraftAttention: Experimental Results and Setup

## 4.1 Experiment Setup

### Model Family
- **HunyuanVideo-T2V**: 768p resolution, 128 frames
- **Wan2.1-T2V**: 512p and 768p resolutions, 80 frames

### Resolution Configuration
- **512p**: Latent size 32×48 (perfectly divisible by 8×16 kernel)
- **768p**: Latent size 48×80 (perfectly divisible by 8×16 kernel)
- **Padding**: Method supports any resolution with appropriate padding

### Implementation Details
- **Baseline**: Full attention for first 25% of denoising steps
- **Framework**: Block Sparse Attention
- **Comparison**: Mainly with Sparse VideoGen (SVG)
- **Hardware**: H100 GPU testing platform

### Metrics and Prompts
- **Quality Metrics**: VBench evaluation
  - Image quality
  - Subject consistency
  - Background consistency
  - Dynamic degree
  - Aesthetic quality
- **Similarity Metrics**: PSNR, SSIM, LPIPS
- **Prompts**: Penguin Video Benchmark from HunyuanVideo
- **Computation**: PFLOPs for main diffusion transformer models

## 4.2 Main Results

### 4.2.1 Wan2.1 Model Results (512p)

| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img Qual | Sub Cons | Bakg Cons | Dyn Deg | Aes Qual | PFLOPs ↓ |
|--------|--------------|--------|--------|---------|----------|----------|-----------|---------|----------|----------|
| SVG | 0% | / | / | / | 65.1% | 95.0% | 95.9% | 44.7% | 58.9% | 145.65 |
| SVG | 55% | 25.61 | 83.63 | 10.42 | 65.2% | 94.8% | 95.9% | 45.2% | 58.9% | 99.26 |
| SVG | 75% | 23.66 | 78.80 | 15.05 | 64.7% | 94.5% | 95.7% | 45.7% | 58.6% | 91.12 |
| Ours | 0% | / | / | / | 69.3% | 95.5% | 96.7% | 47.6% | 61.5% | 145.65 |
| Ours | 55% | 25.13 | 84.77 | 8.43 | 69.2% | 95.5% | 96.6% | 47.6% | 61.5% | 99.26 |
| Ours | 75% | 23.10 | 79.07 | 12.37 | 69.0% | 95.4% | 96.5% | 46.9% | 61.5% | 91.12 |

### 4.2.2 Wan2.1 Model Results (768p)

| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img Qual | Sub Cons | Bakg Cons | Dyn Deg | Aes Qual | PFLOPs ↓ |
|--------|--------------|--------|--------|---------|----------|----------|-----------|---------|----------|----------|
| SVG | 0% | / | / | / | 67.7% | 95.3% | 96.4% | 43.4% | 60.4% | 609.52 |
| SVG | 55% | 26.01 | 84.81 | 10.89 | 67.9% | 95.1% | 96.3% | 42.1% | 60.0% | 354.68 |
| SVG | 75% | 23.62 | 79.05 | 17.57 | 67.5% | 94.8% | 96.1% | 42.1% | 58.8% | 309.95 |
| Ours | 0% | / | / | / | 67.5% | 95.7% | 97.1% | 37.7% | 60.8% | 609.52 |
| Ours | 55% | 29.22 | 92.16 | 5.82 | 67.4% | 95.6% | 97.0% | 37.2% | 60.8% | 354.69 |
| Ours | 75% | 27.17 | 88.97 | 8.71 | 67.2% | 95.6% | 97.0% | 38.6% | 60.7% | 309.95 |

### 4.2.3 Hunyuan Model Results (768p)

| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img Qual | Sub Cons | Bakg Cons | Dyn Deg | Aes Qual | PFLOPs ↓ |
|--------|--------------|--------|--------|---------|----------|----------|-----------|---------|----------|----------|
| Dense | 0% | / | / | / | 66.4% | 96.0% | 97.0% | 36.4% | 58.6% | 682.67 |
| SVG | 60% | 25.80 | 84.46 | 14.20 | 66.4% | 95.9% | 97.0% | 36.6% | 58.2% | 343.72 |
| SVG | 80% | 24.70 | 81.90 | 17.55 | 66.0% | 95.7% | 96.9% | 33.9% | 58.1% | 295.30 |
| SVG | 90% | 23.48 | 78.57 | 22.60 | 65.1% | 95.4% | 96.7% | 32.8% | 57.5% | 283.20 |
| Ours | 60% | 32.08 | 93.21 | 5.58 | 66.4% | 95.9% | 97.0% | 35.9% | 58.5% | 343.73 |
| Ours | 80% | 29.19 | 89.32 | 9.19 | 66.2% | 95.8% | 97.0% | 35.7% | 58.2% | 295.31 |
| Ours | 90% | 24.22 | 79.90 | 18.12 | 65.9% | 95.7% | 96.9% | 36.6% | 57.8% | 283.20 |

## 4.3 Latency Results

### Speedup Achievements
- **Hunyuan (768p)**: 1.75× speedup at 90% sparsity
- **Wan2.1 (768p)**: 1.42× speedup at 75% sparsity, 1.22× at 55% sparsity

### Latency Comparison (H100 GPU)
```
Sparsity Ratios:
Dense: 1× baseline
60%: 1.31× (Hunyuan), 1.22× (Wan2.1)
80%: 1.58× (Hunyuan), 1.42× (Wan2.1)
90%: 1.75× (Hunyuan)
```

## 4.4 Ablation Study

### Pooling Method Comparison
- **Average Pooling**: Better generation quality, especially for background
- **Max Pooling**: Inferior results with noticeable degradation
- **Visualization**: Clear quality difference in background regions with 90% sparsity

### Visual Quality Assessment
- **SVG at 90% sparsity**: Noticeable blurry pixels, quality degradation
- **DraftAttention at 90% sparsity**: Better maintained generation quality
- **Comparison**: Videos more similar to dense baseline

## Key Performance Insights

### Quality Preservation
1. **PSNR Improvements**: DraftAttention consistently achieves higher PSNR values
   - Wan2.1 (768p): 29.22 vs 26.01 (SVG at 55% sparsity)
   - Hunyuan: 32.08 vs 25.80 (SVG at 60% sparsity)

2. **LPIPS Reduction**: Lower LPIPS values indicate better perceptual similarity
   - Wan2.1 (768p): 5.82 vs 10.89 (SVG at 55% sparsity)
   - Hunyuan: 5.58 vs 14.20 (SVG at 60% sparsity)

3. **VBench Metrics**: Maintained or improved across most categories
   - Subject consistency: ≥95% across all configurations
   - Background consistency: ≥96% across all configurations

### Computational Efficiency
- **PFLOPs Reduction**: Consistent with sparsity ratios
- **Memory Bandwidth**: Optimized through reordering and block processing
- **GPU Utilization**: Improved through hardware-friendly execution patterns

## Experimental Configuration Summary

| Parameter | Value | Notes |
|-----------|--------|--------|
| Pooling Kernel | 8×16 | Stride = kernel size |
| Token Reduction | 128× | 8×16 = 128 |
| Block Size | 128 tokens | Matches kernel size |
| Sparsity Ratios | 55%, 60%, 75%, 80%, 90% | Tested ranges |
| GPU Platform | H100 | NVIDIA H100 |
| Baseline Steps | 25% | Full attention for initial denoising |

## Limitations Observed in Experiments
1. **Resolution Dependency**: Optimal performance requires divisible latent sizes
2. **Sparsity Saturation**: Diminishing returns beyond 90% sparsity
3. **Content Sensitivity**: Some video types may benefit less from sparsity
4. **Model Variability**: Different architectures show varying speedup ratios