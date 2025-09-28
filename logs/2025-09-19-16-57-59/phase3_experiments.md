# Phase 3: Experimental Details and Results - DraftAttention

## 1. Experiment Setup

### 1.1 Model Family
- **HunyuanVideo-T2V**: 768p resolution, 128 frames
- **Wan2.1-T2V**: 
  - 512p resolution, 80 frames
  - 768p resolution, 80 frames

### 1.2 Resolution Alignment
- **512p**: Latent size 32×48 (divisible by 8×16 pooling kernel)
- **768p**: Latent size 48×80 (divisible by 8×16 pooling kernel)
- **Padding**: Applied for non-divisible resolutions

### 1.3 Implementation Details
- **Framework**: Block Sparse Attention [18]
- **GPU**: H100 GPU for latency measurements
- **Baseline Comparison**: Sparse VideoGen (SVG) [16]
- **Attention Strategy**: Full attention for first 25% of denoising steps, then DraftAttention

### 1.4 Evaluation Metrics
- **VBench [33]**: Comprehensive video generation quality metrics
  - Image Quality
  - Subject Consistency
  - Background Consistency
  - Dynamic Degree
  - Aesthetic Quality
- **Similarity Metrics**:
  - Peak Signal-to-Noise Ratio (PSNR) ↑
  - Structural Similarity Index Measure (SSIM) ↑
  - Learned Perceptual Image Patch Similarity (LPIPS) ↓
- **Computational Cost**: PFLOPs (including main diffusion transformer models)
- **Prompts**: Penguin Video Benchmark [5] from HunyuanVideo

## 2. Main Results

### 2.1 Wan2.1 Model Results (512p)

| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img Qual. | Sub Cons. | Bakg Cons. | Dyn Deg. | Aes Qual. | PFLOPs ↓ |
|--------|--------------|--------|--------|---------|-----------|-----------|------------|----------|-----------|----------|
| SVG | 0% | - | - | - | 65.1% | 95.0% | 95.9% | 44.7% | 58.9% | 145.65 |
| SVG | 55% | 25.61 | 83.63 | 10.42 | 65.2% | 94.8% | 95.9% | 45.2% | 58.9% | 99.26 |
| SVG | 75% | 23.66 | 78.80 | 15.05 | 64.7% | 94.5% | 95.7% | 45.7% | 58.6% | 91.12 |
| **Ours** | 0% | - | - | - | 69.3% | 95.5% | 96.7% | 47.6% | 61.5% | 145.65 |
| **Ours** | 55% | 25.13 | 84.77 | 8.43 | 69.2% | 95.5% | 96.6% | 47.6% | 61.5% | 99.26 |
| **Ours** | 75% | 23.10 | 79.07 | 12.37 | 69.0% | 95.4% | 96.5% | 46.9% | 61.5% | 91.12 |

### 2.2 Wan2.1 Model Results (768p)

| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img Qual. | Sub Cons. | Bakg Cons. | Dyn Deg. | Aes Qual. | PFLOPs ↓ |
|--------|--------------|--------|--------|---------|-----------|-----------|------------|----------|-----------|----------|
| SVG | 0% | - | - | - | 67.7% | 95.3% | 96.4% | 43.4% | 60.4% | 609.52 |
| SVG | 55% | 26.01 | 84.81 | 10.89 | 67.9% | 95.1% | 96.3% | 42.1% | 60.0% | 354.68 |
| SVG | 75% | 23.62 | 79.05 | 17.57 | 67.5% | 94.8% | 96.1% | 42.1% | 58.8% | 309.95 |
| **Ours** | 0% | - | - | - | 67.5% | 95.7% | 97.1% | 37.7% | 60.8% | 609.52 |
| **Ours** | 55% | 29.22 | 92.16 | 5.82 | 67.4% | 95.6% | 97.0% | 37.2% | 60.8% | 354.69 |
| **Ours** | 75% | 27.17 | 88.97 | 8.71 | 67.2% | 95.6% | 97.0% | 38.6% | 60.7% | 309.95 |

### 2.3 Hunyuan Model Results (768p)

| Method | Sparse Ratio | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Img Qual. | Sub Cons. | Bakg Cons. | Dyn Deg. | Aes Qual. | PFLOPs ↓ |
|--------|--------------|--------|--------|---------|-----------|-----------|------------|----------|-----------|----------|
| Dense | 0% | - | - | - | 66.4% | 96.0% | 97.0% | 36.4% | 58.6% | 682.67 |
| SVG | 60% | 25.80 | 84.46 | 14.20 | 66.4% | 95.9% | 97.0% | 36.6% | 58.2% | 343.72 |
| SVG | 80% | 24.70 | 81.90 | 17.55 | 66.0% | 95.7% | 96.9% | 33.9% | 58.1% | 295.30 |
| SVG | 90% | 23.48 | 78.57 | 22.60 | 65.1% | 95.4% | 96.7% | 32.8% | 57.5% | 283.20 |
| **Ours** | 60% | 32.08 | 93.21 | 5.58 | 66.4% | 95.9% | 97.0% | 35.9% | 58.5% | 343.73 |
| **Ours** | 80% | 29.19 | 89.32 | 9.19 | 66.2% | 95.8% | 97.0% | 35.7% | 58.2% | 295.31 |
| **Ours** | 90% | 24.22 | 79.90 | 18.12 | 65.9% | 95.7% | 96.9% | 36.6% | 57.8% | 283.20 |

## 3. Latency Results

### 3.1 Speedup Achievements
- **Hunyuan Model (768p)**:
  - 60% sparsity: 1.31× speedup
  - 80% sparsity: 1.58× speedup
  - 90% sparsity: 1.75× speedup
- **Wan2.1 Model (768p)**:
  - 55% sparsity: 1.22× speedup
  - 75% sparsity: 1.42× speedup

### 3.2 Runtime Complexity Comparison

#### Baseline (Full Attention)
- **Matrix Multiplication**: [n, n, d]
- **Computation**: O(n²d)
- **Memory**: O(n²)

#### Proposed (DraftAttention)
- **Draft Phase**: [g, g, d] where g = n/128
- **Sparse Phase**: [n, n·r, d] where r is sparsity ratio
- **Total**: O(n²rd + g²d)

#### Practical Runtime Examples
- **90% Sparsity (r=0.1)**: 
  - Theoretical speedup: 10× for sparse phase
  - Actual speedup: 1.75× (including draft overhead and memory operations)
- **Memory Transfer**: [n, d] for reordering operations (minimal overhead)

## 4. Ablation Studies

### 4.1 Pooling Strategy Comparison
- **Average Pooling**: Better generation quality, especially for background
- **Max Pooling**: Inferior results with noticeable artifacts
- **Visualization**: Figure 6 shows clear quality difference at 90% sparsity

### 4.2 Sparsity Ratio Analysis
- **Quality vs Speed Trade-off**: Higher sparsity → faster speed but lower quality
- **Sweet Spot**: 80-90% sparsity provides good balance
- **Degradation**: DraftAttention shows slower quality degradation compared to SVG

## 5. Visualization Results

### 5.1 Qualitative Comparison
- **SVG at 90% sparsity**: Noticeable blurry pixels, quality degradation
- **DraftAttention at 90% sparsity**: Maintains sharpness and detail preservation
- **Prompt Examples**: 
  - "The banks of the Thames, as the camera moves vertically from low to high"
  - "A blue long dress fell from the balcony clothes rack and dropped into the water"

### 5.2 Failure Cases
- **High Motion Scenes**: Some temporal consistency issues at 90% sparsity
- **Fine Details**: Very fine textures may be lost at extreme sparsity ratios
- **Edge Cases**: Complex lighting conditions show more degradation

## 6. Reproducibility Details

### 6.1 Code Availability
- **GitHub**: https://github.com/shawnricecake/draft-attention
- **Framework**: PyTorch with Block Sparse Attention
- **Hardware**: H100 GPU (80GB memory)

### 6.2 Hyperparameters
- **Pooling Kernel**: 8×16 fixed
- **Sparsity Ratios**: 55%, 60%, 75%, 80%, 90%
- **Full Attention Steps**: First 25% of denoising process
- **Batch Size**: 1 (single video generation)
- **Precision**: FP16 for computation

## 7. Limitations and Future Work

### 7.1 Current Limitations
- **Resolution Dependency**: Optimal for resolutions divisible by 8×16
- **Temporal Coherence**: Some degradation in long sequences
- **Memory Overhead**: Reordering requires additional memory

### 7.2 Future Improvements
- **Adaptive Pooling**: Dynamic kernel size based on content
- **Quantization Integration**: Combine with INT8/INT4 quantization
- **Multi-GPU Scaling**: Distributed sparse attention across GPUs