# Phase 3: Experiments and Results - DraftAttention Paper

## 4.1 Experiment Setup

### Models Tested
1. **HunyuanVideo-T2V** [5]
   - Resolution: 768p
   - Frames: 128
   - Latent size: 48×80 (perfectly divisible by 8×16 kernel)

2. **Wan2.1-T2V** [6]
   - Resolutions: 512p and 768p
   - Frames: 80
   - Latent sizes: 32×48 (512p) and 48×80 (768p)

### Implementation Details
- **Pooling kernel**: 8×16 with stride equal to kernel size
- **Token reduction**: Factor of 128
- **Attention framework**: Block Sparse Attention [18]
- **Baseline comparison**: Sparse VideoGen (SVG) [16]
- **Full attention**: Retained for first 25% of denoising steps
- **Hardware**: H100 GPU
- **Evaluation prompts**: Penguin Video Benchmark [5]

### Evaluation Metrics
- **VBench** [33]: Image quality, subject consistency, background consistency, dynamic degree, aesthetic quality
- **Similarity metrics**: PSNR, SSIM, LPIPS [34]
- **Computation cost**: PFLOPs (main diffusion transformer models)

## 4.2 Main Results

### Table 1: Comprehensive Results Comparison

#### Wan2.1 (512p)
| Method | Sparse Ratio | PSNR↑ | SSIM↑ | LPIPS↓ | Img Qual↑ | Sub Cons↑ | Bakg Cons↑ | Dyn Deg↑ | Aes Qual↑ | PFLOPs↓ |
|--------|--------------|--------|--------|---------|-----------|-----------|------------|----------|-----------|---------|
| SVG    | 0%   | - | - | - | 65.1% | 95.0% | 95.9% | 44.7% | 58.9% | 145.65 |
| SVG    | 55%  | 25.61 | 83.63 | 10.42 | 65.2% | 94.8% | 95.9% | 45.2% | 58.9% | 99.26 |
| SVG    | 75%  | 23.66 | 78.80 | 15.05 | 64.7% | 94.5% | 95.7% | 45.7% | 58.6% | 91.12 |
| **Ours** | 0%   | - | - | - | 69.3% | 95.5% | 96.7% | 47.6% | 61.5% | 145.65 |
| **Ours** | 55%  | 25.13 | 84.77 | 8.43 | 69.2% | 95.5% | 96.6% | 47.6% | 61.5% | 99.26 |
| **Ours** | 75%  | 23.10 | 79.07 | 12.37 | 69.0% | 95.4% | 96.5% | 46.9% | 61.5% | 91.12 |

#### Wan2.1 (768p)
| Method | Sparse Ratio | PSNR↑ | SSIM↑ | LPIPS↓ | Img Qual↑ | Sub Cons↑ | Bakg Cons↑ | Dyn Deg↑ | Aes Qual↑ | PFLOPs↓ |
|--------|--------------|--------|--------|---------|-----------|-----------|------------|----------|-----------|---------|
| SVG    | 0%   | - | - | - | 67.7% | 95.3% | 96.4% | 43.4% | 60.4% | 609.52 |
| SVG    | 55%  | 26.01 | 84.81 | 10.89 | 67.9% | 95.1% | 96.3% | 42.1% | 60.0% | 354.68 |
| SVG    | 75%  | 23.62 | 79.05 | 17.57 | 67.5% | 94.8% | 96.1% | 42.1% | 58.8% | 309.95 |
| **Ours** | 0%   | - | - | - | 67.5% | 95.7% | 97.1% | 37.7% | 60.8% | 609.52 |
| **Ours** | 55%  | 29.22 | 92.16 | 5.82 | 67.4% | 95.6% | 97.0% | 37.2% | 60.8% | 354.69 |
| **Ours** | 75%  | 27.17 | 88.97 | 8.71 | 67.2% | 95.6% | 97.0% | 38.6% | 60.7% | 309.95 |

#### Hunyuan (768p)
| Method | Sparse Ratio | PSNR↑ | SSIM↑ | LPIPS↓ | Img Qual↑ | Sub Cons↑ | Bakg Cons↑ | Dyn Deg↑ | Aes Qual↑ | PFLOPs↓ |
|--------|--------------|--------|--------|---------|-----------|-----------|------------|----------|-----------|---------|
| Dense  | 0%   | - | - | - | 66.4% | 96.0% | 97.0% | 36.4% | 58.6% | 682.67 |
| SVG    | 60%  | 25.80 | 84.46 | 14.20 | 66.4% | 95.9% | 97.0% | 36.6% | 58.2% | 343.72 |
| SVG    | 80%  | 24.70 | 81.90 | 17.55 | 66.0% | 95.7% | 96.9% | 33.9% | 58.1% | 295.30 |
| SVG    | 90%  | 23.48 | 78.57 | 22.60 | 65.1% | 95.4% | 96.7% | 32.8% | 57.5% | 283.20 |
| **Ours** | 60%  | 32.08 | 93.21 | 5.58 | 66.4% | 95.9% | 97.0% | 35.9% | 58.5% | 343.73 |
| **Ours** | 80%  | 29.19 | 89.32 | 9.19 | 66.2% | 95.8% | 97.0% | 35.7% | 58.2% | 295.31 |
| **Ours** | 90%  | 24.22 | 79.90 | 18.12 | 65.9% | 95.7% | 96.9% | 36.6% | 57.8% | 283.20 |

### Key Performance Highlights

#### Quality Preservation
- **Wan2.1 (768p)**: 29.22 vs 26.01 PSNR improvement at 55% sparsity
- **LPIPS improvement**: 5.82 vs 10.89 at 55% sparsity (768p)
- **Minimal degradation**: Less than 0.5% drop in VBench metrics at 90% sparsity

#### Speedup Results
- **90% sparsity**: 1.75× end-to-end acceleration on H100 GPU
- **80% sparsity**: 1.58× acceleration
- **60% sparsity**: 1.31× acceleration
- **Linear scaling**: Speedup proportional to sparsity ratio

### 4.3 Ablation Study

#### Pooling Kernel Comparison
- **Average pooling** (proposed): Better background quality preservation
- **Max pooling**: Noticeable degradation in background details
- **Visualization**: Figure 6 shows clear quality difference at 90% sparsity

#### Design Choices
- **Per-frame pooling**: Preserves feature map completeness
- **8×16 kernel**: Optimal for tested resolutions (32×48 and 48×80)
- **Block processing**: 128 tokens per block for efficient GPU execution

## Runtime Analysis

### Baseline Full Attention
- **Matrix multiplication**: [n, d, n] where n = F×H×W
- **Example**: [81920, 1152, 81920] for 768p×80 frames
- **Time complexity**: O(n²d)

### DraftAttention
- **Draft computation**: [g, d, g] where g = n/128
- **Example**: [640, 1152, 640] for draft attention
- **Sparse attention**: [n, d, r×n] where r = sparsity ratio
- **Total complexity**: O((r + 1/128)n²d)

### Communication Overhead
- **Reordering**: O(n) memory operations (negligible)
- **No distributed communication** in single-GPU setup
- **Memory coalescing**: Achieved through reordering algorithm

## Experimental Insights

### Quality vs Speed Trade-off
- **Sweet spot**: 80% sparsity for 1.58× speedup with <1% quality loss
- **Aggressive sparsity**: 90% sparsity acceptable for real-time applications
- **Model agnostic**: Consistent improvements across different architectures

### Practical Deployment
- **No retraining**: Direct deployment on pre-trained models
- **Resolution flexibility**: Supports any resolution with appropriate padding
- **Hardware efficiency**: Optimized for modern GPU architectures