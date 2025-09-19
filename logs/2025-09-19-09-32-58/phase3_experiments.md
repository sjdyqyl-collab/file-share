# Phase 3: Experiments Extraction

## 4.1 Experiment Setup

### Models Tested
1. **HunyuanVideo-T2V**
   - Resolution: 768p
   - Frames: 128
   - Latent size: 48×80 (divisible by 8×16 kernel)

2. **Wan2.1-T2V**
   - Resolutions: 512p and 768p
   - Frames: 80
   - Latent sizes: 32×48 (512p), 48×80 (768p)

### Implementation Details
- **Framework**: Block Sparse Attention
- **Pooling**: 8×16 kernel, stride=kernel size
- **Token Reduction**: 128× (8×16=128)
- **Full Attention**: Retained for first 25% of denoising steps
- **Hardware**: H100 GPU
- **Codebase**: Open-source implementations

### Evaluation Metrics
- **Quality**: VBench suite
  - Image quality
  - Subject consistency
  - Background consistency
  - Dynamic degree
  - Aesthetic quality
- **Similarity**: 
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)
- **Efficiency**: 
  - PFLOPs (total computation)
  - Latency (seconds)

### Dataset
- **Prompts**: Penguin Video Benchmark (HunyuanVideo release)
- **Fair comparison**: Full attention baseline for both methods

## 4.2 Main Results

### Table 1: Comprehensive Results

#### Wan2.1 (512p)
| Method | Sparse Ratio | PSNR↑ | SSIM↑ | LPIPS↓ | Img Qual. | Sub Cons. | Bakg Cons. | Dyn Deg. | Aes Qual. | PFLOPs↓ |
|--------|--------------|--------|--------|---------|-----------|-----------|------------|----------|-----------|---------|
| SVG | 0% | - | - | - | 65.1% | 95.0% | 95.9% | 44.7% | 58.9% | 145.65 |
| SVG | 55% | 25.61 | 83.63 | 10.42 | 65.2% | 94.8% | 95.9% | 45.2% | 58.9% | 99.26 |
| SVG | 75% | 23.66 | 78.80 | 15.05 | 64.7% | 94.5% | 95.7% | 45.7% | 58.6% | 91.12 |
| Ours | 0% | - | - | - | 69.3% | 95.5% | 96.7% | 47.6% | 61.5% | 145.65 |
| Ours | 55% | 25.13 | 84.77 | 8.43 | 69.2% | 95.5% | 96.6% | 47.6% | 61.5% | 99.26 |
| Ours | 75% | 23.10 | 79.07 | 12.37 | 69.0% | 95.4% | 96.5% | 46.9% | 61.5% | 91.12 |

#### Wan2.1 (768p)
| Method | Sparse Ratio | PSNR↑ | SSIM↑ | LPIPS↓ | Img Qual. | Sub Cons. | Bakg Cons. | Dyn Deg. | Aes Qual. | PFLOPs↓ |
|--------|--------------|--------|--------|---------|-----------|-----------|------------|----------|-----------|---------|
| SVG | 0% | - | - | - | 67.7% | 95.3% | 96.4% | 43.4% | 60.4% | 609.52 |
| SVG | 55% | 26.01 | 84.81 | 10.89 | 67.9% | 95.1% | 96.3% | 42.1% | 60.0% | 354.68 |
| SVG | 75% | 23.62 | 79.05 | 17.57 | 67.5% | 94.8% | 96.1% | 42.1% | 58.8% | 309.95 |
| Ours | 0% | - | - | - | 67.5% | 95.7% | 97.1% | 37.7% | 60.8% | 609.52 |
| Ours | 55% | 29.22 | 92.16 | 5.82 | 67.4% | 95.6% | 97.0% | 37.2% | 60.8% | 354.69 |
| Ours | 75% | 27.17 | 88.97 | 8.71 | 67.2% | 95.6% | 97.0% | 38.6% | 60.7% | 309.95 |

#### Hunyuan (768p)
| Method | Sparse Ratio | PSNR↑ | SSIM↑ | LPIPS↓ | Img Qual. | Sub Cons. | Bakg Cons. | Dyn Deg. | Aes Qual. | PFLOPs↓ |
|--------|--------------|--------|--------|---------|-----------|-----------|------------|----------|-----------|---------|
| Dense | 0% | - | - | - | 66.4% | 96.0% | 97.0% | 36.4% | 58.6% | 682.67 |
| SVG | 60% | 25.80 | 84.46 | 14.20 | 66.4% | 95.9% | 97.0% | 36.6% | 58.2% | 343.72 |
| SVG | 80% | 24.70 | 81.90 | 17.55 | 66.0% | 95.7% | 96.9% | 33.9% | 58.1% | 295.30 |
| SVG | 90% | 23.48 | 78.57 | 22.60 | 65.1% | 95.4% | 96.7% | 32.8% | 57.5% | 283.20 |
| Ours | 60% | 32.08 | 93.21 | 5.58 | 66.4% | 95.9% | 97.0% | 35.9% | 58.5% | 343.73 |
| Ours | 80% | 29.19 | 89.32 | 9.19 | 66.2% | 95.8% | 97.0% | 35.7% | 58.2% | 295.31 |
| Ours | 90% | 24.22 | 79.90 | 18.12 | 65.9% | 95.7% | 96.9% | 36.6% | 57.8% | 283.20 |

### 4.3 Latency Results (Figure 4)
- **Hunyuan (768p)**: 
  - 60% sparsity: 1.31× speedup
  - 80% sparsity: 1.58× speedup  
  - 90% sparsity: 1.75× speedup
- **Wan2.1 (768p)**:
  - 55% sparsity: 1.22× speedup
  - 75% sparsity: 1.42× speedup

## Key Findings

### Quality Preservation
- **Wan2.1 (768p)**: Ours achieves 29.22 PSNR vs 26.01 (SVG) at 55% sparsity
- **LPIPS**: Ours 5.82 vs 10.89 (SVG) at 55% sparsity for Wan2.1 (768p)
- **Hunyuan**: Ours maintains 66.4% image quality at 60% sparsity vs 66.4% for SVG

### Efficiency Gains
- **Computational**: Same PFLOPs as SVG (fair comparison)
- **Latency**: Up to 1.75× end-to-end acceleration
- **Scalability**: Better performance at higher sparsity ratios

### Visual Quality
- **Figure 5**: Ours maintains sharp details where SVG shows blur
- **Background consistency**: Ours preserves background better
- **Temporal coherence**: Better maintained in dynamic scenes

## 4.4 Ablation Study

### Pooling Kernel Comparison (Figure 6)
- **Average Pooling**: Better background quality, smoother transitions
- **Max Pooling**: Sharper edges but more artifacts, worse background
- **Conclusion**: Average pooling superior for draft attention

### Design Choices Impact
- **Per-frame design**: Preserves feature map completeness
- **Contiguous memory**: Enables efficient block processing
- **Deterministic reordering**: Critical for hardware efficiency

## Runtime Analysis

### Baseline (Full Attention)
- **Computation**: [n, n, d] matrix multiplication
- **Memory**: O(n²) attention matrix storage
- **Complexity**: O(n²d) total

### Proposed Method
- **Draft Attention**: [g, g, d] where g = n/128
- **Sparse Attention**: [n, n·r, d] where r = sparsity ratio
- **Reordering**: O(n) linear operations
- **Total**: O(n²rd + g²d + n)

### Communication Overhead
- **GPU Memory**: Minimal additional storage for reordering indices
- **Kernel Launches**: Reduced due to block-wise processing
- **Memory Coalescing**: Improved via contiguous layout