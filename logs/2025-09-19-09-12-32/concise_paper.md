# DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance - Concise Version

## Abstract
Diffusion transformer-based video generation models suffer from high computational costs, with attention mechanisms accounting for over 80% of total latency. We propose DraftAttention, a training-free framework that accelerates video diffusion transformers using dynamic sparse attention. By downsampling feature maps via 8×16 average pooling to create low-resolution draft attention maps, we achieve 128× token reduction while maintaining generation quality. The method provides up to 1.75× end-to-end speedup on GPUs without retraining.

## 1. Problem Statement
- **Challenge**: Generating 8 seconds of 720p video takes tens of minutes
- **Root Cause**: Attention mechanism has O(n²d) complexity with hundreds of thousands of tokens
- **Opportunity**: Significant redundancy exists in video attention patterns

## 2. Methodology

### 2.1 Core Innovation: Draft Attention
**Two-stage process:**
1. **Draft Stage**: Compute low-resolution attention map
   - Input: [n, d] tokens → Output: [n/128, n/128, d] draft attention
   - Runtime: [n/128, n/128, d] (128× reduction)

2. **Guided Sparse Stage**: Apply sparsity to full resolution
   - Use draft map to create binary mask M ∈ {0,1}^(g×g)
   - Full attention: [r·n, n, d] where r = sparsity ratio
   - Total: [n/128, n/128, d] + [r·n, n, d]

### 2.2 Hardware Optimization
**Reordering Algorithm:**
- Groups 128 tokens into contiguous memory blocks
- Enables coalesced GPU memory access
- Compatible with FlashAttention and Block Sparse Attention
- Overhead: O(n) for permutation operations

### 2.3 Theoretical Guarantees
**Error Bounds:**
- Draft attention error: ||S - S_draft||_F ≤ δn
- Sparsity error: ||S - S⊙M̃||_F ≤ n(δ + t)√(1-r)
- Where δ = max local deviation, t = sparsity threshold

## 3. Experimental Results

### 3.1 Setup
- **Models**: HunyuanVideo-T2V (768p, 128f), Wan2.1-T2V (512p/768p, 80f)
- **Hardware**: H100 GPU
- **Metrics**: PSNR, SSIM, LPIPS, VBench quality scores

### 3.2 Performance Comparison
| Model | Sparsity | Speedup | PSNR↑ | SSIM↑ | LPIPS↓ |
|-------|----------|---------|-------|-------|--------|
| Wan2.1-768p | 75% | 1.42× | 27.17 | 88.97 | 8.71 |
| Hunyuan-768p | 90% | 1.75× | 24.22 | 79.90 | 18.12 |

### 3.3 Key Findings
- **Quality**: Maintains >95% of dense model quality
- **Speed**: Up to 1.75× end-to-end acceleration
- **Memory**: 128× token reduction in draft computation
- **Training**: Zero additional training required

## 4. Limitations and Improvements

### 4.1 Current Limitations
1. Fixed 8×16 kernel size
2. Static sparsity ratios
3. Limited temporal modeling
4. Hardware-specific optimization

### 4.2 Proposed Improvements

#### 4.2.1 Adaptive Kernel Selection
- **Method**: Content-aware kernel size selection
- **Runtime**: [n, d, k] → [n/g'(x), d, k'(x)]
- **Expected**: 5-10% quality improvement

#### 4.2.2 Dynamic Sparsity Scheduling
- **Method**: Step-wise sparsity adjustment
- **Runtime**: [g, g, d] + [r(t)·n, n, d], r(t) ∈ [0.5, 0.95]
- **Expected**: Additional 1.2× speedup

#### 4.2.3 Quantization Integration
- **Method**: INT4 quantization for attention weights
- **Runtime**: [n/128, n/128, d/4] + [r·n, n, d/4]
- **Expected**: 50-75% memory reduction

#### 4.2.4 Distributed Inference
- **Method**: Multi-GPU draft attention
- **Runtime**: [n/128, n/128, d/p] + [r·n/p, n, d] + all-reduce[n²/128², 1]
- **Expected**: Linear scaling with GPU count

## 5. Conclusion
DraftAttention achieves significant acceleration of video diffusion transformers through low-resolution attention guidance while maintaining generation quality. The method's training-free nature and hardware-friendly design make it immediately deployable for practical video generation applications.

## Runtime Summary
- **Dense Baseline**: [n, n, d] - O(n²d)
- **DraftAttention**: [n/128, n/128, d] + [r·n, n, d] - O(n²d/128² + r·n²d)
- **Improved**: [n/g'(x), n/g'(x), d] + [r(t)·n, n, d] - Dynamic complexity
- **Quantized**: [n/128, n/128, d/4] + [r·n, n, d/4] - 4× memory reduction