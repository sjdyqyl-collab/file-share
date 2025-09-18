# DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance (Condensed)

## Abstract
Diffusion transformer-based video generation models achieve excellent quality but suffer from high computational costs, with attention accounting for over 80% of total latency. We propose DraftAttention, a training-free framework that accelerates video diffusion transformers using dynamic sparse attention on GPUs. Our method computes low-resolution draft attention maps via downsampling, identifies spatial and temporal redundancy, and guides full-resolution sparse attention through hardware-friendly reordering. Theoretical analysis shows the low-resolution draft closely approximates full attention with bounded error. Experimental results demonstrate up to 1.75× end-to-end speedup on GPUs while outperforming existing sparse attention methods in video generation quality.

## 1. Introduction
Video diffusion transformers (DiTs) generate high-quality videos using spatiotemporal 3D full attention but face significant computational bottlenecks due to quadratic complexity with respect to context length. For Hunyuan Video, attention consumes over 80% of computation when generating videos longer than 16 seconds, limiting practical applications.

Existing sparse attention methods suffer from limitations: Sparse VideoGen uses static patterns with significant quality degradation, while AdaSpa provides prompt-dependent but still static patterns during diffusion. We address these limitations with DraftAttention, which adapts sparse attention patterns dynamically for each specific attention module.

## 2. Methodology

### 2.1 Draft Attention Framework

**Full Attention**: Given hidden states X ∈ ℝ^(n×d):
```
Attn(X) = Softmax(QK^⊤/√d)V
```

**Draft Attention**: Partition sequence into g regions {R_i}:
```
˜Q_i = (1/|R_i|) Σ_{j∈R_i} Q_j
˜K_i = (1/|R_i|) Σ_{j∈R_i} K_j
A_draft = Softmax(˜Q˜K^⊤/√d) ∈ ℝ^(g×g)
```

**Sparse Attention**: Extract top-r interactions, create mask M, compute:
```
SparseAttn(X) = Softmax((QK^⊤/√d) ⊙ ˜M)V
```

### 2.2 Theoretical Analysis

**Theorem 1 (Draft Error)**: For equal-sized regions:
```
‖S - S_draft‖_F ≤ δn
```

**Theorem 2 (Sparsity Error)**: Additional error from masking:
```
‖S - S⊙˜M‖_F ≤ n(δ+t)√(1-r)
```

### 2.3 Hardware Optimization

**Reordering Algorithm**: Groups spatially adjacent tokens contiguously in memory, enabling efficient block-wise computation and coalesced memory access. Uses 8×16 pooling kernel with 128× token reduction, matching efficient block sizes in attention frameworks.

## 3. Experiments

### 3.1 Setup
- **Models**: HunyuanVideo-T2V (768p, 128 frames), Wan2.1-T2V (512p/768p, 80 frames)
- **Metrics**: VBench, PSNR, SSIM, LPIPS, PFLOPs, latency (H100 GPU)
- **Baselines**: Sparse VideoGen (SVG), full attention
- **Sparsity ratios**: 55%, 60%, 75%, 80%, 90%

### 3.2 Results

**Quality Preservation**: DraftAttention consistently outperforms SVG at same sparsity levels:
- Wan2.1 768p at 75% sparsity: 27.17 vs 23.62 PSNR, 8.71 vs 17.57 LPIPS
- Hunyuan 768p at 90% sparsity: 24.22 vs 23.48 PSNR, 18.12 vs 22.60 LPIPS

**Speedup**: Up to 1.75× end-to-end acceleration on H100 GPU at 90% sparsity, with speedup increasing with sparsity ratio while maintaining quality comparable to dense baseline.

**Ablation**: Average pooling significantly outperforms max pooling for background preservation and overall quality.

## 4. Limitations and Improvements

### Current Limitations
1. **Fixed pooling kernel**: 8×16 kernel may not be optimal for all video types
2. **Uniform sparsity**: Same sparsity ratio applied across all attention layers
3. **No temporal adaptation**: Pooling treats temporal and spatial dimensions equally
4. **Limited to DiTs**: Method designed specifically for transformer-based models
5. **Hardware dependency**: Optimized for GPU execution patterns

### Proposed Improvements

#### 4.1 Adaptive Pooling Kernels
**Current**: Fixed 8×16 kernel
**Improvement**: Learn optimal kernel sizes per layer based on content characteristics
**Runtime**: [n, k, n] → [n/α, k/β, n/α] where α,β are learned scaling factors
**Benefit**: Better preserve fine details in high-motion regions

#### 4.2 Layer-wise Dynamic Sparsity
**Current**: Uniform sparsity ratio r across layers
**Improvement**: Dynamic sparsity based on layer importance and content complexity
**Runtime**: [n, k, n] → [n, k·r_i, n] where r_i varies per layer
**Benefit**: Allocate computation budget more efficiently

#### 4.3 Motion-Aware Temporal Pooling
**Current**: Uniform average pooling across time
**Improvement**: Adaptive pooling weights based on motion magnitude
**Runtime**: [n_t×n_s, k, n_t×n_s] → [n_t'×n_s', k, n_t'×n_s'] with motion-guided sampling
**Benefit**: Better preserve temporal dynamics in high-motion sequences

#### 4.4 Hybrid Attention Mechanisms
**Current**: Only sparse attention after draft guidance
**Improvement**: Combine sparse and dense attention based on content importance
**Runtime**: [n, k, n] → α·sparse + (1-α)·dense with learned α
**Benefit**: Maintain quality for complex scenes while accelerating simple ones

#### 4.5 Multi-Scale Draft Attention
**Current**: Single low-resolution draft
**Improvement**: Hierarchical draft attention at multiple resolutions
**Runtime**: [n, k, n] → Σ_i [n/2^i, k/2^i, n/2^i] for multiple scales
**Benefit**: Capture both local and global patterns more effectively

## 5. Runtime Analysis

### Baseline Method (Full Attention)
**Computation**: [n, k, n] where n = F×H×W (frames×height×width)
**Example**: For 768p video with 128 frames, n ≈ 128×48×80 = 491,520 tokens
**Complexity**: O(n²d) = O((4.9×10⁵)²×d) ≈ 2.4×10¹¹×d operations

### Proposed Method (DraftAttention)
**Stage 1 - Draft Attention**: [n/128, k/128, n/128] = [3840, k/128, 3840]
**Stage 2 - Sparse Attention**: [n, k·r, n] where r is sparsity ratio (0.1 for 90% sparse)
**Total**: O(n²d/128² + r·n²d) ≈ O(1.5×10⁷×d + 2.4×10¹⁰×d) for r=0.1
**Speedup**: ~10× theoretical reduction in attention computation

### Improved Method (Multi-Scale + Adaptive)
**Multi-scale drafts**: Σ_i [n/2^i, k/2^i, n/2^i] for i=1,2,3
**Adaptive sparse**: [n, k·r_i, n] with layer-specific r_i ∈ [0.05, 0.3]
**Expected**: Additional 1.5-2× speedup over current method with better quality
**Communication**: Minimal overhead for reordering operations O(n)

## 6. Conclusion

DraftAttention provides an effective training-free approach to accelerate video diffusion transformers by leveraging low-resolution draft attention for sparse computation guidance. With theoretical guarantees and practical speedup of 1.75× on GPUs, it represents a significant advancement in efficient video generation. Future improvements through adaptive kernels, dynamic sparsity, and motion-aware processing promise even greater efficiency gains while maintaining generation quality.