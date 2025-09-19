# DraftAttention: Fast Video Diffusion via Low-Resolution Attention Guidance

## Abstract
Video diffusion transformers achieve excellent generation quality but suffer from prohibitive computational costs, with attention consuming over 80% of total latency. We propose DraftAttention, a training-free framework that accelerates video diffusion transformers using low-resolution attention guidance. Our method computes lightweight draft attention maps via average pooling (8×16 kernel, 128× token reduction), identifies critical regions through structured sparsity, and applies deterministic reordering for hardware-efficient execution. Theoretical analysis proves bounded approximation error, while experiments demonstrate up to 1.75× speedup on GPUs with superior quality compared to existing sparse attention methods.

## 1. Introduction
Video diffusion transformers (DiTs) excel in generation quality but face computational bottlenecks due to quadratic attention complexity. Generating 8 seconds of 720p video requires tens of minutes, limiting practical deployment. While sparse attention methods exist, they either use static patterns or lack fine-grained adaptation.

**Key Insight:** Video content exhibits spatial and temporal redundancy that can be exploited through hierarchical attention patterns. We propose using low-resolution draft attention to guide sparse full-resolution computation.

## 2. Methodology

### 2.1 Problem Formulation
Given hidden states X ∈ ℝ^(n×d) across F frames with spatial resolution H×W, where n = F×H×W. Full attention computes:
```
Attn(X) = Softmax(QK^⊤/√d) V,  complexity: O(n²d) = [n, n, d]
```

### 2.2 Draft Attention Framework

#### Stage 1: Low-Resolution Draft
```
1. Downsample queries/keys via average pooling:
   eQ_i = (1/|R_i|) Σ_{j∈R_i} Q_j
   eK_i = (1/|R_i|) Σ_{j∈R_i} K_j
   
2. Compute draft attention:
   A_draft = Softmax(eQ eK^⊤/√d) ∈ ℝ^(g×g)
   
   Complexity: [n/128, n/128, d] (128× reduction)
```

#### Stage 2: Structured Sparsity
```
1. Create sparsity mask M ∈ {0,1}^(g×g) by selecting top-r entries
2. Lift to token resolution: M̃_uv = M_ij for u ∈ R_i, v ∈ R_j
3. Compute sparse attention: SparseAttn = Softmax((QK^⊤/√d) ⊙ M̃) V

Complexity: [n, n, d] with sparsity ratio r ∈ [0.5, 0.9]
```

#### Stage 3: Reordering for Efficiency
Deterministic token reordering ensures spatial patches are contiguous in memory, enabling efficient block-wise computation.

### 2.3 Theoretical Analysis

**Theorem 1 (Draft Error):** ∥S - S_draft∥_F ≤ δn
**Theorem 2 (Sparsity Error):** ∥S - S⊙M̃∥_F ≤ n(δ + t)√(1-r)

These bounds guarantee controlled approximation error while achieving significant computational savings.

## 3. Experiments

### 3.1 Setup
- **Models:** HunyuanVideo-T2V (768p, 128f), Wan2.1-T2V (512p/768p, 80f)
- **Hardware:** H100 GPU with Block Sparse Attention
- **Metrics:** PSNR, SSIM, LPIPS, VBench quality scores

### 3.2 Results

| Model | Method | Sparsity | PSNR↑ | SSIM↑ | LPIPS↓ | Speedup |
|-------|--------|----------|--------|--------|--------|---------|
| Wan2.1-768p | SVG | 75% | 23.62 | 79.05 | 17.57 | 1.42× |
| Wan2.1-768p | **Ours** | 75% | **27.17** | **88.97** | **8.71** | **1.42×** |
| Hunyuan-768p | SVG | 90% | 23.48 | 78.57 | 22.60 | 1.75× |
| Hunyuan-768p | **Ours** | 90% | **24.22** | **79.90** | **18.12** | **1.75×** |

**Key Achievements:**
- Up to 1.75× end-to-end speedup on H100 GPU
- Superior quality preservation at 90% sparsity
- No training required, plug-and-play integration

## 4. Limitations and Future Work

### Current Limitations
1. Fixed 8×16 pooling kernel may not suit all content
2. Static sparsity ratios throughout denoising
3. Limited temporal modeling for fast motion
4. Full-precision computation only

### Proposed Enhancements: AdaptiveDraftAttention

#### 4.1 Adaptive Kernel Selection
Dynamic kernel sizing based on resolution and content complexity:
```
Runtime: [n/k, n/k, d] where k ∈ [64, 256] adapts to content
Expected speedup: 2.3× vs 1.75× original
```

#### 4.2 Dynamic Sparsity Scheduling
Time-varying sparsity r(t) based on denoising progress:
```
Runtime: [n, n, d] with r(t) ∈ [0.5, 0.95]
Quality improvement: Better preservation across all steps
```

#### 4.3 Multi-Scale Processing
Hierarchical attention with quantization:
```
Runtime: [n/k₁, n/k₁, d] + [n/k₂, n/k₂, d] + [n, n, d] with mixed precision
Memory savings: 40-50% additional reduction
```

## 5. Conclusion
DraftAttention addresses the computational bottleneck in video diffusion transformers through intelligent use of low-resolution attention guidance. With theoretical guarantees and practical efficiency, it enables scalable video generation while maintaining quality. Future enhancements promise even greater speedups and broader applicability.

## Runtime Summary
- **Baseline:** [n, n, d] full attention
- **DraftAttention:** [n/128, n/128, d] + [n, n, d] with sparsity r → 1.75× speedup
- **Enhanced:** [n/k, n/k, d] + [n, n, d] with dynamic r(t) → 2.3× projected speedup