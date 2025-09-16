# Training-free and Adaptive Sparse Attention for Efficient Long Video Generation

## Abstract
Generating high-fidelity long videos with Diffusion Transformers (DiTs) is often hindered by significant latency, primarily due to the computational demands of attention mechanisms. For instance, generating an 8-second 720p video (110K tokens) with HunyuanVideo takes about 600 PFLOPs, with around 500 PFLOPs consumed by attention computations. To address this issue, we propose AdaSpa, the first Dynamic Pattern and Online Precise Search sparse attention method. Firstly, to realize the Dynamic Pattern, we introduce a blockified pattern to efficiently capture the hierarchical sparsity inherent in DiTs. This is based on our observation that sparse characteristics of DiTs exhibit hierarchical and blockified structures between and within different modalities. This blockified approach significantly reduces the complexity of attention computation while maintaining high fidelity in the generated videos. Secondly, to enable Online Precise Search, we propose the Fused LSE-Cached Search with Head-adaptive Hierarchical Block Sparse Attention. This method is motivated by our finding that DiTs' sparse pattern and LSE vary w.r.t. inputs, layers, and heads, but remain invariant across denoising steps. By leveraging this invariance across denoising steps, it adapts to the dynamic nature of DiTs and allows for precise, real-time identification of sparse indices with minimal overhead. AdaSpa is implemented as an adaptive, plug-and-play solution and can be integrated seamlessly with existing DiTs, requiring neither additional fine-tuning nor a dataset-dependent profiling. Extensive experiments validate that AdaSpa delivers substantial acceleration across various models while preserving video quality, establishing itself as a robust and scalable approach to efficient video generation.

## 1. Introduction

Diffusion Transformers (DiTs) have revolutionized video generation but face computational bottlenecks due to the O(n²) complexity of attention mechanisms. For an 8-second 720p video, attention computations consume ~83% of total FLOPs. Current sparse attention methods fall into two categories:

- **Static Patterns**: Fixed patterns (e.g., sliding window) that cannot adapt to DiTs' dynamic sparsity
- **Dynamic Patterns**: Require expensive offline search or inaccurate online approximation

We propose AdaSpa, the first method combining Dynamic Pattern with Online Precise Search, achieving 1.78× speedup while maintaining quality.

## 2. Preliminaries

### 2.1 Diffusion Transformers and 3D Full Attention
DiTs use 3D Full Attention integrating spatial, temporal, and cross-modal dependencies. Sequence length L = f·h·w + t, where f=frames, h×w=spatial resolution, t=text tokens.

### 2.2 FlashAttention
Block-wise attention computation reducing memory from O(L²) to O(Lb) where b is block size.

### 2.3 Sparse Attention
Defined by masking function M ∈ {0,1}^(L×L). Effectiveness measured by Recall = Σ W_attn(sparse) / Σ W_attn(dense).

## 3. Sparse Pattern Characteristics in DiTs

**Observation 1**: Hierarchical structure makes continuous patterns ineffective. DiTs exhibit blockified sparsity patterns that achieve better Recall (0.93-1.0 vs 0.12-0.54 for continuous patterns).

**Observation 2**: Sparse patterns vary with inputs, layers, and heads, but remain invariant across denoising steps. This enables LSE caching for efficient online search.

## 4. Methodology

### 4.1 Problem Formulation
Block Sparse Attention partitions sequence into L/B chunks, achieving complexity reduction from O(L²d) to O((1-sparsity)L²d).

### 4.2 AdaSpa Design

**Fused LSE-Cached Online Search**:
- **Phase 1**: Fused Online Search (two-pass) computes full attention and caches LSE
- **Phase 2**: LSE-Cached Search (one-pass) reuses cached LSE for subsequent steps

**Head-adaptive Hierarchical Block Sparse Attention**:
- Adjusts sparsity per-head based on Recall performance
- Maintains average sparsity while optimizing individual heads

### 4.3 Implementation
- 2,000+ lines Python, 1,000+ lines Triton
- Plug-and-play interface: single line replacement
- Default: sparsity=0.8, block_size=64, Ts={10,30}

## 5. Experiments

### 5.1 Main Results
| Method | VBench↑ | PSNR↑ | Speedup |
|--------|---------|--------|---------|
| HunyuanVideo (Full) | 80.10 | - | 1.00× |
| + MInference | 79.17 | 22.53 | 1.27× |
| + Sparse VideoGen | 79.39 | 27.61 | 1.58× |
| + AdaSpa (ours) | 80.13 | 29.07 | 1.78× |

| Method | VBench↑ | PSNR↑ | Speedup |
|--------|---------|--------|---------|
| CogVideoX1.5 (Full) | 81.16 | - | 1.00× |
| + MInference | 65.30 | 10.31 | 1.39× |
| + Sparse VideoGen | 79.40 | 18.98 | 1.52× |
| + AdaSpa (ours) | 81.90 | 23.25 | 1.66× |

### 5.2 Ablation Studies
- **Quality-Sparsity Trade-off**: AdaSpa maintains quality across sparsity levels
- **Warmup Impact**: 10-step warmup optimal for quality-speed balance
- **Search Strategy**: {10,30} search steps achieve best quality-efficiency trade-off

### 5.3 Scaling Study
Achieves up to 4.01× speedup for 24-second videos, demonstrating excellent scalability.

## 6. Conclusion
AdaSpa provides the first effective Dynamic Pattern + Online Precise Search solution for DiTs, achieving 1.78× speedup with negligible quality loss through blockified patterns and LSE-cached search.

## References
[Complete reference list from original paper]