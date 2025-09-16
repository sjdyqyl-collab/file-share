# AdaSpa: Training-free and Adaptive Sparse Attention for Efficient Long Video Generation

**Yifei Xia¹², Suhan Ling¹², Fangcheng Fu¹, Yujie Wang¹², Huixia Li², Xuefeng Xiao², Bin Cui¹**  
¹Peking University, ²ByteDance  
*arXiv:2502.21079*

## Abstract

Generating high-fidelity long videos with Diffusion Transformers (DiTs) is hindered by significant latency from attention mechanisms. Generating an 8-second 720p video (110K tokens) with HunyuanVideo requires ~600 PFLOPs, with ~500 PFLOPs consumed by attention. We propose AdaSpa, the first Dynamic Pattern and Online Precise Search sparse attention method. AdaSpa introduces a blockified pattern to efficiently capture hierarchical sparsity inherent in DiTs, and Fused LSE-Cached Search with Head-adaptive Hierarchical Block Sparse Attention. By leveraging invariance across denoising steps, it enables precise, real-time identification of sparse indices with minimal overhead. AdaSpa is training-free, data-free, and seamlessly integrates with existing DiTs. Extensive experiments validate substantial acceleration across models while preserving video quality.

## 1. Introduction

Diffusion Transformers (DiTs) achieve state-of-the-art video generation but suffer from O(n²) attention complexity. For long videos, attention dominates computation costs. While sparse attention reduces redundancy in LLMs, existing methods fail for DiTs due to:

1. **Static Patterns** are inflexible for DiTs' dynamic, irregular sparsity
2. **Dynamic Pattern methods** cannot accurately identify DiT sparse patterns due to hierarchical structure and dispersed attention distribution

We propose AdaSpa, the first effective Dynamic Pattern + Online Precise Search method for DiTs. Key contributions:

- **Comprehensive Analysis**: Reveals DiTs exhibit hierarchical, blockified sparsity invariant across steps but adaptive to prompts/heads
- **Novel Framework**: Training-free, data-free sparse attention with online precise search and head-adaptive hierarchical blocks
- **Strong Results**: Consistent speedups (1.66-1.78×) with negligible quality loss

## 2. Preliminaries

**3D Full Attention**: Integrates video and text tokens into unified sequence of length L = f·h·w + t, applying self-attention across spatial, temporal, and modal dimensions.

**FlashAttention**: Processes attention blockwise to reduce memory from O(L²) to O(Lb), crucial for long sequences.

**Sparse Attention**: Uses masking function M ∈ {0,1}^(L×L) to ignore low-weight interactions, evaluated by Recall = Σ(selected weights)/Σ(all weights).

## 3. Sparse Pattern Characteristics in DiTs

**Observation 1**: DiTs exhibit hierarchical structure within and between modalities, making continuous patterns unsuitable. Attention weights decompose as:

```
W_attn = [W_video-video  W_video-text]
        [W_text-video   W_text-text]
```

Within W_video-video, attention structured into f×f frame regions R_i,j. Hierarchical boundaries disrupt continuous patterns, but blockified patterns achieve superior recall (0.93 vs 0.52 for continuous).

**Observation 2**: Sparse patterns vary with inputs, layers, and heads, but remain invariant across denoising steps. LSE distribution also stable across steps, enabling LSE caching for efficient search.

## 4. Methodology

### 4.1 Problem Formulation

**Blockified Sparse Attention**: Partition L into L/B chunks (B=64). Define block-level sparse pattern MS ∈ {0,1}^(L/B × L/B). Optimal sparse indices S* maximize:

```
S* = argmax_S Σ Wsum_attn(MS)
```

where Wsum_attn contains summed attention weights per block. Complexity reduces from O(L²d) to O((1-sparsity)L²d).

### 4.2 Design of Adaptive Sparse Attention

**Fused LSE-Cached Online Search**:
- **Phase 1**: Fused search computes full FlashAttention + caches LSE, then computes Wsum_attn using cached LSE
- **Phase 2**: LSE-cached search reuses cached LSE for subsequent steps, reducing search time by 50%

**Head-adaptive Hierarchical Block Sparse Attention**:
- Compute recall per head at fixed sparsity
- High-recall heads (recall > 0.8): increase sparsity to (1+sparsity)/2
- Low-recall heads: decrease sparsity to (3×sparsity-1)/2
- Maintains average sparsity while optimizing per-head performance

### 4.3 Implementation

- **Default**: sparsity=0.8, block_size=64, Ts={10,30}
- **Optimizations**: Text sink, row-wise uniform selection
- **Integration**: Single-line replacement for existing attention
- **Code**: 2000+ lines Python, 1000+ lines Triton

## 5. Experiments

### Setup
- **Models**: HunyuanVideo (13B), CogVideoX1.5-5B
- **Baselines**: Sparse VideoGen (static), MInference (dynamic)
- **Metrics**: VBench, PSNR, SSIM, LPIPS, latency
- **Config**: 720p videos, 50 steps, A100-80GB

### Main Results

| Method | HunyuanVideo |  | CogVideoX1.5-5B |  |
|--------|--------------|--|-----------------|--|
|        | VBench↑ | Speedup | VBench↑ | Speedup |
| Full Attention | 80.10 | 1.00× | 81.16 | 1.00× |
| MInference | 79.17 | 1.27× | 65.30 | 1.39× |
| Sparse VideoGen | 79.39 | 1.58× | 79.40 | 1.52× |
| **AdaSpa** | **80.13** | **1.78×** | **81.90** | **1.66×** |

AdaSpa achieves best quality-efficiency trade-off across all metrics.

### Analysis
- **Quality-Sparsity**: Maintains highest quality across sparsity levels 0.7-0.9
- **Warmup**: Consistently best across different warmup configurations  
- **Search Strategy**: {10,30} steps optimal; more searches can hurt quality
- **Scaling**: Speedup increases with video length, reaching 4.01× for 24s videos

## 6. Conclusion

AdaSpa enables efficient long video generation with DiTs through dynamic blockified sparse attention and LSE-cached online search. By exploiting DiTs' hierarchical sparsity and step-invariant patterns, AdaSpa achieves 1.66-1.78× speedup with maintained quality, providing a robust solution for scalable video generation.

## References

[Full reference list available in original paper]