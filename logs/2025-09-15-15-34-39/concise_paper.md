# Training-free and Adaptive Sparse Attention for Efficient Long Video Generation

Yifei Xia¹², Suhan Ling¹², Fangcheng Fu¹, Yujie Wang¹²  
Huixia Li², Xuefeng Xiao², Bin Cui¹  
¹Peking University ²ByteDance

## Abstract

Generating high-fidelity long videos with Diffusion Transformers (DiTs) is often hindered by significant latency, primarily due to the computational demands of attention mechanisms. For instance, generating an 8-second 720p video (110K tokens) with HunyuanVideo takes about 600 PFLOPs, with around 500 PFLOPs consumed by attention computations. To address this issue, we propose AdaSpa, the first Dynamic Pattern and Online Precise Search sparse attention method. Firstly, to realize the Dynamic Pattern, we introduce a blockified pattern to efficiently capture the hierarchical sparsity inherent in DiTs. This is based on our observation that sparse characteristics of DiTs exhibit hierarchical and blockified structures between and within different modalities. This blockified approach significantly reduces the complexity of attention computation while maintaining high fidelity in the generated videos. Secondly, to enable Online Precise Search, we propose the Fused LSE-Cached Search with Head-adaptive Hierarchical Block Sparse Attention. This method is motivated by our finding that DiTs' sparse pattern and LSE vary w.r.t. inputs, layers, and heads, but remain invariant across denoising steps. By leveraging this invariance across denoising steps, it adapts to the dynamic nature of DiTs and allows for precise, real-time identification of sparse indices with minimal overhead. AdaSpa is implemented as an adaptive, plug-and-play solution and can be integrated seamlessly with existing DiTs, requiring neither additional fine-tuning nor a dataset-dependent profiling. Extensive experiments validate that AdaSpa delivers substantial acceleration across various models while preserving video quality, establishing itself as a robust and scalable approach to efficient video generation.

## 1. Introduction

Diffusion Transformers (DiTs) have emerged as powerful frameworks for video generation but face computational bottlenecks due to O(n²) attention complexity. While sparse attention mechanisms have shown success in LLMs, existing approaches fail for DiTs due to their unique characteristics:

- **Static Patterns** lack flexibility to capture DiT's dynamic sparsity
- **Dynamic Patterns** with offline search fail due to input-dependent variations
- **Online Approximate Search** cannot handle DiT's hierarchical and dispersed attention patterns

We propose AdaSpa, the first effective Dynamic Pattern + Online Precise Search method that addresses these limitations through blockified patterns and LSE-cached search.

## 2. Preliminaries

### 2.1 Diffusion Transformers and 3D Full Attention
- **Sequence Length**: L = f·h·w + t (video frames × height × width + text tokens)
- **3D Full Attention**: Integrates spatial, temporal, and cross-modal dependencies in unified sequence
- **Computational Complexity**: O(L²) for attention, becoming prohibitive for long videos

### 2.2 Sparse Attention
- **Definition**: Attention with masking function M ∈ {0,1}^(L×L) where M_ij=1 indicates attended pairs
- **Recall**: Σ_(i,j)∈sparse indices W(i,j)_attn / Σ_i,j W(i,j)_attn - measures preservation of original attention

## 3. Sparse Pattern Characteristics in DiTs

### Observation 1: Hierarchical and Blockified Structure
- **Hierarchical Organization**: Attention matrix decomposes into video-video (f·h·w × f·h·w), video-text, text-video, text-text (t × t) blocks
- **Frame-level Structure**: Within video-video attention, clear f×f frame regions with local continuity
- **Blockified Patterns**: Achieve 0.93-1.0 recall vs 0.12-0.96 for continuous patterns (col, diag)

### Observation 2: Invariance Across Denoising Steps
- **Pattern Invariance**: Sparse patterns remain consistent across denoising steps for given layer/head
- **LSE Stability**: Log-Sum-Exp distribution stable across steps (Figure 6b)
- **Input Adaptation**: Patterns vary significantly with different inputs, layers, and heads

## 4. Methodology

### 4.1 Problem Formulation
**Blockified Sparse Attention**:
- Partition sequence into L/B blocks with size B=64
- Block-level sparse pattern MS ∈ {0,1}^(L/B × L/B)
- Optimal sparse indices: S* = argmax_S Wsum_attn(MS) via top-k selection

### 4.2 AdaSpa Design

#### Fused LSE-Cached Online Search
- **Phase 1 (Warmup)**: Fused Online Search at steps Ts={10,30}
  1. Full FlashAttention with LSE storage
  2. Block-wise Wsum_attn computation using cached LSE
- **Phase 2 (Cached)**: LSE-Cached Search between warmup steps
  1. Reuse cached LSE for single-pass Wsum_attn computation
  2. Generate new block masks with ~50% time reduction

#### Head-adaptive Hierarchical Strategy
1. **Assessment**: Compute Recall per head at base sparsity=0.8
2. **Adaptation**: 
   - High-recall heads: increase sparsity to (1+0.8)/2 = 0.9
   - Low-recall heads: decrease sparsity to (3×0.8-1)/2 = 0.7
3. **Uniformity**: Maintain average sparsity while optimizing per-head performance

### 4.3 Implementation
- **Parameters**: sparsity=0.8, block_size=64, warmup=10 steps, Ts={10,30}
- **Optimizations**: Text sink inclusion, row-wise uniform selection
- **Integration**: One-line replacement `adaspa_attention_handler()`

## 5. Experiments

### Setup
- **Models**: HunyuanVideo (13B), CogVideoX1.5-5B
- **Baselines**: Sparse VideoGen (static), MInference (dynamic)
- **Metrics**: VBench, PSNR, SSIM, LPIPS, Latency, Speedup
- **Hardware**: Single A100 GPU-80GB

### Results

#### HunyuanVideo (8s, 720p)
| Method | VBench↑ | PSNR↑ | SSIM↑ | LPIPS↓ | Latency(s) | Speedup |
|--------|---------|-------|-------|--------|------------|---------|
| Full | 80.10 | - | - | - | 3213.76 | 1.00× |
| MInference | 79.17 | 22.53 | 0.7435 | 0.3550 | 2532.80 | 1.27× |
| Sparse VideoGen | 79.39 | 27.61 | 0.8683 | 0.1703 | 2035.59 | 1.58× |
| **AdaSpa** | **80.13** | **29.07** | **0.8905** | **0.1478** | **1810.23** | **1.78×** |

#### CogVideoX1.5-5B (10s, 720p)
| Method | VBench↑ | PSNR↑ | SSIM↑ | LPIPS↓ | Latency(s) | Speedup |
|--------|---------|-------|-------|--------|------------|---------|
| Full | 81.16 | - | - | - | 3135.24 | 1.00× |
| MInference | 65.30 | 10.31 | 0.3113 | 0.6820 | 2258.35 | 1.39× |
| Sparse VideoGen | 79.40 | 18.98 | 0.6465 | 0.3632 | 2061.42 | 1.52× |
| **AdaSpa** | **81.90** | **23.25** | **0.8267** | **0.2067** | **1888.14** | **1.66×** |

### Key Findings
- **Consistent Superiority**: AdaSpa achieves best quality and efficiency across all settings
- **Scaling**: 4.01× speedup for 24-second videos
- **Robustness**: Maintains quality across sparsity levels (0.7-0.9)
- **Ablation**: Head-adaptive mechanism provides +0.5 VBench improvement

## 6. Conclusion

AdaSpa introduces the first effective Dynamic Pattern + Online Precise Search sparse attention method for DiTs. By leveraging hierarchical blockified patterns and LSE-cached search, it achieves 1.78× efficiency improvement while maintaining high video quality. The method's training-free, plug-and-play design makes it immediately applicable to existing DiT models.

## References
[Complete reference list from original paper]