# AdaSpa: Training-free and Adaptive Sparse Attention for Efficient Long Video Generation

**Yifei Xia¹², Suhan Ling¹², Fangcheng Fu¹, Yujie Wang¹², Huixia Li², Xuefeng Xiao², Bin Cui¹**  
¹Peking University, ²ByteDance

## Abstract

Generating high-fidelity long videos with Diffusion Transformers (DiTs) is often hindered by significant latency, primarily due to the computational demands of attention mechanisms. For instance, generating an 8-second 720p video (110K tokens) with HunyuanVideo takes about 600 PFLOPs, with around 500 PFLOPs consumed by attention computations. To address this issue, we propose AdaSpa, the first Dynamic Pattern and Online Precise Search sparse attention method. Firstly, to realize the Dynamic Pattern, we introduce a blockified pattern to efficiently capture the hierarchical sparsity inherent in DiTs. This is based on our observation that sparse characteristics of DiTs exhibit hierarchical and blockified structures between and within different modalities. This blockified approach significantly reduces the complexity of attention computation while maintaining high fidelity in the generated videos. Secondly, to enable Online Precise Search, we propose the Fused LSE-Cached Search with Head-adaptive Hierarchical Block Sparse Attention. This method is motivated by our finding that DiTs' sparse pattern and LSE vary w.r.t. inputs, layers, and heads, but remain invariant across denoising steps. By leveraging this invariance across denoising steps, it adapts to the dynamic nature of DiTs and allows for precise, real-time identification of sparse indices with minimal overhead. AdaSpa is implemented as an adaptive, plug-and-play solution and can be integrated seamlessly with existing DiTs, requiring neither additional fine-tuning nor a dataset-dependent profiling. Extensive experiments validate that AdaSpa delivers substantial acceleration across various models while preserving video quality, establishing itself as a robust and scalable approach to efficient video generation.

## 1. Introduction

Diffusion models have emerged as powerful frameworks for generative tasks, with Diffusion Transformers (DiTs) setting new benchmarks in video generation. However, generating high-quality videos remains computationally expensive due to the O(n²) complexity of attention mechanisms. For an 8-second 720p video, attention consumes ~83% of total computation (500/600 PFLOPs).

Existing sparse attention methods fall into two categories:
- **Static Pattern**: Predetermined sparse indices (e.g., StreamingLLM, BigBird)
- **Dynamic Pattern**: Real-time sparse index determination (e.g., DSA, MInference)

However, DiTs exhibit unique characteristics that make these approaches suboptimal:
1. **Hierarchical Structure**: Attention patterns are fragmented across modalities
2. **Dynamic Nature**: Patterns vary with inputs, layers, and heads
3. **Dispersed Distribution**: Key areas are not concentrated or continuous

We propose AdaSpa, the first Dynamic Pattern + Online Precise Search method that addresses these challenges through hierarchical blockified patterns and efficient online search with LSE caching.

## 2. Key Observations

### Observation 1: Hierarchical Blockified Structure
DiT attention exhibits hierarchical organization:
- **Video-Video**: Spatial-temporal interactions among video tokens
- **Text-Video/Text-Text**: Cross-modal interactions with text sink effects
- **Frame-wise Structure**: Clear boundaries between f×f frame regions

Due to this hierarchy, continuous patterns (col, diag) achieve only 0.12-0.96 recall, while blockified patterns consistently achieve 0.93-1.0 recall.

### Observation 2: Dynamic yet Invariant Properties
- **Dynamic**: Patterns vary with inputs, layers, and heads
- **Invariant**: Patterns and LSE remain consistent across denoising steps
- **Opportunity**: LSE caching enables efficient online search

## 3. Methodology

### 3.1 Blockified Sparse Attention
Partition sequence length L into L/B blocks of size B. Define block-level sparse pattern MS ∈ {0,1}^(L/B × L/B) and expand to gMS ∈ {0,1}^(L×L). The sparse attention becomes:

Wattn(gMS) = Softmax_safe(QK^T/√D - c(1-gMS))

Optimal sparse indices S* are found by maximizing block-wise attention weights:
S* = argmax_k Wsum_attn[k]

Complexity reduces from O(L²d) to O((1-sparsity)L²d).

### 3.2 Fused LSE-Cached Online Search

**Two-Phase Approach:**

1. **Fused Online Search (2-pass)**:
   - Pass 1: Compute FlashAttention outputs and store LSE
   - Pass 2: Use cached LSE to compute Wsum_attn block-wise

2. **LSE-Cached Search (1-pass)**:
   - Reuse LSE from previous search
   - Reduces search time by 50%
   - Overhead <5% of full attention time

### 3.3 Head-adaptive Hierarchical Strategy
- Compute recall for each head at fixed sparsity
- Sort heads by performance
- Adjust sparsity: high-recall heads → higher sparsity, low-recall heads → lower sparsity
- Maintains average sparsity while optimizing per-head performance

## 4. Implementation

**Configuration:**
- Sparsity: 0.8 (default)
- Block size: 64
- Search steps: Ts = {10, 30}
- Warmup: 10 steps full attention

**Optimizations:**
1. Text Sink: Preserve text-video interactions
2. Row-wise: Ensure uniform attention per query row

**Integration:** Single line code change via `adaspa_attention_handler`

## 5. Experiments

### Setup
- **Models**: HunyuanVideo-13B, CogVideoX1.5-5B
- **Videos**: 720p, 8-10 seconds, 50 steps
- **Baselines**: Sparse VideoGen, MInference
- **Metrics**: VBench, PSNR, SSIM, LPIPS, Latency

### Results

| Model | Method | VBench (↑) | PSNR (↑) | Speedup |
|-------|--------|------------|----------|---------|
| HunyuanVideo | Full Attention | 80.10 | - | 1.00× |
| | MInference | 79.17 | 22.53 | 1.27× |
| | Sparse VideoGen | 79.39 | 27.61 | 1.58× |
| | **AdaSpa** | **80.13** | **29.07** | **1.78×** |
| CogVideoX1.5-5B | Full Attention | 81.16 | - | 1.00× |
| | MInference | 65.30 | 10.31 | 1.39× |
| | Sparse VideoGen | 79.40 | 18.98 | 1.52× |
| | **AdaSpa** | **81.90** | **23.25** | **1.66×** |

### Key Findings
- **Superior Quality**: Highest VBench and PSNR across both models
- **Best Speedup**: 1.78× (HunyuanVideo) and 1.66× (CogVideoX1.5-5B)
- **Scalable**: Speedup increases with video length (up to 4.01× for 24s videos)
- **Robust**: Maintains quality across sparsity levels (0.7-0.9)

### Ablation Studies
- **Head-adaptive**: 2-3% quality improvement over uniform sparsity
- **LSE caching**: 5-8% speed improvement over non-cached version
- **Search strategy**: Ts={10,30} optimal; more searches provide diminishing returns

## 6. Limitations and Future Work

### Current Limitations
1. **Fixed Parameters**: Sparsity and block size remain constant
2. **Limited Hardware**: Only tested on A100 GPU
3. **Model Scope**: Evaluated on 2 DiT architectures
4. **Content Type**: General video generation only

### Proposed Improvements
1. **Adaptive Sparsity**: Content-complexity based dynamic sparsity
2. **Multi-Scale Blocks**: Hierarchical block sizes (32, 64, 128)
3. **Hardware Optimization**: Platform-specific kernel optimizations
4. **Temporal Consistency**: Motion-aware sparsity patterns
5. **RL-Based Scheduling**: Learned optimal search timing

### Expected Gains
- **Conservative**: 2.0-2.2× speedup, 2-3% quality improvement
-