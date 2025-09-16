# AdaSpa: Training-free and Adaptive Sparse Attention for Efficient Long Video Generation

**Yifei Xia, Suhan Ling, Fangcheng Fu, Yujie Wang, Huixia Li, Xuefeng Xiao, Bin Cui**

## Abstract
Generating high-fidelity long videos with Diffusion Transformers (DiTs) is often hindered by significant latency, primarily due to the computational demands of attention mechanisms. For instance, generating an 8-second 720p video (110K tokens) with HunyuanVideo takes about 600 PFLOPs, with around 500 PFLOPs consumed by attention computations. To address this issue, we propose AdaSpa, the first Dynamic Pattern and Online Precise Search sparse attention method. Firstly, to realize the Dynamic Pattern, we introduce a blockified pattern to efficiently capture the hierarchical sparsity inherent in DiTs. This is based on our observation that sparse characteristics of DiTs exhibit hierarchical and blockified structures between and within different modalities. This blockified approach significantly reduces the complexity of attention computation while maintaining high fidelity in the generated videos. Secondly, to enable Online Precise Search, we propose the Fused LSE-Cached Search with Head-adaptive Hierarchical Block Sparse Attention. This method is motivated by our finding that DiTs' sparse pattern and LSE vary w.r.t. inputs, layers, and heads, but remain invariant across denoising steps. By leveraging this invariance across denoising steps, it adapts to the dynamic nature of DiTs and allows for precise, real-time identification of sparse indices with minimal overhead. AdaSpa is implemented as an adaptive, plug-and-play solution and can be integrated seamlessly with existing DiTs, requiring neither additional fine-tuning nor a dataset-dependent profiling. Extensive experiments validate that AdaSpa delivers substantial acceleration across various models while preserving video quality, establishing itself as a robust and scalable approach to efficient video generation.

## 1. Introduction

Long video generation with Diffusion Transformers faces a critical computational bottleneck: attention mechanisms consume 83% of total FLOPs. Existing sparse attention methods fall short because:
- **Static patterns** cannot capture DiTs' dynamic sparsity
- **Dynamic patterns** with offline search lack portability across inputs
- **Approximate online search** fails due to hierarchical structure complexity

We propose AdaSpa, the first **Dynamic Pattern + Online Precise Search** method that is training-free and data-free, achieving 1.78× speedup while maintaining video quality.

## 2. Key Observations

### Observation 1: Hierarchical Blockified Structure
- DiTs exhibit hierarchical sparsity between modalities (text vs video) and within video frames
- Continuous patterns (col, diag) fail due to discontinuities
- Blockified patterns achieve superior recall (0.93-1.0 vs 0.12-0.54 for continuous patterns)

### Observation 2: Dynamic but Predictable Patterns
- Patterns vary significantly across inputs, layers, and heads
- **Critical insight**: Patterns remain stable across denoising steps for fixed layer/head
- LSE distributions are stable across steps, enabling caching

## 3. Methodology

### 3.1 Blockified Sparse Attention
**Problem**: Find optimal sparse indices under block-wise sparsity

**Formulation**:
- Partition sequence length L into L/B blocks of size B
- Define block pattern MS ∈ {0,1}^(L/B × L/B)
- Optimal indices: S* = argmax_S ||W_sum_attn(MS)||

**Complexity**: O((1-sparsity)L²d) vs O(L²d) for full attention

### 3.2 Fused LSE-Cached Online Search

#### Two-Phase Approach
1. **Fused Online Search** (warmup steps):
   - Pass 1: Full FlashAttention + LSE caching
   - Pass 2: Block-wise W_sum_attn computation
2. **LSE-Cached Search** (subsequent steps):
   - Single pass using cached LSE
   - 50% time reduction vs fused search

#### Algorithm Overview
```
// Fused Online Search (Algorithm 1)
1. Full attention computation with LSE caching
2. Block-wise attention weight aggregation
3. Top-k block selection for sparsity

// LSE-Cached Search (Algorithm 2)  
1. Reuse cached LSE from previous search
2. Single-pass block weight computation
3. Maintain sparsity constraints
```

### 3.3 Head-adaptive Hierarchical Strategy

#### Adaptive Sparsity Assignment
1. **Initial**: Uniform sparsity across all heads
2. **Evaluation**: Compute recall per head
3. **Adjustment**:
   - High recall heads: Increase sparsity to (1+sparsity)/2
   - Low recall heads: Decrease sparsity to (3×sparsity-1)/2
4. **Constraint**: Maintain average sparsity across all heads

## 4. Implementation

### Configuration
- **Sparsity**: 0.8 (default)
- **Block Size**: 64
- **Search Steps**: {10, 30}
- **Warmup**: 10 steps full attention

### Integration
```python
from adaspa import adaspa_attention_handler
# Single line replacement
out = adaspa_attention_handler(query=q, key=k, value=v)
```

### Optimizations
- **Text Sink**: Manual selection of cross-modal attention indices
- **Row-wise Uniformity**: Ensures consistent attention distribution

## 5. Experiments

### Setup
- **Models**: HunyuanVideo (13B), CogVideoX1.5-5B
- **Baselines**: Sparse VideoGen, MInference
- **Metrics**: VBench, PSNR, SSIM, LPIPS, Latency

### Results

#### HunyuanVideo Performance
| Method | VBench (↑) | PSNR (↑) | Latency (s) | Speedup |
|--------|-------------|----------|-------------|---------|
| Full Attention | 80.10 | - | 3213.76 | 1.00× |
| MInference | 79.17 | 22.53 | 2532.80 | 1.27× |
| Sparse VideoGen | 79.39 | 27.61 | 2035.59 | 1.58× |
| **AdaSpa** | **80.13** | **29.07** | **1810.23** | **1.78×** |

#### CogVideoX1.5 Performance
| Method | VBench (↑) | PSNR (↑) | Latency (s) | Speedup |
|--------|-------------|----------|-------------|---------|
| Full Attention | 81.16 | - | 3135.24 | 1.00× |
| MInference | 65.30 | 10.31 | 2258.35 | 1.39× |
| Sparse VideoGen | 79.40 | 18.98 | 2061.42 | 1.52× |
| **AdaSpa** | **81.90** | **23.25** | **1888.14** | **1.66×** |

### Key Findings
1. **Superior Quality**: Best VBench scores across both models
2. **Maximum Speedup**: 1.78× (HunyuanVideo), 1.66× (CogVideoX1.5)
3. **Scalability**: 4.01× speedup for 24-second videos
4. **Robustness**: Consistent performance across sparsity levels (0.7-0.9)

## 6. Conclusion

AdaSpa addresses the computational bottleneck in DiT-based video generation through:
1. **Blockified patterns** that capture hierarchical sparsity
2. **LSE-cached search** that leverages step invariance
3. **Head-adaptive strategy** for optimal per-head sparsity
4. **Training-free integration** with existing models

Achieving 1.78× speedup with negligible quality loss, AdaSpa establishes a new paradigm for efficient long video generation.

## Runtime Analysis

### Matrix Multiplication Representation
- **Full Attention**: [L, d, L] = O(L²d)
- **AdaSpa**: [(1-sparsity)L, d, L] = O((1-sparsity)L²d)
- **Search Overhead**: [L/B, B², L/B] = O(L²d/B²) per search step

### Practical Performance
- **Search Cost**: <5% of total generation time
- **Memory**: Block-wise processing reduces memory footprint
- **Scalability**: Linear speedup improvement with sequence length