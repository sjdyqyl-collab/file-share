# AdaSpa: Training-free and Adaptive Sparse Attention for Efficient Long Video Generation

## Abstract
Generating high-fidelity long videos with Diffusion Transformers (DiTs) is hindered by significant latency, primarily due to the O(n²) complexity of attention mechanisms. For instance, generating an 8-second 720p video (110K tokens) with HunyuanVideo takes about 600 PFLOPs, with ~500 PFLOPs consumed by attention. We propose AdaSpa, the first Dynamic Pattern and Online Precise Search sparse attention method for DiTs. Our key insight is that DiT attention exhibits hierarchical blockified sparsity patterns that vary across inputs, layers, and heads but remain invariant across denoising steps. AdaSpa introduces: (1) a blockified pattern to efficiently capture hierarchical sparsity, and (2) Fused LSE-Cached Search with Head-adaptive Hierarchical Block Sparse Attention that leverages step-invariance for precise real-time sparse pattern identification with minimal overhead. Implemented as a plug-and-play solution requiring no fine-tuning or profiling, AdaSpa achieves 1.78× speedup on HunyuanVideo and 1.66× on CogVideoX1.5-5B while maintaining video quality.

## 1. Introduction

**Problem**: DiTs suffer from prohibitive computational costs for long video generation due to attention mechanisms. Traditional sparse attention methods fail because:
- **Static patterns** cannot capture dynamic sparsity in DiTs
- **Dynamic patterns** with offline search lack portability across inputs
- **Approximate search** methods cannot handle dispersed attention distributions

**Key Observations**:
1. **Hierarchical Blockified Structure**: Attention patterns exhibit clear boundaries between modalities and frames, making continuous patterns ineffective
2. **Input-Dependent Variation**: Sparse patterns vary significantly across prompts, layers, and heads
3. **Step Invariance**: Patterns and LSE distributions remain stable across denoising steps

**Solution**: AdaSpa combines Dynamic Pattern recognition with Online Precise Search, leveraging step-invariance for efficient real-time sparse pattern identification.

## 2. Methodology

### 2.1 Blockified Sparse Attention

**Problem Formulation**:
- Sequence length: L = f·h·w + t (video frames × spatial resolution + text tokens)
- Block size: B (typically 64)
- Block-level pattern: MS ∈ {0,1}^(L/B × L/B)
- Goal: Select sparse indices S* that maximize attention recall

**Optimal Selection**:
```
S* = argmax_k Wsum_attn[k]
```
Where Wsum_attn computes attention weight sums per block.

### 2.2 Fused LSE-Cached Online Search

**Two-Phase Strategy**:

**Phase 1: Fused Online Search (Warmup)**
```
Algorithm 1: Fused Online Search
1. First pass: Full FlashAttention + store LSE
2. Second pass: Compute Wsum_attn using cached LSE
3. Select top-k blocks for sparse pattern
```

**Phase 2: LSE-Cached Search**
```
Algorithm 2: LSE-Cached Search
1. Reuse LSE from previous step
2. Single pass computation of Wsum_attn
3. Update sparse pattern if needed
```

**Search Overhead**: <5% of full attention time due to LSE caching.

### 2.3 Head-Adaptive Hierarchical Block Sparse Attention

**Adaptive Sparsity Mechanism**:
1. **Initial sparsity**: 0.8 (80% sparsity)
2. **Recall-based adjustment**:
   - Sort heads by recall performance
   - Increase sparsity for high-recall heads: (1+sparsity)/2
   - Decrease sparsity for low-recall heads: (3×sparsity-1)/2
3. **Maintain average sparsity** while optimizing per-head performance

**Implementation**:
- **Configuration**: sparsity=0.8, block_size=64, Ts={10,30}
- **Optimizations**: Text sink, row-wise uniformity
- **Integration**: Single-line code change

## 3. Experiments

### 3.1 Experimental Setup
- **Models**: HunyuanVideo (13B), CogVideoX1.5-5B
- **Resolution**: 720p
- **Duration**: 8s (HunyuanVideo), 10s (CogVideoX1.5-5B)
- **Metrics**: VBench, PSNR, SSIM, LPIPS, latency
- **Baselines**: Full Attention, MInference, Sparse VideoGen

### 3.2 Main Results

#### HunyuanVideo Performance
| Method | VBench (↑) | PSNR (↑) | Speedup |
|--------|------------|----------|---------|
| Full Attention | 80.10 | - | 1.00× |
| MInference | 79.17 | 22.53 | 1.27× |
| Sparse VideoGen | 79.39 | 27.61 | 1.58× |
| AdaSpa (ours) | 80.13 | 29.07 | 1.78× |

#### CogVideoX1.5-5B Performance
| Method | VBench (↑) | PSNR (↑) | Speedup |
|--------|------------|----------|---------|
| Full Attention | 81.16 | - | 1.00× |
| MInference | 65.30 | 10.31 | 1.39× |
| Sparse VideoGen | 79.40 | 18.98 | 1.52× |
| AdaSpa (ours) | 81.90 | 23.25 | 1.66× |

### 3.3 Ablation Studies

#### Component Analysis
| Variant | VBench (Hunyuan) | Speedup | Impact |
|---------|------------------|---------|---------|
| w/o head adaptive | 79.64 | 1.76× | -0.49 VBench |
| w/o LSE cache | 80.16 | 1.71× | -0.07 speedup |

#### Scaling Study
| Video Length | Speedup |
|--------------|---------|
| 8s | 1.78× |
| 16s | 2.79× |
| 24s | 4.01× |

## 4. Computational Analysis

### Runtime Complexity
- **Full Attention**: Get_Time[L, d, L] = O(L²d)
- **AdaSpa**: Get_Time[0.2L, d, L] + Get_Time[L, d, 0.2L] + Search_Cost[0.05×Full]
- **Memory**: O(L·B) vs O(L²) for full attention

### Performance Summary
- **Speedup**: 1.66-1.78× across models
- **Quality**: <1% VBench degradation
- **Scalability**: Linear improvement with video length
- **Integration**: Plug-and-play with single-line change

## 5. Limitations and Future Work

### Current Limitations
1. **Fixed block size**: B=64 may not be optimal for all patterns
2. **Sparsity threshold**: Fixed at 0.8, could be adaptive
3. **Search intervals**: Manual selection of Ts={10,30}
4. **Head grouping**: Coarse-grained adaptation
5. **Memory overhead**: LSE caching requires additional storage

### Proposed Improvements
1. **Dynamic block sizing**: Adaptive block size based on pattern density
2. **Learned sparsity**: ML-based sparsity prediction per layer/head
3. **Progressive search**: Adaptive search intervals based on pattern stability
4. **Fine-grained adaptation**: Individual token-level sparsity
5. **Memory optimization**: Compressed LSE storage

## 6. Conclusion

AdaSpa addresses the computational bottleneck in DiT video generation through novel sparse attention mechanisms. By leveraging hierarchical blockified patterns and step-invariant properties, it achieves substantial speedups while maintaining quality. The method's training-free, plug-and-play nature makes it immediately applicable to existing DiTs, representing a significant advance in efficient video generation.