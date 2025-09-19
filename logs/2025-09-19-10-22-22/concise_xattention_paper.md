# XAttention: Block Sparse Attention with Antidiagonal Scoring

## Abstract
Long-Context Transformer Models (LCTMs) are vital for real-world applications but suffer high computational costs due to attention's quadratic complexity. Block-sparse attention mitigates this by focusing computation on critical regions, yet existing methods struggle with balancing accuracy and efficiency due to costly block importance measurements. In this paper, we introduce XAttention, a plug-and-play framework that dramatically accelerates long-context inference in Transformers models using sparse attention. XAttention's key innovation is the insight that the sum of antidiagonal values (i.e., from the lower-left to upper-right) in the attention matrix provides a powerful proxy for block importance. This allows for precise identification and pruning of non-essential blocks, resulting in high sparsity and dramatically accelerated inference. Across comprehensive evaluations on demanding long-context benchmarks—including RULER and LongBench for language, VideoMME for video understanding, and VBench for video generation—XAttention achieves accuracy comparable to full attention while delivering substantial computational gains. We demonstrate up to 13.5× acceleration in attention computation. These results underscore XAttention's ability to unlock the practical potential of block sparse attention, paving the way for scalable and efficient deployment of LCTMs in real-world applications.

## 1. Introduction

Long-Context Transformer Models face quadratic computational complexity in attention mechanisms. While block-sparse attention offers a promising solution, existing methods struggle with the trade-off between accuracy and efficiency due to expensive block importance measurements.

**Key Insight**: The sum of antidiagonal values in attention blocks serves as a lightweight yet powerful indicator of block importance, enabling precise pruning of non-essential computations.

## 2. Method

### 2.1 Antidiagonal Scoring
For each B×B attention block:
1. Sample elements along antidiagonals with stride S
2. Compute sum of sampled elements as importance score
3. This preserves information from all tokens while detecting vertical/slash patterns

**Runtime**: Pattern selection requires [L/S, d, L/S] operations vs [L, d, L] for full evaluation

### 2.2 Threshold Block Selection
1. **Antidiagonal Summation**: Sum strided antidiagonal elements
2. **Softmax Normalization**: Convert to probability distribution
3. **Block Selection**: Find minimal blocks exceeding threshold τ

**Algorithm**: 
```
find_blocks(A, τ) = argmin_B {|B| : Σ_b∈B Σ_(i,j)∈b A[i,j] ≥ τ}
```

### 2.3 Dynamic Threshold Prediction
Uses dynamic programming to optimize τ per attention head:
- State: D[h][m] = best performance with m adjustments across h heads
- Recurrence: D[h][m] = max(D[h-1][m], P(h,m))
- Reduces average threshold from 0.9 to 0.8

## 3. Experiments

### 3.1 Setup
**Models**: Llama-3.1-8B (language), Qwen2-VL-7B (video understanding), HunyuanVideo (generation)
**Baselines**: FlashAttention, MInference, FlexPrefill, SeerAttention
**Datasets**: RULER, LongBench, VideoMME, VBench

### 3.2 Results

**RULER Benchmark**:
- XAttention S=8: 88.47 avg (outperforms all baselines)
- Maintains accuracy up to 128k tokens

**Efficiency**:
- **Max Speedup**: 13.5× at 256k tokens
- **Min Density**: 6.2% at 128k tokens
- **Pattern Selection**: 24.9× faster than MInference

**Video Generation**:
- High fidelity: PSNR 23.5, SSIM 0.822
- 50%+ sparsity with 5-step warmup

### 3.3 Ablation Studies
- **Antidiagonal vs Random**: +8.29 accuracy improvement
- **Stride Sensitivity**: S=64 degrades accuracy to 81.21
- **Dynamic Threshold**: Improves both accuracy and sparsity

## 4. Limitations and Improvements

### Current Limitations
1. **Warmup Dependency**: 5-step full attention needed for video generation
2. **Stride Sensitivity**: Large strides (S=64) cause accuracy loss
3. **Fixed Block Size**: Uniform blocks across all heads/layers
4. **Threshold Overhead**: 1000 DP iterations for optimization

### Proposed Improvements

#### 1. Adaptive Warmup Strategy
**Method**: Learn optimal warmup steps based on content complexity
**Runtime**: α×[L, d, L] with α∈[0,5] vs fixed 5×[L, d, L]
**Expected**: 2-3× reduction in warmup overhead

#### 2. Multi-Scale Patterns
**Method**: Combine strides S={4,8,16} with learned weights
**Runtime**: 1.2×[L/S, d, L/S] vs single [L/S, d, L/S]
**Expected**: 15-20% better accuracy at same density

#### 3. Content-Adaptive Sparsity
**Method**: Adjust density based on input characteristics
**Runtime**: [f(content)×L, d, f(content)×L] vs fixed density
**Expected**: 20-30% additional speedup on easy inputs

#### 4. Gradient-Based Optimization
**Method**: Use gradients to reduce threshold iterations from 1000→100
**Runtime**: O(H×M/10×evaluation_cost) vs O(H×M×evaluation_cost)
**Expected**: 5-10× faster threshold optimization

## 5. Conclusion

XAttention demonstrates that antidiagonal scoring provides an effective, lightweight method for block importance prediction in sparse attention. With up to 13.5× acceleration and minimal accuracy loss, it enables practical deployment of long-context transformers. Future improvements in adaptive strategies and hardware optimization could yield additional 1.5-3× speedups.

## Runtime Summary
- **Baseline**: [L, d, L] for full attention
- **XAttention**: [density×L, d, density×L] + [L/S, d, L/S] for pattern selection
- **Improved**: [f(content)×L, d, f(content)×L] with adaptive optimizations
- **Best Achieved**: 13.5× speedup at density=6.89% for 256k tokens