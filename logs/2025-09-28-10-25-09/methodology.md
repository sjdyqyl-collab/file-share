# XAttention: Methodology Details

## Overview
XAttention is a three-component framework for efficient block-sparse attention in long-context Transformers:
1. Importance prediction using antidiagonal scoring
2. Block selection via threshold-based algorithm
3. Dynamic threshold prediction per attention head

## 1. Importance Prediction via Antidiagonal Scoring

### Core Insight
The sum of antidiagonal values (lower-left to upper-right) in attention blocks serves as a powerful proxy for block importance. This approach:
- **Preserves information**: Each token contributes to at least one antidiagonal sum
- **Detects patterns**: Antidiagonals intersect all vertical and slash patterns within blocks
- **Efficient computation**: Simple summation vs complex pooling operations

### Mathematical Formulation
For each block of size B×B:
- Define stride S for antidiagonal sampling
- Select elements along antidiagonals with stride S
- Compute sum: `score = Σ antidiagonal_values`

### Advantages over Existing Methods
- **vs Pooling**: More robust when only few significant patterns exist
- **vs Vertical/Slash Detection**: No need for complex search algorithms
- **vs Token-based**: Operates at block level for efficiency

## 2. Block Selection Algorithm

### Process Flow
1. **Antidiagonal Summation**: Extract and sum antidiagonal elements in S×S blocks
2. **Softmax Normalization**: Convert sums to probability distribution
3. **Threshold Selection**: Find minimal block set exceeding threshold τ

### Algorithm 1: Block Selection
```
Input: Q∈R^(L×d), K∈R^(L×d), block size B, stride S, threshold τ
t: Number of blocks NB = ⌊L/B⌋

for b = 0 to NB-1:
    Qslice = Q[bB:(b+1)B,:]
    Qreshaped = reshape_along_antidiagonal(Qslice, S)
    Kreshaped = reshape_along_antidiagonal(Kslice, S)
    Aapprox = Softmax(Qreshaped × Kreshaped^T / √(d·S))
    Mb = find_blocks(Aapprox, τ)
M = concatenate(M0, M1, ..., M_NB-1)
```

### Threshold Selection Formula
```
find_blocks(A, τ) = argmin_B{|B| : Σ_b∈B Σ_(i,j)∈b A_i,j ≥ τ}
```

## 3. Minimum Threshold Prediction

### Dynamic Programming Approach
- **Problem**: Different attention heads have varying sparsity levels
- **Solution**: Per-head threshold optimization via dynamic programming

### Formulation
- Define DP table D[h][m] for h heads and m adjustments
- Recurrence: D[h][m] = max(D[h-1][m], P(h,m))
- Threshold adjustment: th(m) = th(m-1) × 0.9

### Implementation Details
- Start with τ = 0.9
- Allow M = 1000 threshold combinations
- Results in average threshold of 0.8

## Computational Complexity Analysis

### Baseline Full Attention
- Time: [L, L, d] where L = sequence length, d = hidden dimension
- Complexity: O(L²d)

### XAttention Sparse Attention
- **Pattern Selection**: [B, B, d] for each block, where B = block size
- **Sparse Computation**: [L_sparse, L_sparse, d] where L_sparse = τ·L
- **Total Time**: O(L²d·τ² + L·B·d/S) for pattern selection

### Efficiency Gains
- **Pattern Selection**: 24.9× faster than MInference
- **Overall Speedup**: Up to 13.5× for 256K tokens
- **Sparsity**: Achieves 6.2-55.4% density across contexts

## Implementation Considerations

### Parameters
- **Stride S**: Controls sampling granularity (typical: 8 or 16)
- **Block size B**: Usually 64-128 tokens
- **Threshold τ**: 0.9 default, optimized per-head via DP

### Warmup Strategy (for Video Generation)
- Use full attention for first 5 denoising steps
- Switch to XAttention for remaining steps
- Prevents layout shifts in generated videos

### Hardware Optimization
- Compatible with FlashAttention/FlashInfer
- GPU-friendly block-based computation
- Minimal memory overhead for sparse patterns