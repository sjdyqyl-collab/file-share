# Phase 2: Methodology Details of XAttention

## 2.1 Importance Prediction

### Core Insight
The antidiagonal sum serves as a robust indicator of block importance because:
1. **Information Preservation**: Every token contributes to at least one antidiagonal sum
2. **Pattern Detection**: Antidiagonals intersect both vertical and slash patterns within blocks

### Strided Antidiagonal Scoring
For each B×B block in attention map:
- Use stride S to sample elements along antidiagonals
- Sum selected elements to get block importance score
- Visualized in Figure 1 with B=8, S=4

### Mathematical Formulation
```
For block b in attention map A:
    score_b = sum(A[i,j] where (i+j) mod S == k for various k)
```

## 2.2 Threshold Block Selection

### Algorithm Steps
1. **Antidiagonal Summation**: Sum elements along strided antidiagonals
2. **Softmax Normalization**: Convert sums to probability distribution
3. **Block Selection**: Find minimal block set exceeding threshold τ

### Formal Definition
```
find_blocks(A, τ) = argmin_B {|B| : sum_{b∈B} sum_{(i,j)∈b} A[i,j] ≥ τ}
```

### Algorithm 1: Block Selection
```
Input: Q∈R^{L×d}, K∈R^{L×d}, block size B, stride S, head dim dh, threshold τ
Output: Sparse mask M

1: NB ← ⌊L/B⌋
2: for b = 0 to NB-1 do
3:   Qslice ← Q[bB:(b+1)B,:]
4:   Qreshaped ← reshape_along_antidiagonal(Qslice, S)
5:   Kreshaped ← reshape_along_antidiagonal(K[bB:(b+1)B,:], S)
6:   Aapprox ← softmax(Qreshaped × K^T_reshaped / √(dh·S))
7:   Mb ← find_blocks(Aapprox, τ)
8: end for
9: M ← concatenate(M0, M1, ..., MNB-1)
```

## 2.3 Minimum Threshold Prediction

### Dynamic Programming Formulation
- **State**: D[h][m] = best performance with m threshold adjustments across first h heads
- **Recurrence**: D[h][m] = max(D[h-1][m], P(h,m))
- **Threshold Update**: th(m) = th(m-1) × 0.9

### Optimization Process
1. Start with τ = 0.9
2. Allow M = 1000 threshold adjustments
3. Result: Average threshold of 0.8 across heads

## Implementation Details

### Block Size and Stride
- **Block Size (B)**: Typically 8×8 or 16×16
- **Stride (S)**: Tested values: 4, 8, 16, 64
- **Trade-off**: Larger S → lower density but potential accuracy loss

### Threshold Selection
- **Fixed**: τ = 0.9 baseline
- **Dynamic**: Optimized per head via DP
- **Range**: Typically 0.8-0.95 for best results

### Computational Complexity Analysis
- **Pattern Selection Time**: O(L²/S) for antidiagonal scoring
- **Attention Computation**: O(density × L²d)
- **Memory**: O(L²) for full attention → O(density × L²) for sparse

## Domain-Specific Adaptations

### Language Models (Llama-3.1-8B)
- Use dynamic threshold prediction
- Stride S=8 or S=16
- Tested up to 128k tokens

### Video Understanding (Qwen2-VL-7B)
- Fixed τ = 0.9, S = 16
- Frame rate: 1 fps for 1-hour videos
- Handles both short (11s) and long (1h) videos

### Video Generation (HunyuanVideo)
- Non-causal attention in DiT architecture
- Warmup strategy: 5 steps full attention → XAttention
- Thresholds: τ = 0.9 or 0.95
- Resolution: 720×1280, 129 frames, 50 denoising steps