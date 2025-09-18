# Phase 2: Methodology Extraction - XAttention

## 2.1 Importance Prediction with Antidiagonal Scoring

### Core Insight
The sum of antidiagonal values in attention matrices serves as a computationally efficient proxy for block importance, intersecting all vertical and slash patterns within blocks.

### Mathematical Formulation
For each block of size B×B:
- Select elements along antidiagonal using stride S
- Compute sum: `sum = Σ antidiagonal_elements`
- This sum serves as the importance score for the block

### Advantages
1. **Information Preservation**: Each token contributes to at least one antidiagonal sum
2. **Pattern Detection**: Antidiagonals intersect every possible vertical and slash pattern
3. **Computational Efficiency**: Simple summation vs. complex pooling operations

## 2.2 Threshold Block Selection Algorithm

### Algorithm 1: Block Selection Process

```
Require: Query matrix Q∈R^(L×d), Key matrix K∈R^(L×d), 
        block size B, stride S, head dimension d_h, threshold τ
Ensure: Sparse mask M

1: N_B ← ⌊L/B⌋  // Number of blocks
2: for b = 0 to N_B-1 do
3:   Q_slice ← Q[bB:(b+1)B,:]  // Extract Q block
4:   Q_reshaped ← []
5:   for i = S-1 down to 0 do
6:     Q_reshaped.append(Q_slice[i::S,:])  // Reshape along antidiagonals
7:   end for
8:   K_reshaped ← []
9:   for i = 0 to S-1 do
10:    K_reshaped.append(K[i::S,:])  // Reshape along antidiagonals
11:  end for
12:  A_approx ← Softmax(Q_reshaped × K_reshaped^T / √(d_h·S))
13:  M_b ← find_blocks(A_approx, τ)  // Find blocks based on threshold
14: end for
15: M ← concatenate(M_0, M_1, ..., M_{N_B-1})
```

### Block Selection Function
```
find_blocks(A, τ) = argmin_B{|B| | Σ_{b∈B} Σ_{(i,j)∈b} A_{i,j} ≥ τ}
```

Where:
- A: attention map
- B: set of blocks
- |B|: number of blocks in the set
- τ: predefined threshold

## 2.3 Minimum Threshold Prediction via Dynamic Programming

### Problem Formulation
For H attention heads, define DP table D[h][m]:
- h ∈ {1,2,...,H}: head index
- m ∈ {1,2,...,M}: number of threshold adjustments
- D[h][m]: best performance with m adjustments across first h heads

### Dynamic Programming Recurrence
```
D[h][m] = max(D[h-1][m], P(h, m))
```

Where P(h,m) represents performance when h-th head's threshold is adjusted for m-th time.

### Threshold Adjustment Strategy
```
th(m) = th(m-1) × 0.9
```

Gradual 10% reduction per step to balance computation reduction with accuracy preservation.

### Implementation Details
- Start with τ = 0.9
- Set M = 1000 (maximum adjustments)
- Results in refined thresholds with average value 0.8
- Not mandatory - can use fixed threshold τ = 0.9

## Computational Complexity Analysis

### Baseline Full Attention
- Time: O(L²d) where L = sequence length, d = hidden dimension
- Matrix multiplication: [L, d] × [d, L] → [L, L]

### XAttention Sparse Attention
- Time: O(ρL²d) where ρ = density ratio (typically 6-45%)
- Pattern selection overhead: O(Ld log L) for antidiagonal scoring
- Total: O(ρL²d + Ld log L)

### Comparison with Baselines
- **MInference**: Uses vertical/slash detection with O(L²) overhead
- **FlexPrefill**: Uses pooling with O(L²) overhead
- **XAttention**: Antidiagonal scoring with O(Ld log L) overhead

## Parameter Configuration

### Recommended Settings
- **Stride S**: 8 or 16 (balance between accuracy and efficiency)
- **Threshold τ**: 0.9 (fixed) or use dynamic threshold prediction
- **Block size B**: Typically 64-128 (hardware dependent)
- **Warmup steps**: 5 for video generation (0 for text tasks)

### Density Results
- S=8: 6.89% density at 128k tokens
- S=16: 7.32% density at 128k tokens
- Dynamic threshold: 21.09% average density (better accuracy)