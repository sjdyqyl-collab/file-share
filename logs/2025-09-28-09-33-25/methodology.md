# XAttention: Detailed Methodology

## Overview
XAttention is a three-component framework for efficient block-sparse attention:
1. Importance prediction using antidiagonal scoring
2. Block selection based on threshold
3. Dynamic threshold prediction per attention head

## 1. Importance Prediction

### Antidiagonal Scoring Method
For each attention block of size B×B:
- Select elements along antidiagonal using stride S
- Compute sum of these elements as block importance score
- This captures both vertical and slash patterns efficiently

### Mathematical Formulation
Given attention matrix A ∈ ℝ^(L×L):
- Divide into blocks of size B×B
- For each block b, compute:
  `score(b) = Σ_(i,j)∈antidiagonal(b) A[i,j]`

### Pattern Detection
The antidiagonal pattern intersects:
- **Vertical patterns**: Attention between specific query and all keys
- **Slash patterns**: Diagonal attention patterns
- Ensures no crucial patterns are missed

## 2. Block Selection Algorithm

### Algorithm Steps
1. **Antidiagonal Summation**: Compute antidiagonal sums for each S×S sub-block
2. **Softmax Normalization**: Apply softmax to get probability distribution
3. **Threshold-based Selection**: Select minimal blocks exceeding threshold τ

### Formal Definition
```
find_blocks(A, τ) = argmin_B {|B| : Σ_(b∈B) Σ_(i,j)∈b A[i,j] ≥ τ}
```

### Algorithm 1: Block Selection
```
Require: Q ∈ ℝ^(L×d), K ∈ ℝ^(L×d), block size B, stride S, threshold τ
Ensure: Sparse mask M

NB ← ⌊L/B⌋
for b = 0 to NB-1 do
    Q_slice ← Q[bB:(b+1)B,:]
    // Reshape along antidiagonals
    Q_reshaped ← reshape_antidiagonal(Q_slice, S)
    K_reshaped ← reshape_antidiagonal(K, S)
    A_approx ← softmax(Q_reshaped × K_reshaped^T / √(d·S))
    M_b ← find_blocks(A_approx, τ)
end for
M ← concatenate(M_0, M_1, ..., M_{NB-1})
```

## 3. Minimum Threshold Prediction

### Dynamic Programming Approach
For H attention heads with M threshold adjustments:
- Define DP table D[h][m] for best performance with m adjustments over first h heads
- Recurrence: D[h][m] = max(D[h-1][m], P(h,m))
- Threshold update: τ^(m) = τ^(m-1) × 0.9

### Optimization Objective
Maximize accuracy while minimizing computation by:
- Adjusting thresholds per head based on sparsity patterns
- Starting from τ=0.9 and reducing by 10% per step
- Exploring up to M=1000 combinations

## Implementation Details

### Parameters
- **Block Size (B)**: 8 or 16 (recommended)
- **Stride (S)**: 8 or 16 (recommended)
- **Threshold (τ)**: 0.9 default, dynamically optimized
- **Sequence Length**: 4k-256k tokens supported

### Computational Complexity
- **Pattern Selection**: O(L²/S) for antidiagonal scoring
- **Sparse Attention**: O(L²·density) where density ∈ [0.06, 0.55]
- **Overall**: Significantly better than O(L²) full attention

### Memory Efficiency
- Only stores selected blocks in memory
- Reduces memory bandwidth requirements
- Compatible with FlashAttention-style optimizations

## Integration
- **Plug-and-play**: No model retraining required
- **Compatible**: Works with existing Transformer architectures
- **Flexible**: Supports both causal and non-causal attention
- **Scalable**: Effective from 4k to 256k token sequences