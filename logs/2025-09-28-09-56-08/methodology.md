# XAttention: Methodology

## Overview
XAttention is a three-step process for efficient block sparse attention:
1. **Strided Antidiagonal Scoring**: Score blocks by summing antidiagonal values
2. **Block Selection**: Select high-scoring blocks based on threshold
3. **Block Sparse Attention**: Compute attention only on selected blocks

## 1. Importance Prediction via Antidiagonal Scoring

### Core Concept
The antidiagonal (lower-left to upper-right) sum serves as a proxy for block importance in the attention matrix.

### Mathematical Formulation
For each block of size B×B:
- Define stride parameter S
- Select elements along antidiagonal with stride S
- Compute sum: `sum = Σ antidiagonal_elements_with_stride_S`

### Key Properties
1. **Information Preservation**: Every token contributes to at least one antidiagonal sum
2. **Pattern Detection**: Antidiagonal intersects both vertical and slash patterns within blocks
3. **Efficiency**: Simple summation operation with O(B²/S) complexity

### Implementation Details
```
For each B×B block:
    For i = S-1 down to 0:
        Qreshaped.append(Qslice[i::S,:])  // Reshape along antidiagonals
    For i = 0 to S-1:
        Kreshaped.append(K[i::S,:])      // Reshape along antidiagonals
    Aapprox = Softmax(Qreshaped × Kreshapedᵀ / √(dh·S))
```

## 2. Threshold Block Selection

### Algorithm Flow
1. **Antidiagonal Summation**: Sum elements along antidiagonals within each S×S block
2. **Softmax Normalization**: Apply softmax to get probability distribution over antidiagonals
3. **Block Selection**: Find minimal set of blocks whose cumulative probability exceeds threshold τ

### Mathematical Formulation
```
find_blocks(A, τ) = argmin_B {|B| : Σ_{b∈B} Σ_{(i,j)∈b} A_{i,j} ≥ τ}
```

### Algorithm 1: Block Selection
```
Require: Q∈ℝ^{L×d}, K∈ℝ^{L×d}, block size B, stride S, head dimension dh, threshold τ
Ensure: Sparse mask M

1: NB ← ⌊L/B⌋
2: for b = 0 to NB-1 do
3:     Qslice ← Q[bB:(b+1)B,:]
4:     Qreshaped ← []
5:     for i = S-1 down to 0 do
6:         Qreshaped.append(Qslice[i::S,:])
7:     end for
8:     Kreshaped ← []
9:     for i = 0 to S-1 do
10:        Kreshaped.append(K[i::S,:])
11:    end for
12:    Aapprox ← Softmax(Qreshaped × Kreshapedᵀ / √(dh·S))
13:    Mb ← find_blocks(Aapprox, τ)
14: end for
15: M ← concatenate(M0, M1, ..., M_{NB-1})
```

## 3. Minimum Threshold Prediction

### Problem Formulation
- Model has H attention heads
- Dynamic programming table D[h][m] where:
  - h ∈ {1,2,...,H} represents head index
  - m ∈ {1,2,...,M} represents threshold adjustments
  - D[h][m] stores best performance with m adjustments across first h heads

### Dynamic Programming Recurrence
```
D[h][m] = max(D[h-1][m], P(h, m))
```

Where P(h,m) is performance when adjusting h-th head's threshold for m-th time.

### Threshold Adjustment Strategy
- Start with τ = 0.9
- Reduce by 10% at each step: τ(m) = τ(m-1) × 0.9
- Explore M = 1000 combinations
- Results in refined thresholds with average value ≈ 0.8

### Implementation Details
```
For each head h:
    For each adjustment m:
        Evaluate performance with threshold τ(m)
        Update D[h][m] based on accuracy vs. computation trade-off
    Select optimal threshold for each head
```

## Computational Complexity Analysis

### Time Complexity
- **Antidiagonal Scoring**: O(L²/S) where L is sequence length, S is stride
- **Block Selection**: O(NB log NB) where NB is number of blocks
- **Total**: O(L²/S + NB log NB)

### Space Complexity
- **Memory**: O(L²) for attention matrix (sparse storage reduces to O(L²·density))
- **Overhead**: O(L) for antidiagonal computations

## Parameter Selection Guidelines

### Stride S
- **S=4**: Higher accuracy, lower sparsity
- **S=8**: Balanced performance (recommended)
- **S=16**: Higher sparsity, slight accuracy loss
- **S=64**: Too sparse, significant accuracy degradation

### Threshold τ
- **τ=0.9**: Conservative, higher accuracy, lower sparsity
- **τ=0.95**: Very conservative, near-full accuracy
- **Dynamic**: Per-head optimization yields average τ≈0.8

### Block Size B
- Typically 8×8 or 16×16
- Trade-off between granularity and overhead
- Smaller blocks → finer control but higher overhead

## Integration Requirements

### Model Compatibility
- **Architecture**: Compatible with standard Transformer attention
- **Attention Type**: Works with both causal and non-causal attention
- **Models Tested**: Llama-3.1-8B, Qwen2-VL-7B, HunyuanVideo

### Implementation Notes
- **Training-free**: No model retraining required
- **Plug-and-play**: Direct replacement for attention computation
- **Warmup Strategy**: For video generation, 5-step full attention warmup recommended

## Pseudocode Summary
```
function XAttention(Q, K, V, B, S, τ):
    L, d = Q.shape
    NB = L // B
    M = zeros(L, L)
    
    for b in range(NB):
        # Antidiagonal scoring
        Qslice = Q[b*B:(b+1)*B, :]
        Qreshaped = reshape_antidiagonal(Qslice, S)
        Kreshaped = reshape_antidiagonal(K, S)
        
        # Approximate attention
        Aapprox = softmax(Qreshaped @ Kreshaped.T / sqrt(d*S))
        
        # Block selection
        Mb = select_blocks(Aapprox, τ)
        M[b*B:(b+1)*B, :] = Mb
    
    # Sparse attention computation
    return sparse_attention(Q, K, V, M)
```