# Phase 2: Methodology Extraction - XAttention

## Core Method Overview
XAttention is a plug-and-play framework for block-sparse attention that uses antidiagonal scoring to identify important attention blocks efficiently.

## Detailed Method Components

### 1. Antidiagonal Importance Prediction

#### Mathematical Formulation
For each block of size B×B in the attention map:
- **Antidiagonal Selection**: Select elements along antidiagonal with stride S
- **Scoring Function**: Sum of selected antidiagonal values serves as block importance score
- **Pattern Coverage**: Antidiagonal intersects both vertical and slash patterns

#### Algorithm Steps
1. **Query/Key Reshaping**: 
   - Reshape Q and K matrices along antidiagonals with stride S
   - Q_reshaped = [Q[i::S,:] for i in range(S-1, -1, -1)]
   - K_reshaped = [K[i::S,:] for i in range(S)]

2. **Approximate Attention Calculation**:
   - A_approx = Softmax(Q_reshaped × K_reshaped^T / √(d_h·S))
   - Where d_h is head dimension

### 2. Block Selection Algorithm

#### Formal Definition
```
find_blocks(A, τ) = argmin_B {|B| : Σ_{b∈B} Σ_{(i,j)∈b} A_{i,j} ≥ τ}
```

#### Algorithm 1: Block Selection Process
```
Input: Q∈ℝ^{L×d}, K∈ℝ^{L×d}, block size B, stride S, head dimension d_h, threshold τ
Output: Sparse mask M

1: NB ← ⌊L/B⌋  // Number of blocks
2: for b = 0 to NB-1 do
3:   Q_slice ← Q[bB:(b+1)B, :]
4:   // Reshape along antidiagonals with stride S
5:   Q_reshaped ← reshape_antidiagonal(Q_slice, S)
6:   K_reshaped ← reshape_antidiagonal(K_slice, S)
7:   A_approx ← Softmax(Q_reshaped × K_reshaped^T / √(d_h·S))
8:   M_b ← find_blocks(A_approx, τ)
9: end for
10: M ← concatenate(M_0, M_1, ..., M_{NB-1})
```

### 3. Dynamic Threshold Prediction

#### Problem Formulation
- **Variables**: H attention heads, M threshold adjustments
- **DP Table**: D[h][m] = best performance with m adjustments across first h heads
- **Recurrence**: D[h][m] = max(D[h-1][m], P(h,m))

#### Threshold Adjustment Strategy
- **Initial**: τ_start = 0.9
- **Reduction**: τ(m) = τ(m-1) × 0.9
- **Steps**: M = 1000 adjustments
- **Result**: Average threshold = 0.8

## Computational Complexity Analysis

### Baseline Methods
- **Full Attention**: O(L²d) where L is sequence length, d is hidden dimension
- **Matrix Multiplication**: [L, d, L] for Q×K^T

### XAttention Method
- **Pattern Selection**: O(L²/S) where S is stride
- **Sparse Attention**: O(L²·ρ) where ρ is density (6.2-55.4%)
- **Total Computation**: [L, d, L·ρ] for sparse Q×K^T

### Runtime Comparison
- **Full Attention**: [L, d, L] (baseline)
- **XAttention**: [L, d, L·ρ] + [L, d, L/S] (pattern selection)
- **Example**: For L=256K, ρ=6.89%, S=8 → 13.5× speedup

## Implementation Details

### Parameters
- **Block Size (B)**: Typically 8×8 or 16×16
- **Stride (S)**: {4, 8, 16, 64} - 8 and 16 show best balance
- **Threshold (τ)**: 0.9 (fixed) or dynamically predicted (0.8 avg)

### Model Configurations
- **Llama-3.1-8B-Instruct**: S=8,16 with dynamic threshold
- **Qwen2-VL-7B-Instruct**: S=16, τ=0.9
- **HunyuanVideo**: S=8, τ={0.9,0.95} with 5-step warmup

## Communication Time
- **Pattern Selection**: Minimal overhead (24.9× faster than MInference)
- **Sparse Computation**: Reduced due to lower density
- **No additional communication required** (training-free method)